# MoE 路由预测、通信优化与专家放置研究总结

> 本文总结本次对话中围绕 SGLang、DeepEP、MoE 路由预测、专家权重预取、专家放置、复制、跨数据集泛化和 CUDA 求解器展开的全部工作。日期：2026-08-17。

## 1. 研究目标

整个讨论最终聚焦于三个相互关联的问题：

1. 分析 DeepEP `dispatch` / `combine` 各阶段耗时，尽量让耗时集中在真实数据传输，而不是 tensorize、排序、元数据构造等准备工作。
2. 利用上一层激活预测下一层 Top-K/Top-M 专家，提前加载或预取下一层专家权重。
3. 根据离线路由 trace 优化专家 placement 和 replica，在降低远程通信的同时控制计算不均衡，并验证 placement 的跨数据集泛化能力。

最终形成的总体思路是：

- 离线阶段决定稳定的专家主 placement，尽量降低长期通信成本。
- 第一轮在线 planner 使用上一层预测结果预取部分权重。
- 下一层真实 gate 结果出来后，第二轮轻量 planner 只做少量修正。
- 在线调度在通信等价的 rank 中优先选择计算负载较低的 rank。

## 2. SGLang、All-to-All 与 DeepEP

SGLang 的 MoE All-to-All 并不必然固定使用 DeepEP，而是取决于所选 MoE A2A backend。启用 DeepEP backend 时，MoE token 的 `dispatch` 和 `combine` 使用 DeepEP；其他 backend 可能使用不同实现。

本次讨论中希望比较不同 backend 的性能，并寻找类似 `benchmark_deepep_ht_distribution` 的脚本，重点测量：

- dispatch 前的输入整理和 metadata 构造；
- dispatch 的真实通信；
- expert 计算；
- combine 前的准备；
- combine 的真实通信；
- 同步和 stream 等待。

判断优化是否有效时，不能只看 kernel 时间，还要区分：

- CPU tensorize；
- Host-to-Device 拷贝；
- 图构建；
- placement 求解；
- replica 候选生成；
- replay；
- CUDA 同步造成的隐式等待。

曾观察到如下阶段耗时：

```text
tensorize=18.289s
graph_build=1.744s
primary_replay=0.053s
graph_solve=0.482s
wall=25.461s
```

这说明端到端瓶颈主要不在图求解，而在输入 tensorize 和未单独归因的加载、同步或 Python 处理。

## 3. 上一层激活预测下一层专家

### 3.1 预测任务

设模型真实路由 Top-K 为 `K`，预测器输出 Top-M，允许 `M > K`。评估目标不是要求预测顺序完全一致，而是希望真实 Top-K 尽量包含在预测 Top-M 中，从而提前加载一个覆盖率较高的专家候选集。

预测输入为第 `i` 层激活，预测目标为第 `i+1` 层最终路由专家。

### 3.2 指标定义

- `recall`：每个 token 的真实 Top-K 专家中，有多少比例出现在预测 Top-M 中，再对 token 求平均。
- `full`：真实 Top-K 全部被预测 Top-M 覆盖的 token 比例。

例如真实 Top-K 有 8 个专家，预测 Top-16 命中 7 个，则该 token recall 为 `7/8`，但 full 为 0。

因此：

- recall 更适合衡量平均可预取比例；
- full 更适合衡量是否完全不需要补载；
- 权重预取场景中，full 往往比 recall 更苛刻，也更直接影响 fallback 次数。

### 3.3 初始结果

初始 Top-16 验证结果：

| 层对 | tokens | recall | full |
|---|---:|---:|---:|
| L0 -> L1 | 1342 | 83.33% | 34.87% |
| L1 -> L2 | 1342 | 79.84% | 27.72% |
| L2 -> L3 | 1342 | 95.05% | 72.13% |

这说明层间可预测性差异很大，不能只看平均结果。L1 -> L2 是当前主要短板。

### 3.4 训练方案讨论

讨论过的改进包括：

- 使用 LayerNorm 后的激活；
- 冻结目标层 gate；
- 在目标 gate 前增加可训练的两层 residual MLP；
- 直接蒸馏目标 router 的概率分布；
- 使用多标签 BCE、KL、排序损失和 Top-K coverage 相关损失；
- 对难负样本和靠近 Top-K 边界的专家加权；
- 按层分别训练，避免不同层难度互相干扰；
- 增加 `M`，用预取空间换 coverage；
- 使用模型真实 dispatch 结果而不是裸 gate logits 作为标签。

这里的“两层 residual MLP”是对预测输入或 gate logits 的修正，例如：

```text
pred_logits = frozen_next_gate(h + W2(act(W1(LN(h)))))
```

`W1`、`W2` 是预测器参数，不是 MoE 专家权重的增量。它和 LoRA 都是在冻结主模型基础上增加可训练参数，但独立预测头、residual adapter 和 LoRA 的参数化、初始化和归纳偏置不同，不能只因为“都增加额外权重”就认为训练行为完全相同。

### 3.5 为什么不能只用裸 logits Top-K

模型真实 router 可能还包含：

- softmax 或 sigmoid；
- expert bias；
- group routing；
- capacity 或 token dropping；
- route scaling；
- dispatch 前的其他修正。

因此 `next_gate(hidden)` 的裸 logits Top-K 不一定等于真实 dispatch 专家。更可靠的训练标签应复用模型实际 router 的最终选择逻辑。

直接对真实 router 概率分布做 cross-entropy/KL distillation 通常比只拟合 hard Top-K 更平滑，但最终仍应使用 Top-M recall/full 作为验证指标，因为预取关心的是候选集合覆盖，而不是整体分布误差。

### 3.6 尝试过并删除的训练方案

Residual predictor 的一次结果：

| 层对 | recall | full |
|---|---:|---:|
| L0 -> L1 | 83.41% | 34.72% |
| L1 -> L2 | 79.94% | 29.28% |
| L2 -> L3 | 94.95% | 70.12% |

该方案没有明显提升，后来按要求删除。

复现 PROBE 风格训练器后的一次结果：

| 层对 | recall | full |
|---|---:|---:|
| L0 -> L1 | 84.62% | 37.78% |
| L1 -> L2 | 79.85% | 29.43% |
| L2 -> L3 | 95.41% | 73.47% |

提升有限，远未达到 PROBE 文章中 `2 * Top-K` 接近 100% 的覆盖率。随后尝试的 `probe-topk` 效果更差，也按要求删除。

这说明继续堆叠 loss 并不是主要突破口。更值得优先确认的是：

1. 输入激活位置是否与论文一致；
2. 标签是否为模型最终真实路由；
3. 模型、数据分布、Top-K 和论文设置是否一致；
4. 是否按 prefill/decode、层和 token 位置分别建模；
5. 文章是否使用了更强的时序或跨层特征。

### 3.7 数据集支持与常见问题

训练和评估脚本需要支持任意 Hugging Face 数据集或本地数据，而不应只硬编码 ShareGPT、HumanEval、Summary。ShareGPT、HumanEval 和摘要数据集只是例子。

`max-length` 表示 tokenizer 后每条样本允许的最大 token 长度。超过部分截断，不足部分按 batch 策略 padding。它影响：

- 每次 forward 的 token 数；
- 显存；
- 采集到的 prefill/decode 分布；
- 训练样本中的长上下文比例。

遇到过的数据加载错误：

```text
datasets.exceptions.DataFilesNotFoundError:
No (supported) data files found in anon8231489123/ShareGPT_Vicuna_unfiltered
```

这通常表示该 Hub 仓库不是可直接由 `datasets.load_dataset()` 推断的数据仓库，或需要指定具体文件、配置、revision/remote code。脚本应允许显式传入 dataset name、config、split、data files 和文本字段。

加载权重或打印 `[load] moving model to cuda` 后长时间无输出，常见原因包括：

- CPU -> GPU 权重搬运耗时；
- GPU 显存不足导致抖动或 OOM 前等待；
- dtype 转换；
- 多 shard 权重加载；
- CUDA 首次初始化；
- 同步点之前没有进度日志。

训练 loss 持续增大时应优先检查：

- 学习率过高；
- label/logit 定义不一致；
- KL 温度和 reduction；
- 正负样本比例；
- 梯度爆炸；
- 混合精度溢出；
- scheduler 是否方向错误。

## 4. 为什么单专家复制不一定减少通信

一个 token 的 Top-K 中可能有多个专家位于同一个远程 rank。DeepEP 的通信量若按“访问的远程 rank 数”计算，那么只把其中一个专家复制到 source rank 后，token 仍然需要访问原远程 rank 上的其他专家，远程 rank-copy 数不变。

因此复制收益必须按完整 Top-K bundle 评估，而不能只按单专家热度评估。

有价值的复制动作主要有两类：

- `single`：复制单个专家就能消除一次远程 rank 访问；
- `bundle-closure`：复制同一 bundle 在某个远程 rank 上的全部相关专家，才能真正关闭该远程访问。

这也解释了为什么 token 数可能远小于 bundle 数：脚本中的 `tokens` 可能是聚合 count 总数或当前批次 token 数，而 `bundles/candidates` 可能枚举了 source、Top-K 组合、目标 rank、闭包集合和候选动作，二者不是同一概念。

## 5. 通信优化与计算均衡

### 5.1 先优化什么

本次讨论最终采用：

- 离线优化通信：决定稳定的主 placement。
- 在线优化计算：真实 token 到来后，在可选副本之间选择当前负载较低的 rank。

原因是 token 的未来真实路由不能完全离线确定，而长期专家共现关系可以从 trace 估计。

### 5.2 通信等价 rank

当某个 expert 有多个副本时，对一个 token 而言，下列 rank 具有较低或相同的新增通信成本：

1. token 的 source rank；
2. 该 token 因其他 Top-K expert 已经必须访问的 rank；
3. 其他包含该专家副本的 rank。

在线调度策略应先最小化新增通信 rank 数，再在通信成本相同的 rank 中选择当前计算负载最低者。

### 5.3 计算不均衡指标

当前输出中的 `comp` 是 rank 粒度，而不是 expert 粒度。常用定义是：

```text
max_rank_load / average_rank_load
```

专家级负载不均衡会通过 placement 聚合成 rank 级负载不均衡，但最终设备执行时间更直接受 rank 级最大负载影响。

只优化通信可能把多个热门专家聚集到同一 rank，使 `comp` 显著变差。对此讨论过三种约束方式：

- 硬约束：`max_compute_inflation <= cap`；
- 软惩罚：通信目标加上计算方差或最大负载惩罚；
- 通信中性修正：只接受不改变通信目标、但能降低计算不均衡的 swap。

如果要求 `comp` 尽可能接近 graph plan 之前的水平，硬约束最清晰；如果完全关闭计算约束，可以获得更大的通信收益，但计算风险很高。

## 6. Pairwise Graph、Hypergraph 与 Replica

### 6.1 Pairwise graph

将每个专家看作顶点。一个 Top-K bundle 中任意两个专家共同出现一次，就为对应边增加权重。求解目标是在每个 rank 容量受限的条件下，把高共现专家放在一起，降低跨 rank edge cut。

优点：

- 图构建和 swap delta 容易增量更新；
- GPU 上可用 dense tensor 高效计算；
- 对 source-rank 噪声不敏感；
- 实际实验中通常比当前 hypergraph 更快，且跨数据集效果更稳。

缺点：

- Top-K bundle 被投影成若干 pair，丢失高阶结构；
- edge cut 不等于真实 DeepEP remote rank-copy 数。

### 6.2 Source-aware hypergraph

每个真实 Top-K bundle 是一个 hyperedge。原始目标直接计算 token source 之外的不同 destination rank 数：

```text
C_source(P) = sum_t count_t * |{P(e) != source_t : e in TopK(t)}|
```

它更贴近特定 trace 的 DeepEP remote replay，但也把 source-rank 分布编码进 placement。source rank 在不同数据集、batch 组成或 EP 重分片后可能变化，因此容易过拟合训练 trace。

### 6.3 Source-agnostic hypergraph

为改善跨数据集泛化，新增了不依赖 source rank 的目标：

```text
C_agnostic(P) = sum_t count_t * |{P(e) : e in TopK(t)}|
```

它保留完整 Top-K hyperedge，但只优化一个 bundle 被分散到多少 destination rank，具有 source-rank permutation invariance。

该模式通过统一参数启用：

```text
--hypergraph-objective source-aware|source-agnostic
```

默认仍为 `source-aware`，以保持已有命令兼容。

结果对象分别保存：

- `initial_remote` / `final_remote`：真实 source-aware DeepEP remote replay；
- `initial_objective` / `final_objective`：本次实际优化的 hypergraph objective。

不能把 source-agnostic cardinality 伪装成 remote，否则会误判真实通信收益。

### 6.4 为什么 hypergraph 可能不如 pairwise

当前实验中 hypergraph 不仅更慢，有时还不如 pairwise。可能原因包括：

- source-aware 目标过拟合当前 trace 的 source-rank 分布；
- pairwise affinity 是更平滑的统计量，跨数据集方差更小；
- 当前求解器是单次 expert swap 的局部搜索；
- 高阶目标存在更多局部最优；
- pairwise seed 已经较好，后续 hypergraph swap 可能牺牲泛化性；
- 优化目标与最终 replica/在线调度策略不完全一致。

Source-agnostic hypergraph 正是用于验证“问题是否主要来自 source-rank 过拟合”。

### 6.5 Replica

主 placement 规定每个 logical expert 的 home rank。Replica planner 再决定哪些 expert 在哪些 rank 增加副本。

此前 redundant 权重在实验中通常只有不到 1% 的 remote 改善，原因包括：

- replica 数量太少；
- 复制单专家无法闭合整个 bundle；
- EP=4 时基线本身远程 rank 数上限较低；
- placement 已吸收大部分容易获得的收益；
- 热点 bundle 分散，少量副本覆盖不足。

在 EP=32 下，远程 rank 数和可优化空间通常更大，因此可以使用 EP=4 trace 重分片模拟 EP=32，但模拟依赖 source-rank 扩展假设，必须与真实 EP=32 replay 区分。

## 7. 求解器演进与性能

### 7.1 CPU 版本

最初图构建和候选求解主要在 CPU/Python 上进行。由于 bundles 和 candidates 数量大，端到端可能需要数分钟，不满足运行时几百微秒或 5ms 以内的目标。

### 7.2 CUDA-fast / Triton 版本

后来实现 CUDA-fast planner，将以下工作搬到 GPU：

- expert demand 统计；
- pairwise edge 构建；
- hypergraph move delta；
- pair correction；
- placement swap 评分；
- replica replay。

关键优化包括：

- tensor 化输入；
- 增量更新受 swap 影响的 bundle；
- 避免每轮全量重建 hypergraph；
- 只把少量最终决策搬回 CPU；
- pinned host buffer 和异步 H2D；
- CUDA 上批量计算所有 swap candidate。

Replica 求解没有完全自然地转成单个 Triton kernel，因为它包含候选集合更新、容量约束、动作序列依赖和小规模离散选择。适合 GPU 的部分已 tensorize，最终少量控制流仍可留在 host。

### 7.3 Rounds

增加 rounds 只保证搜索更多局部 swap，不保证持续获得收益，也不保证全局最优。`until convergence` 表示直到不存在改善目标的单次跨 rank expert swap。

“基于单次 expert swap 的局部最优”是指：当前 placement 中任何一次两专家交换都不能改善目标，但同时交换多对专家、先接受一步变差再变好，仍可能到达更优 placement。

获得全局最优通常需要 ILP、CP-SAT、完整 branch-and-bound 或指数级枚举，规模大时不可接受。实用方案是：

- 多随机 restart；
- spectral/pairwise seed；
- 小规模 k-swap；
- tabu/annealing；
- 只对热点专家做精确局部求解。

## 8. 指标说明

离线 placement 输出中常见字段：

| 指标 | 含义 |
|---|---|
| `tokens` | trace 中 token count 总量 |
| `bundles` | 聚合后的 `(source_rank, Top-K)` 组合数 |
| `edges` | pairwise graph 中非零专家边数 |
| `baseline` | 基线 placement 的真实 remote rank-copy 数 |
| `graph` | pairwise graph placement replay 的真实 remote 数 |
| `hypergraph` | hypergraph placement replay 的真实 remote 数 |
| `hyper_obj` | 所选 hypergraph objective 的值 |
| `replica` | 增加副本后的真实 remote 数 |
| `graph_delta` | `graph / baseline - 1` |
| `hyper_delta` | `hypergraph / baseline - 1` |
| `obj_delta` | `hyper_obj / baseline_obj - 1` |
| `replica_delta` | `replica / baseline - 1` |
| `cut` | pairwise edge cut 的初始值和最终值 |
| `rounds` | pairwise swap 轮数 |
| `hyper_rounds` | hypergraph 改善 swap 数 |
| `balance_rounds` | 通信中性计算均衡 swap 数 |
| `actions` | replica 动作数 |
| `solve` | 求解阶段耗时 |
| `replay` | 使用最终 placement 重放 trace 的耗时 |

跨数据集 replay 中：

| 指标 | 含义 |
|---|---|
| `transfer` | 在训练数据集求得 placement 后，在测试数据集上的 remote |
| `oracle` | 直接在测试数据集求得 placement 的 remote |
| `transfer_delta` | transfer 相对测试集 baseline 的变化 |
| `oracle_regret` | `transfer / oracle - 1` |
| `retained` | 训练 placement 在测试集上保留了多少 oracle 可获得收益 |
| `comp a->b->c` | baseline、transfer、oracle 的 rank 计算不均衡 |

`hyper_gain` 字段已按要求删除，`planned` 命名改为 `replica`，避免含义不清。

## 9. 两轮 Planner 与权重预取

设第 `i` 层用于预测第 `i+1` 层：

### 第一轮 planner

- 输入：上一层激活产生的预测 Top-M；
- 目标：高 recall、低预取体积；
- 行为：预取一部分最可能使用的 expert 权重；
- 允许保守，即多取少量专家来降低 miss。

### 第二轮 planner

- 输入：第 `i+1` 层真实 gate Top-K；
- 目标：纠正预测错误并平衡当前计算；
- 行为：尽量复用已加载权重，只允许每个 rank 调整 1 到 2 个权重；
- 优先顺序：source rank、已访问 rank、其他副本 rank；
- 在通信等价的 rank 中选计算负载最低者。

两个 planner 可以不同。第一轮需要概率、置信度和预取预算；第二轮只需要处理 miss 和少量负载修正，因此应更简单，可实现为受限候选上的贪心选择，而不必再次运行完整全局 placement。

## 10. 跨数据集泛化

### 10.1 验证方法

要验证数据集 A 上求得的 placement 是否适用于 B：

1. 在 A 上求 placement `A.pt/json`；
2. 在 B 上独立求 oracle placement `B.pt/json`；
3. 冻结 A placement，在 B trace 上 replay；
4. 比较 B baseline、A -> B transfer 和 B oracle；
5. 交换训练/测试数据集，形成完整 cross-dataset matrix；
6. 至少按层统计 remote、regret、retained gain 和 comp。

只看训练集收益不能说明泛化性。

### 10.2 Robust multi-dataset solver

为提高泛化性，加入多数据集 robust solver：

- 每个数据集单独保留真实 Top-K trace；
- 按 baseline 或 token 数归一化，避免大数据集仅因 token 多而主导；
- 优化平均 normalized objective 和 worst-case normalized objective；
- 通过 multiplicative reweighting 提高当前最差数据集的权重；
- 可选 `max_compute_inflation` 硬约束；
- 可使用 `--no-compute-limit` 完全关闭计算约束。

有计算上限 `1.2x` 时的一次结果：

```text
sharegpt:  remote -1.8%, max compute inflation 1.18x
humaneval: remote -0.5%, max compute inflation 1.20x
summary:   remote -1.2%, max compute inflation 1.20x
```

关闭计算限制后：

```text
sharegpt:  remote -26.9%, max compute inflation 2.57x
humaneval: remote -33.4%, max compute inflation 3.94x
summary:   remote -33.7%, max compute inflation 3.74x
```

这组结果明确展示了通信与计算的 Pareto trade-off：大幅通信收益是以严重 rank 负载不均衡为代价的。

### 10.3 Held-out 数据集现象

使用 ShareGPT + Summary 训练、HumanEval 测试时：

- 各层 transfer remote 通常比 baseline 下降约 9% 到 42%；
- 与 HumanEval oracle 相比仍有约 8% 到 50% regret；
- retained gain 随层变化很大，约 26% 到 82%；
- transfer comp 通常明显优于 oracle comp，但部分层仍显著高于 baseline。

这说明 placement 存在一定迁移能力，但后半层和数据集特有路由仍然明显。

### 10.4 改善泛化的优先方向

优先级从高到低：

1. 使用 source-agnostic Top-K hypergraph，去除不稳定 source-rank 相关性；
2. 使用多数据集 normalized robust objective；
3. 保留 pairwise affinity 作为稳定 seed；
4. 加入 held-out validation，而不是按训练目标选最终轮次；
5. 对层、prefill/decode 和请求类型分别统计稳定性；
6. 只对跨数据集稳定的热点结构做静态 placement，剩余部分交给在线调度。

## 11. 与论文方案的关系

### 11.1 PROBE（`2602.00509v2.pdf`）

讨论重点是上一层特征预测下一层路由。文章报告在 `M = 2K` 时覆盖率接近 100%，而当前复现结果明显较低。

当前差距更可能来自特征位置、真实 router 标签、训练数据和模型配置，而不是简单少用了某个 loss。

### 11.2 GRACE-MoE（`2509.25041v4.pdf`）

讨论中归纳的 GRACE 风格通信思路是：

- 根据专家共现构建 affinity graph；
- 使用 spectral clustering 得到初始专家分组；
- 在拓扑和容量约束下放置专家；
- 使用统计稳定的 affinity，而不是完全绑定单次 token source；
- 再结合副本或运行时调度降低通信。

“Spectral clustering 生成 GRACE 风格初始 placement”指先对专家 affinity matrix 做谱嵌入，再将相近专家聚为 rank group，作为局部搜索初始解。

为了避免方法与 GRACE-MoE 过于相似，本项目不把 pairwise spectral clustering 当最终方法，而是把它作为 baseline/initializer；核心差异方向是：

- 使用真实 Top-K hypergraph；
- source-agnostic 高阶目标；
- 两轮预测/真实路由 planner；
- 通信等价 rank 上的在线计算均衡；
- robust cross-dataset objective。

## 12. 当前代码状态

主要文件：

```text
python/sglang/srt/eplb/co_routing_graph_solver.py
python/sglang/srt/eplb/cuda_fast_co_routing_planner.py
benchmark/solve_co_routing_graph.py
benchmark/solve_robust_co_routing_graph.py
test/registered/unit/eplb/test_co_routing_graph_solver.py
test/registered/unit/eplb/test_cuda_fast_co_routing_planner.py
test/registered/unit/eplb/test_robust_co_routing_graph.py
```

当前已实现：

- pairwise graph placement；
- source-aware hypergraph refinement；
- source-agnostic hypergraph refinement；
- CPU 和 CUDA/Triton 两条路径；
- 增量 move delta 和 pair correction；
- communication-neutral balance swaps；
- replica planner 和 replay；
- cross-dataset transfer replay；
- robust multi-dataset solver；
- `--no-compute-limit`；
- `--hypergraph-objective source-aware|source-agnostic`；
- 终端与 JSON 中分别输出真实 remote 和优化 objective。

清理过的无效内容：

- 删除未被使用的 `_InputTrace.raw` 长期字段；
- 删除 `_solve_layer()` 中随后立即被真实 gate 覆盖的临时 `gate`；
- 删除效果不佳的额外 residual predictor 方案；
- 删除效果更差的 `probe-topk` 方案；
- 删除 `hyper_gain` 输出字段；
- 将含义模糊的 `planned` 改为 `replica`。

## 13. 验证状态

已完成：

- Ruff format；
- Ruff check；
- `py_compile`；
- `git diff --check`；
- source-aware 与 source-agnostic 各 200 个随机 trace 的局部最优校验；
- 250 个 source-rank permutation invariance 随机案例；
- 确定性 destination cardinality 测试；
- CPU/CUDA 对齐测试代码已加入。

本机缺少 `pytest`、`torch`、`triton` 和 `orjson`，因此本轮未在本机真正执行项目 pytest 和 CUDA kernel。CUDA/Triton 路径仍需在 GPU 环境执行注册单测和真实 trace benchmark。

## 14. 运行命令

### 14.1 单数据集 source-agnostic hybrid

```bash
PYTHONPATH=python python benchmark/solve_co_routing_graph.py \
  --input trace.pt \
  --num-ranks 32 \
  --source-ep 4 \
  --device cuda \
  --planner cuda-fast \
  --placement-mode hybrid \
  --hypergraph-objective source-agnostic \
  --hypergraph-until-convergence
```

### 14.2 对照 source-aware

```bash
PYTHONPATH=python python benchmark/solve_co_routing_graph.py \
  --input trace.pt \
  --num-ranks 32 \
  --source-ep 4 \
  --device cuda \
  --planner cuda-fast \
  --placement-mode hybrid \
  --hypergraph-objective source-aware \
  --hypergraph-until-convergence
```

### 14.3 ShareGPT + Summary robust 训练

```bash
PYTHONPATH=python python benchmark/solve_robust_co_routing_graph.py \
  --input sharegpt_ep32.pt \
  --input summary_ep32.pt \
  --num-ranks 32 \
  --device cuda \
  --hypergraph-objective source-agnostic \
  --robust-rounds 8 \
  --swaps-per-round 2 \
  --no-compute-limit \
  --save-placement sharegpt_summary_source_agnostic.json
```

### 14.4 HumanEval held-out replay

```bash
PYTHONPATH=python python benchmark/solve_co_routing_graph.py \
  --input humaneval_ep32.pt \
  --num-ranks 32 \
  --device cuda \
  --load-placement sharegpt_summary_source_agnostic.json \
  --replay-only
```

### 14.5 建议的完整对比矩阵

至少运行以下组合：

```text
pairwise
hybrid + source-aware
hybrid + source-agnostic
robust source-aware
robust source-agnostic
```

并分别报告：

```text
真实 remote delta
所选 objective delta
oracle regret
retained gain
rank compute imbalance
tensorize / graph build / solve / replay / wall time
```

## 15. 下一步实验优先级

1. 在真实 CUDA 环境跑新增单测，确认 Triton 的 source-agnostic 分支与 CPU 完全一致。
2. 对同一批 trace 运行 pairwise、source-aware hypergraph 和 source-agnostic hypergraph，比较真实 remote、held-out regret 和耗时。
3. 把 source-agnostic 作为 robust solver 的目标，验证是否改善 ShareGPT/Summary -> HumanEval 的 retained gain。
4. 保持同一 compute cap 比较不同通信目标，避免用计算恶化换通信收益却误判算法更优。
5. 单独统计 prefill 和 decode；混合统计可能掩盖两者完全不同的路由分布。
6. 若 source-agnostic 仍不如 pairwise，优先保留 pairwise 主 placement，把真实 Top-K hypergraph 用于 replica bundle-closure 和第二轮轻量 planner，而不是继续增加复杂 loss 或全局求解器。

## 16. 核心结论

1. 单专家热度不是正确的通信优化单位，真实收益取决于完整 Top-K bundle 是否减少了 destination rank 数。
2. 离线 placement 和在线负载调度应分工：离线抓稳定共现，在线处理真实负载。
3. Source-aware hypergraph 更贴近单份 trace，但可能过拟合 source-rank；pairwise 更平滑；source-agnostic hypergraph用于保留高阶结构同时去除 source 噪声。
4. 通信和计算必须同时报告。关闭计算约束能获得约 27% 到 34% 的通信下降，但可能带来接近 4 倍的最坏计算膨胀。
5. 预测器的主要问题不是 loss 数量，而是特征位置和训练标签是否与模型真实 router 一致。
6. 当前最值得验证的组合是：pairwise 稳定 seed + source-agnostic hypergraph refinement + 通信等价 rank 上的在线计算均衡。
