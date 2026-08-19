# GRACE-Refine 与受约束专家复制

本文档描述当前仓库中面向 MoE Expert Parallelism（EP）的离线专家放置方案。主路径从 GRACE placement 出发，用完整 Top-K bundle 做 source-aware refinement，最后可选地增加少量 hot-expert 副本。

当前主路径是：

```text
routing trace
    -> expert affinity graph
    -> GRACE hierarchical grouping
    -> exact group-to-rank assignment
    -> constrained single-expert moves
    -> sparse exact pair-swaps
    -> optional constrained hot-expert replication
    -> communication/congestion/compute metrics
```

该方案是离线模拟和 placement 生成器，不会自动把保存的副本 placement 接入在线推理。实际收益还需要在线权重加载、token dispatch 和副本结果合并逻辑配合。

## 1. 输入模型

每条路由记录表示一组具有相同来源和 Top-K 专家的 token：

```text
(source_rank, topk_experts, count)
```

- `source_rank`：token 发起 MoE 请求的 EP rank。
- `topk_experts`：该 token 选择的完整 Top-K 专家集合。
- `count`：相同 bundle 的聚合次数。

compact `.pt` trace 直接以 NumPy array 进入求解器，避免为每个 bundle 创建 Python 对象。`--optimizer-bundles 0` 使用全部 bundle；大于 0 时，对 compact trace 做固定随机种子的可复现采样。

如果采集 trace 的 EP size 与目标 EP size 不同，`compare_grace.py` 会在目标 EP 是源 EP 整数倍时重分 source rank 和 count。

## 2. 优化指标

设 bundle `b` 的来源为 `s_b`，其 Top-K 专家最终访问的 rank 集合为 `D_b`。主要通信目标为：

```text
remote = sum_b count_b * |D_b - {s_b}|
```

同一 bundle 在同一个 remote rank 上访问多个专家只产生一个 remote destination，因此只计一次。这个定义比逐专家统计 remote 更接近实际 Top-K token dispatch。

输出指标包括：

| 指标 | 含义 |
| --- | --- |
| `remote` | 所有 bundle 的 remote destination 总量 |
| `weighted` | 按节点内/节点间代价加权后的 remote |
| `max-pair` | 任意 source-destination rank pair 的最大流量 |
| `max-ingress` | 任意目标 rank 的最大总入站流量 |
| `max-egress` | 任意来源 rank 的最大总出站流量 |
| `comp` | 最大计算负载 / 平均计算负载 |
| `experts/rank` | 各 rank 权重数量的最小值和最大值 |
| `solve-ms` | 生成该结果所需的累计求解时间 |

在单个 NVLink/NVSwitch 超节点中通常设置 `--ranks-per-node` 等于 `--num-ranks`，并使用 `--rdma-cost 1`。此时 `remote` 与 `weighted` 相同，但 `max-pair` 和 `max-ingress` 仍可反映链路及端口热点。

## 3. Baseline

Baseline 按专家编号连续、尽量等量地分配专家，不执行图聚类或 trace 优化。它用于回答“不应用 placement plan 时通信量是多少”。

Baseline 的 `solve-ms` 固定为 0，其他指标使用与优化方案相同的 trace 和计算方法。

## 4. GRACE 初始 Placement

### 4.1 Affinity graph

每个专家是一个顶点。两个专家同时出现在一个 Top-K bundle 中时连接一条边，边权累加该 bundle 的 `count`：

```text
affinity(i, j) = sum count of bundles containing both i and j
```

因此，高 affinity 表示两个专家经常被同一批 token 同时访问。把它们放在同一个 rank，可以合并 remote destination。

### 4.2 两级谱聚类

GRACE 使用归一化 affinity matrix 的谱嵌入进行层级分组：

1. 所有专家先聚类到 node groups。
2. 每个 node group 内再次聚类到 GPU/rank groups。

在单超节点中，第一层只有一个 node group，核心工作发生在 GPU/rank 分组阶段。

默认允许每个 rank 的专家数在理想数量附近非均匀变化，范围由 `--grace-ratio` 控制。非均匀分组可以给 affinity 较密集的 group 更多容量。

使用 `--grace-equal-experts` 时：

- 专家总数必须能被 rank 数整除。
- 每个 rank 的专家数量严格相同。
- 谱聚类后通过移出低组内 affinity 专家、补齐不足 group 和等量 pair-swap 恢复固定大小。

默认 GRACE 只利用专家共同激活关系。设置 `--grace-source-affinity-weight` 后，聚类矩阵还会加入同 source 需求重叠：

```text
A(i,j) = coactivation(i,j) + weight * source_overlap(i,j)

source_excess(i,s) = max(demand(i,s) - mean_source_demand(i), 0)
source_overlap(i,j) = sum_s min(source_excess(i,s), source_excess(j,s))
```

这使经常由同一个 source rank 偏好、但不一定出现在同一个 Top-K bundle 中的专家也倾向进入同组。减去每个专家的 source 平均需求可以避免把在所有 rank 上均匀热门的专家误判成具有 locality。`weight=0` 完全保持原 GRACE；该项只改善 group 成员选择，后续 Exact group-to-rank assignment 仍负责把固定 groups 映射到物理 rank。

## 5. GRACE-Refine

`--grace-refine` 以 GRACE placement 为 seed，不从头重新做 hypergraph placement。它依次执行 group-to-rank assignment、单专家 move 和 pair-swap。

### 5.1 Exact group-to-rank assignment

首先保持每个 GRACE group 的成员完全不变，只决定每个 group 对应哪个物理 rank。

对 group `g` 和物理 rank `r` 构造精确 source-aware cost：

```text
cost[g, r] = group g 放到 r 后产生的 bundle remote
```

然后用 Hungarian assignment 求一对一最小成本匹配。这一步只是整体重命名 group：

- 不拆分 group。
- 不改变每个 rank 的专家数量。
- 不改变每个 group 的计算需求。
- 可以降低 source-aware remote。

### 5.2 受约束单专家 move

接下来允许一个专家从旧 rank 移到目标 rank：

```text
old rank: expert count - 1, compute load - demand(expert)
new rank: expert count + 1, compute load + demand(expert)
```

这不是复制，旧 rank 不再保留该专家。

候选 move 必须满足：

1. 旧 rank 移出后不低于最小专家容量。
2. 新 rank 移入后不超过最大专家容量。
3. 最大计算负载不超过当前值与配置上限中的较大者。
4. remote 精确下降；或者在允许的 remote budget 内改善计算负载。

GRACE-refine 当前传入 `remote_budget=0`，所以不会用更多 remote 换计算均衡。`--grace-refine-rounds` 控制 move refinement 轮数。

容量范围由以下规则决定：

- `--grace-equal-experts`：capacity ratio 强制为 0，每个 rank 数量固定，单专家 move 实际不可行。
- 显式设置 `--grace-refine-capacity-ratio`：使用该值。
- 否则沿用 `--grace-ratio`。

例如 64 个专家、8 个 rank，平均 8 个专家。`capacity_ratio=0` 要求每 rank 恰好 8 个；较大的 ratio 才允许一个 rank 少一个、另一个 rank 多一个。

### 5.3 稀疏 exact pair-swap

Pair-swap 交换两个不同 rank 上的专家，因此不改变各 rank 的专家数量：

```text
rank A: remove expert x, add expert y
rank B: remove expert y, add expert x
```

为了限制求解时间，算法不是枚举所有跨 rank expert pair：

1. 先计算每个专家移动到各 rank 的近似 remote delta。
2. 对每个目标 rank，只保留 `--grace-refine-partners` 个候选 partner。
3. 对候选 pair 使用完整 Top-K bundle 重新计算 exact delta。
4. 每轮对未被锁定的专家应用可接受 swap。
5. 最多执行 `--grace-refine-swaps` 轮。

接受顺序为：

- 优先接受 `remote` 严格下降的 swap。
- `remote` 相同时，接受降低 `max-ingress/max-pair/max-egress` 的 swap。
- 通信指标完全相同时，可以接受改善最大计算负载的 swap。

默认不允许 swap 增加最大计算负载。加上：

```bash
--grace-refine-allow-load-worsening
```

后，remote 下降的 swap 可以增大计算负载。这个参数只是取消负载否决条件，不会为了让负载变差而接受没有通信收益的 swap。

如果仍需要计算上限，可以显式指定：

```bash
--grace-refine-swap-compute-limit 1.25
```

未指定时，`allow-load-worsening` 没有隐含的 `1.25x` 上限。指定上限后，如果当前 seed 已经超过上限，swap 不得继续增加当前最大负载，但仍可以逐步改善它。

该阶段是有界稀疏局部搜索，不包含三专家 cycle、ejection chain 或 exhaustive all-pair search，因此不保证全局最优。

## 6. 受约束 Hot-Expert Replication

`--grace-replication` 在单副本 placement 之后增加少量权重副本：

- 启用 `grace-refine` 时，以 refined placement 为 primary seed。
- 否则以 GRACE placement 为 primary seed。

### 6.1 路由模型

当前模拟器使用确定性的 source-local-secondary 策略：

```text
if source rank has a replica of expert:
    route to the source-local replica
else:
    route to the expert's primary rank
```

第一个 rank 始终是 primary。新增副本只服务来自自身 rank 的请求，不使用 weighted-random、least-load 或跨 rank 副本选择。

这个限制带来三个性质：

- 新副本不会增加 bundle remote。
- 计算迁移量可由 `demand[expert, source_rank]` 精确得到。
- 求解和最终路由是确定性的，不依赖在线负载预测。

### 6.2 候选和 exact gain

首先统计：

```text
source_demand[expert, rank]
```

并选择总 demand 最高的 `--grace-replication-hot-experts` 个专家。对于候选 `expert e @ source rank s`，exact gain 是新增本地副本后从 bundle destination set 中消失的 remote rank 数量。

只有当 `e` 当前所在的 destination rank 在该 bundle 中只出现一次时，把 `e` 路由到本地才会减少 remote；如果同一个 bundle 的另一个 Top-K 专家仍在该 remote rank，remote destination 不会消失。因此算法计算完整 bundle-level gain，而不是简单使用 expert token 数。

第一次通过分块 NumPy 扫描计算全部候选的 exact gain。每增加一个副本，只对受影响的同 source、同 primary bundle 增量更新 gain，不为每个候选重放完整 trace。

### 6.3 约束

候选必须同时满足：

1. `gain > 0`，即 remote 严格降低。
2. 总副本数不超过 `--grace-replication-budget`。
3. 目标 rank 的额外权重数不超过 `--grace-replication-max-extra-per-rank`。
4. 目标位于该专家按 exact gain 排名前 `--grace-replication-candidates` 的 rank 中。
5. 新的最大计算负载不超过：

```text
max(current max load, average load * replication compute limit)
```

其中计算上限由 `--grace-replication-compute-limit` 设置。

### 6.4 贪心选择

每一步按以下顺序选择最佳副本：

```text
1. remote gain 最大
2. 新的最大计算负载更低
3. 搬移到本地的 source demand 更大
4. expert id 和 rank id，保证结果确定
```

添加副本后：

- primary 权重仍然保留。
- 该 source rank 的此专家请求改由本地副本处理。
- primary rank 的计算负载减少相同 demand。
- replica rank 的计算负载增加相同 demand。
- remote 只减不增。

达到总预算或不存在满足约束的正收益候选时停止，因此实际增加的副本数可能小于预算。

## 7. 完整运行示例

采集单机 8 卡 trace：

```bash
PYTHONPATH=python python benchmark/benchmark_ep_trace.py \
  --model Qwen/Qwen3-30B-A3B \
  --tp-size 8 --dp-size 8 --ep-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend none \
  --dataset sharegpt --num-samples 128 \
  --batch-size 8 --max-new-tokens 1 \
  --output /tmp/qwen3_ep8_trace.pt
```

使用完整 trace 运行 GRACE、refine 和 replication：

```bash
PYTHONPATH=python python benchmark/compare_grace.py \
  --input /tmp/qwen3_ep8_trace.pt \
  --num-ranks 8 --ranks-per-node 8 --rdma-cost 1 \
  --optimizer-bundles 0 \
  --grace-ratio 0.15 \
  --grace-source-affinity-weight 1.0 \
  --grace-refine \
  --grace-refine-rounds 8 \
  --grace-refine-swaps 2 \
  --grace-refine-partners 8 \
  --grace-replication \
  --grace-replication-budget 8 \
  --grace-replication-hot-experts 16 \
  --grace-replication-candidates 4 \
  --grace-replication-max-extra-per-rank 1 \
  --grace-replication-compute-limit 1.25 \
  --save-grace grace.json \
  --save-grace-refine grace-refined.json \
  --save-grace-replication grace-replicated.json
```

若想观察通信优先、允许计算变差的 pair-swap：

```bash
--grace-refine-allow-load-worsening
```

若需要同时限制其计算负载：

```bash
--grace-refine-allow-load-worsening \
--grace-refine-swap-compute-limit 1.25
```

使用 SGLang 的 DeepEP normal all-to-all 对同一层、同一批采样 token 测量 plan 前后的真实通信时间：

```bash
PYTHONPATH=python python benchmark/benchmark_a2a_plan.py \
  --input /tmp/qwen3_ep8_trace.pt \
  --plan grace-refined.json \
  --layer 0 --num-ranks 8 \
  --hidden 2048 --tokens-per-rank 1024
```

脚本测量 `get_dispatch_layout + dispatch + combine` 的 CUDA 时间，并分别输出 layout、dispatch、combine、总时间和按 remote destination 估算的双向 BF16 A2A 字节数。`--hidden` 必须填写被测模型的实际 hidden size。Baseline 与 plan 使用相同的 token 样本、hidden size、DeepEP config 和物理 slot 上限；非均匀 placement 会用空 slot 填齐，replication plan 按 source-local 规则选择副本。可先用 `--dry-run` 在无 GPU 环境检查 plan、层名、slot 数和理论 remote。

## 8. 输出 Placement 格式

GRACE 和 GRACE-refine 保存单 rank：

```json
{
  "layer.name": {
    "0": 3,
    "1": 7
  }
}
```

Replication 保存 primary 和额外副本 rank；数组第一个元素为 primary：

```json
{
  "layer.name": {
    "0": [3, 0],
    "1": [7]
  }
}
```

表格中的 `grace-replicated solve-ms` 是 GRACE、可选 refine 和 replication 的累计时间。JSON 结果还包含纯副本阶段的 `replication_seconds`。

## 9. 复杂度与大 Trace

设：

- `B`：bundle 数量。
- `K`：Top-K。
- `E`：专家数量。
- `R`：rank 数量。
- `H`：参与复制的 hot expert 数。
- `P`：每个目标 rank 的 pair-swap partner 数。

主要成本为：

| 阶段 | 主要时间成本 | 备注 |
| --- | --- | --- |
| affinity graph | 约 `O(B*K^2)` | 构造共同激活边 |
| source affinity | `O(B*K + E^2*R)` | 可选；统计 source demand 并补充偏好重叠边 |
| spectral grouping | 主要由 `E` 的矩阵分解决定 | 通常 `E` 远小于 `B` |
| group assignment | `O(R^3)` | Hungarian assignment |
| move refinement | 与候选专家、rank 和其倒排 bundle 数相关 | 使用 expert-to-bundle index |
| sparse pair-swap | 约受 `E*R*P` 候选控制 | 候选再做 exact bundle delta |
| replication initial gain | `O(B*K^2)` | 只做一次完整分块扫描 |
| replication incremental update | 每个副本约 `O(B*K)` 过滤 | 只更新受影响 source/primary bundle |
| final metrics | 约 `O(B*(K+R))` | 分块计算 traffic 和 compute |

所有 replication 全量扫描使用固定大小的 NumPy chunk，峰值临时内存不随 `B` 线性增长。实测合成的 100 万 bundle、Top-K=8、128 experts、8 ranks、8 个副本，副本阶段约 1 秒；两千万 bundle 应近似线性增长，但真实耗时取决于 Top-K 分布、CPU 和内存带宽。

大 trace 的建议：

- 最终结论使用 `--optimizer-bundles 0`。
- 参数搜索可先用 2 万到 20 万 bundle 样本。
- 建议先比较 source-affinity weight `0/0.25/0.5/1.0`，再对最优值使用全量 trace。
- 在相同采样和 seed 下比较不同方法。
- replication budget 通常不要超过实际可加载的额外权重数量。

## 10. 参数速查

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--optimizer-bundles` | `20000` | 求解使用的 compact bundles；0 表示全部 |
| `--grace-ratio` | `0.15` | GRACE 非均匀 group 容量比例 |
| `--grace-source-affinity-weight` | `0` | 聚类中的同 source 需求重叠权重 |
| `--grace-equal-experts` | false | 强制各 rank 专家数相同 |
| `--grace-refine-rounds` | `4` | 单专家 move 轮数 |
| `--grace-refine-swaps` | `2` | pair-swap 轮数 |
| `--grace-refine-partners` | `8` | 每个目标 rank 的候选 partner 数 |
| `--grace-refine-capacity-ratio` | GRACE ratio | move 的专家容量比例 |
| `--grace-refine-allow-load-worsening` | false | 允许通信改善的 swap 增加最大负载 |
| `--grace-refine-swap-compute-limit` | 无 | allow-load-worsening 的可选负载上限 |
| `--grace-replication-budget` | `0` | 每层最多额外副本数 |
| `--grace-replication-hot-experts` | `16` | 参与副本搜索的 hottest experts 数 |
| `--grace-replication-candidates` | `4` | 每个 expert 保留的目标 rank 数 |
| `--grace-replication-max-extra-per-rank` | `1` | 每个 rank 最多新增的权重数 |
| `--grace-replication-compute-limit` | `1.25` | 副本路由的最大计算不均衡约束 |

## 11. 方法之间的关系

`compare_grace.py` 还保留两条独立实验路径：

- `CABLE`：不从 GRACE 出发，直接使用 source-aware Top-K bundle 做贪心 placement 和 refinement。
- `hypergraph`：不使用 GRACE seed，从多个 greedy seed 出发直接优化 fixed-terminal hypergraph connectivity。

它们用于对照实验，不属于 `GRACE -> GRACE-refine -> replication` 主链。`--cable-only` 或 `--hypergraph-only` 会跳过 GRACE；此时不能启用 grace-refine 或 grace-replication。

## 12. 当前边界

1. GRACE-refine 和 replication 都是启发式算法，不保证全局最优。
2. Pair-swap 使用稀疏 partner pool，没有 exhaustive all-pair search。
3. Replication 只模拟 source-local secondary，不模拟 least-load 或 weighted-random 在线选择。
4. Replication 的 compute 模型假设该 source 的全部专家请求转移到本地副本。
5. 离线 trace 可能与在线 workload 漂移，需要重新采集或周期性更新 placement。
6. 单超节点中 `remote` 代表跨 GPU/NVLink 通信，不等同于跨节点 RDMA。
7. 保存 replication JSON 不等于运行时已经支持该 placement，需要单独完成权重加载和 dispatch 接入。

## 13. 测试与实现位置

相关实现：

```text
benchmark/compare_grace.py
python/sglang/srt/eplb/expert_affinity_graph.py
python/sglang/srt/eplb/grace_expert_placement.py
python/sglang/srt/eplb/hypergraph_expert_placement.py
python/sglang/srt/eplb/cable_expert_placement.py
python/sglang/srt/eplb/replicated_expert_placement.py
```

相关测试：

```text
test/registered/unit/eplb/test_cable_expert_placement.py
test/registered/unit/eplb/test_moe_bundle_trace.py
```

运行：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/eplb/test_cable_expert_placement.py \
  test/registered/unit/eplb/test_moe_bundle_trace.py
```
