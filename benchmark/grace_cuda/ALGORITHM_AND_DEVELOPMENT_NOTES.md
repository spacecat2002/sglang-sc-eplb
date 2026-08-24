# GRACE 通信与计算联合优化：算法及开发记录

本文总结围绕 `simulate_remote_top2k.py`、Python 参考实现和
`grace_cuda` CUDA 实现的完整讨论。内容包括目标、算法演变、性能问题、
已实现方案、失败原因、当前代码状态和后续工作。

本文中的“当前实现”以仓库代码为准；讨论过但尚未完整落地的方案会明确标为
“设想”或“待实现”。

## 1. 背景和目标

本项目研究 MoE Expert Parallelism 场景下的专家放置、复制和 quota 分配。
它与同级目录中的 UltraEP 不是同一种策略：UltraEP 更强调极低运行时开销，
本项目希望同时满足两个约束：

1. 计算负载尽可能均衡。计算均衡是硬目标和第一优先级。
2. 在满足计算目标的可行解中，尽量减少 Top-K bundle 的远程通信量，并控制
   `remote`、`max-pair`、`max-ingress` 和 `max-egress`。

最终目标是把规划过程用于实际运行时，因此不仅要求结果质量，还要求 GPU 常驻、
避免 CPU 往返，并尽量把每个阶段压缩到微秒级。典型配置包括 EP=4、K=8，
但算法和实现必须考虑 EP=32 甚至更大的情况。

主要入口和实现文件：

- `benchmark/simulate_remote_top2k.py`：仿真和分阶段计时。
- `python/sglang/srt/eplb/grace_plus_replication.py`：Python 参考逻辑。
- `python/sglang/srt/eplb/gpu_replication.py`：CUDA 路径的 Python 编排。
- `python/sglang/srt/eplb/grace_plus_expert_placement.py`：分组和专家放置参考。
- `benchmark/grace_cuda/csrc/*.cu`：CUDA kernels。
- `benchmark/grace_cuda/csrc/compute_v2.cu`：新增的独立 `capacity-v2` 路径。
- `benchmark/grace_cuda/test_correctness.py`：CUDA 正确性测试。

## 2. 指标含义

结果表中主要指标如下：

- `remote`：去重后的远程 rank 访问总量。一个 token 的多个 Top-K 专家落到同一
  远程 rank 时，只形成一次 bundle 级远程访问。
- `max-pair`：任意 source/destination rank 对之间的最大通信量。
- `max-ingress`：单个目标 rank 的最大入站通信量。
- `max-egress`：单个源 rank 的最大出站通信量。
- `comp`：最大 rank 计算量相对平均计算量的比值。`1.00x` 最均衡。
- `extra/rank`：通信副本数范围。
- `compute-copies`：为计算均衡新增的副本数量。
- `aff-group-ms`：亲和图聚类和分组耗时。
- `comm-repl-ms`：通信副本选择耗时。
- `compute-repl-ms`：计算副本选择耗时。
- `quota-solve-ms`：quota 求解耗时。
- `quota-alloc/eval-ms`：把聚合 quota 映射回 token 并评估的耗时。
- `eval-ms`：基线或完整方案评估耗时。

## 3. 最初的复制和 quota 流程

原始方案大致分为以下三个逻辑阶段。

### 3.1 通信副本放置

先依据每个 `(expert, source rank)` 的需求，选择通信收益较高的专家副本。
目标是让更多 token 的专家在 source rank 本地执行，从而减少远程 bundle 访问。

早期采用类似 Top-N/Top-64 的复制策略。它通常能显著降低通信，但会复制较多专家，
也没有天然保证最终计算均衡。

### 3.2 计算副本选择

在已有通信副本之上，识别过载 execution rank，并给低负载 rank 增加计算副本。
这里必须先确定“复制哪个专家到哪个 rank”，因为 quota 只能分配给已经存在的专家
实例。讨论中明确了以下优先级：

1. 首先选择足以解除过载的专家和目标 rank。
2. 计算贡献相同的候选中，优先已有副本，避免消耗新增副本槽。
3. 再比较 Top-K bundle 通信代价，优先本地化或复用 bundle 已访问的目标 rank。

这不是单纯逐副本 greedy 的论文复刻，而是 capacity/export 联合求解思路：
副本选择和可导出的 quota 数量必须一起判断，防止复制了专家却无法搬走足够负载。

### 3.3 Quota 分配

`quota[source, expert, execution_rank]` 表示来自某个 source rank、属于某个专家的
请求，有多少交给某个 execution rank 执行。它必须满足两个守恒条件：

1. 对 execution rank 求和后，等于原始 `demand[expert, source]`。
2. 对 source 和 expert 求和后，得到每个 execution rank 的实际计算负载。

quota 分配的首要目标是让所有 rank 不超过计算 capacity。在可行候选中，再用以下
通信规则选择目标：

- 优先 source-local，即目标 rank 等于请求的 source rank。
- 否则优先 Top-K bundle 已经访问的远程 rank，避免新增远端目的地。
- 只有在清空原远端目的地的全部相关 quota 时，才能兑现“删除一次远程访问”的
  通信收益。
- 部分迁移不能重复使用同一份静态通信 gain。

## 4. 为什么出现负数候选 key

Python 排序中曾使用 `(-demand[e, s], e)`。这是因为 Python 默认按升序排序，
而需求量需要降序排列。把 demand 取负后，需求越大，负数越小，就会越早被处理；
第二个元素 `e` 用于需求相同时按 expert id 稳定决胜。负数没有业务含义，只是升序
排序实现降序优先级的常见技巧。

## 5. UltraEP 为什么能达到微秒级

对 UltraEP 的讨论得到的主要结论不是“换成 CUDA 就会快”，而是其运行时问题规模、
状态表达和 kernel 结构更适合微秒级执行：

- 热路径数据常驻 GPU，不在每层反复构造 Tensor 或同步到 CPU。
- kernel 数量少，阶段融合，避免 Python 调度和多次 launch。
- 使用固定大小、结构规则的状态，减少动态排序和通用求解。
- 不在热路径反复进行全量候选扫描、二分、重建 quota 和精确 token 级评估。
- 把部分复杂工作放在离线或低频阶段，运行时只执行紧凑决策。

本项目早期虽然把操作搬到了 CUDA，但仍然保留了较重的求解过程，因此出现
`comm-repl-ms`、`compute-repl-ms` 和 `quota-solve-ms` 为数毫秒甚至数十毫秒的情况。
“GPU kernel”与“微秒级 kernel”不是同义词。

## 6. 观察到的性能和效果问题

开发过程中反复出现以下现象：

1. 第一层包含 CUDA 初始化、缓存和编译预热，曾出现约 200 ms 的异常耗时。
2. 稳态下旧实现仍有约 8 ms `compute-repl-ms`、约 5 ms `quota-solve-ms`，完整方案
   约 19 ms。
3. 后续优化一度降到约 1 ms `comm-repl-ms`、4--5 ms `compute-repl-ms`、
   2--6 ms `quota-solve-ms`，但离微秒级仍很远。
4. 强制增加了全部允许的计算副本后，`comp` 仍可能为 `1.04x`--`1.07x`，说明
   “复制完成”不等于“quota 正确均衡”。
5. 某些改动改善计算后显著增加通信；另一些改动降低通信却导致副本槽被小收益候选
   占用，最终无法达到计算 capacity。
6. affinity grouping、谱聚类、匈牙利匹配和后续 refinement 曾导致求解极慢，甚至
   看起来卡住。用户随后明确要求去掉 swap 过程。
7. 新 grouping 的通信效果没有达到最初 Top-64 通信复制版本。

这些结果说明问题不在单个计时 API，而在算法状态和执行复杂度。

## 7. Affinity grouping 路线

讨论并实现过另一种初始专家放置方法：

1. 对每个 token 的 Top-K 专家两两累加亲和边权。
2. 得到专家 affinity graph。
3. 使用归一化谱嵌入和谱聚类，将专家分为等容量 group。
4. 使用匈牙利算法，将 group 映射到 rank，目标包含 remote、max ingress/egress
   等通信拥塞指标。
5. 在分组之后只增加少量通信副本，再进行计算均衡。

参考实现是 `grace_plus_expert_placement.py`。严格对齐目标包括谱聚类、等容量修复和
匈牙利映射，而不是用简化 greedy 替代。

该路线的优点是从 primary placement 层面减少 Top-K 专家跨 rank；缺点是谱分解、
聚类和 assignment 本身较重，不适合作为每层高频运行时步骤。它更适合离线或低频
重配置。CUDA 版本虽然减少了 CPU 往返，但没有消除算法本身的高复杂度。

## 8. Capacity/Export solver 的含义

Capacity/Export solver 联合处理两个问题：

- Capacity：每个 rank 最多允许承载多少计算量。
- Export：过载 rank 的哪些 `(source, expert)` quota 可以导出到哪些低负载 rank。

传统逐副本 greedy 每次选择一个专家副本，再局部搬运 quota，容易产生三个问题：

1. 新副本的可搬运量太小，却占用了稀缺副本槽。
2. 早期局部最优选择阻塞后续全局可行解。
3. 通信 gain 与实际 token bundle 分段不一致。

讨论中希望借鉴 UltraEP 的紧凑 solver 形式，但保留本算法自身的通信约束和论文差异：
计算 capacity 是硬约束，通信代价是可行解中的次级目标。

## 9. 通信约束如何进入计算均衡

通信不能只在计算均衡完成后补救，而应参与目标 rank 和 quota 的选择。
候选移动 `(source, expert, over -> target, amount)` 可估算：

- `amount`：本次能搬走的 quota。
- `potential`：若为该 expert 在 target 建立实例，后续最多还能解除多少过载。
- `gain`：完全移除旧远端目的地时减少的 bundle 通信。
- `cover`：目标 rank 已经被同一 Top-K bundle 访问，因此移动不会新增目的地的量。
- `penalty = amount - cover - gain`：保守的通信增量估计。

当前目标顺序应为：

1. 优先处理负载最高的过载 rank。
2. 优先已有专家实例。
3. 对新增实例，优先 `potential` 大、能真正完成均衡的专家。
4. 同等计算贡献下，比较单位 quota 通信代价 `penalty / amount`。
5. 再以 gain、amount 和稳定 id 决胜。

这回答了“选择低负载 rank 时如何考虑通信”：低负载只是可行性条件；在多个低负载
target 中，应优先 source-local 或已经被 bundle 覆盖的 rank，而不是只按 rank load
最小选择。

## 10. 二分与增量搜索

讨论过把 capacity 二分改成“每次增加一个微小量后重新判断”。结论是逐步增加通常
更慢：若 capacity 范围很大，它需要大量重复可行性检查。二分能用较少轮次找到可行
阈值，但每轮若运行完整 greedy/quota 重建，仍然很贵。

更适合运行时的方向是：

- 直接从目标 capacity 开始做一次增广式求解。
- 若副本容量确实不可行，返回未解决过载量，而不是反复重建整个状态。
- 使用已有 quota 和 load 原地更新。
- 将搜索轮数设为问题规模上界，避免不可控循环。

当前 `capacity-v2` 没有采用逐微量 capacity 搜索。

## 11. Python 与 CUDA 语义对齐问题

“输出计算总量相同”不足以证明 Python 与 CUDA 完全语义一致。严格对齐至少包括：

- 相同 source-aware demand。
- 相同 replica 可达集合。
- 相同 quota 守恒。
- 相同并列候选顺序。
- 相同 Top-K bundle 去重通信定义。
- 相同 quota 到 token 的确定性映射。

CUDA `quota_traffic_kernel` 使用 ordinal 将聚合 quota 映射回 token。每个
`(source, expert)` 的目标顺序为：source rank、primary rank、原通信副本、按
`addition_order` 排列的计算副本。不同专家的 quota 边界可能切分同一个 Top-K
bundle 的不同位置，因此聚合级静态 `gain/cover` 与最终精确 token 通信可能存在偏差。

这也是“quota 看起来均衡，但最终通信变化很大”的重要原因。若要求严格最小化
Top-K 通信，需要 bundle-level assignment；但对完整 trace 做精确组合优化不太可能
同时达到微秒级，因此当前路线使用保守启发式。

## 12. CUDA 安装问题

最初安装失败信息为：

```text
namespace "at::cuda" has no member "getDefaultCUDAStream"
```

原因是新版本 PyTorch CUDA C++ API 不再从该命名空间暴露旧调用。当前代码使用：

```cpp
c10::cuda::getCurrentCUDAStream(device)
```

并包含 `c10/cuda/CUDAStream.h`。

安装慢主要来自每次编译多个 `.cu` 文件和为多个架构生成代码。可通过以下方式降低
本地开发安装时间：

- 只设置实际 GPU 架构，例如 `TORCH_CUDA_ARCH_LIST=9.0a`。
- 使用 Ninja 并行编译。
- 开发时使用 inplace 增量构建，不反复清空 build 目录。
- 避免同时生成 `sm_89`、`sm_90a`、`sm_100a`、`sm_120a`，除非发布包确实需要。

## 13. GPU 常驻 fused runtime

为接近 UltraEP 的运行时形式，代码引入 `GraceCudaRuntime`，预分配并复用：

- demand、replicas 和 routing。
- quota、instance 和 rank loads。
- replica gain、addition order 和 candidate workspace。
- affinity、谱嵌入和匈牙利工作区。

目标是避免每层动态分配、CPU materialization 和重复 Tensor 构造。CUDA Event 用于
异步记录阶段耗时，最后统一同步。

但“常驻”目前主要解决调度和内存分配开销。若单个 kernel 内部仍反复扫描
`experts * ranks * ranks` 候选并串行执行多轮移动，它仍然无法自然达到微秒级。

## 14. 独立的 capacity-v2 路径

为不破坏 legacy solver，新增了可选择的第二条路径：

```bash
--compute-solver capacity-v2
```

legacy 实现在 `quota.cu`，v2 主要实现在 `compute_v2.cu`。v2 当前流程为：

1. 根据当前通信副本放置计算 bundle `gain` 和 `cover`。
2. 在 GPU 上初始化精确的 source-aware quota、instance、load 和 routing。
3. 从过载 execution rank 向未满 target rank 搬运 quota。
4. 必要时建立计算副本，并记录 `addition_order`。
5. 直接把求解中的 quota 作为最终输出，不再求解后重建 quota。
6. 用 `quota_traffic_kernel` 映射回 token 并精确评估通信和计算。

此前 v2 的两个关键错误已经修复：

- 早期聚合掉 source，随后猜测 quota 来源，导致计算和通信预测都错误。
- 早期只检查 `routing[source, expert]`，遗漏同一 quota 已拆分到其他过载 rank 的情况。

最近一次修复进一步处理了副本槽浪费和通信收益高估：

- 新副本按可解除的总过载量 `potential` 排序。
- 计算可行性优先于通信代价。
- 通信比较使用单位 quota 代价，而不是偏爱绝对增量小但只能搬少量 quota 的候选。
- 只有完全清空原执行目的地时才计入静态 bundle gain。
- 新增了“唯一副本槽不能被微小通信候选抢占”的 CUDA 回归测试。

## 15. 当前仍未完全解决的问题

以下问题不能视为已经完成：

1. 尚未在目标 CUDA 机器重新编译并跑完整 trace 验证最近 v2 修复。
2. 本地环境没有 `nvcc` 和可用 CUDA，因此目前只有静态检查和 Python 语法检查。
3. v2 仍是单 block、多轮全候选扫描。EP 增大时，其复杂度离微秒目标仍有明显距离。
4. 静态 bundle gain/cover 在 quota 多次移动后会变旧。
5. 聚合 quota 到 token ordinal 的映射会造成不同专家的 bundle 分段不完全一致。
6. 副本槽不足时，需要显式输出 unresolved overload，区分算法失败和约束本身不可行。
7. 尚未实现严格保持 rank load 不变的通信 refinement。
8. affinity 谱聚类和匈牙利算法不适合作为每层高频热路径，应考虑低频运行。
9. Python 参考实现与最新 v2 的每个候选优先级仍需逐项对齐测试。

## 16. 推荐的后续实现顺序

### 16.1 先验证当前修复

在 CUDA 机器上重新构建并运行：

```bash
cd benchmark/grace_cuda
python setup.py build_ext --inplace
cd ../..
PYTHONPATH=python:benchmark/grace_cuda \
  python benchmark/grace_cuda/test_correctness.py
```

然后使用相同 trace 对比 legacy、`capacity-v2` 和原 Top-64 结果。至少记录：

- 最终 `comp` 是否满足 limit。
- 实际新增计算副本数和各 rank 副本槽使用情况。
- quota 计算总量是否与精确 token evaluation 一致。
- 通信增加发生在 replica selection 还是 quota allocation。
- 稳态中位数，而不是第一层冷启动时间。

### 16.2 增加不可行性输出

kernel 完成后计算：

```text
unresolved = sum(max(load[rank] - capacity, 0))
```

若 `unresolved > 0`，benchmark 必须明确报告“当前 replica cap 下不可行”，不能只输出
一个较差 `comp` 让它看起来像普通求解结果。

### 16.3 分离硬均衡和通信优化

第一阶段只保证 quota 达到 capacity，通信作为候选 tie-breaker。达到计算约束后，
第二阶段只能做等量交换，确保每个 rank 的 load 完全不变，再改善通信。这样通信优化
不会破坏已经得到的计算均衡。

### 16.4 真正降低运行时复杂度

若要接近 UltraEP 的微秒级，需要减少问题本身，而不只是继续优化当前扫描循环：

- 把 affinity grouping 和 primary placement 移出逐层热路径。
- 只维护 top overloaded ranks 和 top candidate experts 的紧凑列表。
- 用一个固定轮数 fused kernel 完成 load、candidate 和 quota 更新。
- 避免每轮扫描完整 `E * R * R`。
- 运行时只做局部增量修正，完整重规划改为低频操作。
- 使用 CUDA Graph 或长期 resident kernel 降低 launch 开销，但前提是 kernel 内工作量
  已足够小。

## 17. 核心结论

整个讨论最终形成了以下原则：

1. 必须先决定专家实例是否存在，再向该实例分配 quota；但副本选择必须依据可导出的
   quota 共同判断，不能与 capacity 求解脱节。
2. 计算均衡是硬约束，通信是计算可行解中的第二目标。
3. 通信优化必须按 Top-K bundle 的远程目的地去重语义计算，不能只看逐专家流量。
4. 更多计算副本可能提高可行性，但不会自动降低通信；副本位置和 quota 边界同样关键。
5. 旧 v1 solver 在通信副本放置变化后效果下降，是因为其候选顺序、可达图和 quota
   假设依赖旧放置结构，而不是 solver 本身在所有 placement 下都等价有效。
6. 精确 bundle-level 全局优化与微秒级运行时存在直接冲突，实际系统需要离线全局
   放置加运行时局部修正的两层结构。
7. 当前 `capacity-v2` 是独立实验路径，legacy 行为应保持不变，直到 v2 经过目标 GPU
   上的效果和性能验证。

