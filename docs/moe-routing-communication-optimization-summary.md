# GRACE+ MoE 专家放置

本文对应当前实现，不包含已经删除的 CABLE 和独立 Hypergraph 实验路径。`benchmark/compare_grace.py` 每层固定输出三种 placement：

1. `baseline`：按 expert id 连续、均匀地映射到 rank。
2. `grace`：GRACE co-activation grouping，按聚类生成顺序映射到 rank。
3. `grace+`：以原始 GRACE placement 为起点，执行目标感知 group assignment、move、swap 和受约束复制。

## 1. 输入与通信模型

一条压缩 bundle 由三个字段组成：

```text
source_rank: 产生 token 的 rank
topk_experts: token 选择的完整 Top-K expert 集合
count: 相同 source 和 Top-K 组合的 token 数量
```

内部使用 `RoutedArrays` 保存三个 NumPy 数组。`--optimizer-bundles N` 对 compact `.pt` trace 的每层做固定 seed 采样；`0` 使用所有 bundle。JSON 输入不采样。

单副本 placement 为 `expert -> rank`。一个 bundle 只向 Top-K 涉及的每个不同目标 rank 发送一次，目标等于 source 时不计通信。因此 traffic matrix 为：

```text
T[source, destination] = 发送到该 destination 的 bundle count
T[r, r] = 0
```

输出指标：

| 指标 | 定义 |
|---|---|
| `remote` | `sum(T)`，所有远端目标 rank 的总 token-bundle 数 |
| `max-pair` | `max(T)` |
| `max-ingress` | `max(sum(T, axis=0))` |
| `max-egress` | `max(sum(T, axis=1))` |
| `comp` | 最大 expert demand rank load / 平均 load |

## 2. 两种统一目标

所有目标感知阶段都接受同一个 `--objective`：

```text
ingress-egress（默认）:
  min max(max-ingress, max-egress)
  -> min max-pair
  -> min remote

remote:
  min remote
  -> min max-ingress
  -> min max-pair
```

比较采用字典序，不使用需要调参的加权和。计算不均衡作为通信指标完全相同时的最后 tie-break，并受显式 compute limit 约束。

GRACE 的 spectral grouping 本身只有 co-activation cut，并不接受该 objective。`--objective` 从 GRACE+ 的 group-to-rank assignment 开始生效，并统一用于 move、swap 和 replication。

## 3. Baseline

`_baseline_placement()` 按 expert 排序位置计算：

```python
rank = floor(expert_index * num_ranks / num_experts)
```

它不求解，`solve-ms` 固定为 0，用于表示不应用 plan 时的原始权重顺序。

## 4. GRACE

### 4.1 Affinity graph

`build_co_routing_graph()` 遍历每个 bundle：

- 每个 Top-K expert 的 `demand += count`。
- 每一对共同激活 expert 的无向边权 `edge[e1, e2] += count`。
- 可选 `--source-affinity-weight` 会增加具有相似 source-demand excess 的 expert pair 权重；默认 0。

compact 路径直接在 NumPy 数组上用 `np.add.at` 聚合，不创建逐 bundle Python 对象。

### 4.2 Spectral grouping

`grace_expert_placement()` 直接把 experts 聚成 `num_ranks` 个 rank groups：

1. 构造对称 affinity matrix `A`。
2. 计算 normalized affinity `D^-1/2 A D^-1/2`。
3. 取最大的 `num_groups` 个特征向量作为 embedding。
4. 用确定性的 farthest-first 初始化中心。
5. 最多执行 32 轮 k-means，并修复空 group。
6. 按 `--grace-ratio` 限制每组 expert 数量；`--equal-experts` 则强制完全相等。
7. equal 模式通过正 affinity gain 的 pair-swap 改善组内亲密度，同时不改变组大小。

### 4.3 按顺序映射 rank

GRACE 不读取 source-to-destination traffic matrix，也不求解 group-to-rank assignment。代码按 group 返回顺序映射：

```python
rank = group_index
```

表格中的 `grace` 到此结束，因此不会把 GRACE+ 的 source-aware 优化计入论文基线。

## 5. GRACE+

入口是 `grace_plus_expert_placement()`。benchmark 必须传入 GRACE 的 `rank_by_expert`，不会从随机 seed 或独立 greedy placement 开始。

### 5.1 索引与状态

`_prepare()` 一次性构建：

```text
source[B]                 bundle source rank
topk[B, K]                expert 的连续内部索引
count[B]                  bundle count
token_indexes[E]          每个 expert 出现在哪些 bundle
demand[E]                 每个 expert 的总计算需求
```

`_placement_state()` 从 GRACE placement 构造：

```text
ranks[E]                  expert 当前 rank
slots[R]                  每个 rank 的 expert 数
bundle_rank_counts[B, R]  bundle 的 Top-K 中有多少 expert 位于该 rank
```

`bundle_rank_counts[b, r] > 0` 表示 bundle `b` 必须发送到 rank `r`。move/swap 只更新受影响 bundle 的两列，不重新路由全 trace。

### 5.2 容量和计算约束

`_capacity_bounds()` 根据 `--capacity-ratio` 计算每个 rank 可容纳 expert 数的 `[minimum, maximum]`。`--equal-experts` 会把 ratio 设为 0。

expert demand 是所有选择该 expert 的 `count` 之和。move 后的最大 rank load 不能超过：

```text
max(--compute-limit * average_load, current_max_load)
```

因此 refinement 不会被迫修复 GRACE 已存在的负载峰值，但默认不会继续把最大值推得更坏。swap 默认也要求最大 load 不增加；`--allow-load-worsening` 可放宽，`--swap-compute-limit` 可设置硬上限。

### 5.3 Group-to-rank assignment

GRACE groups 已固定，但 group 编号不等于物理 rank。

`remote` 调用 `_align_groups_to_ranks()`：计算每个 `group -> rank` 的精确 total remote 成本，用 Hungarian assignment 找最小总代价的一一映射。

`ingress-egress` 调用 `_align_groups_to_congestion()`：

1. 为每个 `group -> rank` 计算 ingress、source egress、max-pair 和 remote。
2. 用 perfect-matching feasibility + 二分阈值，找全局最小可行 bottleneck。
3. 在该阈值内继续找最小可行 max-pair。
4. 在允许边上用 Hungarian assignment 最小化 total remote。

该阶段不会拆散 GRACE group，也不会改变每 rank expert 数。

### 5.4 单专家 move

每轮按 expert demand 从高到低检查候选目标 rank。

`remote` 的 `_refine_moves()` 通过 `bundle_rank_counts` 精确计算：

- 从旧 rank 移除 expert 后，旧列从 1 变 0 的 bundle 减少 remote。
- 加到新 rank 后，新列从 0 变 1 的 bundle 增加 remote。
- 只接受 total remote 改善，并满足容量与计算限制。

一轮可锁定并应用多个互不冲突的改善 move，最多执行 `--rounds` 轮。

`ingress-egress` 的 `_refine_congestion_moves()` 为候选更新 traffic 的旧/新目标列，按统一 congestion key 选择一轮中最佳 move。每轮只应用一个全局最佳改善，最多执行 `--rounds` 轮。

### 5.5 两专家 swap

`_refine_swaps()` 为每个 expert、每个目标 rank 只保留 `--partners` 个候选 partner。候选预排序使用两个单向 move delta 的和；最终接受仍调用 `_exact_swap_delta()` 扫描两个 expert 涉及的 bundle union，计算 exact delta。

- `remote`：接受 remote 降低；remote 相同时可接受更低 traffic key 或 compute load。
- `ingress-egress`：为每个候选精确更新 traffic matrix，按 congestion key 选择本轮唯一最佳交换。

最多执行 `--swaps` 轮。默认从历史的 2 降为 1，因为完整 trace 上 congestion swap 是主要耗时。

### 5.6 受约束 hot-expert replication

`replicate_hot_experts()` 接收 move/swap 后的单副本 placement，并采用确定性的 source-local-first 路由：如果 source rank 有该 expert 的副本，则在本地执行；否则使用列表中的第一个 rank，即 primary。

第一次 trace 扫描同时构造：

```text
source_demand[E, R]  每个 expert 来自各 source rank 的计算需求
traffic[R, R]        当前 source-to-destination bundle traffic
gains[E, R]          将 expert E 复制到 source R 后可精确消除的 remote
```

`gains` 只统计旧 destination 在完整 Top-K destination set 中彻底消失的 bundle。如果同一 bundle 的另一个 expert 仍使用旧 destination，复制不会被误算为通信收益。

每轮在 demand 最高的 `--hot-experts` 中枚举 `(expert, target rank)`，并要求：

- target 尚无该 expert；
- target 新增权重数小于 `--max-comm-expert-per-rank`；
- gain 大于 0 且统一 objective 严格改善；
- source-local demand 从 primary 转到 target 后，计算不均衡满足 `--replica-compute-limit`。

`remote` 依次比较 total remote、ingress/egress bottleneck、max-pair；`ingress-egress` 依次比较 bottleneck、max-pair、remote。每轮添加全局最佳候选，并增量更新受影响 gain，直到每个 rank 都达到 `--max-comm-expert-per-rank`，或者没有可行改善。全局最多新增 `num_ranks * max_comm_expert_per_rank` 个副本，不需要单独的全局预算参数。

GRACE+ JSON 保存：

```json
{
  "replicas": {"0": [2, 0], "1": [1]},
  "quota": [
    [[70, 30], [1]],
    [[0, 100], [1]]
  ]
}
```

第一个 rank 是 primary，后续为 secondary；quota 最后一维使用相同顺序。`--max-comm-expert-per-rank 0` 时通信阶段不复制，计算阶段仍可由 `--max-comp-expert-per-rank` 单独开启。

### 5.7 计算均衡复制与 quota 分发

通信复制完成后，`balance_replica_compute()` 再执行计算优先的第二阶段。它不把通信复制的上限和计算复制混用：

```text
--max-comm-expert-per-rank  通信收益复制阶段的每 rank 上限
--max-comp-expert-per-rank  计算均衡复制阶段的每 rank 上限
```

第二阶段把每个 `(source_rank, expert)` 的需求拆成整数 quota，可将同一需求分给多个副本：

1. 在 `expert -> replica rank` 二分图上二分 rank 容量并求整数最大流，得到当前副本集合下全局最小的 `max compute load`；
2. 计算复制只枚举有 source demand 的新增权重 `(expert, target)`，并且必须严格改善 `(max_load, sum(load^2))`；通信量作为 tie-break；
3. 副本集合确定后，一次性重新 water-fill 所有专家，不逐 quota 迭代迁移；
4. source-local-first 填充各副本容量，剩余需求优先复用通信阶段的 routing destination；
5. quota 按物理 rank 顺序构造 prefix；同一 token 的完整 Top-K 共享同一个位置，使相同 quota 区间尽量共用 destination；
6. 最后对完整 Top-K bundle 做一次对应的期望通信评估。

通信复制和计算复制分别只受 `max_comm_expert_per_rank` 与 `max_comp_expert_per_rank` 的每-rank 权重槽预算约束。quota 生成本身没有迭代次数，计算复制迭代最多为 `num_ranks * max_comp_expert_per_rank`。

因此计算阶段不会为了降低通信而接受更差的计算峰值，也尽量不产生新的 remote destination。最终保存 `quota[source][expert][replica_index]`，最后一维与该专家的 `replicas` 顺序一致。A2A benchmark 参考 UltraEP 的 quota-prefix 路由，为每个 token 生成一个共享位置，再对各 Top-K expert 的 prefix 查找副本；旧的 routing-only plan 仍兼容。

### 5.8 最终选择与输出

最终 placement 用 `evaluate_placement()` 重算完整指标。benchmark 输出：

```text
layer method    remote max-pair max-ingress max-egress comp experts/rank solve-ms
0     baseline  ...
0     grace     ...
0     grace+    ...
```

`grace+ solve-ms` 包含 GRACE 图构造、聚类、GRACE+ assignment/refinement 和 replication。JSON 中分别记录 `refine_seconds` 与 `replication_seconds`。

## 6. 命令行

默认优化 ingress/egress bottleneck：

```bash
python benchmark/compare_grace.py \
  --input trace.pt \
  --num-ranks 8 \
  --optimizer-bundles 20000 \
  --objective ingress-egress \
  --rounds 4 \
  --swaps 1 \
  --partners 8 \
  --max-comm-expert-per-rank 1 \
  --max-comp-expert-per-rank 1 \
  --save-grace grace.json \
  --save-grace-plus grace-plus.json
```

以 total remote 为目标：

```bash
python benchmark/compare_grace.py \
  --input trace.pt --num-ranks 8 \
  --objective remote
```

使用全部 bundle：

```bash
--optimizer-bundles 0
```

核心参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--objective` | `ingress-egress` | 所有目标感知阶段的统一目标 |
| `--optimizer-bundles` | `20000` | 每层求解 bundle 数；0 为全部 |
| `--grace-ratio` | `0.15` | GRACE group expert 数浮动比例 |
| `--equal-experts` | false | 强制每 rank expert 数相同 |
| `--source-affinity-weight` | `0` | affinity graph 的 source overlap bonus |
| `--capacity-ratio` | `0.15` | GRACE+ move 的 expert 容量浮动 |
| `--compute-limit` | `2.0` | move 最大计算不均衡上限 |
| `--rounds` | `4` | 单专家 move 轮数 |
| `--swaps` | `1` | pair-swap 轮数 |
| `--partners` | `8` | 每个 expert/目标 rank 的候选 partner 数 |
| `--allow-load-worsening` | false | 允许 swap 增大最大 compute load |
| `--swap-compute-limit` | 无 | 放宽 swap 时的计算不均衡硬上限 |
| `--hot-experts` | `16` | 复制候选中的热门 expert 数 |
| `--replica-candidates` | `4` | 每个 expert 保留的候选 target rank 数 |
| `--replica-compute-limit` | `1.25` | 复制路由的最大计算不均衡 |
| `--max-comm-expert-per-rank` | `0` | 通信优化阶段单 rank 最多新增的专家权重数；0 禁用该阶段 |
| `--max-comp-expert-per-rank` | `0` | 计算均衡阶段单 rank 最多新增的专家权重数；0 不新增权重，但仍生成 quota |

## 7. DeepEP 实测

保存 GRACE+ plan 后，用 `benchmark/benchmark_a2a_plan.py` 对比 baseline 与 plan：

```bash
torchrun --standalone --nproc-per-node=8 benchmark/benchmark_a2a_plan.py \
  --input trace.pt \
  --plan grace-plus.json \
  --model MODEL_OR_PATH \
  --layer 0 \
  --tokens-per-rank 1024 \
  --num-sms 0
```

该 benchmark 从模型 config 读取 `hidden_size`，用相同 sampled logical Top-K 构造 baseline 和 plan 的 physical expert id，然后交替测量 DeepEP layout、dispatch、combine 和 total 时间。

## 8. 复杂度与性能边界

affinity graph 构造约为 `O(B * K^2)`；GRACE spectral decomposition 主要依赖 expert 数，而不是 bundle 数。

GRACE+ move 约为 `O(rounds * E * R * average_occurrence)`。pair-swap 对候选 expert pair 的 bundle union 做 exact 扫描；`ingress-egress` 还要复制并更新 `R x R` traffic matrix，因此通常是最大热点。Replication 初始 gain 构造约为 `O(B * (R + K^2))`，每增加一个副本只重扫来自目标 source 且包含该 expert 的相关 bundle。计算均衡先用一次 `O(B*K)` 聚合得到 source-expert demand，候选循环不再依赖 bundle 数，最后用一次 `O(B*K)` 精确评估。

20M bundles 不建议在每次调参时全量执行前面的 grouping、move 和 swap。先用默认 20k 或更大样本选参数，再用 `--optimizer-bundles 0` 做最终 placement/评估。计算均衡阶段已不再逐候选扫描完整 trace；若仍需进一步压缩运行时延迟，应把 source-expert 聚合和候选归约移入现有 UltraEP CUDA kernel。

## 9. 代码位置

```text
benchmark/compare_grace.py
python/sglang/srt/eplb/expert_affinity_graph.py
python/sglang/srt/eplb/grace_expert_placement.py
python/sglang/srt/eplb/grace_plus_expert_placement.py
python/sglang/srt/eplb/grace_plus_refinement.py
python/sglang/srt/eplb/grace_plus_replication.py
benchmark/benchmark_a2a_plan.py
test/registered/unit/eplb/test_grace_plus_expert_placement.py
```
