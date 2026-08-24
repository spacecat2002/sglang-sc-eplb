# GRACE+ Source Replication 算法说明

本文只描述 Python 参考实现中的 source-aware replication planner，主要对应：

- `python/sglang/srt/eplb/grace_plus_replication.py`
- `python/sglang/srt/eplb/gpu_replication.py`

文档不依赖 CUDA kernel 的实现细节。CUDA 路径的目标是复现这里的 placement、quota、routing 和指标语义。

## 1. 算法要解决的问题

MoE 层中，每个 token 会被送到 Top-K experts。一个 rank 产生的 token，如果它要访问的 expert 不在本 rank，就需要跨 rank 通信。我们希望同时满足：

1. 热门的远端 expert 可以被复制到产生请求的 rank，减少通信；
2. expert 的计算负载在 rank 之间尽量均衡；
3. 一个 expert 的多个副本能够按 source rank 和需求量进行确定性分流；
4. 在指定通信预算时，在不损失计算均衡的前提下控制 ingress、egress 或 pair traffic；
5. 对 compact trace 中的聚合 token 保持精确语义，而不是只用平均概率估计。

这里的核心选择顺序是：先决定副本集合（通信副本和计算副本），再决定每个
source-expert 需求分给哪些副本，最后根据 quota 模拟真实 token occurrence 的目标
rank。所有候选都使用同一个计算优先目标：先最小化最大 rank load，再最小化 load
平方和；只有这两个计算指标相同，才比较通信预算超额和通信总量。临时 instance
quota 只用于比较候选，不是最终 quota；最终 quota 只在副本集合固定后求解一次。

## 2. 输入表示

Python 实现不要求每个 token 都单独展开，而是使用一个 compact routing bundle：

```text
source_rank   产生这批 token 的 rank
topk_experts  这批 token 的完整 Top-K expert 列表
count         具有相同 source 和 Top-K 列表的 token 数量
```

对应的数据结构是 `RoutedToken` 和 `RoutedArrays`：

```text
source_rank:  [B]
topk_experts: [B, K]
count:        [B]
```

其中：

- `B` 是 bundle 数量，不一定等于 token 数量；
- `K` 是 Top-K；
- 每行 `topk_experts` 不允许重复；
- `count` 表示该行 bundle 代表多少个 token；
- 所有数组使用整数类型。

后文使用以下符号：

```text
E  expert 数量
R  rank 数量
K  Top-K 大小
B  bundle 数量
s  source rank
e  expert
r  目标 rank
```

## 3. 中间状态和输出

### 3.1 Primary placement

输入 `primary[e]` 表示 expert `e` 的原始主副本所在 rank。每个 expert 至少有一个 primary。

### 3.2 Replica set

`replicas[e]` 是 expert `e` 的副本 rank 有序列表：

```text
replicas[e][0] == primary[e]
replicas[e][1:]       secondary replicas
```

同一个 expert 不能在同一个 rank 上重复出现。

内部常用布尔矩阵：

```text
replica_mask[e, r] = True  当 expert e 在 rank r 有副本
```

### 3.3 Default routing

在没有 quota 时，source `s` 访问 expert `e` 的默认目标是：

```text
如果 replica_mask[e, s]：目标为 s
否则：目标为 primary[e]
```

因此只要 source rank 有本地副本，就优先本地执行。

### 3.4 Quota

quota 是三维数组：

```text
quota[s, e, r]
```

含义是：source `s` 对 expert `e` 的需求中，有多少 occurrence 分配给 rank `r`。只有 `r` 属于 `replicas[e]` 时该项才允许非零。

对每个 `(s, e)`，必须满足：

```text
sum_r quota[s, e, r] == source_demand[e, s]
```

最后一维的 rank 顺序与 `replicas[e]` 的顺序一致，便于保存和执行时使用 prefix。

### 3.5 ReplicaPlacement

最终返回 `ReplicaPlacement`，主要字段为：

```text
replicas_by_expert  expert -> ordered tuple of ranks
routing_by_source   [source, expert] 的默认/最终目标
quota_by_source     可选的 quota 序列化结果
metrics             通信和计算指标
extra_copies        通信复制阶段增加的副本数
balance_copies      计算均衡阶段增加的副本数
source_demand       [expert, source] 需求矩阵
```

## 4. 从 trace 到 source demand

第一步是构造 source-aware demand 矩阵：

```text
source_demand[e, s]
    = sum(count[b]
          for b in bundles
          if source_rank[b] == s and e in topk_experts[b])
```

实现上，`_source_demand()` 把每个 bundle 的 Top-K 展平，然后执行一次二维 indexed accumulation。一个 bundle 的 `count` 会同时加到该 bundle 的每个 Top-K expert 上。

这个矩阵同时表达两件事：

1. source `s` 对 expert `e` 的计算需求有多大；
2. 如果把 expert `e` 复制到 source `s`，最多有多少需求可以变成本地执行。

注意：`source_demand` 是按 expert occurrence 统计的，不是通信量。一个 bundle 的多个 expert 可能最终落在同一个目标 rank，因此通信统计时还需要按 bundle 去重。

## 5. 阶段一：Source-aware Top-N 通信复制

入口是 `replicate_source_top_experts()`。

### 5.1 初始化

先让每个 expert 只有 primary：

```text
replicas[e] = (primary[e],)
```

然后逐个 source rank 处理。

### 5.2 每个 source rank 的候选

对于 source `s`，候选 expert 满足：

```text
source_demand[e, s] > 0
primary[e] != s
```

候选按以下确定性 key 排序：

```text
(-source_demand[e, s], e)
```

也就是优先复制 source `s` 请求最多的远端 expert；需求相同时选择 expert id 较小者。

最多选择 `max_extra_per_rank` 个候选，并把 source `s` 加入这些 expert 的 replica set。

### 5.3 为什么采用 Source Top-N

这个阶段只解决通信复制，不尝试同时解决计算 quota：

- 复制到请求发生地可以直接消除远端访问；
- 每个 rank 有独立副本槽位上限，搜索空间简单且可预测；
- 按 source demand 排序能优先覆盖最有价值的远端需求；
- 先确定副本集合，后续 quota 才有明确的可用目标集合。

该阶段的副本数上限是：

```text
最多 E 个 primary，加上每个 rank 至多 max_extra_per_rank 个 secondary
```

### 5.4 默认路由

副本集合确定后，生成 `routing[source, expert]`：

```text
routing[s, e] = s       如果 s 是 e 的副本
routing[s, e] = primary[e]  否则
```

如果没有启用计算均衡和 quota，算法到这里即可直接评估并返回 placement。

需要区分两种“计算均衡”：`replicate_source_top_experts()` 中的
`compute_imbalance_limit` 只会在已有副本集合上重新分配 quota；真正增加计算副本由
`balance_replica_compute()` 的 `max_extra_per_rank` 参数控制。

## 6. 通信和计算指标

对于普通单目标路由，`_route()` 分块扫描 bundle。每个 Top-K occurrence 都会产生一次 compute load，但通信按照一个 bundle 中的不同目标 rank 去重：

```text
compute[r] += count[b]       对每个 Top-K occurrence
traffic[s, r] += count[b]    只要该 bundle 至少有一个 occurrence 发往 r
```

本地目标 `r == s` 不计入 remote traffic。

最终指标为：

```text
remote          traffic.sum()
max_pair        traffic.max()
max_ingress     traffic.sum(axis=0).max()
max_egress      traffic.sum(axis=1).max()
compute_load    每个 rank 的 occurrence 计算量
```

`compute_imbalance` 定义为：

```text
max(compute_load) / mean(compute_load)
```

## 7. 阶段二：计算 quota 的基本构造

当设置了 `compute_imbalance_limit`，或者要求增加计算副本时，需要把 expert demand 分配到 replica ranks。

### 7.1 Expert total demand

先计算：

```text
expert_demand[e] = sum_s source_demand[e, s]
```

然后生成稳定排序：

```text
demand_order = experts sorted by (-expert_demand[e], e)
```

稳定排序保证相同需求时结果可复现。

### 7.2 Greedy instance quota

`_greedy_instance_quotas()` 先忽略 source 维度，只把每个 expert 的总需求拆到它的 replica ranks。

处理顺序是：

```text
(len(replicas[e]) > 1, -expert_demand[e], e)
```

因此单副本 expert 先处理，多副本 expert 后处理；同一类别中需求大的先处理。

原因是：单副本 expert 没有可选择空间，先放置它们可以确定固定负载；多副本 expert 再利用剩余副本进行均衡。

### 7.3 Waterfill

`_waterfill(loads, ranks, demand)` 在给定副本集合中执行整数均衡：

1. 按当前 rank load 从小到大排序，rank id 作为并列 tie-break；
2. 尝试把低负载 rank 提升到下一个负载水平；
3. 如果剩余 demand 不足以填平下一个水平，就在当前最小的一组 rank 中做整数商分配；
4. 余数按 rank 排序顺序逐个加一。

结果满足：

- 总分配量等于 expert demand；
- 只分给该 expert 的 replicas；
- 在当前 replica 集合内尽量降低最大 instance load；
- 完全确定性。

输出：

```text
instance_quota[e, r]
loads[r] = sum_e instance_quota[e, r]
```

## 8. 阶段三：按 source 分配 quota

`_source_quotas()` 把 `instance_quota` 进一步拆成 `quota[source, expert, rank]`。

对每个 expert：

### 8.1 先满足本地需求

对每个 source `s`：

```text
local = min(source_demand[e, s], capacity[s])
quota[s, e, s] = local
```

其中 `capacity` 初始等于该 expert 的 instance quota。

这一步体现 local-first：如果 source 本身有副本并且仍有容量，就优先在 source 本地执行。

### 8.2 分配剩余需求

剩余 source 按 remaining demand 降序、source id 升序处理。

每次从仍有容量的 replica ranks 中选择目标，排序 key 为：

```text
(rank != preferred_rank, -remaining_capacity, rank)
```

含义是：

1. 如果 source 的默认 routing 目标仍有容量，优先使用它；
2. 否则选择剩余容量最大的 replica；
3. 容量相同时选择 rank id 较小者。

每次移动：

```text
moved = min(remaining_source_demand, target_capacity)
```

直到该 source 的需求全部分配完。

### 8.3 全局计算 rebalance

逐 expert waterfill 是快速启发式，并不保证得到 replica graph 上的全局最小峰值。
例如某个副本集合可以达到 `[10, 10, 10]`，逐 expert 放置却可能先得到
`[11, 9, 10]`。因此 source quota 构造后还要执行一次全局 rebalance：

1. 计算目标容量 `ceil(total_demand / R * compute_imbalance_limit)`；
2. 按 rank id 扫描所有过载 rank，并分别在 residual replica graph 上做 BFS；
3. 一条边表示某个 `(source, expert)` quota 当前在前一 rank 执行，而该 expert 也能在
   后一 rank 执行；
4. 找到欠载 rank 后，按整条增广路径的最小剩余 quota 搬运；
5. 中间 rank 同时接收和转出等量 quota，负载不变；起点减载，终点增载；
6. 某个过载 rank 没有路径时继续检查其他过载 rank，不能提前停止；
7. 重复直到所有 rank 满足容量，或所有过载 rank 都不存在增广路径。

这一步使计算容量成为真正的全局第一目标，而不是依赖逐 expert greedy 恰好得到均衡
结果。只有 replica graph 本身不可行时，最终负载才允许高于目标容量。

## 9. 阶段四：计算不均衡约束下的 quota localization

当设置 `compute_imbalance_limit=L` 时，允许的 rank capacity 为：

```text
capacity = ceil(total_demand / R * L)
```

该 capacity 是 quota localization 的硬上限。若单副本 expert 的固定负载本身就超过
capacity，或者副本集合无法提供足够的可行目标，算法只能返回最接近的可行结果，
并由最终 `compute_imbalance` 指标暴露不可行性；它不会伪造满足约束的结果。

然后重复执行以下过程，直到没有任何移动：

1. 按 source rank 顺序处理；
2. `room = capacity - compute[source]`；
3. 找出在 source 有副本的 expert，按 `(-source_demand[e, source], e)` 排序；
4. 对每个 expert，按 primary 优先、其他 replica rank 升序检查其余 replicas；
5. 把 `quota[source, expert, target]` 移到 `quota[source, expert, source]`，最多移动 `room`；
6. 同步更新 source 和 target 的 compute load。

移动只会把已有 quota 从远端副本拉回 source 本地，不会改变副本集合，也不会创造新的通信目标。

完成后，`routing[source, expert]` 取该 quota 行中 quota 最大的 rank；相同最大值保留较小 rank，和 `argmax` 的确定性语义一致。这个顺序只用于
localization tie-break；真正的 prefix routing 仍按“primary、静态副本、按 addition
order 排列的计算副本”执行。

### 9.1 精确的联合 quota 求解

代码还提供 `_joint_quotas()` 作为需要同时考虑全局 rank capacity 和 source-expert
通信代价时的精确求解器。它不是简单地逐 expert waterfill，而是分两层建图：

1. `_instance_quotas()` 对候选 rank capacity 做二分搜索；每次用 expert 到 replica rank
   的 max-flow 判断所有 expert demand 是否可装下，从而得到最小可行的最大 rank load；
2. 单副本 expert 的需求直接固定到唯一 rank；
3. 对多副本 expert，为每个非零 `(source, expert)` 建 pair node；
4. pair node 可以走本地 source edge（cost 0），也可以先汇入 expert node，再流向任一 replica
   rank（默认 remote cost 1）；
5. rank 到 sink 的容量是全局 rank capacity 减去单副本 expert 已占用的固定负载；
6. 用逐次最短增广路径完成 min-cost flow，得到同时满足容量、尽量本地化且可均衡的 quota。

`_quota_for_replicas()` 在没有通信预算时使用这一联合 quota 路径；通信预算路径则使用后文的四轮快速通信感知 quota，以减少重复的精确 flow。

## 10. 阶段五：计算均衡副本搜索

`balance_replica_compute()` 用于在已有 placement 上继续添加专门服务计算均衡的副本。它与通信复制使用不同的上限：

```text
max_extra_per_rank          通信复制槽位上限
max_compute_extra_per_rank  计算均衡复制槽位上限
```

### 10.1 初始状态

从已有 placement 构造：

```text
replicas
primary
replica_mask
routing
source_demand
expert_demand
```

计算副本搜索采用与 UltraEP placement solver 相同形态的 capacity/export
求解过程，但保留 GRACE 的 source-aware 通信目标。它不枚举一个副本、重算一次全局
placement，而是在一个候选 rank capacity 下批量构造完整 export plan。

### 10.2 候选生成

1. 从最低通信 quota 开始：source 有本地副本就本地执行，否则在 primary 执行；
2. 计算理想容量 `ceil(total / R)`，先直接运行一次 feasibility oracle；
3. 若理想容量不可行，只在 `[ideal, initial_max_load]` 中二分最小可行 threshold；
4. oracle 为每个 rank 维护 `excess=max(load-threshold, 0)`、目标 slack 和新增副本槽位；
5. 从过载 rank 的 `(source, expert)` quota 中选择可导出的正负载，一次更新 source
   excess、target slack、replica occupancy 和 export quota；
6. 已有副本不消耗计算槽位；新副本只有在实际承接正 quota 时才占用一个槽位；
7. 所有过载 rank 都消除时，该 threshold 可行，并得到完整 export plan。

因此计算目标直接是最小可行 threshold，而不是依赖逐副本 greedy 的局部 load key。

### 10.3 Tie-break

先用 capacity-first oracle 确定最小可行 threshold，再在同一 threshold 重建一次
communication-first plan。它优先把 quota 导出到产生该 expert 请求的 source rank，
因为此时新增副本可以把一次远程执行变成本地执行；其次比较可搬运 quota、是否可复用
已有副本和稳定的 expert/rank id。若通信优先的构造因槽位选择而失败，则回退到同一
threshold 的 capacity-first plan，计算均衡不会被通信目标破坏。

最终 plan 同时给出 `(expert, target, quota)` export。副本集合先由其中承接正 quota 的
新边确定，source quota 随后按相同 replica graph 构造，并只用 residual augmenting
path 做容量正确性修复，不再让 quota repair 承担主要副本规划。

## 11. 通信预算下的 quota 选择

通信预算由一个 baseline placement 的指标乘以比例得到：

```text
budget = ceil(baseline_metric * communication_budget_ratio)
```

预算包含四项：

```text
(remote, max_pair, max_ingress, max_egress)
```

### 11.1 四轮通信感知 quota

`_communication_aware_quotas()` 不删除已经存在的 replicas，只改变 quota 和 preferred routing。

CUDA 常驻 runtime 在 `communication_budget_ratio` 为 `None` 或 `1.0` 且启用计算副本时，
直接消费 capacity/export solver 已生成的 source quota，不再运行下面的旧 quota 重求解；
其他预算比例仍保留多候选路径用于比较预算 violation。

每轮执行：

1. 用 `_fast_communication_quota()` 根据 instance quota 生成 source quota；
2. 计算 quota 的廉价通信摘要；
3. 计算每项指标相对 budget 的 violation；
4. 用以下字典序选择当前最佳方案。计算负载是第一目标，通信预算超额是第二目标：

```text
max(compute_load)
sum(compute_load ** 2)
违反项数量
违反项总量
最大单项违反
remote、pair、ingress、egress 各项违反
remote
max_pair
```

5. 根据当前 traffic 更新 rank cost：

```text
rank_cost[r] = ingress[r] + egress[r]
```

下一轮在选择 preferred replica 时优先使用 rank cost 较小的 rank。

最多运行四轮，因为该阶段的目的不是求解一般 min-cost-flow，而是用少量反馈迭代降低通信瓶颈，同时保留计算均衡。

### 11.2 Budget 分支的整体顺序

在 `balance_replica_compute(..., communication_budget_ratio=...)` 中，副本搜索和
quota 搜索使用同一组 lexicographic 目标：计算负载改善优先，通信预算超额作为
次级代价；随后用精确 `_route_quota()` 重新评估 bundle 边界。如果 quota 估计因
bundle crossing 超出预算，会计算不带 quota 的 safe routing 作为候选。只有 safe
routing 的计算 key 不差于当前方案且满足通信预算时才回退；否则保留计算更均衡的
方案，并让最终指标明确显示通信预算被放宽。这是“计算第一”的明确语义，而不是
隐式覆盖结果。

## 12. 精确 quota 路由评估

quota 不是简单的静态 expert-level 比例。一个 compact bundle 的 `count` 可能跨过 quota 的边界，因此必须按 occurrence ordinal 处理。

### 12.1 Prefix

对每个 `(source, expert)`，先按 source-local-first 顺序排列 replicas，然后计算累计 quota：

```text
prefix[source, expert, rank_i]
    = quota 在该顺序下的累计和
```

给定 occurrence ordinal，就能通过 prefix 找到目标 replica。

### 12.2 Bundle ordinal

`_route_quota()` 按 `(source, expert)` 对 bundle 内的 occurrence 编号：

1. 对当前 chunk 的 `(source, expert)` key 做 stable sort；
2. 对相同 key 的 count 做 cumulative sum；
3. 加上前一个 chunk 的 `counters`，保证 chunk 边界连续；
4. 得到每个 Top-K occurrence 的 ordinal。

这样即使一个 bundle 的 count 很大，也不会把它错误地当成一个不可拆分单位。

### 12.3 普通 bundle 和 crossing bundle

大多数 bundle 的所有 Top-K occurrence 都落在固定的 quota 区间，可以直接向量化计算目标。

只有当某个 bundle 的 occurrence 区间跨过一个或多个 prefix boundary 时，才进入 crossing 修正：

1. 先扣除把整个 bundle 当成同一目标时产生的重复通信；
2. 收集所有 quota boundary 和 bundle 两端 offset；
3. 按 boundary 把 bundle 切成若干 segment；
4. 每个 segment 独立计算 Top-K 的目标集合；
5. 对 segment 内不同远端目标各计一次通信。

因此最终 traffic 仍遵守“同一 bundle 到同一目标 rank 只发送一次”的语义，同时支持 quota 在 bundle 内切换。

## 13. 不同入口的关系

### `replicate_source_top_experts`

这是 source-top 主路径：

```text
source demand
 -> 每个 source 的 Top-N replicas
 -> instance quota
 -> source quota
 -> 可选 compute capacity localization
 -> exact route evaluation
```

### `balance_replica_compute`

这是在已有 placement 上增加计算副本的路径：

```text
已有 placement
 -> greedy compute replica search
 -> fast quota
 -> exact quota/traffic evaluation
```

### `replicate_hot_experts`

这是另一条旧的全局 hot-expert replication 路径。它先扫描一次 trace 得到：

```text
gains[e, target]
traffic
source_demand
```

然后在全局最热 expert 中选择能降低通信 objective 且不违反 compute limit 的复制。它与 source-top 主路径的候选顺序和目标函数不同，不能把两者的输出直接视为同一算法。

### `gpu_replication.py`

CUDA wrapper 的 Python 侧流程与上述阶段一致：

1. 规范化输入，并把 source、Top-K、count 保持在 CUDA；
2. 用 fused source-demand/Top-N kernel 构造 demand、replica mask 和默认 routing；
3. 若启用计算副本，selector 接收纯 `demand_order`，按当前副本状态动态生成
   fixed/flexible 顺序；
4. 副本集合确定后，重建 post-replication `expert_order`，再执行 fused quota solve；
5. 若启用通信预算，最多进行四轮 preferred-rank 反馈，但候选排序仍以计算为第一目标；
6. 对最终 quota 计算 bundle ordinals，并用 crossing-aware evaluator 统计 exact traffic
   和 compute；
7. 只有在 safe routing 既满足通信预算又不损失计算均衡时才回退；
8. runtime 模式复用预分配的 demand/quota/routing workspace，返回前才 materialize
   Python placement 和可选 quota tuple。

CUDA 阶段计时使用 CUDA events，单次 plan 结束时统一同步一次，避免每个阶段单独
`synchronize()` 把测量本身变成额外的 host/device barrier。

## 14. 正确性不变量

实现应始终满足以下不变量：

1. 每个被 trace 使用的 expert 都有 primary；
2. replica rank 不重复，且都在 `[0, R)`；
3. `quota[s, e, :]` 的总和等于 `source_demand[e, s]`；
4. quota 非零位置必须是该 expert 的 replica；
5. `routing[s, e]` 必须是该 expert 的 replica；
6. `sum(compute_load) == sum(source_demand)`；
7. traffic 对每个 bundle 的同一目标 rank 只计一次；
8. source-local traffic 不计入 remote；
9. 所有排序和 tie-break 都是稳定且确定性的。

## 15. 复杂度和性能含义

设 `M = B * K`：

- source demand 至少需要处理每个 Top-K occurrence，规模为 `O(M)`；
- 普通 route evaluation 按 chunk 扫描 `O(M)`；
- quota route 还需要对每个 chunk 的 `(source, expert)` occurrence 做 stable ordering；
- compute replica search 最多尝试 `R * max_compute_extra_per_rank` 次添加，每次需要重新计算当前 instance quota 并检查候选；
- communication-aware quota 最多四轮；
- 最终 placement 和 quota 序列化会把 GPU/NumPy 结果转换成 Python tuple/dict。

因此该 Python 参考实现追求的是语义完整和确定性，而不是只针对已聚合 `expert_loads` 的微秒级 placement kernel。UltraEP 类 benchmark 如果只输入 `[expert, source]` 的预聚合负载，并且不做 compact trace 的 crossing-route 精确评估，工作量明显更小，不能直接与这里的端到端时间比较。

## 16. 一个小例子

假设 `R=2`，expert `e=3` 的 primary 在 rank 0：

```text
source_demand[3, 0] = 10
source_demand[3, 1] = 30
replicas[3] = (0, 1)
```

如果两个副本当前负载相同，waterfill 可能得到：

```text
instance_quota[3] = [20, 20]
```

source 0 的 10 个需求先放本地：

```text
quota[0, 3] = [10, 0]
```

source 1 的 30 个需求先放本地 rank 1：

```text
quota[1, 3] = [0, 20]
```

还剩 10 个需求时，再按照 preferred routing、剩余容量和 rank tie-break 分配。最终 quota 的每个 source 行总和仍分别为 10 和 30，且总容量不会超过两个 instance quota。
