# GRACE+ Source-aware 副本与 Quota 算法

本文描述当前代码实际执行的算法，重点是 CUDA 常驻 fast path，同时说明 Python
参考实现和兼容路径的差异。主流程只有三个步骤：

```text
trace -> source demand 与通信副本
      -> capacity/export 联合求解计算副本和 quota
      -> 按 quota 路由并精确评估
```

对应代码：

- `python/sglang/srt/eplb/gpu_replication.py`：CUDA runtime 的 Python 调用链；
- `benchmark/grace_cuda/csrc/quota.cu`：CUDA capacity/export solver；
- `python/sglang/srt/eplb/grace_plus_replication.py`：Python 参考 solver；
- `benchmark/simulate_remote_top2k.py`：benchmark 入口。

## 1. 目标、输入和输出

### 1.1 优化目标

算法按以下优先级做决策：

1. 以最小化所有 rank 中的最大计算负载为首要目标；
2. 在相同计算上限下，尽量不增加远程通信；
3. 优先复用已有副本，只在 direct export 或增广路径需要时扩展副本图；
4. 最后用 expert、source 和 rank id 保证结果确定。

因此通信预算不是可以牺牲计算均衡来满足的硬约束。若“计算更均衡”和“通信预算
不超限”无法同时满足，代码保留计算更均衡的方案，并在最终指标中暴露通信超额。

### 1.2 Compact trace

输入 trace 使用聚合 bundle：

```text
source_rank:  [B]       bundle 来自哪个 rank
topk_experts: [B, K]    bundle 的 Top-K experts
count:        [B]       该 bundle 代表的 token 数
```

算法先聚合为：

```text
demand[e, s]
    = source rank s 对 expert e 的 occurrence 数量
```

这里统计的是 expert occurrence。一个 token 的 K 个 expert 都会贡献计算量，但它们
若发往同一个目标 rank，通信评估时只发送一份 token 数据。

### 1.3 核心状态

```text
primary[e]             expert e 的主副本 rank
replicas[e, r]         expert e 是否存在于 rank r
quota[s, e, r]         source s 对 expert e 的多少 occurrence 在 rank r 执行
load[r]                sum_{s,e} quota[s,e,r]
addition_order[e, r]   计算副本的创建顺序，0 表示不是计算阶段新增
```

quota 必须始终满足：

```text
sum_r quota[s,e,r] == demand[e,s]
quota[s,e,r] > 0  =>  replicas[e,r] == True
```

最终返回 `ReplicaPlacement`，其中：

```text
replicas_by_expert  最终副本集合
routing_by_source   每个 (source, expert) 的主要目标
quota_by_source     source-specific quota，可按需物化
extra_copies        通信阶段增加的副本数
balance_copies      计算阶段增加的副本数
metrics             通信量和各 rank 计算负载
```

## 2. 步骤一：Demand 与通信副本

CUDA runtime 调用 `fused_source_topn_into()`，一次完成 demand 聚合、通信副本选择和
默认路由构造。

初始时每个 expert 只有 primary。对每个 source rank `s`，从它当前需要远程访问的
experts 中，按以下顺序选出最多 `max_extra_per_rank` 个：

```text
(-demand[e,s], e)
```

选中的 expert 被复制到 source `s`。这个阶段只关注通信收益，不负责最终计算均衡。
它产生的副本称为通信副本。

默认路由为：

```text
source 有该 expert 的副本 -> source 本地
否则                    -> primary[e]
```

这也是下一步 solver 的起始方案，因为它在当前副本集合上具有最低的 occurrence-level
远程通信量。

通信副本上限和计算副本上限相互独立：

```text
max_extra_per_rank          通信副本上限
max_compute_extra_per_rank  计算副本上限
```

## 3. 步骤二：Capacity/Export 联合求解

这是当前算法的主体。它不是“添加一个副本、重新 waterfill、再添加下一个副本”的
逐副本 greedy，而是在给定计算 threshold 下构造一份完整 export plan。

### 3.1 从最低通信 quota 开始

solver 先物化默认路由对应的 quota：

```text
if replicas[e,s]:
    quota[s,e,s] = demand[e,s]
else:
    quota[s,e,primary[e]] = demand[e,s]
```

然后计算初始 `load[r]`。计算均衡从最低通信方案出发，只移动消除过载所必需的
quota，而不是先做任意的远程均衡。

### 3.2 搜索最小可行 threshold

理论最小值为：

```text
ideal = ceil(sum(demand) / num_ranks)
```

CUDA kernel 和 Python 参考实现首先直接探测 `ideal`。若 ideal 可行，就不做二分；
若不可行，才在 ideal 和初始最大负载之间二分。二者寻找的都是相同 oracle 能构造出
的最低 threshold。

这里的 feasibility oracle 是确定性的启发式构造，不是对所有 export 组合做穷举或
max-flow。因此“最低”指当前 oracle 判定可行的最低 threshold，不代表一般组合优化
问题的数学全局最优。二分还假设该启发式的可行性随 threshold 单调；这是当前代码的
速度与最优性取舍。

当启用计算副本时，solver 会主动寻找它所能达到的最小 threshold。也就是说，fast
path 的目标比 `compute_imbalance_limit` 更强：该参数仍用于兼容 quota 路径，而联合
solver 本身直接最小化峰值，不会满足到指定比例后提前停止。

### 3.3 Feasibility oracle

对一个候选 threshold `T`：

```text
excess[r] = max(load[r] - T, 0)
slack[r]  = max(T - load[r], 0)
```

CUDA 和 Python oracle 都按初始负载降序处理 execution rank `over`，负载相同时 rank
id 更小者优先；在进入下一个 rank 前消除当前 rank 的 excess。然后枚举该 rank 上
实际存在的 `(source, expert)` quota 和所有有 slack 的 target。

一次 export 的最大搬运量是：

```text
amount = min(
    load[over] - T,
    quota[source, expert, over],
    T - load[target],
)
```

target 若已有该 expert 的副本，不消耗计算副本槽位；否则必须满足：

```text
added_by_rank[target] < max_compute_extra_per_rank
```

direct-export 副本只有在 `amount > 0` 且该 export 真正被采用时才创建。若局部 plan
停在高于 ideal 的 threshold，后续增广修复可以再增加少量图连接边。

应用 export 后同步更新：

```text
quota[source,expert,over]   -= amount
quota[source,expert,target] += amount
load[over]                  -= amount
load[target]                += amount
```

如果所有 rank 都不超过 `T`，则该 threshold 可行。若仍有过载 rank，但不存在合法
export，则该 threshold 不可行。

### 3.4 计算优先与通信 tie-break

threshold 搜索先运行 capacity-first oracle。候选按以下 key 选择：

```text
-amount
is_new_replica
communication_delta
expert
source
target
```

其中：

```text
communication_delta
    = (target != source) - (over != source)
```

它的取值含义是：

```text
-1  远程执行变成本地执行
 0  通信状态不变
 1  本地执行变成远程执行
```

capacity-first 先保证 solver 找到尽可能低的 oracle-feasible threshold。确定该
threshold 后，
再在同一 threshold 下运行 communication-first oracle，其 key 为：

```text
communication_delta
-amount
is_new_replica
expert
source
target
```

因此计算峰值不变时，优先把 quota 搬到请求来源 rank；这类 export 通常既均衡计算，
又降低通信。如果 communication-first 的局部选择耗尽槽位而无法完成整个 plan，代码
回退到同一 threshold 的 capacity-first plan，不牺牲计算均衡。

### 3.5 一次输出副本、quota 和 routing

最终 export plan 同时确定：

```text
replicas          哪些新计算副本被创建
addition_order    新副本顺序
quota             每个 (source, expert) 的实际分配
load              各 rank 计算负载
routing           quota 最大的目标；无 demand 时使用本地副本或 primary
```

在 `GraceCudaRuntime`、启用计算副本并且 `communication_budget_ratio` 为 `None` 或
`1.0` 时，这份 quota 被后续步骤直接消费。不会再运行旧的逐 expert waterfill/quota
solver，也不会再计算 `expert_order` 和 `source_order`。

Python 的 `_capacity_export_plan()` 实现相同的 source-specific oracle，并由
`_balance_replica_compute_hybrid()` 直接使用其 quota。若 direct export 停在高于 ideal
的局部 threshold，Python 和 CUDA 会在同一副本槽位限制内扩展一条 replica edge，
然后在整个 replica graph 上寻找多跳增广路径。这样可以处理“先从 rank B 搬到 C，
才能从 A 搬到 B”的计算均衡，而不重新运行逐 expert waterfill。

## 4. 步骤三：Quota 路由与精确评估

quota 通过 occurrence ordinal 转换为实际目标。对每个 `(source, expert)`，副本顺序
是：

```text
source 本地副本
primary
其他通信副本
按 addition_order 排列的计算副本
```

沿这个顺序对 quota 做 prefix sum。一个 occurrence 的 ordinal 落在哪个 prefix 区间，
它就路由到哪个副本。

compact bundle 的 `count` 可能跨越 quota 边界，所以不能只看 quota 最大的 rank。
评估器会计算每个 `(source, expert)` 的稳定 ordinal；若一个 bundle 跨越多个 prefix
区间，则按边界拆成 segments，再对每个 segment 的 Top-K 目标去重。

最终指标为：

```text
remote       所有非本地 token 传输量
max_pair     最大 source-target pair 流量
max_ingress  最大 rank 入站流量
max_egress   最大 rank 出站流量
compute      各 rank 的 expert occurrence 数
comp         max(compute) / mean(compute)
```

这里的 exact traffic evaluation 扫描 trace，并不属于副本 placement solver 本身。
因此 benchmark 中的 `quota-alloc/eval-ms` 不能直接与 UltraEP 论文只报告 placement
update kernel 的微秒耗时等同。

## 5. 通信预算如何处理

预算由通信副本方案的精确指标乘以比例得到：

```text
budget = ceil(baseline_metric * communication_budget_ratio)
```

预算包含 `remote`、`max_pair`、`max_ingress` 和 `max_egress`。

当前 `ratio=1.0` 的 CUDA fast path 直接使用 communication-first export quota，然后做
精确 bundle 评估。如果结果超预算，代码还会构造不使用 quota 的 source-local safe
routing。只有 safe routing 同时满足预算，并且其计算 key 不差于当前方案时才采用：

```text
compute_key = (max(load), sum(load ** 2))
```

否则保留计算更均衡的 export plan。这是明确的“计算第一、通信第二”语义。

其他预算比例，或没有 `GraceCudaRuntime` 的 CUDA 调用，仍进入兼容的多候选 quota
路径。该路径会使用 waterfill、residual rebalance 和最多四轮 rank-cost 反馈；它不是
当前 benchmark 的 `ratio=1.0` 常驻 fast path。

## 6. Python、CUDA 与兼容代码的关系

### 当前 CUDA benchmark 热路径

当同时满足：

```text
使用 GraceCudaRuntime
max_compute_extra_per_rank > 0
communication_budget_ratio is None or == 1.0
```

benchmark 先从 trace 聚合 `demand[expert, source]`，使 planner 与 UltraEP 一样从已
聚合的 load view 开始计时。实际 planner 调用链是：

```text
select_topn_routing_into
-> select_compute_replicas_into
     -> ideal probe / optional binary search
     -> capacity-first export plan
     -> communication-first export plan
     -> replicas + quota + routing
-> quota_traffic
```

### Python 参考路径

`simulate_remote_top2k.py` 的非 CUDA 分支只调用一次
`replicate_source_top_experts()`。启用计算副本时，该入口直接进入
`_capacity_export_plan()`，不会先运行旧 quota solver 再二次 balance。Python 与 CUDA
使用相同的 threshold 上界、当前最重 rank 选择、slot 约束和候选 key，并直接消费
solver 产出的 quota。

`balance_replica_compute()` 仍作为独立 API 保留，但它调用的是同一个 capacity/export
实现。

### 仍保留的兼容 helpers

以下函数仍被旧 API、无计算副本模式或特殊预算比例使用，但不应再被描述为当前联合
solver 的阶段：

```text
_greedy_instance_quotas / waterfill
_source_quotas
_rebalance_quota_compute
_joint_quotas / min-cost flow
_communication_aware_quotas
solve_quota / fused_quota_kernel
```

它们是兼容或回退实现，不是 capacity/export fast path 的顺序组成部分。这也是旧文档
看起来“阶段很多”且与当前代码不一致的主要原因。

## 7. 正确性约束

实现和测试重点检查以下性质：

1. 每个 `(source, expert)` 的 quota 总量严格等于 demand；
2. quota 只能分配给已有或本次新增的副本；
3. 每个 target rank 的计算新增副本数不超过配置上限；
4. `balance_copies` 包含 direct export 和增广路径修复创建的计算副本边；
5. capacity-first 和 communication-first 使用同一个 oracle-feasible threshold；
6. routing 对无 demand 的 `(source, expert)` 仍指向合法的本地副本或 primary；
7. Python 与 CUDA 对 export 候选使用相同的确定性 key；
8. exact traffic 按 bundle destination 去重，而不是把 Top-K occurrence 直接相加。

如果最终 `comp` 仍高于目标，可能是副本槽位或 replica graph 限制，也可能是启发式
oracle 没有找到实际存在的更低 threshold plan；代码不会再因为“现有副本可通过远程
quota 均衡”这一条件直接停止复制。若 oracle 的候选中存在能在相同 threshold 降低
通信的 source-local 计算副本，communication-first plan 会优先选择它。
