# GRACE CUDA 复制、分组与负载均衡优化：对话总结

更新时间：2026-08-28

本文总结本次对话到目前为止围绕 GRACE CUDA expert placement、通信副本、
affinity grouping、计算负载均衡和 CUDA solver 性能优化所做的讨论、实验、
实现、回退与当前结论。它不是一份新的算法规范；历史结果和当前结果会明确
区分，避免把不同代码版本、不同 EP 或不同 `solver_sms` 配置直接比较。

## 1. 最终目标和评价准则

对话的核心目标经历了多次实验，但最终稳定为以下几项：

1. 使用尽量少的 expert 副本，获得尽量低的跨 rank 通信。
2. 最终计算负载需要严格均衡，目标通常为
   `max_rank_load / average_rank_load = 1.0`。
3. 通信质量主要看
   `max(max_ingress, max_egress)`，而不是只看总 `remote`。
4. pure replication 和 grouping 是两条独立路径。pure replication 不应隐式
   执行 grouping、`compute-v2` 或 legacy grouping 逻辑。
5. grouping 常用 `top16`，对照 pure replication 的 `top32`，而不是用
   `top60` 掩盖副本数量差异。
6. solver 必须尽可能快，并限制使用的 GPU 资源，以便与其他使用 SM 的任务
   并发。默认希望单 CTA/约一个 SM，同时允许通过参数提高并行度。
7. 输出只保留真实 solver 端到端路径中的时间；pure replication 不显示
   affinity 阶段；quota allocation/evaluation 不计入 solver 时间。

## 2. 需求与实验演变

### 2.1 从 pure Top-N replication 开始

最初关注的是纯复制热门 expert：

- 比较 `top48`、`top16` 等不同副本预算的差距。
- 希望用更少副本达到与更多副本相同的通信下降。
- 讨论过 compute-aware replication，即通信副本选完后增加少量只用于计算
  均衡的副本。
- 指出纯按全局 remote traffic 排序不一定正确：一个副本的价值还应考虑它能
  消除哪些 source-destination rank 通信，以及是否降低 ingress/egress
  bottleneck。

随后实现和测试过 source-destination rank grouping：先按 source-destination
rank 对候选通信分组，再在组内选择 expert。该思路没有稳定优于原始 Top-N，
因此没有取代默认 pure replication，只保留为独立实验路径
`--rank-group-replication`。

### 2.2 grouping 思路进入 placement

由于 pure top16 与更大副本预算仍有明显差距，讨论转向先对共路由 expert
分组，再将 group 映射到 rank：

- grouping 可以在不复制副本时改变 primary placement。
- affinity-primary 在部分 layer 能直接改善通信。
- 仅优化 affinity 通信会造成计算不均衡；强行在 primary 阶段严格计算均衡，
  又可能损失通信。
- 因此形成两阶段思路：先产生通信友好的 primary/group placement，再通过
  capacity solver 和少量 compute-only replicas 完成严格计算均衡。

讨论和实现过的选择机制包括：

- 谱嵌入和确定性分组。
- exact-size group repair。
- group-to-rank 的 congestion-aware assignment。
- affinity-primary 收益不足时回退 pure replication 的 adaptive 机制。
- 在 group 内同时考虑 expert affinity 和 compute load。
- 对计算不均衡较高但通信明显更好的 layer 保留 affinity placement，然后把
  负载修复交给后续 capacity-v2。

layer 32 与 layer 38 的差异说明：primary compute imbalance 高并不必然意味着
最终通信差。通信结果同时由 group 结构、source-rank 分布、group-to-rank
mapping 和可用副本边决定，因此不能只依据 primary imbalance 选择方案。

### 2.3 恢复历史 pure 路径并与 grouping 隔离

对话中曾多次要求删除实验代码并恢复最开始状态，最终要求被澄清为：

- pure replication 和 grouping 完全分开，不共用 grouping、compute-v2 或
  legacy 的控制流。
- pure Python/quota/bindings/test 曾要求以 `f191c1bde^` 为恢复参考。
- benchmark/setup 曾要求以 `65a0f91bc` 为恢复参考。
- grouping 方案本身继续保留，两条路径共存。
- 不接受为了兼容实验而出现含义不清晰的 `-legacy` 方案名。

这项隔离原则仍是当前设计约束。当前 Python runtime 分别暴露
`plan_pure()` 和 `plan_grouped()`；CLI 也要求
`--rank-group-replication` 与 `--affinity-placement` 互斥。

### 2.4 “没有计算复制”问题

曾观察到如下参数几乎所有 layer 的 `compute-copies=0`：

```bash
--max-extra-experts-per-rank 60
--max-compute-extra-experts-per-rank 4
--compute-imbalance-limit 1.0
```

当时当前结果相对历史参考同时出现通信变差和 compute replica 消失。例如
layer 0 的历史参考为：

```text
remote=172192, extra/rank=60-64, compute-copies=8, comp=1.00x
```

而回退后的异常结果约为：

```text
remote=193789, extra/rank=60-60, compute-copies=0, comp=1.00x
```

这推动了以下设计澄清：

- `max-extra-experts-per-rank` 是通信副本预算。
- `max-compute-extra-experts-per-rank` 是后续计算均衡额外预算。
- `compute-imbalance-limit` 约束最终 quota/compute assignment 的
  `max_load / average_load`，不是通信 Top-N 的排序目标。
- 某些 layer 可以仅通过重新分配已有通信副本的 quota 达到 1.0，因此不一定
  必须新增 compute replica；`compute-copies=0` 本身不代表失败。
- 但如果所有 layer 突然从稳定存在 compute replica 变为零，同时通信也明显
  变差，应检查代码路径、extension 重建和 workspace/binding 是否匹配，而
  不能只归因于命令错误。

## 3. 当前两条算法路径

### 3.1 Pure replication

Pure 路径不运行 affinity histogram、embedding、partition 或 group mapping。
它以原始 primary placement 为基础，根据 source demand 选择通信副本，然后
进行 routing 和可选的计算均衡。

当前入口：

```python
GraceCudaRuntime.plan_pure(...)
```

可选的 source-destination rank 分组实验通过
`--rank-group-replication` 显式启用，不应影响默认 pure Top-N。

### 3.2 Grouping

Grouping 路径当前大致包含：

1. affinity histogram：统计 expert co-routing affinity 和
   expert-by-source-rank demand。
2. affinity embedding：当前对 E<=256 使用四维 CUDA subspace embedding。
3. partition：按 embedding 分组，并执行 exact-size repair 和 affinity swap。
4. balance：在 group 内进行 load-aware 调整。
5. group-source histogram：统计每个 group 来自各 source rank 的流量。
6. group-to-rank mapping：EP<=16 使用 exact Hungarian；EP>16 使用
   bottleneck-aware greedy mapping。
7. communication replication：在 affinity primary 上选择通信副本。
8. capacity-v2：在已有 replica graph 上分配负载，必要时增加少量
   compute-only replicas。
9. quota/export plan：产生最终 source-expert-target assignment。

当前入口：

```python
GraceCudaRuntime.affinity_primary(...)
GraceCudaRuntime.plan_grouped(...)
```

## 4. 当前关键参数语义

| 参数 | 含义 |
|---|---|
| `--max-extra-experts-per-rank` | 每个 rank 的通信副本上限 |
| `--max-compute-extra-experts-per-rank` | 每个 rank 为计算均衡新增的副本上限 |
| `--compute-imbalance-limit` | 最终最大 rank load / 平均 load 上限 |
| `--compute-solver capacity-v2` | 使用 CUDA capacity/export solver |
| `--affinity-placement` | 启用 grouping primary placement |
| `--solver-sms N` | grouping trace scan 最多使用的 CUDA blocks/近似 SM 数 |
| `--affinity-primary-min-improvement R` | affinity primary bottleneck 改善不足 R 时回退 pure |
| `--rank-group-replication` | pure 路径启用 source-destination rank 分组候选 |

通信比较应使用：

```text
bottleneck = max(max_ingress, max_egress)
```

`remote` 可作为辅助统计，但不能代替 bottleneck。

## 5. Capacity-v2 与 grouping 负载均衡

Capacity-v2 与 affinity-primary 的职责不同：

- affinity-primary 决定每个 expert 的 primary rank，兼顾 affinity、
  group capacity 和一定的负载信息。
- capacity-v2 在 primary 和通信副本构成的 replica graph 上分配实际
  expert demand。
- 当现有图无法在给定 capacity 下完成严格均衡时，capacity-v2 选择新的
  compute replica edge。
- 最终 export quota 优先 source-local target，同时必须满足目标 rank
  capacity。

因此，即使 affinity-primary 已经比较均衡，后续计算副本仍可能有意义：

- primary 的 expert 数量相等不等于 token/compute demand 相等。
- 通信副本可能让 quota 重分配而无需新增副本。
- 只有现有 replica graph 缺少可行边时，compute replica 才是必要的。

讨论过“grouping 阶段完全去掉计算限制，最后再均衡”。该方向可获得更自由的
通信 placement，但后续 capacity-v2 可能需要更多副本或产生更差的
ingress/egress。当前折中是 grouping 内保留轻量 load-aware balance，严格
1.0 则由后续 capacity-v2 保证。

## 6. 本轮保留的大 EP 优化

当前工作区保留了以下已验证优化：

1. 添加 `--solver-sms N`，默认值为 1。
2. affinity embedding 固定为最多四维，避免维数随 EP 线性增长。
3. EP32/64 使用 shared group-source histogram。
4. EP>16 使用并行 bottleneck-aware greedy group mapping；EP<=16 保留
   exact Hungarian。
5. EP>16 使用 aggregate capacity-v2 fast path。
6. fast capacity 直接 materialize quota/export plan。
7. communication Top-N 使用 one-warp-per-source。
8. EP>16 group mapping 并行计算 rank candidate。
9. group balance 对 top-three load 执行 exact optimization。
10. quota move materialization 改为 per-expert parallel。
11. fast capacity 修复 workspace ownership 和 current bundle gain 计算。
12. `aggregate_capacity_kernel` 使用 warp shuffle reduction，减少 CTA barrier。
13. 增加 `current_bundle_gains_fast_into`，避免生成未使用的
    `EP x E x EP` cover tensor。
14. EP>16 capacity candidate 使用 amount-first 排序，减少大量小的顺序 move。
15. EP>16 load-aware group balance 限制为 16 轮；EP4 仍为 32 轮。
16. `map_groups_kernel` 只预计算一次与 rank assignment 无关的 group order。
17. farthest-first center selection 改为增量维护 nearest-center distance，
    将该阶段从 `O(E * EP^2 * D)` 降为 `O(E * EP * D)`，选择结果不变。

## 7. 已测试并撤销的实验

以下实验没有收益或影响通信质量，已从源代码中删除，不应原样重复：

| 实验 | 结果 |
|---|---|
| aggregate capacity 的 shared-memory instance/load state | EP32 compute 约 6.09→7.00 ms，EP64 也略回退 |
| 128-bit vector quota clearing | materialization 仍约 1.24 ms |
| spectral iteration 固定为 8 | 无收益，原流程通常已在 8 轮内收敛 |
| group balance 从 16/32 降到 8 轮 | 更快，但 bottleneck 下降约 0.5%–1.1% |
| 仅为使用 TMA/PTX/tcgen05 而改写离散 solver | 当前热点不匹配这些指令，暂不实施 |

## 8. EP4 通信与正确性参考

当前 grouping 命令在 ShareGPT EP4 trace 上保留以下关键结果：

| Layer | 方法 | bottleneck | compute copies | comp |
|---|---|---:|---:|---:|
| 32 | affinity + remote-top16 + compute | 49,135 | 4 | 1.00x |
| 38 | affinity + remote-top16 + compute | 43,809 | 8 | 1.00x |

完整 40 层优化结果均达到 `comp=1.00x`，最终副本范围通常为
`extra/rank=16-20`。

历史 layer 39 的 adaptive-affinity 参考结果是：

```text
remote=159030
max_ingress=43138
max_egress=44079
comp=1.00x
extra/rank=16-18
compute-copies=4
solver=3.04 ms
```

该结果用于性能对照，但不能直接当作单 SM 结果。

## 9. `solver_sms=1` 与 `solver_sms=10` 测试

### 9.1 单 CTA/约一个 SM

最近的严格 `solver_sms=1` 结果约为：

| 配置 | solver time | bottleneck | compute copies | imbalance |
|---|---:|---:|---:|---:|
| EP4 layer 39 | 11.69 ms | 43,701 | 8 | 1.000000 |
| EP32 layer 0 | 9.50 ms | 18,356 | 25 | 1.000000 |
| EP64 layer 0 | 12.70 ms | 9,754 | 40 | 1.000000 |

EP32/64 使用 ShareGPT layer 0 的 expert/count，并使用
`source = arange(tokens) % EP` 生成对应 EP 的 source rank。这是大 EP
scaling 测试，不是原生 EP32/64 trace。

### 9.2 十个 blocks/近似十个 SM

按用户要求使用 `solver_sms=10` 重新测试：

| 配置 | solver time | 相对 SM=1 | bottleneck | compute copies | imbalance |
|---|---:|---:|---:|---:|---:|
| EP4 layer 39 | 3.775 ms | 3.10x faster | 43,701 | 8 | 1.000000 |
| EP32 layer 0 | 5.496 ms | 1.73x faster | 18,356 | 25 | 1.000000 |
| EP64 layer 0 | 9.425 ms | 1.35x faster | 9,754 | 40 | 1.000000 |

EP4 的 40 层多数为 3.2–4.0 ms；首次 layer 0 为 4.727 ms。

当前 EP4 layer 39 与历史 3.04 ms 的分阶段差异为：

| 阶段 | 历史结果 | 当前 SM=10 | 差值 |
|---|---:|---:|---:|
| affinity histogram | 0.15 ms | 0.412 ms | +0.262 ms |
| embedding | 0.29 ms | 0.290 ms | 基本不变 |
| partition | 0.88 ms | 0.913 ms | +0.033 ms |
| balance | 0.79 ms | 0.804 ms | +0.014 ms |
| group-source/map | 0.42 ms | 0.716 ms | +0.296 ms |
| communication replication | 0.10 ms | 0.102 ms | 基本不变 |
| compute replication | 0.40 ms | 0.535 ms | +0.135 ms |

历史 kernel 对 histogram 和 group-source 使用
`ceil(tokens / 256)` 个 CTA。当前实现严格按照 `solver_sms` 限制 trace
scan blocks，因此历史 3.04 ms 与当前严格 SM=10 并不是完全同一执行配置。
当前 EP4 剩余回退主要在 histogram、group-source/map 和 compute stage。

## 10. 最新大 EP profiler 结果

最近 EP64 单 CTA profile 的主要 kernel 约为：

| Kernel/stage | 时间 |
|---|---:|
| spectral exact groups | 2.85 ms，增量 center 后 partition 总体约 3.12 ms |
| affinity histogram | 2.15 ms |
| aggregate capacity | 1.69 ms |
| group balance | 1.49 ms |
| materialize quota | 1.24 ms |
| group map | 0.87 ms |
| affinity swaps | 0.56 ms |
| current bundle gain | 0.55 ms |
| group-source | 0.49 ms |
| communication bundle gain | 0.44 ms |
| subspace embedding | 0.29 ms |
| communication Top-N | 0.28 ms |
| quota traffic | 0.22 ms |

早期 EP64 placement 约 23.34 ms，旧 capacity-v2 曾达到约 61.09 秒。当前已经
消除了数量级问题，但完整单 CTA EP64 solver 仍是毫秒级，不是微秒级。

## 11. Kernel fusion、TMA、PTX、CTA 与 tcgen05 结论

### 11.1 可考虑的 fusion

最合理的候选是：

```text
current_bundle_gain_fast_kernel -> aggregate_capacity_kernel
```

两者相邻且都是单 CTA。融合可以省一次 launch 和一次 `E x EP` 中间结果的
global write/read，但不能消除 current-gain 的全部计算时间。只有 profiler
证明实际收益足够覆盖复杂度时才应实现。

### 11.2 不适合直接融合的阶段

- histogram 与 grouping：group label 必须等 embedding/partition 完成后才存在。
- communication bundle gain 与 grouping：必须先得到 primary mapping。
- aggregate capacity 与 dense quota materialization：只能省一个 launch，
  `EP x E x EP` 清零和输出规模仍然存在。

### 11.3 TMA

TMA 适合规则的大块 dense transfer。当前 trace gather、co-routing histogram、
离散 candidate selection 和 atomic update 都是不规则访问，因此不是主要收益
点。dense quota 写出虽然规则，但 vector store 实验已经表明带宽不是唯一瓶颈。

### 11.4 tcgen05 / tensor core

本地 CUDA 13.2 和 SM100a 支持 tcgen05，但当前主要热点是整数 atomics、
min/max、排序、分支和依赖 move，不是 dense matrix multiply。tcgen05 最可能
用于窄维 subspace 或 group-affinity dense 运算，而这些阶段目前不是最大热点，
并且现有数据大量使用 int64/float64。当前不应为了使用指令而改写算法。

### 11.5 大 EP 的现实结论

在 EP32/64 下保持微秒级，靠 TMA/PTX 或 launch fusion 不够。真正需要的是：

- 使用 UltraEP 风格 sparse export plan，避免 dense
  `EP x E x EP` quota。
- 减少或近似 grouping 的全量扫描、repair 和 dependent move。
- 复用跨 step 状态，避免每次从完整 trace 重建所有统计。
- 明确区分在线 hot path 与低频 placement refresh；低频 solver 不应冒充每
  token 端到端延迟。

## 12. 输出与计时约定

当前 benchmark 输出规则：

- pure replication 不输出 `aff-*` 列。
- grouping 输出 histogram、embedding、partition、balance、map、
  communication replication、compute replication 和 quota solve。
- 已删除 quota allocation/evaluation 时间，因为它不是 solver 求解时间。
- `solver-ms` 是保留阶段之和。
- CUDA timing 使用 event 延迟收集并在最后统一 synchronize，避免每个阶段都
  强制 CPU/GPU 同步。

任何性能比较必须同时记录：

- EP、E、K 和 trace。
- `solver_sms`。
- communication replica cap。
- compute replica cap。
- compute imbalance limit。
- compute solver。
- 是否启用 affinity/adaptive/rank-group。
- 是否包含首次运行、extension lazy load 或 GPU warm-up。

## 13. 构建与运行命令

### 13.1 构建 CUDA extension

```bash
cd /home/admin/workspace/sglang/benchmark/grace_cuda
TORCH_CUDA_ARCH_LIST="10.0a" MAX_JOBS=16 \
  uv pip install -e . --no-build-isolation
```

### 13.2 Grouping top16 + capacity-v2，单 CTA

```bash
PYTHONPATH=../../python \
../../.venv/bin/python ../simulate_remote_top2k.py \
  --input ../../sharegpt_ep4_trace.pt \
  --cuda \
  --affinity-placement \
  --solver-sms 1 \
  --max-extra-experts-per-rank 16 \
  --max-compute-extra-experts-per-rank 4 \
  --compute-imbalance-limit 1.0 \
  --compute-solver capacity-v2
```

### 13.3 Grouping top16 + capacity-v2，十个 blocks

```bash
PYTHONPATH=../../python \
../../.venv/bin/python ../simulate_remote_top2k.py \
  --input ../../sharegpt_ep4_trace.pt \
  --cuda \
  --affinity-placement \
  --solver-sms 10 \
  --max-extra-experts-per-rank 16 \
  --max-compute-extra-experts-per-rank 4 \
  --compute-imbalance-limit 1.0 \
  --compute-solver capacity-v2
```

### 13.4 Pure top32

```bash
PYTHONPATH=../../python \
../../.venv/bin/python ../simulate_remote_top2k.py \
  --input ../../sharegpt_ep4_trace.pt \
  --cuda \
  --max-extra-experts-per-rank 32 \
  --max-compute-extra-experts-per-rank 4 \
  --compute-imbalance-limit 1.0
```

Pure 路径不要添加 `--affinity-placement` 或 `--solver-sms`。

### 13.5 Adaptive affinity

```bash
PYTHONPATH=../../python \
../../.venv/bin/python ../simulate_remote_top2k.py \
  --input ../../sharegpt_ep4_trace.pt \
  --cuda \
  --affinity-placement \
  --solver-sms 10 \
  --affinity-primary-min-improvement 0.05 \
  --max-extra-experts-per-rank 16 \
  --max-compute-extra-experts-per-rank 4 \
  --compute-imbalance-limit 1.0 \
  --compute-solver capacity-v2
```

### 13.6 Pure source-destination rank grouping 实验

```bash
PYTHONPATH=../../python \
../../.venv/bin/python ../simulate_remote_top2k.py \
  --input ../../sharegpt_ep4_trace.pt \
  --cuda \
  --rank-group-replication \
  --max-extra-experts-per-rank 32 \
  --max-compute-extra-experts-per-rank 4 \
  --compute-imbalance-limit 1.0
```

## 14. 正确性与已知测试状态

大 EP 临时验证已经检查 EP32 和 EP64：

- `quota.sum(dim=2).t() == demand`。
- quota 非负。
- quota target 必须拥有对应 replica。
- materialized loads 与 aggregate solver loads 一致。
- 每 rank compute copies 不超过 4。
- 最大 load 不超过 `ceil(total / EP)`。
- 最终 imbalance 为 1.0。

EP32/64 fast path 上述检查均通过。

当前正确性命令：

```bash
PYTHONPATH=../../python ../../.venv/bin/python test_correctness.py
```

它仍在 `test_correctness.py:361` 的历史精确 quota fixture 断言失败：

```python
assert quota.cpu().tolist() == ...
```

该失败在本轮大 EP 优化前已存在；此前的测试均通过。仍建议增加一个永久的
最小 EP32 fast-capacity 测试，覆盖 quota conservation、replica validity、
aggregate/materialized load equality、copy cap 和 1.0 imbalance。

## 15. 当前文件边界

与本轮算法相关的主要文件：

```text
csrc/affinity.cu
csrc/bindings.cpp
csrc/compute_v2.cu
csrc/placement.cu
csrc/runtime.cu
../../python/sglang/srt/eplb/gpu_replication.py
../simulate_remote_top2k.py
test_correctness.py
```

工作区原本已经包含其他未提交修改和生成文件，例如 `DeepEP`、`setup.py`、
`csrc/pure_compute.cu`、trace、JSON、SQLite 和 checkpoint。后续修改不能用
`git reset --hard` 或整体 checkout 覆盖这些用户文件。

## 16. 后续优化优先级

按预期收益和实现风险排序：

1. 优化受 `solver_sms` 限制的 affinity histogram，重点是 warp-aggregated
   atomics 和减少同 key 冲突；只在 profiler 证明 trace 中重复 key 足够多时保留。
2. 优化 EP4 的 group-source/map。当前 SM=10 相对历史仍多约 0.30 ms。
3. 将 dense quota/export plan 改为 sparse representation，消除
   `EP x E x EP` clear/materialization。
4. 减少 spectral overflow repair 中的 per-group CTA barrier 和重复 E scan。
5. 在 current-gain + aggregate fusion 上做一次 A/B profile；收益不足则撤销。
6. 为 EP32/64 增加永久正确性测试和原生大 EP trace benchmark。
7. 把 online latency 目标拆成 fast incremental refresh 与 full regrouping。
   完整 grouping 在单 CTA 下不应被不现实地描述为微秒级。

## 17. 当前结论

- grouping 对减少副本预算有价值，但必须和 pure replication 独立维护。
- communication 应以 ingress/egress bottleneck 为主指标。
- capacity-v2 可以在保留通信 placement 的同时实现严格 1.0 计算均衡，但
  replica graph 不足时仍需要 compute-only replicas。
- `solver_sms` 是明显的性能/资源占用旋钮：EP4 从 1 提高到 10 可获得约
  3.10 倍加速；EP32/64 收益逐渐降低，因为 partition 和 dependent capacity
  move 成为主导。
- 当前 kernel fusion 只能带来局部收益。达到大 EP 微秒级需要改变状态和输出
  表示，而不是单纯堆叠 TMA、PTX、CTA 或 tcgen05 技术。
- 当前最直接的性能问题是受 block 数限制后的 trace histogram 和
  group-source 扫描吞吐；这是下一轮优化的首要目标。

## 18. 远端崩溃后的恢复背景

本轮对话开始时，用户说明远端服务器在此前优化尚未 `git push` 时崩溃，
因此远端会话中完成的代码没有保留下来。恢复工作的输入包括：

1. 本文档此前的版本。
2. Codex 附件 `pasted-text.txt`，其中保存了服务器崩溃前的终端输出、实现说明、
   benchmark 和正确性结果。
3. 当前本地工作区中的未提交修改。

恢复过程中始终遵守以下原则：

- 不使用 `git reset --hard`、整体 checkout 或其他会覆盖用户改动的操作。
- 先检查当前工作区，再根据历史文本重新实现缺失代码。
- 远端日志只能作为历史参考；没有在当前代码和当前环境重新验证的结果不能
  当作本轮已验证结果。
- 本地没有 `.codegraph/` 索引，因此按仓库常规方式使用 `rg` 和定点文件读取。

用户在恢复过程中多次询问为什么此前能够修改、后面却出现“没有读写工具”的
说法。最终要求很明确：不再停留在说明层面，要直接继续修改代码。当前工作区
确实可读写，后续修改均已在 `/Users/zwh/Workspace/sglang` 中完成。

## 19. 本轮需求演进

本轮对话中的优化请求按时间顺序大致为：

1. 根据崩溃前文本重新恢复优化和代码。
2. 确认并实现 UltraEP 形式的 sparse quota。
3. 分析 GRACE 与 UltraEP demand 采集方式差异，以及 GRACE 是否因此更慢。
4. 实现 demand/gain 增量更新，避免每次完整扫描 trace。
5. 所有 EP size 尽量使用当前最快路径，而不是只优化 EP32/64。
6. 去掉 ordinal 全量排序并实现真正 CSR sparse quota。
7. 优化 aggregate capacity solver 和压缩增量索引。
8. 所有 EP 均支持 single CTA、multi CTA 和 shared-memory 路径；EP 最大值统一为
   64，不再保留 128-rank 上限。
9. 继续执行后续编号 2-7 的优化。
10. 最终聚焦并要求实现新的优化项 2、3、4、5：

| 编号 | 优化项 |
|---|---|
| 2 | 真正 incidence CSR，用于增量 current-gain 更新 |
| 3 | capacity solver active candidate cache |
| 4 | CSR quota metadata 压缩为 int32 |
| 5 | group-source warp-aggregated update |

本节中的编号来自连续对话中的优化列表，不等同于本文早期章节编号。

## 20. 崩溃前日志中记录的历史实现与结果

附件记录显示，远端崩溃前曾实现和验证过以下内容，但这些代码当时没有 push，
因此需要在当前工作区重新实现或核对：

- UltraEP 风格 sparse quota-prefix/export plan。
- `materialize_quota=False` 时跳过 dense `EP x E x EP` quota。
- `fused_source_topn_into` 在一次 trace 扫描中同时产生 demand 和初始
  communication gain。
- K=8 专用寄存器缓存，减少第二遍读取 `topk` 和 replica state。
- `current_bundle_gain_fast_kernel` 的 K=8 特化。
- sparse 和 dense 的 placement、routing、quota、traffic、compute load 严格
  A/B 对比。

附件中的历史测试配置为 32K token、E=256、K=8、单 SM。记录的结果为：

| EP | dense quota path | sparse quota path |
|---:|---:|---:|
| 32 | 约 7.8 ms | 约 3.7 ms |
| 64 | 约 13.3 ms | 约 5.0 ms |

附件同时指出，即使 sparse quota 已经显著降低 quota 阶段耗时，完整路径仍受
`current_bundle_gains_fast_into` 的第二次 trace 扫描限制：

| EP | current gain | quota + capacity |
|---:|---:|---:|
| 32 | 约 1.6 ms | 约 0.34 ms |
| 64 | 约 1.7 ms | 约 0.87 ms |

通信采集本身的历史数据约为 EP32 110 us、EP64 135 us。由此形成的重要结论是：

- sparse quota 已不是唯一瓶颈。
- 要接近 200 us，需要避免或增量化 current-gain 的完整 trace 重扫。
- 完整 regrouping 和在线 incremental refresh 必须区分。

这些数字来自崩溃前的远端版本，仅作为性能基线，不代表本轮本地代码已经复现。

## 21. EP 与执行路径统一

用户明确要求不要按 EP<=16、EP32/64 分裂成互斥算法，而是所有 EP 1-64 都应
支持：

- single CTA。
- cooperative multi CTA。
- shared-memory fast path。
- 相同的算法语义和确定性 tie-breaking。

当前统一约束为：

```text
1 <= EP <= 64
```

`kMaxEpSize` 已集中定义为 64。single CTA 是所有 EP 的合法执行模式；
`solver_sms > 1` 时可使用 cooperative persistent kernel，如果设备不支持
cooperative launch 或驻留 CTA 不足，则回退 single CTA。shared memory 用于
loads、added counters、当前 overload expert instance 等紧凑状态，而不是复制
完整 `E x EP` 大矩阵。

## 22. Demand、gain 与增量更新

### 22.1 GRACE 与 UltraEP 的采集差异

讨论中的核心区别是：

- UltraEP 更倾向直接消费 dispatcher/通信阶段已经存在的计数或布局状态。
- GRACE 如果从 `(source, topk, count)` 原始 trace 重新统计 demand 和 gain，
  会额外执行全 trace scan。
- GRACE 常规非 pure 且没有外部 `demand_tensor` 时已经使用 fused demand+initial
  gain，一次扫描同时产生两个统计量；真正额外的成本是通信副本确定后再次计算
  current gain。

因此，GRACE 可能比直接复用 dispatcher counters 的 UltraEP 更慢，但根因不是
demand 与 initial gain 必然分成两次 kernel，而是 refresh 阶段仍可能重新读取
完整 Top-K trace。

### 22.2 Runtime 增量接口

当前 runtime 已增加或保留：

```python
GraceCudaRuntime.refresh_incremental(...)
GraceCudaRuntime.regroup(...)
```

目标是明确区分：

- `refresh_incremental`：trace 拓扑未变化，只更新 changed counts/primary/replica
  相关统计。
- `regroup`：重新执行完整 affinity grouping 和 placement。

runtime 使用 tensor identity、data pointer、shape 和 PyTorch `_version` 构造 trace
index signature。`source` 或 `topk` 的底层存储、shape 或版本发生变化时，缓存
自动失效并重建。

## 23. 本轮最终实现的优化 2-5

### 23.1 真正 incidence CSR

此前增量 current-gain 使用：

```text
bundle_heads[expert, source] -> bundle_next[entry] -> ...
```

虽然 heads/next 已压缩为 int32，但链表遍历不连续，访存 locality 较差。本轮新增
真正的 incidence CSR：

```text
row = source * experts + expert
offsets[row] : offsets[row + 1] -> contiguous entry ids
entry id = token * K + column
```

构建过程分为：

1. 对每个 `(source, expert)` 统计 entry count。
2. prefix scan 生成 int32 offsets。
3. 使用独立 cursor workspace 将 entry id 写入连续数组。

新增 CUDA 绑定：

```text
build_bundle_incidence_csr_into
incremental_bundle_gains_csr_fast_into
```

增量 gain kernel 支持 K=1/2/4/8/16 模板特化和 arbitrary-K fallback。runtime 在
新扩展可用时只构建 CSR，不再同时支付 linked-list 构建成本；旧 linked-list API
继续保留，用于旧扩展和兼容测试。

### 23.2 Active candidate cache

aggregate capacity solver 原先在每一轮 move 中重新扫描完整 `E x EP` 候选。
本轮将当前 overload rank 上 `instance[expert, over] > 0` 的 expert 压缩到 shared
memory 的 active expert 数组，然后只评分：

```text
active_experts x target_ranks
```

这项优化已接入：

- single-CTA aggregate capacity kernel。
- fused current-gain + single-CTA aggregate kernel。
- cooperative multi-CTA aggregate capacity kernel。

候选比较、amount/gain 排序、copy cap、capacity 校验和确定性 tie-breaking 保持
不变。对于 E>256，保留完整矩阵 fallback，避免 shared array 越界。

### 23.3 CSR quota metadata int32

本轮将真正 CSR quota 的索引元数据改为：

| 字段 | dtype | 原因 |
|---|---|---|
| `offsets` | int32 | nnz 已显式限制不超过 `INT_MAX` |
| `targets` | int32 | rank 最大为 63 |
| `boundaries` | int64 | 累计 token/count 可能超过 int32 |

因此没有把所有 CSR 字段机械地压成 int32。`boundaries` 保留 int64 是为了避免
长 trace 或大权重下溢出。Python runtime、materializer、traffic consumer 和
correctness fixture 均同步更新 dtype。

### 23.4 Group-source warp aggregation

group-source shared kernel 原先每个 token/group 都执行一次 shared-memory
`atomicAdd`。本轮使用 warp 内 key：

```text
key = group * MaxRanks + source
```

并通过 `__match_any_sync` 找出相同 key 的 lanes，在 warp 内累加 weight，只有
leader lane 执行一次 shared atomic。每个 token 内仍使用 `seen` bitmask 对 group
去重，因此统计语义不变。

固定 K dispatch 继续覆盖 K=1/2/4/8/16，其他 K 使用 generic fallback。

## 24. Ordinal 与 sparse quota 结论

对话要求“去掉 ordinal 全量排序”和“真正 CSR sparse quota”。当前实现的方向是：

- 不再为 quota 生成全局 token-entry 排序。
- bundle ordinal 使用按 source-rank 的计数过程生成，而不是通用全量 sort。
- quota export 使用每个 `(source, expert)` row 的 CSR boundaries/targets。
- `materialize_quota=False` 时，metrics 直接消费 CSR，不需要重建 dense quota。

需要注意，quota routing 为了确定一个 token 的 weight 区间落在哪个 target，仍需
使用 entry ordinal。优化目标是移除昂贵的全量排序和 dense materialization，
而不是删除所有 ordinal 语义。

## 25. 截图中的旧扩展错误

用户提供的截图显示远端运行：

```text
AttributeError: module 'grace_cuda._C' has no attribute
'select_bundle_topn_routing_index_into'
```

调用栈位于 `gpu_replication.py` 的 grouped planner，说明 Python 已更新，但加载的
`grace_cuda._C` 仍是旧 `.so`，没有导出新绑定。

本轮做了两层处理：

1. Python 使用 `hasattr` 检测 indexed、CSR、incremental 和 fused-capacity
   capability。旧扩展缺少新符号时自动回退已有 full-scan API，不再直接崩溃。
2. 新 C++ binding 正式导出 incidence CSR 和 incremental CSR API。

兼容 fallback 只用于避免版本错配崩溃。要实际启用本轮 CUDA 优化，远端仍必须
重新编译扩展，并确认 Python 加载的是新生成的 `.so`。

## 26. 本轮修改文件

本轮最终直接修改：

```text
benchmark/grace_cuda/csrc/affinity.cu
benchmark/grace_cuda/csrc/bindings.cpp
benchmark/grace_cuda/csrc/compute_v2.cu
benchmark/grace_cuda/csrc/demand.cu
benchmark/grace_cuda/csrc/quota.cu
benchmark/grace_cuda/test_correctness.py
python/sglang/srt/eplb/gpu_replication.py
```

此前对话中还涉及但本轮没有重新覆盖的文件包括：

```text
benchmark/grace_cuda/csrc/placement.cu
benchmark/grace_cuda/csrc/runtime.cu
benchmark/grace_cuda/csrc/launch.cuh
benchmark/grace_cuda/csrc/limits.cuh
benchmark/grace_cuda/csrc/pure_compute.cu
benchmark/grace_cuda/csrc/traffic.cu
```

`limits.cuh` 中统一最大 EP 为 64。当前工作区仍是未提交状态，不能把本文当作
git commit 记录。

## 27. 正确性测试扩展

当前 `test_correctness.py` 已覆盖或计划覆盖：

- K=1/2/4/8/16/10 的 demand/current-gain 等价性。
- linked-list incremental gain 与 full current-gain 等价性。
- incidence CSR incremental gain 与 full current-gain 等价性。
- EP32/EP64 的 group-source single/multi block 等价性。
- single CTA 与 multi CTA aggregate solver 输出等价性。
- dense、prefix sparse 和 CSR quota 的 reconstructed quota、traffic、compute
  等价性。
- CSR offsets/targets 的 int32 dtype。

本轮本地已通过：

```bash
git diff --check
python3 -m py_compile \
  python/sglang/srt/eplb/gpu_replication.py \
  benchmark/grace_cuda/test_correctness.py
```

本机环境没有 `nvcc`，当前 Python 环境也没有 PyTorch，因此本轮不能在本地完成：

- CUDA extension 编译。
- pybind symbol load 验证。
- GPU correctness test。
- single/multi CTA A/B benchmark。

因此，本轮新增 CUDA 代码当前状态是“已实现并通过静态检查，等待远端 CUDA
编译和运行验证”，不能描述为已经通过 GPU 回归。

## 28. 远端重新构建与验证

远端首先重新构建扩展：

```bash
cd /home/admin/workspace/sglang/benchmark/grace_cuda
TORCH_CUDA_ARCH_LIST="10.0a" MAX_JOBS=16 \
  uv pip install -e . --no-build-isolation
```

也可在对应远端虚拟环境中使用：

```bash
pip install -e .
```

然后确认新符号：

```bash
python - <<'PY'
from grace_cuda import _C
for name in (
    "select_bundle_topn_routing_index_into",
    "build_bundle_incidence_csr_into",
    "incremental_bundle_gains_csr_fast_into",
    "materialize_fast_csr_quota_into",
    "csr_quota_traffic",
):
    print(name, hasattr(_C, name))
PY
```

正确性测试：

```bash
cd /home/admin/workspace/sglang/benchmark/grace_cuda
PYTHONPATH=../../python ../../.venv/bin/python test_correctness.py
```

建议矩阵：

```text
EP:          1, 2, 4, 8, 16, 32, 64
K:           1, 2, 4, 8, 10, 16
solver CTAs: 1, 2, 4, 8
quota:       dense, prefix sparse, CSR sparse
refresh:     unchanged source/topk; changed source/topk/count/primary
```

重点断言：

- replicas、addition order、move plan 和 routing 完全一致。
- dense 与 CSR traffic/compute metrics 完全一致。
- quota conservation：每个 `(source, expert)` 的 quota sum 等于 demand。
- quota target 都拥有对应 replica。
- single CTA 与 cooperative multi CTA 结果一致。
- EP64 不发生 shared-memory 越界或 int32 metadata 溢出。
- 第二次相同 trace refresh 使用 CSR incremental path，而不是 full trace scan。

## 29. 当前结论与后续优先级

截至 2026-08-28，本轮请求的优化 2-5 已在当前工作区实现：incidence CSR、
active expert candidate cache、CSR quota int32 metadata 和 group-source warp
aggregation。同时增加了旧 CUDA extension 的 capability fallback，解决截图中的
AttributeError。

在远端 GPU 回归通过之前，最重要的下一步不是继续扩大重构，而是：

1. 编译并检查所有 pybind symbols。
2. 跑完整 `test_correctness.py`。
3. 对 EP1-64、K8、single/multi CTA 做确定性 A/B。
4. profile 第二次相同 trace refresh，确认 incidence CSR 实际减少了链表随机访存
   和 full current-gain scan。
5. profile active expert 数量。如果 overload rank 通常覆盖几乎全部 expert，active
   cache 收益会有限；如果 active set 明显稀疏，则应保留。
6. 比较 warp aggregation 前后 group-source atomic throughput，确认
   `__match_any_sync` 的循环归约成本低于减少的 atomic 冲突。

本轮没有进行 git commit 或 push。远端验证通过后，应先审阅完整 diff，再提交，
避免再次因服务器状态丢失未保存优化。
