# GRACE CUDA 复制、分组与负载均衡优化：对话总结

更新时间：2026-08-26

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
