# GRACE/CABLE 离线专家放置

GRACE 从每层 Top-K 路由构造 expert co-activation affinity graph，再做层级谱聚类和受控非均匀分组。CABLE 独立使用完整 Top-K bundle 做 NumPy 贪心放置，再执行受约束 swap，并直接约束 remote、端口拥塞和计算负载。

GRACE 的完整论文方案还包括动态专家复制和在线 locality-aware routing；本仓库当前实现的是离线 grouping 部分。

## 运行

真实 SGLang 离线采集（单机 8 卡 NVLink/NVSwitch）：

```bash
PYTHONPATH=python python benchmark/benchmark_ep_trace.py \
  --model Qwen/Qwen3-30B-A3B \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend none \
  --dataset sharegpt \
  --num-samples 128 \
  --batch-size 8 \
  --max-new-tokens 1 \
  --output /tmp/qwen3_ep8_trace.pt
```

运行 Grace。compact trace 超过 `--optimizer-bundles` 时，求解阶段只使用可复现的 bundle 样本，避免创建千万级 Python 对象：

```bash
PYTHONPATH=python python benchmark/compare_grace.py \
  --input /tmp/qwen3_ep8_trace.pt \
  --num-ranks 8 \
  --ranks-per-node 8 \
  --rdma-cost 1 \
  --optimizer-bundles 20000 \
  --grace-ratio 0.15 \
  --save-grace grace.json
```

只追求最快求解速度时跳过 GRACE 图构造，直接运行 CABLE：

```bash
PYTHONPATH=python python benchmark/compare_grace.py \
  --input /tmp/qwen3_ep8_trace.pt \
  --num-ranks 8 \
  --ranks-per-node 8 \
  --rdma-cost 1 \
  --cable-only \
  --save-cable cable.json
```

默认执行两轮受约束 remote-decreasing swap；只测最快的初始贪心时加上 `--cable-refine-swaps 0`。

`--optimizer-bundles 0` 表示使用完整 compact trace，但一千万 bundle 会明显变慢。`--rdma-cost 1` 适用于单机 NVLink/NVSwitch；`none` backend 仅用于采集路由，实际通信收益需要在 token A2A backend 下验证。

## 后续改进方向

- 按论文的 affinity utilization `U(r)` 与 size deviation `S(r)` 自动选择每层非均匀比例，而不是固定 `0.15`。
- 补齐动态 hot-expert replication 和 locality-aware routing。
- 使用 compact tensor 直接构造 affinity，避免 Python bundle 转换。
- 用全量 trace 分块验证最终 placement，不在求解阶段扫描所有 Python 对象。

## 测试

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/eplb/test_moe_bundle_trace.py
```

## 相关文件

```text
python/sglang/srt/eplb/expert_affinity_graph.py
python/sglang/srt/eplb/grace_expert_placement.py
python/sglang/srt/eplb/moe_bundle_trace.py
benchmark/benchmark_ep_trace.py
benchmark/compare_grace.py
test/registered/unit/eplb/test_moe_bundle_trace.py
```
