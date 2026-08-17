# Pairwise 与 GRACE-MoE 离线专家放置

当前实验只保留两种主专家放置算法：Pairwise 和 GRACE-MoE。两者使用相同的路由 trace、通信 replay 和负载指标，不包含专家复制、在线路由或其他 refinement。

## Pairwise

Pairwise 将每个专家视为顶点，将同一 Top-K bundle 中的专家两两连接。边权是专家对的共现次数。

求解流程：

1. 按加权度和专家负载生成容量受限的初始 placement。
2. 用 pairwise edge-cut delta 筛选跨 rank 交换，默认保留最优的 32 个候选。
3. 仅重放包含候选专家的 bundle，按真实节点内/跨节点通信代价重排。
4. 接受通信代价下降且计算不均衡不超过限制的交换。
5. 达到轮数上限或不存在改进交换时停止。

Pairwise 默认保持每个 rank 的专家容量一致。通信重排默认使用 `rdma_cost=4`，计算不均衡上限为 `1.2`。

## GRACE-MoE

GRACE-MoE 使用相同的专家共现图，但通过谱聚类生成分层 placement：

1. 节点级使用非均匀谱聚类，优先减少跨节点专家共现。
2. 节点内再次谱聚类，将专家映射到 GPU。
3. 使用论文 Algorithm 2 的受控非均匀范围限制每个 GPU 的专家数量。
4. 默认非均匀比例为 `r=0.15`。

GRACE-MoE 允许不同 GPU 保存不同数量的专家，因此可能降低通信，也可能增加计算和显存不均衡。

## 对比指标

统一脚本输出：

| 指标 | 含义 |
|---|---|
| `remote` | token 访问的不同远程 rank 数量 |
| `weighted` | 节点内通信权重为 1、跨节点通信权重为 `rdma_cost` 的通信量 |
| `cut` | Pairwise 共现图的跨 rank 边权 |
| `comp` | 最大 rank 计算负载除以平均负载 |
| `experts/rank` | 每个 rank 的最少和最多专家数 |

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

单独运行 Pairwise。单机全部使用 NVLink/NVSwitch，因此 `ranks-per-node=8`、`rdma-cost=1`：

```bash
PYTHONPATH=python python benchmark/compare_pairwise_grace.py \
  --input /tmp/qwen3_ep8_trace.pt \
  --num-ranks 8 \
  --ranks-per-node 8 \
  --rdma-cost 1 \
  --method pairwise \
  --pairwise-candidates 32 \
  --pairwise-max-imbalance 1.2
```

单独运行 GRACE 时将最后三个 Pairwise 参数替换为 `--method grace`。不传 `--method` 仍会依次运行两种方案。

采集脚本使用 SGLang 自带的 `return_routed_experts`，source rank 直接取每个请求返回的真实 `dp_rank`。脚本要求 `tp=dp=ep`，并关闭 CUDA graph 和 overlap schedule。

`moe-a2a-backend=none` 可以用于采集路由，但实际运行的是 all-gather/reduce-scatter，专家重排不会减少这条通信路径的通信量。Pairwise/GRACE 的通信收益需要切换到 token A2A backend 后验证。

运行定向测试：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/eplb/test_moe_bundle_trace.py \
  test/registered/unit/eplb/test_pairwise_grace_placement.py
```

保存两个 placement：

```bash
PYTHONPATH=python python benchmark/compare_pairwise_grace.py \
  --input /tmp/qwen3_ep8_trace.pt \
  --num-ranks 8 \
  --ranks-per-node 8 \
  --rdma-cost 1 \
  --save-pairwise pairwise.json \
  --save-grace grace.json
```

相关文件：

```text
python/sglang/srt/eplb/co_routing_graph_solver.py
python/sglang/srt/eplb/grace_expert_placement.py
python/sglang/srt/eplb/moe_bundle_trace.py
benchmark/benchmark_ep_trace.py
benchmark/compare_pairwise_grace.py
test/registered/unit/eplb/test_moe_bundle_trace.py
test/registered/unit/eplb/test_pairwise_grace_placement.py
```
