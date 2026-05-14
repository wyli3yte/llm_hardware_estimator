# Methodology

## 显存模型

总显存由权重显存、KV Cache 峰值显存、运行时开销和生产余量组成。

```text
weight_memory_gb = params_billion * 1e9 * bytes_per_param * quant_overhead / 1024^3
kv_cache_peak_gb = 2 * layers * kv_heads * head_dim * (input_tokens + output_tokens) * concurrency * bytes_per_kv / 1024^3
runtime_overhead_gb = max(absolute_gb, weight_memory_gb * weight_ratio + kv_cache_peak_gb * kv_cache_ratio)
production_memory_gb = (weight_memory_gb + kv_cache_peak_gb + runtime_overhead_gb) * (1 + production_margin_ratio)
```

Embedding 和 Reranker 不分配自回归 KV Cache。MoE 权重显存按总参数量计算，FLOPs 和带宽按 active params 估算。

## 多卡口径

工具会按候选卡数生成保守并行策略：KV heads 能被卡数整除时优先 Tensor Parallel，否则使用 Pipeline Parallel 口径估算。若用户显式传入不合法 TP 配置，核心估算会标记 `supported_parallel_config=false`。

## 推荐排序

推荐先硬过滤精度和容量，再按目标偏好评分。`budget` 更关注卡数和来源可信度，`balanced` 综合显存余量、带宽、算力和平台能力，`performance` 更关注带宽、算力和余量。
