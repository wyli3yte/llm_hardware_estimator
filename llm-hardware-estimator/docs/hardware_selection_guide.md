# Hardware Selection Guide

## 选型顺序

1. 先看 `fit_status` 和 `recommendation_level`，排除 Not Fit。
2. 再看 `suggested_gpu_count`、`memory_margin_ratio` 和 `multi_gpu_risk`。
3. 对采购候选，优先保留 `source_confidence=high/medium` 的方案。
4. 对长上下文和高并发场景，优先选择更高显存和更高带宽的GPU。

## 风险解读

- `capacity`：容量是第一瓶颈，直接影响是否 OOM。
- `bandwidth`：decode 阶段可能受显存带宽限制。
- `compute`：prefill 或 forward 阶段可能受算力限制。
- `multi_gpu`：需要实测并行策略、通信和框架调度。
- `data_missing`：规格缺失，不能作为最终采购依据。
