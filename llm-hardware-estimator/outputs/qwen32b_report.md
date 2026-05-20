# LLM推理硬件需求估算报告

## 一句话结论
qwen3-32b 场景下，推荐优先评估 2 × 曦云C500 64GB（沐曦），等级 C。

## 输入参数快照
- 输入tokens：4096
- 输出tokens：512
- 并发：4
- 权重精度：bf16
- KV精度：bf16

## 显存估算
- 权重显存：59.60 GB
- KV Cache平均显存：4.25 GB
- KV Cache峰值显存：4.50 GB
- 运行时预留：6.64 GB
- 总显存：70.74 GB
- 生产建议显存：91.96 GB

## GPU候选
| rank | vendor | gpu_model | level | cards | margin | bottleneck | confidence | note |
|---:|---|---|---|---:|---:|---|---|---|
| 1 | 沐曦 | 曦云C500 64GB | C | 2 | 7% | compute | medium | 建议进入方案比选 |
| 2 | 华为昇腾 | 910B 64GB平台 | C | 2 | 7% | compute | medium | 建议进入方案比选 |
| 3 | 摩尔线程 | MTT S4000 | B | 3 | 19% | compute | high | 建议进入方案比选 |
| 4 | 寒武纪 | MLU370-X8 | B | 3 | 19% | compute | high | 建议进入方案比选 |
| 5 | 昆仑芯 | R200-8F 32GB | Not Fit | 4 | 5% | compute | high | 不建议作为当前配置候选 |

## 场景对比
| scenario | input | output | concurrency | production_gb | suggested_cards | suggested_gpu | note |
|---|---:|---:|---:|---:|---:|---|---|
| chat | 1024 | 512 | 2 | 86.36 | 2 | 曦云C500 64GB | 可作为生产试点 |
| demo | 2048 | 256 | 1 | 86.08 | 2 | 曦云C500 64GB | 可作为生产试点 |
| embedding | 1024 | 0 | 16 | 91.21 | 2 | 曦云C500 64GB | 可作为生产试点 |
| long_doc | 32768 | 1024 | 2 | 109.90 | 3 | 曦云C500 64GB | 可作为生产试点 |
| rag | 4096 | 512 | 4 | 91.96 | 2 | 曦云C500 64GB | 可作为生产试点 |
| reranker | 1024 | 0 | 8 | 88.22 | 2 | 曦云C500 64GB | 可作为生产试点 |

## 方法与边界
- 显存由权重、KV Cache、运行时开销和生产余量组成。
- TTFT/TPOT仅给出理论瓶颈提示，正式采购前需要目标硬件压测。
- 国产GPU公开规格不完整时，报告保留来源可信度和待确认风险。
