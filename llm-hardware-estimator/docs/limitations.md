# Limitations

- 不自动联网抓取GPU规格，所有硬件数据来自本地配置。
- 不预测可交付级 TTFT/TPOT/E2E，只给理论下界和瓶颈提示。
- 不评估量化后的模型质量。
- 不模拟复杂多卡拓扑、专家并行 All-to-All、speculative decoding 或 chunked prefill。
- 国产GPU部分公开规格不完整，低/中可信度条目需要供应商确认。
