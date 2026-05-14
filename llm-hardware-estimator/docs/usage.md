# LLM Hardware Estimator Usage

## 快速开始

```bash
python3 llm-hardware-estimator/estimator.py validate-configs
python3 llm-hardware-estimator/estimator.py recommend --model qwen3-32b --scenario rag --concurrency 4
```

生成报告和表格：

```bash
python3 llm-hardware-estimator/estimator.py recommend \
  --model qwen3-32b \
  --scenario rag \
  --concurrency 4 \
  --export-md llm-hardware-estimator/outputs/qwen32b_report.md \
  --export-json llm-hardware-estimator/outputs/qwen32b_result.json \
  --export-csv llm-hardware-estimator/outputs/qwen32b_csv \
  --export-xlsx llm-hardware-estimator/outputs/qwen32b_compare.xlsx
```

## 常用命令

- `estimate`：指定模型和GPU，输出容量估算和该GPU可行性。
- `recommend`：按国产GPU库排序推荐候选。
- `buy`：输出成本优先、均衡、性能优先三档采购建议。
- `compare-precision`：比较 BF16、W8A16、W4A16、KV INT8 等精度组合。
- `validate`：读取 1.7B benchmark CSV，计算显存误差和验证结论。
- `validate-configs`：校验默认模型、硬件、场景配置。

## 配置说明

配置位于 `llm-hardware-estimator/configs/`。文件使用 JSON-compatible YAML：保留 `.yaml` 扩展名，但内容也能被 Python 标准库 `json` 直接读取；如果安装了 PyYAML，也兼容普通 YAML。

## 重要边界

工具默认做容量和理论瓶颈初筛，不承诺真实 TTFT/TPOT/E2E。正式采购或上线前，仍需要在目标硬件、目标框架和目标并发下压测。
