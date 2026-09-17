# LLM推理硬件需求估算工具使用说明

## 1. 工具用途

本工具用于根据模型、业务场景、并发规模和推理精度，估算推理部署所需 GPU 显存、推荐卡数和候选硬件方案，并导出 Excel 对比报告。

典型用途包括：

- 评估指定模型在不同业务场景下需要多少张 GPU。
- 对比不同 GPU 型号的显存余量、风险和适配程度。
- 输出可用于方案沟通的 Excel 报告。
- 结合压测结果校准理论估算。

## 2. 仓库内容

仓库主要包含：

- `llm-hardware-estimator/START_WEB.command`：macOS Web 门户启动脚本。
- `llm-hardware-estimator/web_app.py`、`llm-hardware-estimator/web/`：Web 门户服务和页面文件。
- `llm-hardware-estimator/estimator.py`、`llm-hardware-estimator/estimator/`：估算核心程序。
- `llm-hardware-estimator/configs/`：模型、场景、硬件和平台配置。
- `llm-hardware-estimator/tools/benchmark_openai.py`：OpenAI-compatible 接口压测脚本。
- `llm-hardware-estimator/outputs/qwen32b_compare.xlsx`：示例 Excel 报告。
- `README.md`：本说明文档。

仓库不包含账号、密码、VPN 信息、API Key、内网地址、个人资料、临时报表或系统隐藏文件。

## 3. 快速使用

### 3.1 macOS 双击启动

1. 解压交付包。
2. 进入 `llm-hardware-estimator` 文件夹。
3. 双击 `START_WEB.command`。
4. 终端窗口出现访问地址后，打开浏览器访问：

```text
http://127.0.0.1:8080
```

5. 在页面中选择参数并点击“生成方案”。
6. 点击“下载Excel报告”保存结果。

如系统提示脚本无法打开，可右键点击 `START_WEB.command`，选择“打开”。

### 3.2 命令行启动

如需手动启动门户，可在项目目录执行：

```bash
python3 web_app.py --host 0.0.0.0 --port 8080
```

浏览器访问：

```text
http://127.0.0.1:8080
```

如 8080 端口被占用，可改用 8081：

```bash
python3 web_app.py --host 0.0.0.0 --port 8081
```

## 4. 门户操作指引

Web 门户左侧为输入区，右侧为结果区。

### 4.1 输入区

- 模型：选择待估算模型，例如 `qwen3-32b`。
- 场景：选择业务场景，例如 `rag`、`chat`、`long_doc`。
- 并发：填写同时处理的请求数量，例如 `4`。
- 精度：选择推理精度，例如 `bf16`、`w8a16`、`w4a16`。
- 决策目标：选择方案偏好，包括成本优先、均衡、性能优先。
- 硬件范围：选择候选硬件范围，国产 GPU 优先选择 `domestic`。

填写完成后点击“生成方案”。

### 4.2 结果区

结果区会展示：

- 推荐结论：推荐 GPU 型号、卡数和推荐等级。
- 生产建议显存：考虑运行时开销和生产余量后的显存需求。
- 主要风险：包括多卡并行、显存余量、规格可信度等风险。
- 三档方案：成本优先、均衡、性能优先。
- GPU 候选表：展示不同硬件的适配状态、推荐卡数、显存余量和采购建议。

点击“下载Excel报告”可导出完整对比表。

## 5. Excel 报告说明

Excel 报告包含以下工作表：

- `Decision`：推荐结论、输入摘要、关键风险和下一步验证建议。
- `Options`：三档选型方案。
- `GPU_Comparison`：候选 GPU 方案对比。
- `What_If`：不同精度和场景下的估算变化。
- `Appendix`：配置假设、数据来源、测试指引和边界说明。

报告结果用于方案初筛。正式选型前，应结合目标环境压测结果进行校准。

## 6. 局域网访问

如需让同一局域网内其他终端访问门户：

1. 在运行工具的主机上启动 `START_WEB.command`。
2. 查询主机局域网 IP，例如 `192.168.1.23`。
3. 访问终端打开：

```text
http://192.168.1.23:8080
```

如无法访问，应检查防火墙、网络策略和端口占用情况。

## 7. 命令行生成报告

也可不打开门户，直接生成 Excel：

```bash
python3 estimator.py recommend --model qwen3-32b --scenario rag --concurrency 4 --output outputs/qwen32b_compare.xlsx
```

常用参数：

- `--model`：模型 ID。
- `--scenario`：场景 ID。
- `--concurrency`：并发数。
- `--output`：输出文件路径。

## 8. 如何迭代配置

### 8.1 新增或修改模型

编辑：

```text
configs/models.yaml
```

重点维护以下信息：

- 模型参数量
- 层数
- hidden size
- attention heads / kv heads
- 上下文长度
- 默认精度

模型结构参数越完整，显存估算越准确。

### 8.2 新增或修改场景

编辑：

```text
configs/scenarios/
```

重点维护以下信息：

- 输入 token 数
- 输出 token 数
- 并发数
- KV Cache 精度
- 生产余量

长上下文、高并发和高输出 token 场景通常会显著增加显存需求。

### 8.3 新增或修改硬件

编辑：

```text
configs/hardware.yaml
```

重点维护以下信息：

- GPU 型号
- 单卡显存
- 显存带宽
- 理论算力
- 数据来源
- 来源可信度
- 平台适配说明

硬件规格不完整时，应在配置中保留可信度说明，避免将待确认信息作为确定结论。

## 9. 压测与校准

估算结果应通过目标环境压测校准。压测前准备：

- OpenAI-compatible API endpoint
- API Key
- 模型名称
- 代表性输入文本

示例命令：

```bash
python3 tools/benchmark_openai.py --base-url <endpoint> --api-key "<api-key>" --model <model_name> --input-tokens 2048 --output-tokens 256 --concurrency 4 --repeat 5 --output data/benchmark_1.7b.csv
```

导入校准：

```bash
python3 estimator.py validate --benchmark data/benchmark_1.7b.csv --model qwen-1.7b --scenario rag --output outputs/validation.xlsx
```

校准规则：

- 显存误差小于等于 5%：通过。
- 显存误差大于 5% 且小于等于 10%：可接受，建议补测。
- 显存误差大于 10%：不通过，应核对模型参数、场景配置或运行环境。

## 10. 常见问题

### 10.1 双击脚本无法启动

先确认 Python 3 是否可用：

```bash
python3 --version
```

如未安装 Python 3，应先安装后再启动。

### 10.2 端口被占用

可改用其他端口：

```bash
PORT=8081 ./START_WEB.command
```

或手动执行：

```bash
python3 web_app.py --host 0.0.0.0 --port 8081
```

### 10.3 页面可以打开但生成失败

建议检查：

- 模型 ID 是否存在于 `configs/models.yaml`。
- 场景 ID 是否存在于 `configs/scenarios/`。
- 并发是否为正整数。
- 终端窗口是否显示错误信息。

## 11. 注意事项

- 交付包不应写入账号、密码、VPN、API Key、内网地址等敏感信息。
- Excel 报告为估算和方案初筛结果，不构成最终采购承诺。
- 真实 TTFT、TPOT、吞吐、稳定性应通过目标环境压测确认。
- 临时报表、本地私有配置和系统隐藏文件不属于交付范围。
