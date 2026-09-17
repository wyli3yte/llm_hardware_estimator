import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .export_xlsx import write_xlsx


def _clean(value):
    if is_dataclass(value):
        return _clean(asdict(value))
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def result_payload(
    model_id,
    scenario,
    memory,
    candidates=None,
    precision_comparison=None,
    scenario_comparison=None,
    sources=None,
    validation=None,
):
    return _clean(
        {
            "model_id": model_id,
            "scenario": scenario,
            "memory": memory,
            "gpu_candidates": [item.to_dict() for item in candidates or []],
            "precision_comparison": precision_comparison or [],
            "scenario_comparison": scenario_comparison or [],
            "sources": sources or [],
            "validation": validation or [],
        }
    )


def render_markdown(payload, title="LLM推理硬件需求估算报告", buyer_options=None):
    memory = payload["memory"]
    candidates = payload.get("gpu_candidates", [])
    best = candidates[0] if candidates else None
    lines = [
        "# %s" % title,
        "",
        "## 一句话结论",
    ]
    if best:
        lines.append(
            "%s 场景下，推荐优先评估 %s × %s（%s），等级 %s。"
            % (
                payload["model_id"],
                best["suggested_gpu_count"],
                best["gpu_model"],
                best["vendor"],
                best["recommendation_level"],
            )
        )
    else:
        lines.append("%s 当前配置未找到可推荐GPU候选。" % payload["model_id"])

    lines.extend(
        [
            "",
            "## 输入参数快照",
            "- 输入tokens：%s" % payload["scenario"].get("input_tokens"),
            "- 输出tokens：%s" % payload["scenario"].get("output_tokens", 0),
            "- 并发：%s" % payload["scenario"].get("concurrency"),
            "- 权重精度：%s" % payload["scenario"].get("weight_precision"),
            "- KV精度：%s" % payload["scenario"].get("kv_cache_precision"),
            "",
            "## 显存估算",
            "- 权重显存：%.2f GB" % memory["weight_memory_gb"],
            "- KV Cache平均显存：%.2f GB" % memory["kv_cache_avg_gb"],
            "- KV Cache峰值显存：%.2f GB" % memory["kv_cache_peak_gb"],
            "- 运行时预留：%.2f GB" % memory["runtime_overhead_gb"],
            "- 总显存：%.2f GB" % memory["total_memory_gb"],
            "- 生产建议显存：%.2f GB" % memory["production_memory_gb"],
            "",
            "## GPU候选",
            "| rank | vendor | gpu_model | level | cards | margin | bottleneck | confidence | note |",
            "|---:|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for item in candidates:
        lines.append(
            "| {rank} | {vendor} | {gpu_model} | {recommendation_level} | {suggested_gpu_count} | {memory_margin_ratio:.0%} | {main_bottleneck} | {source_confidence} | {decision_note} |".format(
                **item
            )
        )
    if buyer_options:
        lines.extend(["", "## 采购三档方案"])
        for option in buyer_options:
            lines.append(
                "- %s：%s × %s，%s"
                % (option["name"], option["gpu_count"], option["gpu_model"], option["note"])
            )
    if payload.get("precision_comparison"):
        lines.extend(
            [
                "",
                "## 精度对比",
                "| precision | weight_gb | kv_peak_gb | production_gb | quality_risk |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in payload["precision_comparison"]:
            lines.append(
                "| {precision} | {weight_memory_gb:.2f} | {kv_cache_peak_gb:.2f} | {production_memory_gb:.2f} | {quality_risk} |".format(
                    **row
                )
            )
    if payload.get("scenario_comparison"):
        lines.extend(
            [
                "",
                "## 场景对比",
                "| scenario | input | output | concurrency | production_gb | suggested_cards | suggested_gpu | note |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in payload["scenario_comparison"]:
            lines.append(
                "| {scenario} | {input_tokens} | {output_tokens} | {concurrency} | {production_memory_gb:.2f} | {suggested_gpu_count} | {suggested_gpu_model} | {note} |".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "## 方法与边界",
            "- 显存由权重、KV Cache、运行时开销和生产余量组成。",
            "- TTFT/TPOT仅给出理论瓶颈提示，正式采购前需要目标硬件压测。",
            "- 国产GPU公开规格不完整时，报告保留来源可信度和待确认风险。",
        ]
    )
    return "\n".join(lines) + "\n"


def export_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def export_markdown(path, payload, buyer_options=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload, buyer_options=buyer_options), encoding="utf-8")


def _pct(value):
    if value is None:
        return ""
    return "%.0f%%" % (float(value) * 100)


def _decision_rows(payload, buyer_options=None):
    scenario = payload.get("scenario", {})
    candidates = payload.get("gpu_candidates", [])
    best = candidates[0] if candidates else {}
    risks = []
    if best.get("multi_gpu_risk") and best.get("multi_gpu_risk") != "low":
        risks.append("多卡并行需验证")
    if best.get("source_confidence") != "high":
        risks.append("硬件规格需供应商确认")
    if best.get("memory_margin_ratio", 0) < 0.10:
        risks.append("显存余量偏低")
    risks.append("真实TTFT/TPOT需目标环境压测")
    next_steps = [
        "确认候选GPU规格和可采购型号",
        "使用目标API运行1.7B或目标模型benchmark",
        "验证量化质量和长上下文稳定性",
    ]
    option_text = "；".join(
        "%s：%s × %s" % (item["name"], item["gpu_count"], item["gpu_model"])
        for item in (buyer_options or [])
    )
    return [
        {"项目": "报告标题", "内容": "LLM推理硬件需求估算与国产GPU选型报告"},
        {"项目": "报告用途", "内容": "面向模型容量估算、国产GPU初筛和测试验证"},
        {"项目": "模型", "内容": payload.get("model_id", "")},
        {"项目": "场景", "内容": scenario.get("name", "")},
        {"项目": "并发", "内容": scenario.get("concurrency", "")},
        {"项目": "精度", "内容": "%s / KV %s" % (scenario.get("weight_precision", ""), scenario.get("kv_cache_precision", ""))},
        {
            "项目": "推荐结论",
            "内容": "%s × %s（%s），等级%s"
            % (
                best.get("suggested_gpu_count", ""),
                best.get("gpu_model", ""),
                best.get("vendor", ""),
                best.get("recommendation_level", ""),
            )
            if best
            else "未找到推荐候选",
        },
        {"项目": "生产建议显存", "内容": "%.2f GB" % payload.get("memory", {}).get("production_memory_gb", 0)},
        {"项目": "首选显存余量", "内容": _pct(best.get("memory_margin_ratio")) if best else ""},
        {"项目": "三档方案", "内容": option_text},
        {"项目": "关键风险Top3", "内容": "；".join(risks[:3])},
        {"项目": "下一步验证Top3", "内容": "；".join(next_steps)},
        {"项目": "实测校准状态", "内容": "已有验证数据" if payload.get("validation") else "暂无实测校准数据"},
    ]


def _option_rows(buyer_options=None):
    rows = []
    for item in buyer_options or []:
        rows.append(
            {
                "方案": item["name"],
                "GPU": item["gpu_model"],
                "卡数": item["gpu_count"],
                "为什么选": item["note"],
                "主要风险": "需目标硬件压测确认",
                "适合场景": "采购比选 / 客户演示 / 生产试点",
            }
        )
    return rows or [{"方案": "暂无", "GPU": "", "卡数": "", "为什么选": "当前约束下未找到可行方案", "主要风险": "", "适合场景": ""}]


def _gpu_rows(payload):
    rows = []
    for item in payload.get("gpu_candidates", []):
        rows.append(
            {
                "排名": item.get("rank"),
                "推荐等级": item.get("recommendation_level"),
                "厂商": item.get("vendor"),
                "GPU型号": item.get("gpu_model"),
                "是否满足": item.get("fit_status"),
                "推荐卡数": item.get("suggested_gpu_count"),
                "单卡显存GB": item.get("memory_per_gpu_gb"),
                "显存余量%": _pct(item.get("memory_margin_ratio")),
                "主要瓶颈": item.get("main_bottleneck"),
                "多卡风险": item.get("multi_gpu_risk"),
                "数据可信度": item.get("source_confidence"),
                "采购建议": item.get("decision_note"),
            }
        )
    return rows


def _what_if_rows(payload):
    rows = []
    for item in payload.get("precision_comparison", []):
        rows.append(
            {
                "对比类型": "precision",
                "方案/场景": item.get("precision"),
                "输入tokens": "",
                "输出tokens": "",
                "并发": "",
                "生产建议显存GB": item.get("production_memory_gb"),
                "推荐GPU": "",
                "推荐卡数": "",
                "说明": "质量风险：%s" % item.get("quality_risk", ""),
            }
        )
    for item in payload.get("scenario_comparison", []):
        rows.append(
            {
                "对比类型": "scenario",
                "方案/场景": item.get("scenario"),
                "输入tokens": item.get("input_tokens"),
                "输出tokens": item.get("output_tokens"),
                "并发": item.get("concurrency"),
                "生产建议显存GB": item.get("production_memory_gb"),
                "推荐GPU": item.get("suggested_gpu_model"),
                "推荐卡数": item.get("suggested_gpu_count"),
                "说明": item.get("note"),
            }
        )
    return rows


def _appendix_rows(payload):
    rows = [{"类别": "assumption", "项目": key, "内容": value} for key, value in payload.get("scenario", {}).items()]
    rows.extend(
        {
            "类别": "source",
            "项目": "%s %s" % (item.get("vendor", ""), item.get("gpu_model", "")),
            "内容": "%s | %s | %s" % (item.get("source_type", ""), item.get("confidence", ""), item.get("source_url", "")),
        }
        for item in payload.get("sources", [])
    )
    if payload.get("validation"):
        rows.extend(
            {
                "类别": "validation",
                "项目": item.get("run_id", ""),
                "内容": "显存误差=%s 结论=%s" % (item.get("memory_error_ratio", ""), item.get("conclusion", "")),
            }
            for item in payload.get("validation", [])
        )
    else:
        rows.append({"类别": "validation", "项目": "校准数据", "内容": "暂无实测校准数据"})
    rows.append({"类别": "test_guide", "项目": "显存误差判定", "内容": "显存误差≤5%：通过；5%-10%：可接受；>10%：不通过"})
    rows.append({"类别": "test_guide", "项目": "样本数规则", "内容": "单场景样本数≥10输出p50/p90/p95；样本数<10只看p50/min/max"})
    rows.append({"类别": "test_guide", "项目": "连通性测试", "内容": "先用OpenAI-compatible /v1/chat/completions或/v1/embeddings做小token流式请求，确认模型、API key和endpoint可用"})
    rows.append({"类别": "test_guide", "项目": "压测命令模板", "内容": "python3 llm-hardware-estimator/tools/benchmark_openai.py --base-url <endpoint> --api-key \"$OPENAI_API_KEY\" --model <model_name> --input-tokens 2048 --output-tokens 256 --concurrency 4 --repeat 5 --output llm-hardware-estimator/data/benchmark_1.7b.csv"})
    rows.append({"类别": "test_guide", "项目": "结果导入模板", "内容": "python3 llm-hardware-estimator/estimator.py validate --benchmark llm-hardware-estimator/data/benchmark_1.7b.csv --model qwen-1.7b --scenario rag --output llm-hardware-estimator/outputs/validation.xlsx"})
    rows.append({"类别": "limitation", "项目": "性能预测", "内容": "TTFT/TPOT需目标硬件、目标框架和目标并发压测确认"})
    rows.append({"类别": "limitation", "项目": "敏感信息", "内容": "VPN、账号、API key等测试环境信息不写入公开仓库输出样例"})
    return rows


def export_csv_dir(path, payload):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tables = {
        "gpu_comparison.csv": payload.get("gpu_candidates", []),
        "precision_comparison.csv": payload.get("precision_comparison", []),
        "scenario_comparison.csv": payload.get("scenario_comparison", []),
        "sources.csv": payload.get("sources", []),
        "validation.csv": payload.get("validation", []),
        "assumptions.csv": [{"key": key, "value": value} for key, value in payload.get("scenario", {}).items()],
    }
    for filename, rows in tables.items():
        with (path / filename).open("w", encoding="utf-8", newline="") as fh:
            if not rows:
                fh.write("")
                continue
            keys = []
            for row in rows:
                for key in row.keys():
                    if key not in keys:
                        keys.append(key)
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def export_xlsx(path, payload, buyer_options=None):
    write_xlsx(
        path,
        {
            "Decision": _decision_rows(payload, buyer_options),
            "Options": _option_rows(buyer_options),
            "GPU_Comparison": _gpu_rows(payload),
            "What_If": _what_if_rows(payload),
            "Appendix": _appendix_rows(payload),
        },
    )
