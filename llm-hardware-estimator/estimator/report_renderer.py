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


def result_payload(model_id, scenario, memory, candidates=None, precision_comparison=None, validation=None):
    return _clean(
        {
            "model_id": model_id,
            "scenario": scenario,
            "memory": memory,
            "gpu_candidates": [item.to_dict() for item in candidates or []],
            "precision_comparison": precision_comparison or [],
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


def export_csv_dir(path, payload):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tables = {
        "gpu_comparison.csv": payload.get("gpu_candidates", []),
        "precision_comparison.csv": payload.get("precision_comparison", []),
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


def export_xlsx(path, payload):
    write_xlsx(
        path,
        {
            "Summary": [
                {"metric": "model_id", "value": payload.get("model_id")},
                {"metric": "production_memory_gb", "value": payload.get("memory", {}).get("production_memory_gb")},
            ],
            "GPU_Comparison": payload.get("gpu_candidates", []),
            "Precision_Comparison": payload.get("precision_comparison", []),
            "Assumptions": [{"key": key, "value": value} for key, value in payload.get("scenario", {}).items()],
            "Validation": payload.get("validation", []),
        },
    )
