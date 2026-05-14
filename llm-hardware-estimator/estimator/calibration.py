import csv
import statistics
from pathlib import Path

from .memory_model import estimate_memory


def _percentile(values, ratio):
    ordered = sorted(values)
    if not ordered:
        return None
    index = int(round((len(ordered) - 1) * ratio))
    return ordered[index]


def validate_benchmark_csv(path, model, base_scenario):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            scenario = dict(base_scenario)
            scenario["input_tokens"] = int(float(row.get("input_tokens") or scenario.get("input_tokens", 0)))
            scenario["output_tokens"] = int(float(row.get("output_tokens") or scenario.get("output_tokens", 0)))
            scenario["concurrency"] = int(float(row.get("concurrency") or scenario.get("concurrency", 1)))
            scenario["weight_precision"] = row.get("weight_precision") or scenario.get("weight_precision", "bf16")
            scenario["kv_cache_precision"] = row.get("kv_cache_precision") or scenario.get("kv_cache_precision", "bf16")
            estimate = estimate_memory(model, scenario)
            observed = float(row.get("peak_memory_gb") or 0)
            error = abs(estimate.total_memory_gb - observed) / observed if observed else None
            if error is None:
                conclusion = "缺少实测显存"
            elif error <= 0.15:
                conclusion = "通过"
            elif error <= 0.30:
                conclusion = "需解释差异"
            else:
                conclusion = "不通过"
            rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "input_tokens": scenario["input_tokens"],
                    "output_tokens": scenario["output_tokens"],
                    "concurrency": scenario["concurrency"],
                    "ttft_ms": float(row.get("ttft_ms") or 0),
                    "tpot_ms": float(row.get("tpot_ms") or 0),
                    "e2e_ms": float(row.get("e2e_ms") or 0),
                    "output_tokens_per_s": float(row.get("output_tokens_per_s") or 0),
                    "estimated_memory_gb": estimate.total_memory_gb,
                    "peak_memory_gb": observed,
                    "memory_error_ratio": error,
                    "conclusion": conclusion,
                }
            )
    return rows


def summarize_validation(rows):
    if not rows:
        return {"samples": 0}
    ttft = [row["ttft_ms"] for row in rows if row["ttft_ms"]]
    summary = {"samples": len(rows), "ttft_p50_ms": statistics.median(ttft) if ttft else None}
    if len(ttft) >= 10:
        summary["ttft_p95_ms"] = _percentile(ttft, 0.95)
    return summary
