import math
from dataclasses import asdict, dataclass

from .memory_model import estimate_memory
from .performance_model import estimate_performance
from .precision import precision_supported_by_hardware, normalize_precision


GOAL_WEIGHTS = {
    "budget": {
        "min_gpu_count_score": 0.35,
        "memory_margin_score": 0.10,
        "bandwidth_score": 0.10,
        "compute_score": 0.10,
        "source_confidence_score": 0.20,
        "platform_profile_score": 0.10,
        "multi_gpu_risk_score": 0.05,
    },
    "balanced": {
        "min_gpu_count_score": 0.10,
        "memory_margin_score": 0.25,
        "bandwidth_score": 0.20,
        "compute_score": 0.15,
        "source_confidence_score": 0.15,
        "platform_profile_score": 0.10,
        "multi_gpu_risk_score": 0.05,
    },
    "performance": {
        "min_gpu_count_score": 0.05,
        "memory_margin_score": 0.30,
        "bandwidth_score": 0.30,
        "compute_score": 0.25,
        "source_confidence_score": 0.05,
        "platform_profile_score": 0.05,
        "multi_gpu_risk_score": 0.00,
    },
}

CONFIDENCE_SCORE = {"high": 1.0, "medium": 0.65, "low": 0.30}


@dataclass
class HardwareCandidate:
    rank: int
    hardware_id: str
    vendor: str
    gpu_model: str
    region_type: str
    recommendation_level: str
    fit_status: str
    suggested_gpu_count: int
    min_gpu_count: int
    memory_per_gpu_gb: float
    total_memory_gb: float
    usable_memory_gb: float
    required_memory_gb: float
    production_memory_gb: float
    memory_margin_gb: float
    memory_margin_ratio: float
    memory_bandwidth_gbps: float
    fp16_tflops: float
    bf16_tflops: float
    int8_tops: float
    precision_support: str
    main_bottleneck: str
    multi_gpu_risk: str
    source_confidence: str
    score: float
    pros: str
    cons: str
    decision_note: str

    def to_dict(self):
        return asdict(self)


def _scenario_for_count(scenario, count, model):
    cloned = dict(scenario)
    parallel = dict(cloned.get("parallel") or {})
    if count <= 1:
        parallel.update(
            {
                "strategy": "none",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "parallel_overhead_factor": 1.0,
            }
        )
    elif model.get("task_type") == "generation" and model.get("num_kv_heads") and int(model["num_kv_heads"]) % count == 0:
        parallel.update(
            {
                "strategy": "tensor_parallel",
                "tensor_parallel_size": count,
                "pipeline_parallel_size": 1,
                "parallel_overhead_factor": 1.10 if count <= 4 else 1.15,
            }
        )
    else:
        parallel.update(
            {
                "strategy": "pipeline_parallel",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": count,
                "parallel_overhead_factor": 1.08 if count <= 4 else 1.12,
            }
        )
    cloned["parallel"] = parallel
    return cloned


def _find_count(model, scenario, hardware, max_gpus):
    usable = float(hardware.get("memory_gb", 0)) * float(scenario.get("gpu_utilization_target", 0.85))
    best_estimate = None
    for count in range(1, max_gpus + 1):
        estimate = estimate_memory(model, _scenario_for_count(scenario, count, model))
        if best_estimate is None or estimate.per_gpu_required_gb < best_estimate.per_gpu_required_gb:
            best_estimate = estimate
        if estimate.supported_parallel_config and estimate.per_gpu_required_gb <= usable:
            return count, estimate
    return max_gpus, best_estimate


def _level(fit_status, margin_ratio, confidence, count):
    if fit_status != "Fit":
        return "Not Fit"
    if margin_ratio >= 0.25 and confidence == "high" and count <= 2:
        return "A"
    if margin_ratio >= 0.10:
        return "B"
    if margin_ratio >= 0:
        return "C"
    return "D"


def _multi_gpu_risk(count):
    if count <= 1:
        return "low"
    if count <= 4:
        return "medium"
    return "high"


def recommend_hardware(model, scenario, hardware_db, goal="balanced", target="domestic", max_gpus=8):
    goal = goal if goal in GOAL_WEIGHTS else "balanced"
    weights = GOAL_WEIGHTS[goal]
    weight_precision = normalize_precision(scenario.get("weight_precision", "bf16"))
    candidates = []
    max_bandwidth = max([float(gpu.get("memory_bandwidth_gbps") or 0) for gpu in hardware_db.values()] or [1])
    max_compute = max([float(gpu.get("bf16_tflops") or gpu.get("fp16_tflops") or 0) for gpu in hardware_db.values()] or [1])

    for hardware_id, gpu in hardware_db.items():
        if target != "all" and gpu.get("region_type", "domestic") != target:
            continue
        precision_ok = precision_supported_by_hardware(weight_precision, gpu)
        count, estimate = _find_count(model, scenario, gpu, max_gpus)
        usable = float(gpu.get("memory_gb", 0)) * float(scenario.get("gpu_utilization_target", 0.85))
        fit_status = "Fit" if precision_ok and estimate and estimate.per_gpu_required_gb <= usable else "Not Fit"
        if not precision_ok:
            fit_status = "Not Fit"
        required = estimate.per_gpu_required_gb if estimate else math.inf
        margin = usable - required if math.isfinite(required) else -math.inf
        margin_ratio = margin / usable if usable > 0 and math.isfinite(margin) else -1.0
        confidence = gpu.get("confidence", "low")
        bandwidth = float(gpu.get("memory_bandwidth_gbps") or 0)
        compute = float(gpu.get("bf16_tflops") or gpu.get("fp16_tflops") or 0)
        multi_risk = _multi_gpu_risk(count)
        source_score = CONFIDENCE_SCORE.get(confidence, 0.3)
        score = (
            weights["min_gpu_count_score"] * max(0.0, 1 - (count - 1) / max_gpus)
            + weights["memory_margin_score"] * max(0.0, min(1.0, margin_ratio))
            + weights["bandwidth_score"] * (bandwidth / max_bandwidth if max_bandwidth else 0)
            + weights["compute_score"] * (compute / max_compute if max_compute else 0)
            + weights["source_confidence_score"] * source_score
            + weights["platform_profile_score"] * (1.0 if gpu.get("software_profile") else 0.4)
            + weights["multi_gpu_risk_score"] * {"low": 1.0, "medium": 0.55, "high": 0.2}[multi_risk]
        )
        if fit_status != "Fit":
            score *= 0.15

        perf = estimate_performance(model, scenario, gpu, estimate)
        level = _level(fit_status, margin_ratio, confidence, count)
        pros = []
        cons = []
        if confidence == "high":
            pros.append("规格来源可信")
        if bandwidth:
            pros.append("带宽%.0fGB/s" % bandwidth)
        if count > 1:
            cons.append("需要%d卡并行验证" % count)
        if not precision_ok:
            cons.append("不支持%s" % weight_precision)
        if margin_ratio < 0.1:
            cons.append("显存余量偏低")
        if model.get("model_type") == "moe":
            cons.append("MoE路由和专家并行需实测")
        note = "建议进入方案比选" if fit_status == "Fit" else "不建议作为当前配置候选"
        candidates.append(
            HardwareCandidate(
                rank=0,
                hardware_id=hardware_id,
                vendor=gpu.get("vendor", ""),
                gpu_model=gpu.get("model", hardware_id),
                region_type=gpu.get("region_type", "domestic"),
                recommendation_level=level,
                fit_status=fit_status,
                suggested_gpu_count=count if fit_status == "Fit" else max(count, 1),
                min_gpu_count=count,
                memory_per_gpu_gb=float(gpu.get("memory_gb", 0)),
                total_memory_gb=float(gpu.get("memory_gb", 0)) * max(count, 1),
                usable_memory_gb=usable,
                required_memory_gb=required,
                production_memory_gb=estimate.production_memory_gb if estimate else math.inf,
                memory_margin_gb=margin,
                memory_margin_ratio=margin_ratio,
                memory_bandwidth_gbps=bandwidth,
                fp16_tflops=float(gpu.get("fp16_tflops") or 0),
                bf16_tflops=float(gpu.get("bf16_tflops") or 0),
                int8_tops=float(gpu.get("int8_tops") or 0),
                precision_support=",".join(gpu.get("precision_support", [])),
                main_bottleneck=perf["main_bottleneck"],
                multi_gpu_risk=multi_risk,
                source_confidence=confidence,
                score=score,
                pros="；".join(pros) or "容量估算可作为初筛",
                cons="；".join(cons) or "需目标框架压测确认TTFT/TPOT",
                decision_note=note,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    for idx, item in enumerate(candidates, 1):
        item.rank = idx
    return candidates
