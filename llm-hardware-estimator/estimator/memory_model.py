import math
from dataclasses import dataclass, field

from .precision import bytes_per_param, normalize_precision, quant_overhead


GIB = 1024 ** 3


DEFAULT_RUNTIME_OVERHEAD = {
    "absolute_gb": 2.0,
    "weight_ratio": 0.10,
    "kv_cache_ratio": 0.15,
}

DEFAULT_PARALLEL = {
    "strategy": "none",
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "parallel_overhead_factor": 1.00,
}


@dataclass
class MemoryEstimate:
    task_type: str
    weight_memory_gb: float
    kv_cache_avg_gb: float
    kv_cache_peak_gb: float
    runtime_overhead_gb: float
    total_memory_gb: float
    production_memory_gb: float
    per_gpu_weight_gb: float
    per_gpu_kv_cache_gb: float
    per_gpu_runtime_gb: float
    per_gpu_required_gb: float
    weight_shard_factor: int
    kv_shard_factor: int
    supported_parallel_config: bool = True
    warnings: list = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)


def _as_positive_int(value, name):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer" % name)
    if result < 0:
        raise ValueError("%s must be non-negative" % name)
    return result


def _runtime_overhead(weight_gb, kv_gb, runtime):
    return max(
        float(runtime.get("absolute_gb", 2.0)),
        weight_gb * float(runtime.get("weight_ratio", 0.10))
        + kv_gb * float(runtime.get("kv_cache_ratio", 0.15)),
    )


def _parallel_factors(model, task_type, parallel):
    strategy = parallel.get("strategy", "none")
    tp = max(1, int(parallel.get("tensor_parallel_size", 1)))
    pp = max(1, int(parallel.get("pipeline_parallel_size", 1)))
    warnings = []
    supported = True

    if strategy == "none":
        return 1, 1, supported, warnings
    if strategy == "tensor_parallel":
        weight_shard = tp
        kv_shard = tp
    elif strategy == "pipeline_parallel":
        weight_shard = pp
        kv_shard = pp
    elif strategy == "hybrid":
        weight_shard = tp * pp
        kv_shard = tp * pp
    else:
        raise ValueError("unsupported parallel strategy: %s" % strategy)

    if task_type == "generation" and strategy in ("tensor_parallel", "hybrid"):
        kv_heads = model.get("num_kv_heads")
        if not kv_heads or int(kv_heads) % tp != 0:
            supported = False
            warnings.append(
                "num_kv_heads=%s is not divisible by tensor_parallel_size=%s"
                % (kv_heads, tp)
            )

    return weight_shard, kv_shard, supported, warnings


def estimate_memory(model, scenario):
    task_type = model.get("task_type", "generation")
    runtime = dict(DEFAULT_RUNTIME_OVERHEAD)
    runtime.update(scenario.get("runtime_overhead") or {})
    parallel = dict(DEFAULT_PARALLEL)
    parallel.update(scenario.get("parallel") or {})

    input_tokens = _as_positive_int(scenario.get("input_tokens", 0), "input_tokens")
    output_tokens = _as_positive_int(scenario.get("output_tokens", 0), "output_tokens")
    concurrency = max(1, _as_positive_int(scenario.get("concurrency", 1), "concurrency"))
    production_margin = float(scenario.get("production_margin_ratio", 0.30))
    weight_precision = normalize_precision(scenario.get("weight_precision", "bf16"))
    kv_precision = normalize_precision(scenario.get("kv_cache_precision", weight_precision))
    quantization = normalize_precision(scenario.get("quantization", weight_precision))
    if quantization in ("fp32", "fp16", "bf16"):
        quantization = "none"

    params_billion = float(model.get("params_billion", 0))
    weight_memory_gb = (
        params_billion
        * 1e9
        * bytes_per_param(weight_precision)
        * quant_overhead(quantization)
        / GIB
    )

    warnings = []
    kv_cache_avg_gb = 0.0
    kv_cache_peak_gb = 0.0
    if task_type == "generation":
        required_fields = ("num_layers", "hidden_size", "num_attention_heads", "num_kv_heads")
        missing = [key for key in required_fields if not model.get(key)]
        if missing:
            warnings.append("missing model fields for KV cache estimate: %s" % ", ".join(missing))
        else:
            layers = int(model["num_layers"])
            hidden = int(model["hidden_size"])
            heads = int(model["num_attention_heads"])
            kv_heads = int(model["num_kv_heads"])
            head_dim = hidden / heads
            avg_tokens = input_tokens + output_tokens / 2.0
            peak_tokens = input_tokens + output_tokens
            if model.get("attention_type") == "sliding_window" and model.get("sliding_window_size"):
                window = int(model["sliding_window_size"])
                avg_tokens = min(avg_tokens, window)
                peak_tokens = min(peak_tokens, window)
            kv_cache_avg_gb = (
                2 * layers * kv_heads * head_dim * avg_tokens * concurrency * bytes_per_param(kv_precision) / GIB
            )
            kv_cache_peak_gb = (
                2 * layers * kv_heads * head_dim * peak_tokens * concurrency * bytes_per_param(kv_precision) / GIB
            )

    runtime_overhead_gb = _runtime_overhead(weight_memory_gb, kv_cache_peak_gb, runtime)
    total_memory_gb = weight_memory_gb + kv_cache_peak_gb + runtime_overhead_gb
    production_memory_gb = total_memory_gb * (1 + production_margin)

    weight_shard, kv_shard, supported, parallel_warnings = _parallel_factors(model, task_type, parallel)
    warnings.extend(parallel_warnings)

    per_gpu_weight_gb = weight_memory_gb / weight_shard
    per_gpu_kv_gb = kv_cache_peak_gb / kv_shard
    per_gpu_runtime_gb = _runtime_overhead(per_gpu_weight_gb, per_gpu_kv_gb, runtime)
    parallel_overhead = float(parallel.get("parallel_overhead_factor", 1.0))
    per_gpu_required_gb = (
        (per_gpu_weight_gb + per_gpu_kv_gb + per_gpu_runtime_gb)
        * parallel_overhead
        * (1 + production_margin)
    )
    if not supported:
        per_gpu_required_gb = math.inf

    return MemoryEstimate(
        task_type=task_type,
        weight_memory_gb=weight_memory_gb,
        kv_cache_avg_gb=kv_cache_avg_gb,
        kv_cache_peak_gb=kv_cache_peak_gb,
        runtime_overhead_gb=runtime_overhead_gb,
        total_memory_gb=total_memory_gb,
        production_memory_gb=production_memory_gb,
        per_gpu_weight_gb=per_gpu_weight_gb,
        per_gpu_kv_cache_gb=per_gpu_kv_gb,
        per_gpu_runtime_gb=per_gpu_runtime_gb,
        per_gpu_required_gb=per_gpu_required_gb,
        weight_shard_factor=weight_shard,
        kv_shard_factor=kv_shard,
        supported_parallel_config=supported,
        warnings=warnings,
        assumptions={
            "weight_precision": weight_precision,
            "kv_cache_precision": kv_precision,
            "quantization": quantization,
            "production_margin_ratio": production_margin,
            "parallel": parallel,
        },
    )
