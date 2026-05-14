from .precision import bytes_per_param, normalize_precision


def estimate_performance(model, scenario, hardware, memory_estimate):
    task_type = model.get("task_type", "generation")
    active_params = float(model.get("active_params_billion", model.get("params_billion", 0))) * 1e9
    input_tokens = int(scenario.get("input_tokens", 0))
    output_tokens = int(scenario.get("output_tokens", 0))
    weight_precision = normalize_precision(scenario.get("weight_precision", "bf16"))
    kv_precision = normalize_precision(scenario.get("kv_cache_precision", weight_precision))
    bandwidth = float(hardware.get("memory_bandwidth_gbps") or 0) * 1e9
    compute_tflops = float(
        hardware.get("%s_tflops" % weight_precision)
        or hardware.get("bf16_tflops")
        or hardware.get("fp16_tflops")
        or 0
    )
    effective_compute = compute_tflops * 1e12 * 0.30 if compute_tflops else 0
    effective_bandwidth = bandwidth * 0.30 if bandwidth else 0

    prefill_flops = 2 * active_params * max(input_tokens, 1)
    prefill_compute_s = prefill_flops / effective_compute if effective_compute else None

    decode_memory_s = None
    decode_compute_s = None
    attention_ratio = None
    if task_type == "generation":
        layers = model.get("num_layers")
        heads = model.get("num_attention_heads")
        hidden = model.get("hidden_size")
        kv_heads = model.get("num_kv_heads")
        context = input_tokens + output_tokens
        if layers and heads and hidden and kv_heads:
            head_dim = float(hidden) / float(heads)
            attention_flops = 4 * int(layers) * int(heads) * head_dim * max(context, 1)
            weight_flops = 2 * active_params
            decode_flops = weight_flops + attention_flops
            decode_compute_s = decode_flops / effective_compute if effective_compute else None
            attention_ratio = attention_flops / decode_flops if decode_flops else None
            weight_read = active_params * bytes_per_param(weight_precision)
            kv_read = 2 * int(layers) * int(kv_heads) * head_dim * max(context, 1) * bytes_per_param(kv_precision)
            decode_memory_s = (weight_read + kv_read) / effective_bandwidth if effective_bandwidth else None

    times = [item for item in (prefill_compute_s, decode_compute_s, decode_memory_s) if item is not None]
    if not times:
        bottleneck = "data_missing"
    elif decode_memory_s is not None and decode_memory_s >= max(times):
        bottleneck = "bandwidth"
    elif decode_compute_s is not None and decode_compute_s >= max(times):
        bottleneck = "compute"
    else:
        bottleneck = "compute"

    if memory_estimate.per_gpu_required_gb > float(hardware.get("memory_gb", 0)):
        bottleneck = "capacity"

    return {
        "prefill_compute_seconds_lower_bound": prefill_compute_s,
        "decode_compute_seconds_per_token_lower_bound": decode_compute_s,
        "decode_memory_seconds_per_token_lower_bound": decode_memory_s,
        "attention_flops_ratio": attention_ratio,
        "main_bottleneck": bottleneck,
        "confidence": "medium" if compute_tflops and bandwidth else "low",
    }
