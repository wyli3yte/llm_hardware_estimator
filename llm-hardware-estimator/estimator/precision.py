from dataclasses import dataclass


BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "w8a8": 1.0,
    "w8a16": 1.0,
    "int4": 0.5,
    "w4a16": 0.5,
}

QUANT_OVERHEAD = {
    "none": 1.00,
    "fp32": 1.00,
    "fp16": 1.00,
    "bf16": 1.00,
    "fp8": 1.05,
    "int8": 1.10,
    "w8a8": 1.10,
    "w8a16": 1.10,
    "int4": 1.25,
    "w4a16": 1.25,
}

QUALITY_RISK = {
    "fp32": "low",
    "fp16": "low",
    "bf16": "low",
    "fp8": "medium",
    "int8": "medium",
    "w8a8": "medium",
    "w8a16": "medium",
    "int4": "high",
    "w4a16": "high",
}


@dataclass(frozen=True)
class PrecisionBundle:
    weight_precision: str
    activation_precision: str
    kv_cache_precision: str
    quantization: str


def normalize_precision(name):
    if not name:
        return "bf16"
    value = str(name).strip().lower().replace("-", "")
    aliases = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
        "none": "bf16",
        "kvint8": "int8",
    }
    return aliases.get(value, value)


def bytes_per_param(precision):
    key = normalize_precision(precision)
    if key not in BYTES_PER_PARAM:
        raise ValueError("unsupported precision: %s" % precision)
    return BYTES_PER_PARAM[key]


def quant_overhead(quantization):
    key = normalize_precision(quantization)
    if key not in QUANT_OVERHEAD:
        raise ValueError("unsupported quantization: %s" % quantization)
    return QUANT_OVERHEAD[key]


def parse_precision_bundle(value):
    raw = normalize_precision(value)
    if "_kv_" in raw:
        weight_part, kv_part = raw.split("_kv_", 1)
        weight_precision = normalize_precision(weight_part)
        kv_cache_precision = normalize_precision(kv_part)
    else:
        weight_precision = raw
        kv_cache_precision = raw if raw in ("fp32", "fp16", "bf16") else "bf16"

    if weight_precision not in BYTES_PER_PARAM:
        raise ValueError("unsupported weight precision: %s" % weight_precision)
    if kv_cache_precision not in BYTES_PER_PARAM:
        raise ValueError("unsupported KV precision: %s" % kv_cache_precision)

    return PrecisionBundle(
        weight_precision=weight_precision,
        activation_precision="bf16" if weight_precision not in ("fp32", "fp16", "bf16") else weight_precision,
        kv_cache_precision=kv_cache_precision,
        quantization=weight_precision if weight_precision in QUANT_OVERHEAD else "none",
    )


def precision_supported_by_hardware(precision, hardware):
    supported = set(normalize_precision(item) for item in hardware.get("precision_support", []))
    normalized = normalize_precision(precision)
    if normalized in supported:
        return True
    if normalized in ("w8a16", "w8a8") and "int8" in supported:
        return True
    if normalized == "w4a16" and ("int4" in supported or "int8" in supported):
        return True
    return False
