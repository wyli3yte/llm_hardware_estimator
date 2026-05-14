def build_buyer_options(candidates):
    fit = [item for item in candidates if item.fit_status == "Fit"]
    if not fit:
        return []
    budget = sorted(fit, key=lambda item: (item.suggested_gpu_count, item.memory_per_gpu_gb, -item.score))[0]
    balanced = fit[0]
    performance = sorted(fit, key=lambda item: (item.memory_margin_ratio, item.memory_bandwidth_gbps, item.score), reverse=True)[0]
    return [
        {
            "name": "成本优先",
            "hardware_id": budget.hardware_id,
            "gpu_model": budget.gpu_model,
            "gpu_count": budget.suggested_gpu_count,
            "note": "资源占用较少，但需重点验证余量和并行风险",
        },
        {
            "name": "均衡",
            "hardware_id": balanced.hardware_id,
            "gpu_model": balanced.gpu_model,
            "gpu_count": balanced.suggested_gpu_count,
            "note": "综合显存余量、数据可信度和平台能力的默认建议",
        },
        {
            "name": "性能优先",
            "hardware_id": performance.hardware_id,
            "gpu_model": performance.gpu_model,
            "gpu_count": performance.suggested_gpu_count,
            "note": "优先高显存、高带宽和扩展空间",
        },
    ]
