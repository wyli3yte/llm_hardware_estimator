import argparse
import sys

from .buyer_advisor import build_buyer_options
from .calibration import validate_benchmark_csv
from .config_loader import (
    ConfigError,
    load_project_configs,
    load_scenario,
    merge_scenario,
    resolve_key,
    validate_project_configs,
)
from .hardware_selector import recommend_hardware
from .memory_model import estimate_memory
from .precision import QUALITY_RISK, parse_precision_bundle
from .report_renderer import (
    export_csv_dir,
    export_json,
    export_markdown,
    export_xlsx,
    render_markdown,
    result_payload,
)


def _add_common(parser):
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", default="rag")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--weight-precision")
    parser.add_argument("--kv-cache-precision")
    parser.add_argument("--quantization")
    parser.add_argument("--goal", default=None, choices=["budget", "balanced", "performance"])
    parser.add_argument("--target", "--target-hardware", dest="target", default="domestic")
    parser.add_argument("--output", dest="output_md")
    parser.add_argument("--export-md")
    parser.add_argument("--export-json")
    parser.add_argument("--export-csv")
    parser.add_argument("--export-xlsx")


def _load_context(args):
    configs = load_project_configs(args.config_dir)
    model_id, model = resolve_key(configs["models"], args.model, "模型")
    scenario = load_scenario(configs, args.scenario)
    scenario = merge_scenario(
        scenario,
        {
            "concurrency": args.concurrency,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "weight_precision": args.weight_precision,
            "kv_cache_precision": args.kv_cache_precision,
            "quantization": args.quantization,
            "decision_goal": args.goal,
        },
    )
    if not scenario.get("kv_cache_precision"):
        scenario["kv_cache_precision"] = scenario.get("weight_precision", "bf16")
    if not scenario.get("quantization"):
        scenario["quantization"] = "none"
    goal = args.goal or scenario.get("decision_goal", "balanced")
    return configs, model_id, model, scenario, goal


def _export(args, payload, buyer_options=None):
    md_path = args.export_md or args.output_md
    if md_path:
        export_markdown(md_path, payload, buyer_options=buyer_options)
    if args.export_json:
        export_json(args.export_json, payload)
    if args.export_csv:
        export_csv_dir(args.export_csv, payload)
    if args.export_xlsx:
        export_xlsx(args.export_xlsx, payload)


def cmd_estimate(args):
    configs, model_id, model, scenario, _ = _load_context(args)
    hardware_id, hardware = resolve_key(configs["hardware"], args.hardware, "硬件")
    memory = estimate_memory(model, scenario)
    candidates = recommend_hardware(model, scenario, {hardware_id: hardware}, goal="balanced", target="all")
    payload = result_payload(model_id, scenario, memory, candidates)
    _export(args, payload)
    print(render_markdown(payload))
    return 0


def cmd_recommend(args):
    configs, model_id, model, scenario, goal = _load_context(args)
    memory = estimate_memory(model, scenario)
    candidates = recommend_hardware(model, scenario, configs["hardware"], goal=goal, target=args.target)
    payload = result_payload(model_id, scenario, memory, candidates)
    _export(args, payload)
    if candidates:
        best = candidates[0]
        print(
            "推荐：%s × %s（%s），等级%s，显存余量%.0f%%"
            % (
                best.suggested_gpu_count,
                best.gpu_model,
                best.vendor,
                best.recommendation_level,
                best.memory_margin_ratio * 100,
            )
        )
    else:
        print("未找到候选GPU")
    return 0


def cmd_buy(args):
    configs, model_id, model, scenario, goal = _load_context(args)
    memory = estimate_memory(model, scenario)
    candidates = recommend_hardware(model, scenario, configs["hardware"], goal=goal, target=args.target)
    buyer_options = build_buyer_options(candidates)
    payload = result_payload(model_id, scenario, memory, candidates)
    _export(args, payload, buyer_options=buyer_options)
    if not buyer_options:
        print("当前约束下未找到采购候选。")
        return 1
    for option in buyer_options:
        print("%s：%s × %s，%s" % (option["name"], option["gpu_count"], option["gpu_model"], option["note"]))
    return 0


def cmd_compare_precision(args):
    configs, model_id, model, scenario, goal = _load_context(args)
    rows = []
    for item in args.precisions.split(","):
        bundle = parse_precision_bundle(item.strip())
        variant = dict(scenario)
        variant["weight_precision"] = bundle.weight_precision
        variant["activation_precision"] = bundle.activation_precision
        variant["kv_cache_precision"] = bundle.kv_cache_precision
        variant["quantization"] = bundle.quantization
        estimate = estimate_memory(model, variant)
        rows.append(
            {
                "precision": item.strip(),
                "weight_memory_gb": estimate.weight_memory_gb,
                "kv_cache_peak_gb": estimate.kv_cache_peak_gb,
                "runtime_overhead_gb": estimate.runtime_overhead_gb,
                "production_memory_gb": estimate.production_memory_gb,
                "quality_risk": QUALITY_RISK.get(bundle.weight_precision, "medium"),
            }
        )
    memory = estimate_memory(model, scenario)
    candidates = recommend_hardware(model, scenario, configs["hardware"], goal=goal, target=args.target)
    payload = result_payload(model_id, scenario, memory, candidates, precision_comparison=rows)
    _export(args, payload)
    print(render_markdown(payload))
    return 0


def cmd_validate(args):
    configs, model_id, model, scenario, _ = _load_context(args)
    rows = validate_benchmark_csv(args.benchmark, model, scenario)
    memory = estimate_memory(model, scenario)
    payload = result_payload(model_id, scenario, memory, validation=rows)
    _export(args, payload)
    if args.output_md:
        export_markdown(args.output_md, payload)
    print("验证样本：%d" % len(rows))
    return 0


def cmd_validate_configs(args):
    report = validate_project_configs(args.config_dir)
    if report["status"] == "ok":
        print("配置校验通过：models=%d hardware=%d scenarios=%d" % (report["models"], report["hardware"], report["scenarios"]))
        return 0
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 1


def cmd_wizard(args):
    print("LLM硬件估算向导")
    model = input("模型ID（默认 qwen3-32b）：").strip() or "qwen3-32b"
    scenario = input("场景（chat/rag/long_doc，默认 rag）：").strip() or "rag"
    concurrency = input("并发（默认 4）：").strip() or "4"
    goal = input("偏好（budget/balanced/performance，默认 balanced）：").strip() or "balanced"
    target = input("硬件范围（domestic/all，默认 domestic）：").strip() or "domestic"
    args.model = model
    args.scenario = scenario
    args.concurrency = int(concurrency)
    args.goal = goal
    args.target = target
    args.input_tokens = None
    args.output_tokens = None
    args.weight_precision = None
    args.kv_cache_precision = None
    args.quantization = None
    args.export_md = args.export_md
    return cmd_buy(args)


def build_parser():
    parser = argparse.ArgumentParser(description="LLM推理硬件需求估算与国产GPU选型工具")
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", help="估算指定模型和硬件")
    _add_common(estimate)
    estimate.add_argument("--hardware", required=True)
    estimate.set_defaults(func=cmd_estimate)

    recommend = sub.add_parser("recommend", help="推荐GPU候选")
    _add_common(recommend)
    recommend.set_defaults(func=cmd_recommend)

    buy = sub.add_parser("buy", help="采购友好三档建议")
    _add_common(buy)
    buy.set_defaults(func=cmd_buy)

    compare = sub.add_parser("compare-precision", help="精度/量化显存对比")
    _add_common(compare)
    compare.add_argument("--precisions", required=True)
    compare.set_defaults(func=cmd_compare_precision)

    validate = sub.add_parser("validate", help="导入1.7B benchmark CSV做显存校准")
    _add_common(validate)
    validate.add_argument("--benchmark", required=True)
    validate.set_defaults(func=cmd_validate)

    validate_configs = sub.add_parser("validate-configs", help="校验默认配置库")
    validate_configs.add_argument("--config-dir", default=None)
    validate_configs.set_defaults(func=cmd_validate_configs)

    wizard = sub.add_parser("wizard", help="交互式采购向导")
    wizard.add_argument("--config-dir", default=None)
    wizard.add_argument("--export-md")
    wizard.add_argument("--export-json")
    wizard.add_argument("--export-csv")
    wizard.add_argument("--export-xlsx")
    wizard.add_argument("--output", dest="output_md")
    wizard.set_defaults(func=cmd_wizard)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
