#!/usr/bin/env python3
import argparse
import json
import mimetypes
import threading
import time
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

from estimator.buyer_advisor import build_buyer_options
from estimator.cli import (
    _precision_comparison_rows,
    _scenario_comparison_rows,
    _source_rows,
)
from estimator.config_loader import load_project_configs, load_scenario, merge_scenario, resolve_key
from estimator.hardware_selector import recommend_hardware
from estimator.memory_model import estimate_memory
from estimator.precision import parse_precision_bundle
from estimator.report_renderer import export_xlsx, result_payload


APP_ROOT = Path(__file__).resolve().parent
WEB_ROOT = APP_ROOT / "web"
DEFAULT_OUTPUT_DIR = APP_ROOT / "outputs" / "web_reports"


def _json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def _safe_int(value, default):
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _public_candidate_rows(candidates):
    return [
        {
            "rank": item.rank,
            "level": item.recommendation_level,
            "vendor": item.vendor,
            "gpu_model": item.gpu_model,
            "fit_status": item.fit_status,
            "suggested_gpu_count": item.suggested_gpu_count,
            "memory_per_gpu_gb": item.memory_per_gpu_gb,
            "memory_margin_ratio": item.memory_margin_ratio,
            "main_bottleneck": item.main_bottleneck,
            "multi_gpu_risk": item.multi_gpu_risk,
            "source_confidence": item.source_confidence,
            "decision_note": item.decision_note,
        }
        for item in candidates
    ]


def build_recommendation_response(request_payload, output_dir=DEFAULT_OUTPUT_DIR, config_dir=None):
    configs = load_project_configs(config_dir)
    model_id, model = resolve_key(configs["models"], request_payload.get("model", "qwen3-32b"), "模型")
    scenario = load_scenario(configs, request_payload.get("scenario", "rag"))
    scenario = merge_scenario(
        scenario,
        {
            "concurrency": _safe_int(request_payload.get("concurrency"), scenario.get("concurrency", 4)),
            "decision_goal": request_payload.get("goal") or None,
        },
    )
    if request_payload.get("weight_precision"):
        bundle = parse_precision_bundle(request_payload["weight_precision"])
        scenario["weight_precision"] = bundle.weight_precision
        scenario["activation_precision"] = bundle.activation_precision
        scenario["kv_cache_precision"] = bundle.kv_cache_precision
        scenario["quantization"] = bundle.quantization
    if request_payload.get("kv_cache_precision"):
        scenario["kv_cache_precision"] = request_payload["kv_cache_precision"]
    if not scenario.get("kv_cache_precision"):
        scenario["kv_cache_precision"] = scenario.get("weight_precision", "bf16")
    if not scenario.get("quantization"):
        scenario["quantization"] = "none"
    goal = request_payload.get("goal") or scenario.get("decision_goal", "balanced")
    target = request_payload.get("target") or "domestic"

    memory = estimate_memory(model, scenario)
    candidates = recommend_hardware(model, scenario, configs["hardware"], goal=goal, target=target)
    buyer_options = build_buyer_options(candidates)
    payload = result_payload(
        model_id,
        scenario,
        memory,
        candidates,
        precision_comparison=_precision_comparison_rows(model, scenario),
        scenario_comparison=_scenario_comparison_rows(configs, model, target, goal, scenario),
        sources=_source_rows(configs["hardware"], candidates),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "llm_hardware_report_%s_%s.xlsx" % (int(time.time()), uuid.uuid4().hex[:8])
    export_xlsx(output_dir / filename, payload, buyer_options=buyer_options)

    best = candidates[0] if candidates else None
    return {
        "model_id": model_id,
        "scenario": scenario.get("name", ""),
        "decision": {
            "title": "推荐结论",
            "summary": (
                "%s × %s（%s），等级%s"
                % (best.suggested_gpu_count, best.gpu_model, best.vendor, best.recommendation_level)
                if best
                else "当前条件下未找到推荐候选"
            ),
            "memory": "%.2f GB" % memory.production_memory_gb,
            "risk": best.cons if best else "请补充硬件数据",
        },
        "options": buyer_options,
        "gpu_candidates": _public_candidate_rows(candidates),
        "download_url": "/download/%s" % filename,
    }


def options_response(config_dir=None):
    configs = load_project_configs(config_dir)
    return {
        "models": sorted(configs["models"].keys()),
        "scenarios": sorted(configs["scenarios"].keys()),
        "goals": ["budget", "balanced", "performance"],
        "targets": ["domestic", "all"],
        "precisions": ["bf16", "fp16", "w8a16", "w4a16", "w4a16_kv_int8"],
    }


class PortalHandler(BaseHTTPRequestHandler):
    output_dir = DEFAULT_OUTPUT_DIR
    config_dir = None

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path == "/api/options":
            _json_response(self, 200, options_response(self.config_dir))
            return
        if path.startswith("/download/"):
            self._serve_download(path.rsplit("/", 1)[-1])
            return
        if path == "/":
            path = "/index.html"
        self._serve_static(path)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/recommend":
            _json_response(self, 404, {"error": "not found"})
            return
        try:
            payload = _read_json_body(self)
            result = build_recommendation_response(payload, output_dir=self.output_dir, config_dir=self.config_dir)
            _json_response(self, 200, result)
        except Exception as exc:
            _json_response(self, 400, {"error": str(exc)})

    def _serve_static(self, request_path):
        relative = request_path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.exists() or target.is_dir():
            _json_response(self, 404, {"error": "not found"})
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_download(self, filename):
        if "/" in filename or "\\" in filename or not filename.endswith(".xlsx"):
            _json_response(self, 404, {"error": "not found"})
            return
        target = (Path(self.output_dir) / filename).resolve()
        if not str(target).startswith(str(Path(self.output_dir).resolve())) or not target.exists():
            _json_response(self, 404, {"error": "not found"})
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.send_header("Content-Disposition", 'attachment; filename="%s"' % filename)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(host, port, output_dir=DEFAULT_OUTPUT_DIR, config_dir=None):
    class ConfiguredPortalHandler(PortalHandler):
        pass

    ConfiguredPortalHandler.output_dir = Path(output_dir)
    ConfiguredPortalHandler.config_dir = config_dir
    return ThreadingHTTPServer((host, port), ConfiguredPortalHandler)


def start_server_thread(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def main():
    parser = argparse.ArgumentParser(description="LLM硬件估算Web门户")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    server = create_server(args.host, args.port, output_dir=Path(args.output_dir), config_dir=args.config_dir)
    print("Web门户已启动：http://%s:%s" % (args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
