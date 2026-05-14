#!/usr/bin/env python3
import argparse
import csv
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


FIELDS = [
    "run_id",
    "date",
    "model",
    "hardware",
    "framework",
    "weight_precision",
    "kv_cache_precision",
    "input_tokens",
    "output_tokens",
    "concurrency",
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "output_tokens_per_s",
    "total_tokens_per_s",
    "peak_memory_gb",
    "error_rate",
    "notes",
]


def _request_stream(url, api_key, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % api_key,
        },
        method="POST",
    )
    t0 = time.perf_counter()
    first = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if first is None:
                first = time.perf_counter()
            chunks += 1
    tend = time.perf_counter()
    if first is None:
        first = tend
    generated = max(1, min(max_tokens, chunks or max_tokens))
    ttft_ms = (first - t0) * 1000
    e2e_ms = (tend - t0) * 1000
    tpot_ms = ((tend - first) * 1000 / max(generated - 1, 1)) if generated > 1 else 0
    output_tps = (generated - 1) / max(tend - first, 1e-9) if generated > 1 else 0
    return ttft_ms, tpot_ms, e2e_ms, output_tps, generated


def _worker(index, args, rows, lock):
    prompt = args.prompt or ("请用中文概括大模型推理硬件容量估算的关键步骤。" * max(1, args.input_tokens // 40))
    try:
        ttft, tpot, e2e, output_tps, generated = _request_stream(
            args.base_url,
            args.api_key,
            args.model,
            prompt,
            args.output_tokens,
            args.timeout,
        )
        row = {
            "run_id": "%s-%03d" % (args.run_id, index),
            "date": time.strftime("%Y-%m-%d"),
            "model": args.model,
            "hardware": args.hardware,
            "framework": args.framework,
            "weight_precision": args.weight_precision,
            "kv_cache_precision": args.kv_cache_precision,
            "input_tokens": args.input_tokens,
            "output_tokens": generated,
            "concurrency": args.concurrency,
            "ttft_ms": round(ttft, 2),
            "tpot_ms": round(tpot, 2),
            "e2e_ms": round(e2e, 2),
            "output_tokens_per_s": round(output_tps, 2),
            "total_tokens_per_s": "",
            "peak_memory_gb": args.peak_memory_gb or "",
            "error_rate": 0,
            "notes": "",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        row = {
            "run_id": "%s-%03d" % (args.run_id, index),
            "date": time.strftime("%Y-%m-%d"),
            "model": args.model,
            "hardware": args.hardware,
            "framework": args.framework,
            "weight_precision": args.weight_precision,
            "kv_cache_precision": args.kv_cache_precision,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "concurrency": args.concurrency,
            "ttft_ms": "",
            "tpot_ms": "",
            "e2e_ms": "",
            "output_tokens_per_s": "",
            "total_tokens_per_s": "",
            "peak_memory_gb": args.peak_memory_gb or "",
            "error_rate": 1,
            "notes": str(exc),
        }
    with lock:
        rows.append(row)


def append_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible streaming benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", default="current-test-gpu")
    parser.add_argument("--framework", default="openai-compatible")
    parser.add_argument("--weight-precision", default="bf16")
    parser.add_argument("--kv-cache-precision", default="bf16")
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--peak-memory-gb", type=float, default=None)
    parser.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    rows = []
    lock = threading.Lock()
    start = time.perf_counter()
    for batch in range(args.repeat):
        threads = [
            threading.Thread(target=_worker, args=(batch * args.concurrency + idx, args, rows, lock))
            for idx in range(args.concurrency)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    wall = time.perf_counter() - start
    total_output = sum(int(row["output_tokens"]) for row in rows if not row["error_rate"])
    aggregate = total_output / wall if wall > 0 else 0
    for row in rows:
        row["total_tokens_per_s"] = round(aggregate, 2)
    append_rows(args.output, rows)
    print("wrote %d rows to %s" % (len(rows), args.output))


if __name__ == "__main__":
    main()
