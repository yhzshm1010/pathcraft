#!/usr/bin/env python3
"""Generate CWQ decomposition predictions from a LlamaFactory OpenAI API.

The input is the existing CWQ decomposition-prompt file.  Its instruction and
input fields are sent together as one user message, which matches the Alpaca
prompt used during SFT.  The output keeps the original rows and adds a
``prediction``/``raw_prediction`` pair compatible with DecompositionStore.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib import request


def _json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"model output is not JSON: {text[:300]!r}")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    decomposition = value.get("decomposition")
    if isinstance(decomposition, str):
        decomposition = [decomposition]
    if not isinstance(decomposition, list) or not all(isinstance(x, str) and x.strip() for x in decomposition):
        raise ValueError("JSON output must contain a non-empty decomposition array")
    return {"decomposition": [x.strip() for x in decomposition]}


def _call(url: str, model: str, row: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:
    prompt = "\n".join(str(x) for x in (row.get("instruction", ""), row.get("input", "")) if x)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "do_sample": False,
        "temperature": 0.0,
        "max_tokens": 1024,
        "n": 1,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = request.Request(
                url.rstrip("/") + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            # Do not route loopback traffic through an inherited HTTP proxy.
            with request.build_opener(request.ProxyHandler({})).open(req, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw["choices"][0]["message"]["content"]
            parsed = _json_from_text(text)
            output = dict(row)
            output["prediction"] = parsed["decomposition"]
            output["raw_prediction"] = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            output["generated_by"] = "llamafactory-vllm"
            output["validation"] = {
                "json_valid": True,
                "schema_valid": True,
                "non_empty": True,
                "format_passed": True,
            }
            return output
        except Exception as exc:  # noqa: BLE001 - retry transient server errors
            last = exc
            if attempt < retries:
                time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"failed question={row.get('input', '')!r}: {last}") from last


def _atomic_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--model", default="decomposition")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise SystemExit("--input must contain a JSON list")
    output_path = Path(args.output)
    rows = source
    if args.resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(existing, list) and len(existing) == len(source):
            rows = existing
    pending = [i for i, row in enumerate(rows) if not row.get("generated_by") == "llamafactory-vllm"]
    print(f"total={len(rows)} pending={len(pending)} workers={args.workers}", flush=True)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_call, args.url, args.model, rows[i], args.timeout, args.retries): i
            for i in pending
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            rows[index] = future.result()
            completed += 1
            if completed % 32 == 0 or completed == len(pending):
                _atomic_dump(output_path, rows)
                print(f"completed={completed}/{len(pending)}", flush=True)
    _atomic_dump(output_path, rows)
    print(f"saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
