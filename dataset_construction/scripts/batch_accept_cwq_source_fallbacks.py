#!/usr/bin/env python3
"""Batch-gate deterministic CWQ source fallbacks without an LLM dependency.

Every accepted row is rebuilt from the source semantic/compose pair, checked by
the strict semantic/compose validators, and then execution-checked against the
row's gold SPARQL before it is appended to ``accepted.jsonl``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpt54_data_flywheel.cli import (  # noqa: E402
    DEFAULT_CWQ_COMPOSE,
    DEFAULT_CWQ_DECOMPOSE,
    DEFAULT_CWQ_SEMANTIC,
    DEFAULT_CWQ_SOURCE,
)
from gpt54_data_flywheel.pipeline import (  # noqa: E402
    Flywheel,
    FlywheelConfig,
    _accepted_records,
    _append_jsonl,
    load_source_pairs,
    materialize,
)
from validate_and_accept_manual_correction import DEFAULT_EXECUTION_VALIDATOR  # noqa: E402


DEFAULT_REJECTED = (
    PROJECT_ROOT
    / "work/cwq_rejected/active_errors_with_golden_sparql.jsonl"
)


class _NoModel:
    """The fallback does not generate, but Flywheel records model metadata."""

    model = "deterministic-source-fallback"
    api_mode = "none"
    api_url = ""
    json_mode = ""
    responses_url = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--semantic-source", type=Path, default=DEFAULT_CWQ_SEMANTIC)
    parser.add_argument("--compose-source", type=Path, default=DEFAULT_CWQ_COMPOSE)
    parser.add_argument("--decompose-source", type=Path, default=DEFAULT_CWQ_DECOMPOSE)
    parser.add_argument("--dataset-source", type=Path, default=DEFAULT_CWQ_SOURCE)
    parser.add_argument("--execution-validator-command", default=DEFAULT_EXECUTION_VALIDATOR)
    parser.add_argument("--execution-validator-timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--exclude-index", type=int, action="append", default=[])
    parser.add_argument("--no-materialize", action="store_true")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.start < 0 or args.limit < 0:
        raise SystemExit("--start and --limit must be non-negative")

    output_dir = args.accepted_output.resolve()
    accepted_path = output_dir / "accepted.jsonl"
    existing = _accepted_records(accepted_path)
    pairs = load_source_pairs(
        args.semantic_source,
        args.compose_source,
        webqsp_path=args.dataset_source,
        require_sparql=True,
        split_name="train",
        decompose_path=args.decompose_source,
        dataset="cwq",
    )
    pairs_by_index = {pair.compose_index: pair for pair in pairs}
    excluded = set(args.exclude_index)
    rows = [
        row
        for row in _read_jsonl(args.rejected)
        if int(row.get("compose_source_index", -1)) not in excluded
        and str(row.get("error_category", "")) != "02_source_sparql_no_answers"
    ]
    rows.sort(key=lambda row: int(row.get("compose_source_index", -1)))
    rows = rows[args.start :]
    if args.limit:
        rows = rows[: args.limit]

    flywheel = Flywheel(
        client=_NoModel(),
        config=FlywheelConfig(
            output_dir=output_dir,
            dataset="cwq",
            require_execution_validation=True,
            external_validator_command=args.execution_validator_command,
            external_validator_timeout=args.execution_validator_timeout,
        ),
        semantic_source=args.semantic_source,
        compose_source=args.compose_source,
        webqsp_source=args.dataset_source,
        decompose_source=args.decompose_source,
    )
    results_path = output_dir / "deterministic_source_fallback_results.jsonl"
    counts: Counter[str] = Counter()
    for position, row in enumerate(rows, start=1):
        index = int(row.get("compose_source_index", -1))
        source_id = str(row.get("source_id", ""))
        if source_id in existing:
            counts["already_accepted"] += 1
            continue
        pair = pairs_by_index.get(index)
        if pair is None:
            outcome = {"compose_source_index": index, "source_id": source_id, "status": "missing_pair"}
        else:
            print(f"[{position}/{len(rows)}] compose_source_index={index}", file=sys.stderr, flush=True)
            fallback_input = dict(row)
            accepted = flywheel._validated_source_fallback(pair, fallback_input)
            if accepted is None:
                trace = list(fallback_input.get("trace", []) or [])
                diagnostics = (
                    list(trace[-1].get("static_and_execution_errors", []) or [])
                    if trace and isinstance(trace[-1], dict)
                    else []
                )
                outcome = {
                    "compose_source_index": index,
                    "source_id": source_id,
                    "status": "rejected",
                    "errors": diagnostics,
                }
            else:
                _append_jsonl(accepted_path, accepted)
                existing[source_id] = accepted
                outcome = {"compose_source_index": index, "source_id": source_id, "status": "accepted"}
        _append_jsonl(results_path, outcome)
        counts[outcome["status"]] += 1

    materialized: dict[str, Any] | None = None
    if not args.no_materialize and counts["accepted"]:
        materialized = materialize(output_dir, split_name="train")
    print(json.dumps({"selected": len(rows), "counts": dict(counts), "materialized": materialized}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
