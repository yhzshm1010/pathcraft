#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gpt54_data_flywheel.prompts import (  # noqa: E402
    COMPOSE_SFT_INSTRUCTION,
    DECOMPOSE_SFT_INSTRUCTION,
    SEMANTIC_SFT_INSTRUCTION,
    WEBQSP_DECOMPOSE_SFT_INSTRUCTION,
)
from gpt54_data_flywheel.validation import validate_decomposition  # noqa: E402


DEFAULT_SOURCE = PROJECT_ROOT / "work/generated"
DEFAULT_OUTPUT = PROJECT_ROOT / "work/finetune_incremental"
LEDGER_NAME = "accepted_ledger.jsonl"
CONTRACT_KEYS = (
    "contract_version",
    "dataset",
    "semantic_source_sha256",
    "compose_source_sha256",
    "decompose_source_sha256",
    "webqsp_source_sha256",
    "split_name",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally merge accepted flywheel records and build "
            "LLaMA-Factory-compatible fine-tuning datasets."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help=(
            "generation output directory or accepted.jsonl path; repeat for multiple "
            "model stages (default: work/generated)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="persistent incremental dataset directory (default: work/finetune_incremental)",
    )
    parser.add_argument("--split-name", default="train")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sources = args.source or [DEFAULT_SOURCE]
    try:
        summary = build_incremental_dataset(
            sources=sources,
            output_dir=args.output,
            split_name=args.split_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_incremental_dataset(
    *,
    sources: Sequence[Path],
    output_dir: Path,
    split_name: str = "train",
) -> dict[str, Any]:
    if not split_name.strip():
        raise ValueError("split_name must not be empty")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / LEDGER_NAME

    records = _read_accepted(ledger_path, ledger=True) if ledger_path.exists() else {}
    source_contract = _existing_contract(output_dir / "manifest.json")
    seen_indices = _index_owners(records)
    added = 0
    duplicate = 0
    source_summaries: list[dict[str, Any]] = []

    for source in sources:
        accepted_path, source_manifest = _resolve_source(source)
        configuration = _manifest_configuration(source_manifest)
        source_contract = _merge_contract(
            source_contract,
            configuration,
            source=accepted_path,
        )
        incoming = _read_accepted(accepted_path)
        source_added = 0
        source_duplicate = 0
        provenance = {
            "accepted_path": str(accepted_path),
            "configuration": configuration,
        }

        for source_id, candidate in incoming.items():
            existing = records.get(source_id)
            if existing is not None:
                if _training_signature(existing) != _training_signature(candidate):
                    raise ValueError(
                        f"conflicting accepted content for source_id {source_id!r}: "
                        f"{accepted_path} differs from the persistent ledger"
                    )
                duplicate += 1
                source_duplicate += 1
                continue

            source_index = _source_index(candidate, accepted_path)
            owner = seen_indices.get(source_index)
            if owner is not None and owner != source_id:
                raise ValueError(
                    f"source_index {source_index} belongs to both {owner!r} and "
                    f"{source_id!r}; source datasets are not compatible"
                )
            stored = deepcopy(candidate)
            stored["_finetune_provenance"] = provenance
            records[source_id] = stored
            seen_indices[source_index] = source_id
            added += 1
            source_added += 1

        source_summaries.append(
            {
                "accepted_path": str(accepted_path),
                "model": configuration.get("model", "unknown"),
                "records_seen": len(incoming),
                "records_added": source_added,
                "duplicates": source_duplicate,
            }
        )

    ordered = sorted(
        records.values(),
        key=lambda record: (_source_index(record, ledger_path), str(record["source_id"])),
    )
    semantic_rows = [
        _finetune_row(record, field="semantic_row", instruction=SEMANTIC_SFT_INSTRUCTION)
        for record in ordered
    ]
    decompose_rows = [_decompose_row(record) for record in ordered]
    compose_rows = [
        _finetune_row(record, field="compose_row", instruction=COMPOSE_SFT_INSTRUCTION)
        for record in ordered
    ]
    multitask_rows = [*decompose_rows, *semantic_rows, *compose_rows]

    semantic_name = f"semantic_path_{split_name}.json"
    decompose_name = f"decompose_{split_name}.json"
    compose_name = f"compose_{split_name}.json"
    multitask_name = f"multitask_{split_name}.json"
    now = _utc_now()
    previous_manifest = _read_json_object(output_dir / "manifest.json", required=False)
    manifest = {
        "format_version": 1,
        "created_at": previous_manifest.get("created_at", now),
        "updated_at": now,
        "source_contract": source_contract,
        "training_format": {
            "formatting": "alpaca",
            "includes_original_question": True,
            "instruction_profile": "rules_only_no_few_shot",
        },
        "counts": {
            "accepted": len(ordered),
            "added_this_run": added,
            "duplicates_this_run": duplicate,
            "decompose_rows": len(decompose_rows),
            "semantic_rows": len(semantic_rows),
            "compose_rows": len(compose_rows),
            "multitask_rows": len(multitask_rows),
        },
        "sources_scanned_this_run": source_summaries,
        "outputs": {
            "accepted_ledger": LEDGER_NAME,
            "decompose": decompose_name,
            "semantic_path": semantic_name,
            "compose": compose_name,
            "multitask": multitask_name,
            "dataset_info": "dataset_info.json",
        },
    }
    dataset_info = {
        f"decompose_{split_name}": _dataset_entry(decompose_name),
        f"semantic_path_{split_name}": _dataset_entry(semantic_name),
        f"compose_{split_name}": _dataset_entry(compose_name),
        f"multitask_{split_name}": _dataset_entry(multitask_name),
    }

    _atomic_write_jsonl(ledger_path, ordered)
    _atomic_write_json(output_dir / decompose_name, decompose_rows)
    _atomic_write_json(output_dir / semantic_name, semantic_rows)
    _atomic_write_json(output_dir / compose_name, compose_rows)
    _atomic_write_json(output_dir / multitask_name, multitask_rows)
    _atomic_write_json(output_dir / "dataset_info.json", dataset_info)
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest["counts"]


def _resolve_source(source: Path) -> tuple[Path, Path]:
    resolved = source.resolve()
    if resolved.is_dir():
        accepted_path = resolved / "accepted.jsonl"
        manifest_path = resolved / "manifest.json"
    else:
        accepted_path = resolved
        manifest_path = resolved.with_name("manifest.json")
    if not accepted_path.is_file():
        raise FileNotFoundError(f"accepted source does not exist: {accepted_path}")
    return accepted_path, manifest_path


def _read_accepted(path: Path, *, ledger: bool = False) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} contains invalid JSON: {exc}") from exc
            _validate_record(record, path=path, line_number=line_number, ledger=ledger)
            source_id = str(record["source_id"])
            previous = records.get(source_id)
            if previous is not None and _training_signature(previous) != _training_signature(record):
                raise ValueError(f"{path} has conflicting records for source_id {source_id!r}")
            records[source_id] = record
    return records


def _validate_record(
    record: Any,
    *,
    path: Path,
    line_number: int,
    ledger: bool,
) -> None:
    location = f"{path}:{line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{location} must contain a JSON object")
    if record.get("status") != "accepted":
        kind = "ledger" if ledger else "accepted source"
        raise ValueError(f"{location} in {kind} does not have status='accepted'")
    if not str(record.get("source_id", "")).strip():
        raise ValueError(f"{location} has no source_id")
    _source_index(record, path)
    for field in ("semantic_row", "compose_row"):
        row = record.get(field)
        if not isinstance(row, dict):
            raise ValueError(f"{location}.{field} must be an object")
        for row_field in ("instruction", "input", "output"):
            if not isinstance(row.get(row_field), str):
                raise ValueError(f"{location}.{field}.{row_field} must be a string")
        for row_field in ("input", "output"):
            try:
                decoded = json.loads(row[row_field])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{location}.{field}.{row_field} contains invalid JSON: {exc}"
                ) from exc
            if not isinstance(decoded, dict):
                raise ValueError(f"{location}.{field}.{row_field} must decode to an object")
        history = row.get("history", [])
        if not isinstance(history, list):
            raise ValueError(f"{location}.{field}.history must be an array")


def _source_index(record: dict[str, Any], path: Path) -> int:
    value = record.get("source_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: source_index must be a non-negative integer")
    return value


def _training_signature(record: dict[str, Any]) -> str:
    return json.dumps(
        [
            _finetune_row(
                record,
                field="semantic_row",
                instruction=SEMANTIC_SFT_INSTRUCTION,
            ),
            _finetune_row(
                record,
                field="compose_row",
                instruction=COMPOSE_SFT_INSTRUCTION,
            ),
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _finetune_row(
    record: dict[str, Any],
    *,
    field: str,
    instruction: str,
) -> dict[str, Any]:
    row = deepcopy(record.get(field))
    if not isinstance(row, dict):
        raise ValueError(f"source_id {record.get('source_id')!r}: {field} must be an object")
    try:
        payload = json.loads(str(row.get("input", "")))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"source_id {record.get('source_id')!r}: {field}.input contains invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"source_id {record.get('source_id')!r}: {field}.input must decode to an object"
        )

    original_question = str(record.get("question", "")).strip()
    input_question = str(payload.get("question", "")).strip()
    if original_question and input_question and original_question != input_question:
        raise ValueError(
            f"source_id {record.get('source_id')!r}: top-level question and "
            f"{field}.input.question differ"
        )
    question = original_question or input_question
    if not question:
        raise ValueError(
            f"source_id {record.get('source_id')!r}: no original question is available"
        )
    payload["question"] = question
    row["instruction"] = instruction
    row["input"] = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    row.setdefault("history", [])
    return row


def _decompose_row(record: dict[str, Any]) -> dict[str, Any]:
    """Use the corrected generation-stage decomposition when available."""
    instruction = (
        DECOMPOSE_SFT_INSTRUCTION
        if str(record.get("dataset", "webqsp")).casefold() == "cwq"
        else WEBQSP_DECOMPOSE_SFT_INSTRUCTION
    )
    existing = record.get("decompose_row")
    if isinstance(existing, dict):
        try:
            payload = json.loads(str(existing.get("input", "")))
            output = json.loads(str(existing.get("output", "")))
        except json.JSONDecodeError:
            payload = output = None
        if (
            isinstance(payload, dict)
            and isinstance(output, dict)
            and isinstance(output.get("decomposition"), list)
            and output["decomposition"]
            and all(isinstance(item, str) and item.strip() for item in output["decomposition"])
        ):
            decomposition_value = {"decomposition": output["decomposition"]}
            decomposition_errors = validate_decomposition(decomposition_value)
            if decomposition_errors:
                raise ValueError(
                    f"source_id {record.get('source_id')!r}: decompose_row output is invalid: "
                    + "; ".join(decomposition_errors)
                )
            return {
                "instruction": instruction,
                "input": json.dumps(
                    {"question": str(record.get("question", payload.get("question", "")))},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "output": json.dumps(
                    decomposition_value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "history": [],
            }
    semantic_row = _finetune_row(
        record,
        field="semantic_row",
        instruction=SEMANTIC_SFT_INSTRUCTION,
    )
    try:
        payload = json.loads(semantic_row["input"])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"source_id {record.get('source_id')!r}: semantic_row.input contains invalid JSON: {exc}"
        ) from exc
    decomposition = payload.get("decomposition")
    if (
        not isinstance(decomposition, list)
        or not decomposition
        or any(not isinstance(item, str) or not item.strip() for item in decomposition)
    ):
        raise ValueError(
            f"source_id {record.get('source_id')!r}: semantic_row.input.decomposition "
            "must be a non-empty array of non-empty strings"
        )
    decomposition_errors = validate_decomposition({"decomposition": decomposition})
    if decomposition_errors:
        raise ValueError(
            f"source_id {record.get('source_id')!r}: semantic decomposition is invalid: "
            + "; ".join(decomposition_errors)
        )
    return {
        "instruction": instruction,
        "input": json.dumps(
            {"question": payload["question"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "output": json.dumps(
            {"decomposition": decomposition},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "history": [],
    }


def _index_owners(records: dict[str, dict[str, Any]]) -> dict[int, str]:
    owners: dict[int, str] = {}
    for source_id, record in records.items():
        source_index = _source_index(record, Path(LEDGER_NAME))
        previous = owners.get(source_index)
        if previous is not None and previous != source_id:
            raise ValueError(
                f"persistent ledger has multiple source_ids for source_index {source_index}"
            )
        owners[source_index] = source_id
    return owners


def _manifest_configuration(path: Path) -> dict[str, Any]:
    manifest = _read_json_object(path, required=False)
    configuration = manifest.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError(f"{path}: configuration must be an object")
    return configuration


def _existing_contract(path: Path) -> dict[str, Any]:
    manifest = _read_json_object(path, required=False)
    contract = manifest.get("source_contract", {})
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: source_contract must be an object")
    return {key: contract[key] for key in CONTRACT_KEYS if contract.get(key) not in {None, ""}}


def _merge_contract(
    current: dict[str, Any],
    configuration: dict[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    merged = dict(current)
    for key in CONTRACT_KEYS:
        incoming = configuration.get(key)
        if incoming in {None, ""}:
            continue
        existing = merged.get(key)
        if existing not in {None, ""} and existing != incoming:
            raise ValueError(
                f"{source}: incompatible {key}: expected {existing!r}, got {incoming!r}"
            )
        merged[key] = incoming
    return merged


def _read_json_object(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} contains invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
        },
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for value in values
    )
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
