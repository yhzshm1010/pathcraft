#!/usr/bin/env python3
"""Keep the verified 19,518 CWQ rows and append remaining CWQ train rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from source_annotations import load_source_annotations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "data/cwq_19518/decompose_train.json"
DEFAULT_ARCHIVE = PROJECT_ROOT / "inputs/cwq_source_annotations.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/cwq_26944_incremental_source_decomposition"


def _object(value):
    return json.loads(value) if isinstance(value, str) else value


def _question_key(value: str) -> str:
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base_path: Path, archive_path: Path, output_dir: Path, source_member: str = "") -> dict:
    base_rows = json.loads(base_path.read_text(encoding="utf-8"))
    if len(base_rows) != 19518:
        raise ValueError(f"expected 19,518 base rows, found {len(base_rows)}")

    source_rows, selected_member, member_bytes = load_source_annotations(archive_path, source_member)
    if len(source_rows) != 26944:
        raise ValueError(f"expected 26,944 CWQ rows, found {len(source_rows)}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for source_index, row in enumerate(source_rows):
        decomposition = [
            str(item.get("decomposed", "")).strip()
            for item in row.get("reasoning_information", [])
            if str(item.get("decomposed", "")).strip()
        ]
        if not decomposition:
            raise ValueError(f"CWQ source row {source_index} has no decomposition")
        groups[_question_key(row.get("question", ""))].append(
            {
                "source_index": source_index,
                "question": str(row.get("question", "")),
                "decomposition": decomposition,
                "used": False,
            }
        )

    # Match every retained row to one source occurrence. Exact decomposition
    # matching is preferred for repeated question strings, but retained targets
    # are never modified.
    matched_source_indices: list[int] = []
    for row_index, row in enumerate(base_rows):
        question = str(_object(row.get("input", {})).get("question", ""))
        decomposition = _object(row.get("output", {})).get("decomposition", [])
        candidates = [
            item
            for item in groups.get(_question_key(question), [])
            if not item["used"]
        ]
        if not candidates:
            raise ValueError(f"base row {row_index} cannot be matched to CWQ")
        selected = next(
            (item for item in candidates if item["decomposition"] == decomposition),
            candidates[0],
        )
        selected["used"] = True
        matched_source_indices.append(int(selected["source_index"]))

    primary_instruction, primary_count = Counter(
        str(row.get("instruction", "")) for row in base_rows
    ).most_common(1)[0]
    if not primary_instruction:
        raise ValueError("base dataset has an empty primary instruction")

    appended: list[dict] = []
    appended_source_indices: list[int] = []
    for source_index, row in enumerate(source_rows):
        candidates = groups[_question_key(row.get("question", ""))]
        source = next(item for item in candidates if item["source_index"] == source_index)
        if source["used"]:
            continue
        appended.append(
            {
                "history": [],
                "input": _dump({"question": source["question"]}),
                "instruction": primary_instruction,
                "output": _dump({"decomposition": source["decomposition"]}),
            }
        )
        appended_source_indices.append(source_index)

    if len(appended) != 7426:
        raise ValueError(f"expected 7,426 appended rows, found {len(appended)}")
    output = [*deepcopy(base_rows), *appended]
    if len(output) != 26944:
        raise ValueError(f"expected 26,944 output rows, found {len(output)}")
    if output[: len(base_rows)] != base_rows:
        raise ValueError("the retained 19,518-row prefix changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "decompose_train.json"
    output_path.write_text(_dump(output), encoding="utf-8")
    manifest = {
        "dataset": "cwq",
        "split": "train",
        "rows": len(output),
        "retained_rows": len(base_rows),
        "appended_source_rows": len(appended),
        "retained_prefix_unchanged": True,
        "appended_prompt_equals_primary_base_prompt": True,
        "primary_prompt_base_rows": primary_count,
        "source_archive": str(archive_path.resolve()),
        "source_member": selected_member,
        "source_member_sha256": hashlib.sha256(member_bytes).hexdigest(),
        "base_sha256": _sha256(base_path),
        "output_sha256": _sha256(output_path),
        "matched_source_indices": len(matched_source_indices),
        "appended_source_index_min": min(appended_source_indices),
        "appended_source_index_max": max(appended_source_indices),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "cwq_decompose_incremental_source_26944": {
                    "file_name": "decompose_train.json",
                    "formatting": "alpaca",
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                        "history": "history",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-member", default="", help="Optional exact member in the source ZIP.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.base, args.source_archive, args.output_dir, args.source_member),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
