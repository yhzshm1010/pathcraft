#!/usr/bin/env python3
"""Replace CWQ decomposition targets with the supplied source annotations.

The outer Alpaca rows, including instruction, input, and history, are copied
verbatim from the current dataset. Only ``output.decomposition`` is replaced.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from source_annotations import load_source_annotations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = PROJECT_ROOT / "data/cwq_19518/decompose_train.json"
DEFAULT_ARCHIVE = PROJECT_ROOT / "inputs/cwq_source_annotations.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/cwq_19518_source_decomposition"


def _object(value):
    return json.loads(value) if isinstance(value, str) else value


def _question_key(value: str) -> str:
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def _dump(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(current_path: Path, archive_path: Path, output_dir: Path, source_member: str = "") -> dict:
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if not isinstance(current, list):
        raise ValueError(f"{current_path} must contain a JSON array")

    source_rows, selected_member, member_bytes = load_source_annotations(archive_path, source_member)

    groups: dict[str, list[dict]] = defaultdict(list)
    for source_index, row in enumerate(source_rows):
        decomposition = [
            str(item.get("decomposed", "")).strip()
            for item in row.get("reasoning_information", [])
            if str(item.get("decomposed", "")).strip()
        ]
        if not decomposition:
            continue
        groups[_question_key(row.get("question", ""))].append(
            {
                "source_index": source_index,
                "decomposition": decomposition,
                "used": False,
            }
        )

    output: list[dict] = []
    same = 0
    replaced = 0
    source_indices: list[int] = []
    for row_index, row in enumerate(current):
        input_object = _object(row.get("input", {}))
        output_object = _object(row.get("output", {}))
        question = str(input_object.get("question", "")).strip()
        current_decomposition = output_object.get("decomposition", [])
        candidates = [
            item
            for item in groups.get(_question_key(question), [])
            if not item["used"]
        ]
        if not candidates:
            raise ValueError(
                f"row {row_index}: no unused CWQ source row for {question!r}"
            )

        # Repeated CWQ question strings are paired by exact target first. This
        # preserves the original source identity whenever it is recoverable.
        selected = next(
            (
                item
                for item in candidates
                if item["decomposition"] == current_decomposition
            ),
            candidates[0],
        )
        selected["used"] = True
        source_indices.append(int(selected["source_index"]))
        source_decomposition = list(selected["decomposition"])

        copied = deepcopy(row)
        copied_output = dict(output_object)
        copied_output["decomposition"] = source_decomposition
        copied["output"] = (
            _dump(copied_output)
            if isinstance(row.get("output"), str)
            else copied_output
        )
        output.append(copied)
        if source_decomposition == current_decomposition:
            same += 1
        else:
            replaced += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "decompose_train.json"
    output_path.write_text(_dump(output), encoding="utf-8")

    # Verify the prompt-side fields were copied byte-for-value at object level.
    for index, (before, after) in enumerate(zip(current, output)):
        for field in ("history", "input", "instruction"):
            if before.get(field) != after.get(field):
                raise ValueError(f"row {index}: prompt field {field} changed")

    manifest = {
        "dataset": "cwq",
        "split": "train",
        "rows": len(output),
        "unique_question_strings": len(
            {str(_object(row["input"])["question"]) for row in output}
        ),
        "unique_normalized_questions": len(
            {_question_key(_object(row["input"])["question"]) for row in output}
        ),
        "same_as_current_before_replacement": same,
        "replaced_with_source_target": replaced,
        "all_targets_from_source": True,
        "source_archive": str(archive_path.resolve()),
        "source_member": selected_member,
        "source_member_sha256": hashlib.sha256(member_bytes).hexdigest(),
        "current_dataset_sha256": _sha256(current_path),
        "output_sha256": _sha256(output_path),
        "source_index_min": min(source_indices),
        "source_index_max": max(source_indices),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dataset_info = {
        "cwq_decompose_source_19518": {
            "file_name": "decompose_train.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
            },
        }
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-member", default="", help="Optional exact member in the source ZIP.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = build(args.current, args.source_archive, args.output_dir, args.source_member)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
