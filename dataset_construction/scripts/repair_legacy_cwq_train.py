#!/usr/bin/env python3
"""Normalize the historical 19,518 CWQ rows to the current graph_v2 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def obj(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def strip_literal(value: str) -> tuple[str, str]:
    text = str(value).strip()
    match = re.match(r'^"(.*)"@([A-Za-z-]+)$', text)
    if match:
        return match.group(1), f"lang:{match.group(2)}"
    match = re.match(r'^"(.*)"(?:\^\^.*)?$', text)
    if match:
        raw = match.group(1)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
            return raw, "number"
        if re.fullmatch(r"\d{4}(?:-\d{2}-\d{2})?", raw):
            return raw, "year"
        return raw, "string"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return text, "number"
    return text, "string"


def normalize_semantic(value: dict[str, Any]) -> dict[str, Any]:
    paths = []
    for raw_path in value.get("semantic_paths", []):
        steps = []
        for raw_step in raw_path.get("steps", []):
            steps.append(
                {
                    "id": raw_step["id"],
                    "relation_label": list(raw_step["relation_label"]),
                    "direction": raw_step["direction"],
                    "from": raw_step["from"],
                    "to": raw_step["to"],
                }
            )
        if not steps:
            continue
        paths.append(
            {
                "id": raw_path["id"],
                "anchor_ref": raw_path["anchor_ref"],
                "goal": raw_path["goal"],
                "steps": steps,
                "path_output_var": steps[-1]["to"],
            }
        )
    return {"anchors": value.get("anchors", []), "semantic_paths": paths}


def normalize_compose(value: dict[str, Any], operators: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int]:
    output = {
        "entities": value.get("entities", []),
        "triples": [],
        "answer_var": value.get("answer_var", ""),
    }
    generated: list[dict[str, Any]] = []
    for raw in value.get("triples", []):
        if not isinstance(raw, dict) or "subject" not in raw or "object" not in raw or "relation_label" not in raw:
            continue
        subject = str(raw["subject"])
        object_ = str(raw["object"])
        relation = list(raw["relation_label"])
        subject_ok = re.fullmatch(r"(?:E|V)\d+", subject) is not None
        object_ok = re.fullmatch(r"(?:E|V)\d+", object_) is not None
        if subject_ok and object_ok:
            output["triples"].append({"subject": subject, "relation_label": relation, "object": object_})
            continue
        # Historical Compose represented scalar literals as graph nodes. The
        # current contract represents them as EQUAL operators instead.
        if subject_ok and not object_ok:
            value_text, value_type = strip_literal(object_)
            generated.append({
                "type": "EQUAL",
                "inputs": [],
                "input_var": subject,
                "attribute_relation_label": relation,
                "attribute_relation_labels": [],
                "value": value_text,
                "value_type": value_type,
            })
        elif object_ok and not subject_ok:
            value_text, value_type = strip_literal(subject)
            generated.append({
                "type": "EQUAL",
                "inputs": [],
                "input_var": object_,
                "attribute_relation_label": relation,
                "attribute_relation_labels": [],
                "value": value_text,
                "value_type": value_type,
            })
    existing = list(operators.get("operators", []))
    existing.extend(generated)
    existing = [item for item in existing if str(item.get("type", "")).upper() != "AND"]
    return output, {"operators": existing}, len(generated)


def row(instruction: str, input_object: dict[str, Any], output_object: dict[str, Any]) -> dict[str, Any]:
    return {"history": [], "input": compact(input_object), "instruction": instruction, "output": compact(output_object)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[1] / "inputs/cwq")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "work/cwq_strict")
    args = parser.parse_args()
    source = args.input_dir
    semantic_rows = load(source / "semantic_path_train.json")
    compose_rows = load(source / "compose_train.json")
    operator_rows = load(source / "operator_train.json")
    decompose_rows = load(source / "decompose_train.json")
    out_s, out_c, out_o, out_d, evidence = [], [], [], [], []
    repaired = 0
    for index, (drow, srow, crow, orow) in enumerate(zip(decompose_rows, semantic_rows, compose_rows, operator_rows)):
        semantic_input = obj(srow["input"])
        compose_input = obj(crow["input"])
        semantic = normalize_semantic(obj(srow["output"]))
        compose, operators, generated_count = normalize_compose(obj(crow["output"]), obj(orow["output"]))
        compose_input = {"question": semantic_input["question"], **semantic}
        operator_input = {"question": semantic_input["question"], **compose}
        out_d.append(drow)
        out_s.append({**srow, "input": compact({"question": semantic_input["question"], "decomposition": semantic_input.get("decomposition", [])}), "output": compact(semantic)})
        out_c.append({**crow, "input": compact(compose_input), "output": compact(compose)})
        out_o.append({**orow, "input": compact(operator_input), "output": compact(operators)})
        if generated_count or semantic != obj(srow["output"]):
            repaired += 1
        evidence.append({"row_index": index, "legacy_static_normalization": "applied" if generated_count or semantic != obj(srow["output"]) else "unchanged", "generated_literal_operators": generated_count})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("decompose_train.json", out_d), ("semantic_path_train.json", out_s), ("compose_train.json", out_c), ("operator_train.json", out_o), ("repair_evidence.json", evidence)):
        (args.output_dir / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({"dataset": "CWQ 19,518 + strict Gold-first normalized v2", "rows": len(out_s), "rows_normalized": repaired, "source": str(source), "note": "historical literal-node and semantic path fields normalized to current graph_v2 schema"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(out_s), "rows_normalized": repaired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
