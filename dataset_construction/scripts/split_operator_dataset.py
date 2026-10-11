#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split the new training data into three decoupled tasks:

1) Semantic Path (no operator)
   input : question + decomposition
   output: anchors + semantic_paths

2) Compose (no operator)
   input : question + anchors + semantic_paths
   output: entities + triples + answer_var

3) Operator
   input : question + entities + triples + answer_var
   output: operators

By default this script expects:
  semantic_path_train.json
  compose_train.json

Example:
  python split_operator_dataset.py \
      --semantic semantic_path_train.json \
      --compose compose_train.json \
      --out-dir processed
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SEMANTIC_INSTRUCTION = """[TASK=SEMANTIC_PATH]

Convert the original question and its decomposition into canonical Freebase semantic paths.

Return exactly one JSON object containing:
- anchors
- semantic_paths

Each semantic path is an entity-centric chain of JOIN projections.

Direction rules:
- backward: (?x, r, e). step.from=e and step.to=?x.
- forward: (e, r, ?x). step.from=e and step.to=?x.

Each path Pn must use only its own path-local variables Pn.Vm in step.to and
path_output_var. Variables from different paths must never be reused directly.

JOIN is represented only by semantic path steps.

Use canonical relation labels only. A relation_label must be the reversible
[domain, type, property] style label of one Freebase relation.

Do not predict operators, cross-path variable equalities, KB IDs, raw SPARQL,
or dotted Freebase relation IDs.

Return JSON only."""


COMPOSE_INSTRUCTION = """[TASK=COMPOSE]

Build the complete JSON logical query graph from the original question and
the supplied semantic paths.

Map path-local variables Pn.Vm into shared global variables Vn. When different
semantic paths refer to the same logical node, unify them by assigning the same
global variable.

Each semantic path step becomes one triple:
- direction="backward": (?x, r, e)
  step.from is e and step.to is ?x
  => subject=to, relation_label=r, object=from
- direction="forward": (e, r, ?x)
  step.from is e and step.to is ?x
  => subject=from, relation_label=r, object=to

Multi-hop paths remain chains of triples.

Convert anchor entities into global entity constants E0, E1, ...
and return the final answer variable in the global Vn namespace.

Return exactly one JSON object containing:
- entities
- triples
- answer_var

Do not predict or emit operators.
Do not emit path-local variables, selected_paths, variable_equalities, KB IDs,
raw SPARQL, or dotted Freebase relation IDs.

Return JSON only."""


OPERATOR_INSTRUCTION = """[TASK=OPERATOR]

Predict the execution operators required by the original question for the
supplied composed logical query graph.

Input contains:
- question
- entities
- triples
- answer_var

Return exactly one JSON object containing:
- operators

Allowed execution operators are:
COUNT, ARGMAX, ARGMIN, GREATER_THAN, GREATER_OR_EQUAL,
LESS_THAN, LESS_OR_EQUAL, EQUAL, TC, NO_EQUAL.

AND is not an execution operator in this task. Cross-path intersection and
variable unification have already been resolved by COMPOSE through shared
global variables.

Every operator object must contain exactly:
- type
- inputs
- input_var
- attribute_relation_label
- attribute_relation_labels
- value
- value_type

Use global compose variables Vn in input_var.
Unused strings must be "" and unused arrays must be [].
If no execution operator is required, return {"operators": []}.

Return JSON only."""


ALLOWED_OPERATOR_TYPES = {
    "COUNT",
    "ARGMAX",
    "ARGMIN",
    "GREATER_THAN",
    "GREATER_OR_EQUAL",
    "LESS_THAN",
    "LESS_OR_EQUAL",
    "EQUAL",
    "TC",
    "NO_EQUAL",
}


def parse_json_field(value, field_name):
    """The dataset stores input/output as JSON strings; also accept dicts."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON string in field '{field_name}': {exc}") from exc
    if isinstance(value, dict):
        return copy.deepcopy(value)
    raise TypeError(
        f"Field '{field_name}' must be a JSON string or dict, got {type(value).__name__}"
    )


def dump_json_field(obj):
    """Keep the same compact JSON-string style as the source training files."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_dataset(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level JSON array.")
    return data


def save_dataset(path, data, pretty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def question_of(record):
    return parse_json_field(record["input"], "input")["question"]


def validate_alignment(semantic_data, compose_data):
    """Require the same question multiset while preserving duplicate rows."""
    sem_questions = [question_of(r) for r in semantic_data]
    comp_questions = [question_of(r) for r in compose_data]

    sem_counts = Counter(sem_questions)
    comp_counts = Counter(comp_questions)
    if sem_counts != comp_counts:
        only_sem = sorted((sem_counts - comp_counts).elements())
        only_comp = sorted((comp_counts - sem_counts).elements())
        raise ValueError(
            "Semantic/Compose question sets do not match.\n"
            f"Only in semantic: {len(only_sem)}\n"
            f"Only in compose: {len(only_comp)}"
        )


def build_semantic_no_operator(semantic_data):
    result = []

    for record in semantic_data:
        new_record = copy.deepcopy(record)
        output_obj = parse_json_field(record["output"], "output")

        # Core transformation: remove operator supervision.
        output_obj.pop("operators", None)

        expected = {"anchors", "semantic_paths"}
        if set(output_obj.keys()) != expected:
            raise ValueError(
                f"Unexpected semantic output keys for question "
                f"{question_of(record)!r}: {sorted(output_obj.keys())}"
            )

        new_record["instruction"] = SEMANTIC_INSTRUCTION
        new_record["output"] = dump_json_field(output_obj)
        result.append(new_record)

    return result


def build_compose_no_operator(compose_data):
    result = []

    for record in compose_data:
        new_record = copy.deepcopy(record)
        input_obj = parse_json_field(record["input"], "input")
        output_obj = parse_json_field(record["output"], "output")

        # Core transformation: remove operator information from BOTH sides.
        input_obj.pop("operators", None)
        output_obj.pop("operators", None)

        expected_input = {"question", "anchors", "semantic_paths"}
        expected_output = {"entities", "triples", "answer_var"}

        if set(input_obj.keys()) != expected_input:
            raise ValueError(
                f"Unexpected compose input keys for question "
                f"{input_obj.get('question')!r}: {sorted(input_obj.keys())}"
            )
        if set(output_obj.keys()) != expected_output:
            raise ValueError(
                f"Unexpected compose output keys for question "
                f"{input_obj.get('question')!r}: {sorted(output_obj.keys())}"
            )

        new_record["instruction"] = COMPOSE_INSTRUCTION
        new_record["input"] = dump_json_field(input_obj)
        new_record["output"] = dump_json_field(output_obj)
        result.append(new_record)

    return result


def build_operator_dataset(compose_data):
    result = []
    operator_counter = Counter()
    no_operator_count = 0

    for record in compose_data:
        compose_input = parse_json_field(record["input"], "input")
        compose_output = parse_json_field(record["output"], "output")

        operators = copy.deepcopy(compose_output.get("operators", []))

        # The new compose output should already have AND lowered away.
        for op in operators:
            op_type = op.get("type")
            if op_type == "AND":
                raise ValueError(
                    f"AND found in compose output for question "
                    f"{compose_input['question']!r}. "
                    "This script expects AND to have already been lowered by COMPOSE."
                )
            if op_type not in ALLOWED_OPERATOR_TYPES:
                raise ValueError(
                    f"Unexpected operator type {op_type!r} for question "
                    f"{compose_input['question']!r}"
                )
            operator_counter[op_type] += 1

        if not operators:
            no_operator_count += 1

        operator_input = {
            "question": compose_input["question"],
            "entities": copy.deepcopy(compose_output["entities"]),
            "triples": copy.deepcopy(compose_output["triples"]),
            "answer_var": compose_output["answer_var"],
        }

        operator_output = {
            "operators": operators
        }

        result.append({
            "history": [],
            "input": dump_json_field(operator_input),
            "instruction": OPERATOR_INSTRUCTION,
            "output": dump_json_field(operator_output),
        })

    return result, operator_counter, no_operator_count


def main():
    parser = argparse.ArgumentParser(
        description="Split Semantic/Compose training data into no-operator Semantic, "
                    "no-operator Compose, and standalone Operator datasets."
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        default=PROJECT_ROOT / "data/webqsp_2651/semantic_path_train.json",
        help="Path to new semantic_path_train.json",
    )
    parser.add_argument(
        "--compose",
        type=Path,
        default=PROJECT_ROOT / "data/webqsp_2651/compose_train.json",
        help="Path to new compose_train.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "work/operator_split",
        help="Output directory",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the outer JSON arrays (input/output fields remain JSON strings).",
    )
    args = parser.parse_args()

    semantic_data = load_dataset(args.semantic)
    compose_data = load_dataset(args.compose)

    validate_alignment(semantic_data, compose_data)

    semantic_no_op = build_semantic_no_operator(semantic_data)
    compose_no_op = build_compose_no_operator(compose_data)
    operator_data, op_counter, no_op_count = build_operator_dataset(compose_data)

    semantic_out = args.out_dir / "semantic_path_train_no_operator.json"
    compose_out = args.out_dir / "compose_train_no_operator.json"
    operator_out = args.out_dir / "operator_train.json"

    save_dataset(semantic_out, semantic_no_op, pretty=args.pretty)
    save_dataset(compose_out, compose_no_op, pretty=args.pretty)
    save_dataset(operator_out, operator_data, pretty=args.pretty)

    print("Done.")
    print(f"Semantic no-operator : {semantic_out} ({len(semantic_no_op)} records)")
    print(f"Compose no-operator  : {compose_out} ({len(compose_no_op)} records)")
    print(f"Operator dataset     : {operator_out} ({len(operator_data)} records)")
    print()
    print("Operator distribution:")
    for op_type, count in sorted(op_counter.items()):
        print(f"  {op_type:<20} {count}")
    print(f"  {'NO_OPERATOR':<20} {no_op_count}")


if __name__ == "__main__":
    main()
