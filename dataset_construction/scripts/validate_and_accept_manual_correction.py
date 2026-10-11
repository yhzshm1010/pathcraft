#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for import_root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_incremental_finetune_dataset import build_incremental_dataset  # noqa: E402
from gpt54_data_flywheel.cli import (  # noqa: E402
    DEFAULT_COMPOSE,
    DEFAULT_CWQ_COMPOSE,
    DEFAULT_CWQ_DECOMPOSE,
    DEFAULT_CWQ_SEMANTIC,
    DEFAULT_CWQ_SOURCE,
    DEFAULT_SEMANTIC,
    DEFAULT_WEBQSP_SOURCE,
)
from gpt54_data_flywheel.pipeline import (  # noqa: E402
    SourcePair,
    _accepted_records,
    _append_jsonl,
    load_source_pairs,
    materialize,
)
from gpt54_data_flywheel.prompts import (  # noqa: E402
    COMPOSE_SFT_INSTRUCTION,
    KAEDE_DECOMPOSE_SFT_INSTRUCTION,
    KAEDE_SEMANTIC_SFT_INSTRUCTION,
    SEMANTIC_SFT_INSTRUCTION,
)
from gpt54_data_flywheel.validation import (  # noqa: E402
    run_external_validator,
    validate_compose_program,
    validate_decomposition,
    validate_semantic_graph,
)


DEFAULT_ACCEPTED_OUTPUT = PROJECT_ROOT / "work/generated"
DEFAULT_FINETUNE_OUTPUT = PROJECT_ROOT / "work/finetune_incremental_webqsp"
DEFAULT_CWQ_FINETUNE_OUTPUT = PROJECT_ROOT / "work/finetune_incremental_cwq"
DEFAULT_EXECUTION_VALIDATOR = (
    f"{shlex.quote(sys.executable)} "
    f"{shlex.quote(str(PROJECT_ROOT / 'scripts/cwq_execution_validator.py'))}"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a manually corrected semantic/compose pair and append it to "
            "accepted.jsonl only when every required check passes."
        )
    )
    parser.add_argument(
        "correction",
        type=Path,
        help=(
            "JSON file containing compose_source_index, semantic_graph, and "
            "compose_program"
        ),
    )
    parser.add_argument("--dataset", choices=("cwq",), default="cwq")
    parser.add_argument("--decompose-source", type=Path, default=None)
    parser.add_argument("--semantic-source", type=Path, default=None)
    parser.add_argument("--compose-source", type=Path, default=None)
    parser.add_argument(
        "--cwq-source",
        "--dataset-source",
        dest="dataset_source",
        type=Path,
        default=None,
        help="CWQ TRAIN gold source JSON/split directory",
    )
    parser.add_argument("--accepted-output", type=Path, default=None)
    parser.add_argument("--finetune-output", type=Path, default=None)
    parser.add_argument("--split-name", default="train")
    parser.add_argument(
        "--preserve-source-decomposition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "strictly preserve the original KaeDe decomposition and semantic-path "
            "skeleton. By default manual corrections are gold-first: decomposition, "
            "anchors, paths, relations, variables, and operators may be corrected, "
            "while the corrected semantic graph must remain internally consistent "
            "with the corrected decomposition."
        ),
    )
    parser.add_argument(
        "--execution-validator-command",
        default=DEFAULT_EXECUTION_VALIDATOR,
    )
    parser.add_argument("--execution-validator-timeout", type=float, default=180.0)
    parser.add_argument(
        "--require-execution-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "require the candidate to match gold answers through the SPARQL endpoint; "
            "use --no-require-execution-validation only for an explicitly static-only review"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every configured check without writing accepted or training files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    is_cwq = args.dataset == "cwq"
    semantic_source = args.semantic_source or (
        DEFAULT_CWQ_SEMANTIC if is_cwq else DEFAULT_SEMANTIC
    )
    compose_source = args.compose_source or (
        DEFAULT_CWQ_COMPOSE if is_cwq else DEFAULT_COMPOSE
    )
    decompose_source = args.decompose_source or (DEFAULT_CWQ_DECOMPOSE if is_cwq else None)
    dataset_source = args.dataset_source or (
        DEFAULT_CWQ_SOURCE if is_cwq else DEFAULT_WEBQSP_SOURCE
    )
    finetune_output = args.finetune_output or (
        DEFAULT_CWQ_FINETUNE_OUTPUT if is_cwq else DEFAULT_FINETUNE_OUTPUT
    )
    # Manual correction is gold-first by default, including for CWQ/KaeDe.
    # Pass --preserve-source-decomposition only when an exact source-preserving
    # review is explicitly desired.
    preserve_source_decomposition = (
        args.preserve_source_decomposition
        if args.preserve_source_decomposition is not None
        else False
    )
    if is_cwq and not args.dry_run and args.accepted_output is None:
        raise SystemExit(
            "--accepted-output is required for a non-dry-run CWQ correction to avoid "
            "writing into the WebQSP default output"
        )
    accepted_output = args.accepted_output or DEFAULT_ACCEPTED_OUTPUT
    try:
        result = validate_and_accept(
            correction_path=args.correction,
            semantic_source=semantic_source,
            compose_source=compose_source,
            webqsp_source=dataset_source,
            accepted_output=accepted_output,
            finetune_output=finetune_output,
            split_name=args.split_name,
            require_execution_validation=args.require_execution_validation,
            execution_validator_command=args.execution_validator_command,
            execution_validator_timeout=args.execution_validator_timeout,
            dry_run=args.dry_run,
            dataset=args.dataset,
            decompose_source=decompose_source,
            preserve_source_decomposition=preserve_source_decomposition,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def validate_and_accept(
    *,
    correction_path: Path,
    semantic_source: Path,
    compose_source: Path,
    webqsp_source: Path,
    accepted_output: Path,
    finetune_output: Path,
    split_name: str,
    require_execution_validation: bool,
    execution_validator_command: str,
    execution_validator_timeout: float,
    dry_run: bool,
    dataset: str = "webqsp",
    decompose_source: Path | None = None,
    preserve_source_decomposition: bool = False,
) -> dict[str, Any]:
    dataset = dataset.casefold()
    if dataset not in {"webqsp", "cwq"}:
        raise ValueError("dataset must be webqsp or cwq")
    if preserve_source_decomposition and decompose_source is None:
        raise ValueError("preserve_source_decomposition requires decompose_source")
    correction_path = correction_path.resolve()
    correction = _read_correction(correction_path)
    pairs = load_source_pairs(
        semantic_source,
        compose_source,
        webqsp_path=webqsp_source,
        require_sparql=require_execution_validation,
        split_name=split_name,
        decompose_path=decompose_source,
        dataset=dataset,
    )
    pair = _select_pair(pairs, correction)
    _validate_golden_metadata(correction, pair)
    semantic_graph = correction["semantic_graph"]
    compose_program = correction["compose_program"]
    decomposition = _correction_decomposition(
        correction,
        pair=pair,
        required=preserve_source_decomposition,
    )
    # For manual KaeDe corrections, preserve only the source *skeleton* here:
    # decomposition text, semantic-path count/order/ids/goals, and anchor_ref slots.
    # Do not pass the original source graph/program into the generic validators,
    # because doing so also freezes incorrect anchor grounding/surfaces and can
    # prevent a gold-aligned manual correction from mapping to its compose graph.
    source_contract_errors = (
        _validate_source_path_contract(semantic_graph, pair.semantic_output)
        if preserve_source_decomposition else []
    )

    # Always validate the corrected semantic graph against the decomposition
    # that will actually be accepted.  This permits decomposition repair while
    # still enforcing path-count/goal consistency inside the corrected example.
    # In preserved-source mode the original operator contract is intentional.
    # In gold-first mode, however, the original operators may be the rejected
    # defect (for example a truncated ARGMIN property chain). Validate the
    # corrected operators structurally and across semantic/compose here; exact
    # semantic correctness is then gated by gold-answer execution below.
    semantic_expected = (
        pair.expected_operators
        if preserve_source_decomposition
        else list(semantic_graph.get("operators", []) or [])
    )
    compose_expected = (
        pair.compose_expected_operators
        if preserve_source_decomposition
        else list(compose_program.get("operators", []) or [])
    )
    semantic_errors = validate_semantic_graph(
        semantic_graph,
        expected=semantic_expected,
        decomposition=(
            list(decomposition.get("decomposition", []) or [])
            if decomposition is not None else None
        ),
        source_graph=None,
    )
    compose_errors = validate_compose_program(
        compose_program,
        semantic_graph=semantic_graph,
        expected=compose_expected,
        source_graph=None,
        source_program=None,
    )
    errors = [
        *(f"semantic: {error}" for error in source_contract_errors),
        *(f"semantic: {error}" for error in semantic_errors),
        *(f"compose: {error}" for error in compose_errors),
    ]
    if errors:
        raise ValueError(_failure_message(pair, errors))

    execution_errors: list[str] = []
    if require_execution_validation:
        if not execution_validator_command.strip():
            raise ValueError(
                "execution validation is required but execution_validator_command is empty"
            )
        execution_errors = run_external_validator(
            execution_validator_command,
            _validator_payload(pair, semantic_graph, compose_program),
            timeout=execution_validator_timeout,
        )
        if execution_errors:
            raise ValueError(_failure_message(pair, execution_errors, stage="execution"))

    base_result: dict[str, Any] = {
        "source_id": pair.source_id,
        "source_index": pair.index,
        "semantic_source_index": pair.semantic_index,
        "compose_source_index": pair.compose_index,
        "question": pair.question,
        "dataset": dataset,
        "checks": {
            "semantic_static_validation": "passed",
            "compose_static_validation": "passed",
            "execution_validation": (
                "passed" if require_execution_validation else "skipped_by_request"
            ),
        },
    }
    if dry_run:
        return {**base_result, "status": "valid_dry_run"}

    accepted_output = accepted_output.resolve()
    accepted_output.mkdir(parents=True, exist_ok=True)
    accepted_path = accepted_output / "accepted.jsonl"
    existing = _accepted_records(accepted_path)
    if pair.source_id in existing:
        if not _same_candidates(existing[pair.source_id], semantic_graph, compose_program):
            raise ValueError(
                f"source_id {pair.source_id!r} is already accepted with different content; "
                "refusing to append a conflicting record"
            )
        status = "already_accepted"
    else:
        record = _accepted_record(
            pair,
            semantic_graph=semantic_graph,
            compose_program=compose_program,
            decomposition=decomposition,
            dataset=dataset,
            preserve_source_decomposition=preserve_source_decomposition,
            correction_path=correction_path,
            execution_validation_required=require_execution_validation,
        )
        _append_jsonl(accepted_path, record)
        status = "accepted"

    generation_manifest = materialize(accepted_output, split_name=split_name)
    finetune_counts = build_incremental_dataset(
        sources=[accepted_output],
        output_dir=finetune_output,
        split_name=split_name,
    )
    return {
        **base_result,
        "status": status,
        "accepted_count": generation_manifest.get("counts", {}).get("accepted", 0),
        "finetune_counts": finetune_counts,
    }


def _read_correction(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"correction file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} contains invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    for field in ("compose_source_index", "semantic_graph", "compose_program"):
        if field not in value:
            raise ValueError(f"{path} is missing required field {field!r}")
    index = value["compose_source_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"{path}: compose_source_index must be a non-negative integer")
    if not isinstance(value["semantic_graph"], dict):
        raise ValueError(f"{path}: semantic_graph must be an object")
    if not isinstance(value["compose_program"], dict):
        raise ValueError(f"{path}: compose_program must be an object")
    return value


def _select_pair(pairs: Sequence[SourcePair], correction: dict[str, Any]) -> SourcePair:
    compose_index = correction["compose_source_index"]
    matches = [pair for pair in pairs if pair.compose_index == compose_index]
    if len(matches) != 1:
        raise ValueError(
            f"compose_source_index {compose_index} matched {len(matches)} source pairs; expected 1"
        )
    pair = matches[0]
    supplied_source_id = str(correction.get("source_id", "")).strip()
    if supplied_source_id and supplied_source_id != pair.source_id:
        raise ValueError(
            f"correction source_id {supplied_source_id!r} does not match "
            f"resolved source_id {pair.source_id!r}"
        )
    return pair


def _correction_decomposition(
    correction: dict[str, Any],
    *,
    pair: SourcePair,
    required: bool,
) -> dict[str, Any] | None:
    value = correction.get("decomposition")
    if value is None:
        value = pair.decompose_output
    if value is None:
        if required:
            raise ValueError(
                "correction is missing decomposition and no source decomposition is available"
            )
        return None
    if not isinstance(value, dict):
        raise ValueError("correction decomposition must be an object")
    decomposition = {
        "decomposition": list(value.get("decomposition", []) or []),
    }
    errors = validate_decomposition(decomposition)
    if errors:
        raise ValueError("correction decomposition is invalid: " + "; ".join(errors))
    if required and isinstance(pair.decompose_output, dict):
        source_decomposition = {
            "decomposition": list(
                pair.decompose_output.get("decomposition", []) or []
            )
        }
        if decomposition != source_decomposition:
            raise ValueError(
                "correction decomposition must exactly preserve the KaeDe source decomposition"
            )
    return decomposition


def _validate_golden_metadata(
    correction: dict[str, Any],
    pair: SourcePair,
) -> None:
    supplied_sparql = correction.get("golden_sparql")
    source = pair.cwq_source or pair.webqsp_source
    source_sparql = source.sparql if source is not None else ""
    if supplied_sparql is not None and str(supplied_sparql).strip() != source_sparql.strip():
        raise ValueError("correction golden_sparql does not match the selected source pair")

    supplied_source_id = str(correction.get("golden_source_id", "")).strip()
    if not supplied_source_id or source is None:
        return
    expected_source_id = (
        pair.cwq_source.source_id
        if pair.cwq_source is not None
        else pair.webqsp_source.question_id
    )
    if supplied_source_id != expected_source_id:
        raise ValueError(
            f"correction golden_source_id {supplied_source_id!r} does not match "
            f"the selected source id {expected_source_id!r}"
        )


def _validate_source_path_contract(
    candidate_graph: dict[str, Any],
    source_graph: dict[str, Any],
) -> list[str]:
    """Preserve the KaeDe semantic-path skeleton while allowing grounding repair.

    Manual correction is allowed to change anchor ``surface`` values, relations,
    variables, and operators so the candidate can align with the gold SPARQL.
    What remains frozen is the KaeDe path skeleton used by the decomposition:

    * semantic-path count/order/IDs,
    * each path's ``anchor_ref`` slot/ID,
    * each path's ``goal``.

    The full ``anchors`` list is intentionally *not* frozen.  KaeDe/source
    records may contain unused or later-added anchors; manual correction may
    add, remove, or reground those anchors.  Only anchors actually referenced
    by preserved semantic paths are structural.

    Path connectivity and ``path_output_var`` correctness are still checked by
    ``validate_semantic_graph`` against the corrected candidate itself.
    """
    errors: list[str] = []

    # Do not require the complete anchor arrays to have the same length/order.
    # Only path-referenced anchor IDs are part of the preserved KaeDe skeleton.
    candidate_anchors = list(candidate_graph.get("anchors", []) or [])
    candidate_anchor_ids = {
        anchor.get("id")
        for anchor in candidate_anchors
        if isinstance(anchor, dict) and anchor.get("id") is not None
    }

    source_paths = list(source_graph.get("semantic_paths", []) or [])
    candidate_paths = list(candidate_graph.get("semantic_paths", []) or [])
    if len(candidate_paths) != len(source_paths):
        errors.append(
            "semantic_paths count must preserve source semantic path count "
            f"({len(candidate_paths)} != {len(source_paths)})"
        )
        return errors

    for i, (source_path, candidate_path) in enumerate(zip(source_paths, candidate_paths)):
        source_id = source_path.get("id")
        candidate_id = candidate_path.get("id")
        if candidate_id != source_id:
            errors.append(
                f"semantic_paths[{i}].id must preserve source path order "
                f"({candidate_id!r} != {source_id!r})"
            )

        source_anchor_ref = source_path.get("anchor_ref")
        candidate_anchor_ref = candidate_path.get("anchor_ref")
        if candidate_anchor_ref != source_anchor_ref:
            errors.append(
                f"semantic_paths[{i}].anchor_ref must preserve source path order "
                f"({candidate_anchor_ref!r} != {source_anchor_ref!r})"
            )
        elif candidate_anchor_ref not in candidate_anchor_ids:
            errors.append(
                f"semantic_paths[{i}].anchor_ref {candidate_anchor_ref!r} "
                "must exist in correction anchors"
            )

        source_goal = source_path.get("goal")
        candidate_goal = candidate_path.get("goal")
        if candidate_goal != source_goal:
            errors.append(
                f"semantic_paths[{i}].goal must exactly preserve source goal"
            )

    return errors


def _validator_payload(
    pair: SourcePair,
    semantic_graph: dict[str, Any],
    compose_program: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": pair.source_id,
        "source_index": pair.index,
        "attempt": 0,
        "question": pair.question,
        "semantic_graph": semantic_graph,
        "compose_program": compose_program,
        "source_semantic_graph": pair.semantic_output,
        "source_compose_program": pair.compose_output,
        "webqsp_source": pair.webqsp_source.as_context() if pair.webqsp_source else None,
        "cwq_source": pair.cwq_source.as_context() if pair.cwq_source else None,
    }


def _accepted_record(
    pair: SourcePair,
    *,
    semantic_graph: dict[str, Any],
    compose_program: dict[str, Any],
    decomposition: dict[str, Any] | None,
    dataset: str,
    preserve_source_decomposition: bool,
    correction_path: Path,
    execution_validation_required: bool,
) -> dict[str, Any]:
    compose_input = {
        "question": pair.question,
        "anchors": semantic_graph["anchors"],
        "semantic_paths": semantic_graph["semantic_paths"],
        "operators": semantic_graph["operators"],
    }
    correction_sha256 = hashlib.sha256(correction_path.read_bytes()).hexdigest()
    checks = ["semantic_static_validation", "compose_static_validation"]
    if execution_validation_required:
        checks.append("execution_answer_equivalence")
    else:
        checks.append("execution_validation_skipped_by_request")
    return {
        "status": "accepted",
        "acceptance_mode": "manual_correction",
        "dataset": dataset,
        "source_id": pair.source_id,
        "source_index": pair.index,
        "semantic_source_index": pair.semantic_index,
        "compose_source_index": pair.compose_index,
        "question": pair.question,
        "attempts": 0,
        "verifier_issues": [],
        "webqsp_source": pair.webqsp_source.as_context() if pair.webqsp_source else None,
        "cwq_source": pair.cwq_source.as_context() if pair.cwq_source else None,
        "decompose_row": (
            {
                "instruction": KAEDE_DECOMPOSE_SFT_INSTRUCTION,
                "input": _json_text({"question": pair.question}),
                "output": _json_text(decomposition),
                "history": [],
            }
            if decomposition is not None else None
        ),
        "semantic_row": {
            "instruction": (
                KAEDE_SEMANTIC_SFT_INSTRUCTION
                if dataset == "cwq" and decomposition is not None
                else SEMANTIC_SFT_INSTRUCTION
            ),
            "input": _json_text(
                {
                    **pair.semantic_input,
                    **(
                        {"decomposition": decomposition["decomposition"]}
                        if decomposition is not None else {}
                    ),
                }
            ),
            "output": _json_text(semantic_graph),
            "history": [],
        },
        "compose_row": {
            "instruction": COMPOSE_SFT_INSTRUCTION,
            "input": _json_text(compose_input),
            "output": _json_text(compose_program),
            "history": [],
        },
        "trace": [
            {
                "stage": "manual_correction",
                "correction_path": str(correction_path),
                "correction_sha256": correction_sha256,
                "checks": checks,
            }
        ],
        "usage": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _same_candidates(
    accepted: dict[str, Any],
    semantic_graph: dict[str, Any],
    compose_program: dict[str, Any],
) -> bool:
    try:
        accepted_semantic = json.loads(accepted["semantic_row"]["output"])
        accepted_compose = json.loads(accepted["compose_row"]["output"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return accepted_semantic == semantic_graph and accepted_compose == compose_program


def _failure_message(
    pair: SourcePair,
    errors: Sequence[str],
    *,
    stage: str = "static",
) -> str:
    return json.dumps(
        {
            "status": "rejected",
            "stage": stage,
            "source_id": pair.source_id,
            "compose_source_index": pair.compose_index,
            "errors": list(errors),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
