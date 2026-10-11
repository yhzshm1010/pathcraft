#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for import_root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpt54_data_flywheel.cli import (  # noqa: E402
    DEFAULT_CWQ_COMPOSE,
    DEFAULT_CWQ_DECOMPOSE,
    DEFAULT_CWQ_SEMANTIC,
    DEFAULT_CWQ_SOURCE,
)
from gpt54_data_flywheel.openai_client import (  # noqa: E402
    DEFAULT_GLM_BASE_URL,
    OpenAIRequestError,
    ResponsesClient,
)
from gpt54_data_flywheel.pipeline import (  # noqa: E402
    SourcePair,
    _append_jsonl,
    load_source_pairs,
)
from gpt54_data_flywheel.schemas import (  # noqa: E402
    COMPOSE_PROGRAM_SCHEMA,
    DECOMPOSITION_SCHEMA,
    SEMANTIC_GRAPH_SCHEMA,
)
from gpt54_data_flywheel.validation import (  # noqa: E402
    compile_compose_program,
    repair_answer_exclusion_scopes,
    repair_compose_entity_surfaces,
    repair_operator_fidelity,
    repair_semantic_graph_cross_path_variables,
    run_external_validator,
    validate_compose_program,
    validate_semantic_graph,
)
from validate_and_accept_manual_correction import (  # noqa: E402
    DEFAULT_EXECUTION_VALIDATOR,
    _correction_decomposition,
    _validate_golden_metadata,
    _validate_source_path_contract,
    validate_and_accept,
)


CORRECTION_PROMPT = r"""You are correcting one rejected Complex Web Question (CWQ) case.

Your task is to produce a complete correction JSON that passes both:
1. static validation by scripts/validate_and_accept_manual_correction.py
2. execution validation, where the candidate answers must exactly equal the answers of golden_sparql

The golden SPARQL is the final authority. If the question, decomposition, current semantic graph,
or current compose program conflicts with golden_sparql, follow golden_sparql.

You may modify decomposition. The corrected decomposition must be a non-empty object with a
decomposition array. The number of decomposition items must equal semantic_paths. Every
semantic_paths[i].goal must exactly equal decomposition.decomposition[i].

Semantic path rules:
- Every path is continuous: each step.from equals the previous anchor or previous step.to.
- Every step.to is local to that path, such as P0.V0 or P1.V1.
- path_output_var is the final step.to of that path.
- Do not switch anchors inside a path.
- Use canonical relation labels as arrays of at least three strings.
- Use backward for (?x, relation, entity) and forward for (entity, relation, ?x).

Compose rules:
- Compose must be consistently mappable from the semantic graph.
- Variables representing the same gold SPARQL node must map to the same compose variable.
- Do not use compose AND as a replacement for a variable join. Semantic AND is allowed only
  for a true set intersection and is lowered away in compose.
- answer_var must represent the SELECT answer variable, that is the answer-producing graph variable.
- Every compose triple must be connected and use only E* entities, V* variables, or quoted literals.
- Do not copy Freebase MIDs, dotted relation IDs, raw SPARQL, or invented entities into output.

Recover every constraint expressed by golden_sparql:
- missing relations and wrong directions
- CVT or mediator nodes
- FILTER and NO_EQUAL
- entity type constraints
- EQUAL and comparison constraints
- ARGMIN and ARGMAX
- full sorting/property paths behind ORDER BY and LIMIT

For ARGMIN/ARGMAX, do not keep only the last number relation. Preserve the complete attribute
path needed to compute the ordering value.

Do not guess Freebase MIDs from entity names. Use the supplied execution grounding diagnostics
when available, but emit only entity surfaces in the correction JSON.

When answers mismatch, inspect anchor identity, relation direction, missing triples, filters,
sorting path, and answer_var in that order.

Return only one JSON object. No Markdown fences and no explanation outside the JSON.
"""

CATEGORY_PROMPTS: dict[str, str] = {
    "01_execution_answer_mismatch": """Focus on execution equivalence with golden_sparql. Compare the candidate graph with every golden triple and FILTER. Check anchor identity, relation direction, missing mediator/CVT hops, shared-variable joins, exclusions, answer_var, and ordering/limit. A statically valid graph is not sufficient. Do not preserve a candidate path merely because it looks linguistically plausible.""",
    "02_source_sparql_no_answers": """The golden query currently has no answers on the configured Freebase snapshot. Candidate editing cannot prove exact equivalence. This category should normally be skipped before a model call; if explicitly forced, reconstruct the gold query literally and do not invent a non-empty answer-producing alternative.""",
    "03_semantic_compose_mapping": """Focus on variable identity and lowering. Keep Pn.Vm variables path-local in semantic, represent a gold shared SPARQL variable by one shared Vn in Compose, lower true semantic AND into shared variables, keep all triples connected, and set answer_var to the global variable corresponding to the golden SELECT variable.""",
    "04_freebase_grounding": """Focus on exact Freebase grounding. Use canonical relation labels derived from golden_sparql, correct forward/backward direction, preserve mediator/CVT hops, and use unambiguous entity surfaces. Never replace a gold relation with a semantically similar relation. Inspect grounding diagnostics before changing graph structure.""",
    "05_operator_input_var": """Focus on operator scope. Semantic unary operators use exactly one path-local input_var Pn.Vm and empty inputs. Compose unary operators use the mapped global Vn. Semantic AND uses path IDs in inputs and an empty input_var; AND is lowered away in Compose. Keep value, type, attribute path, and comparison direction identical to golden_sparql.""",
    "06_missing_required_operator": """Recover every operator expressed by golden_sparql. Extract FILTER comparisons and equality literals, temporal constraints, exclusions, COUNT, and the complete ORDER BY property path for ARGMIN/ARGMAX. Add the operator to semantic with path-local scope and to Compose with the corresponding global variable; do not add operators absent from gold.""",
}

DEFAULT_CATEGORY_PROMPT = """Diagnose the supplied validator errors against golden_sparql. Make the smallest complete graph correction that restores static validity and exact execution equivalence."""


CORRECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "compose_source_index": {"type": "integer", "minimum": 0},
        "source_id": {"type": "string"},
        "golden_source_id": {"type": "string"},
        "golden_parse_id": {"type": "string"},
        "golden_sparql": {"type": "string", "minLength": 1},
        "decomposition": DECOMPOSITION_SCHEMA,
        "semantic_graph": SEMANTIC_GRAPH_SCHEMA,
        "compose_program": COMPOSE_PROGRAM_SCHEMA,
        "notes": {"type": "string"},
    },
    "required": [
        "compose_source_index",
        "source_id",
        "golden_source_id",
        "golden_parse_id",
        "golden_sparql",
        "decomposition",
        "semantic_graph",
        "compose_program",
        "notes",
    ],
}


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "work/cwq_rejected/active_errors_with_golden_sparql.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "work/cwq_corrections"
DEFAULT_MODEL = "glm-5.2"
SOURCE_SPARQL_NO_ANSWERS_CATEGORY = "02_source_sparql_no_answers"
SOURCE_SPARQL_NO_ANSWERS_ERROR = "source SPARQL executed but returned no answers"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use GLM to correct active CWQ rejected cases, then gate every result with "
            "the manual correction static and execution validators."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--semantic-source", type=Path, default=DEFAULT_CWQ_SEMANTIC)
    parser.add_argument("--compose-source", type=Path, default=DEFAULT_CWQ_COMPOSE)
    parser.add_argument("--decompose-source", type=Path, default=DEFAULT_CWQ_DECOMPOSE)
    parser.add_argument("--dataset-source", type=Path, default=DEFAULT_CWQ_SOURCE)
    parser.add_argument("--model", default=os.environ.get("GLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url",
        "--responses-url",
        dest="base_url",
        default=os.environ.get("GLM_BASE_URL") or os.environ.get("GLM_RESPONSES_URL") or DEFAULT_GLM_BASE_URL,
        help="GLM base URL or Responses URL; chat completions are derived automatically",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--network-retries", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--json-mode",
        choices=("auto", "prompt"),
        default="auto",
        help="prompt is useful for gateways that reject structured-output fields",
    )
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"), default="low")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--compose-strategy",
        choices=("validate_then_compile", "compile", "model"),
        default="validate_then_compile",
        help="keep a statically valid model Compose and compile only as fallback",
    )
    parser.add_argument(
        "--skip-source-sparql-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "skip rows whose gold/source SPARQL has no answers before calling GLM; "
            "use --no-skip-source-sparql-failures to retry them"
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
    )
    parser.add_argument(
        "--preserve-source-decomposition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="freeze the source KaeDe path skeleton; default is gold-first correction",
    )
    parser.add_argument(
        "--accept",
        action="store_true",
        help="formally append passed corrections through validate_and_accept",
    )
    parser.add_argument("--accepted-output", type=Path, default=None)
    parser.add_argument("--finetune-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    if args.start < 0 or args.limit < 0:
        raise SystemExit("--start and --limit must be non-negative")
    if args.accept and args.accepted_output is None:
        raise SystemExit("--accept requires --accepted-output")
    if args.accept and args.finetune_output is None:
        raise SystemExit("--accept requires --finetune-output")

    api_key = args.api_key or os.environ.get("GLM_API_KEY") or os.environ.get("BIGMODEL_API_KEY", "")
    if not api_key.strip():
        raise SystemExit("GLM_API_KEY or BIGMODEL_API_KEY is required")

    summary = run_auto_correction(args, api_key=api_key)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_auto_correction(args: argparse.Namespace, *, api_key: str) -> dict[str, Any]:
    records = _read_jsonl(args.input)
    pairs = load_source_pairs(
        args.semantic_source,
        args.compose_source,
        webqsp_path=args.dataset_source,
        require_sparql=args.require_execution_validation,
        split_name="train",
        decompose_path=args.decompose_source,
        dataset="cwq",
    )
    pairs_by_index = {pair.compose_index: pair for pair in pairs}
    records = [
        record
        for record in records
        if not args.category or str(record.get("error_category", "")) in set(args.category)
    ]
    records.sort(key=lambda record: int(record.get("compose_source_index", -1)))
    records = records[args.start :]
    if args.limit:
        records = records[: args.limit]

    output_dir = args.output_dir.resolve()
    attempt_dir = output_dir / "attempts"
    correction_dir = output_dir / "corrections"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    correction_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    completed = _completed_results(results_path)

    client = ResponsesClient(
        api_key=api_key,
        model=args.model,
        responses_url=args.base_url,
        timeout=args.timeout,
        max_network_retries=args.network_retries,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        json_mode=args.json_mode,
        api_mode="chat",
    )

    counts: Counter[str] = Counter()
    for position, record in enumerate(records, start=1):
        source_id = str(record.get("source_id", "")).strip()
        compose_index = int(record.get("compose_source_index", -1))
        previous = completed.get(_result_key(record))
        if previous and previous.get("status") in {
            "passed",
            "accepted",
            "skipped_source_sparql",
        }:
            counts["skipped"] += 1
            continue
        if previous and not args.retry_failed:
            counts["skipped_failed"] += 1
            continue
        if args.skip_source_sparql_failures and _is_source_sparql_failure(record):
            print(
                f"[{position}/{len(records)}] compose_source_index={compose_index} "
                f"category={record.get('error_category', '')}",
                file=sys.stderr,
                flush=True,
            )
            print(
                "  skipped: gold/source SPARQL returned no answers; no GLM attempts",
                file=sys.stderr,
                flush=True,
            )
            result = _result_record(
                record,
                status="skipped_source_sparql",
                errors=[str(error) for error in record.get("errors", []) or []],
            )
            _append_jsonl(results_path, result)
            counts["skipped_source_sparql"] += 1
            continue
        pair = pairs_by_index.get(compose_index)
        if pair is None:
            result = _result_record(record, status="failed", errors=["source pair not found"])
            _append_jsonl(results_path, result)
            counts["failed"] += 1
            continue
        print(
            f"[{position}/{len(records)}] compose_source_index={compose_index} "
            f"category={record.get('error_category', '')}",
            file=sys.stderr,
            flush=True,
        )
        result = _process_record(
            record,
            pair=pair,
            args=args,
            client=client,
            attempt_dir=attempt_dir,
            correction_dir=correction_dir,
        )
        _append_jsonl(results_path, result)
        counts[str(result["status"])] += 1

    summary = {
        "input": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "model": args.model,
        "require_execution_validation": args.require_execution_validation,
        "accept": args.accept,
        "selected": len(records),
        "counts": dict(sorted(counts.items())),
        "current_state_counts": _current_state_counts(results_path),
        "results": str(results_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _process_record(
    record: dict[str, Any],
    *,
    pair: SourcePair,
    args: argparse.Namespace,
    client: ResponsesClient,
    attempt_dir: Path,
    correction_dir: Path,
) -> dict[str, Any]:
    feedback: list[str] = [str(error) for error in record.get("errors", []) or []]
    current = _initial_correction(record, pair)
    semantic_expected, compose_expected = _repair_operator_contract(record, pair)
    attempt_summaries: list[dict[str, Any]] = []
    for attempt in range(1, args.max_attempts + 1):
        feedback_before = list(feedback)
        _print_attempt_header(pair, attempt, args.max_attempts)
        _print_feedback("feedback sent to GLM", feedback_before)
        prompt_payload = _prompt_payload(
            record,
            pair,
            current=current,
            feedback=feedback,
            attempt=attempt,
        )
        try:
            generation = client.generate_json(
                schema_name="cwq_manual_correction",
                schema=CORRECTION_SCHEMA,
                instructions=_correction_prompt(record),
                payload=prompt_payload,
                output_tokens=args.max_output_tokens,
            )
            candidate = _normalize_model_correction(generation.value, record, pair)
            candidate, repair_notes = _repair_candidate(
                candidate,
                pair,
                compose_strategy=args.compose_strategy,
                semantic_expected=semantic_expected,
                compose_expected=compose_expected,
            )
            if repair_notes:
                _print_feedback("deterministic source repairs applied", repair_notes)
        except (OpenAIRequestError, ValueError, TypeError, json.JSONDecodeError) as exc:
            error_text = f"model correction failed: {type(exc).__name__}: {exc}"
            feedback = [*feedback_before, error_text]
            attempt_summary = {
                "attempt": attempt,
                "status": "model_error",
                "feedback_before": feedback_before,
                "errors": feedback,
                "feedback_after": list(feedback),
            }
            attempt_summaries.append(attempt_summary)
            _print_feedback("errors for next attempt", feedback)
            continue

        candidate_path = attempt_dir / f"compose_source_index_{pair.compose_index}_attempt_{attempt}.json"
        _write_json(candidate_path, candidate)
        errors = _validate_candidate(
            candidate,
            pair=pair,
            args=args,
            attempt=attempt,
            semantic_expected=semantic_expected,
            compose_expected=compose_expected,
        )
        attempt_summary: dict[str, Any] = {
            "attempt": attempt,
            "status": "validation_failed" if errors else "passed",
            "correction_path": str(candidate_path),
            "feedback_before": feedback_before,
            "errors": errors,
            "feedback_after": list(errors),
            "response_id": generation.response_id,
            "transport_mode": generation.transport_mode,
            "usage": generation.usage,
            "deterministic_repairs": repair_notes,
        }
        attempt_summaries.append(attempt_summary)
        current = candidate
        if errors:
            print(
                f"  attempt {attempt}/{args.max_attempts}: candidate validation failed",
                file=sys.stderr,
                flush=True,
            )
            feedback = errors
            _print_feedback("errors for next attempt", feedback)
            continue

        final_path = correction_dir / f"compose_source_index_{pair.compose_index}.json"
        _write_json(final_path, candidate)
        print(
            f"  attempt {attempt}/{args.max_attempts}: static + execution validation passed",
            file=sys.stderr,
            flush=True,
        )
        accepted_status = "passed"
        accept_result: dict[str, Any] | None = None
        if args.accept:
            try:
                accept_result = validate_and_accept(
                    correction_path=final_path,
                    semantic_source=args.semantic_source,
                    compose_source=args.compose_source,
                    webqsp_source=args.dataset_source,
                    accepted_output=args.accepted_output,
                    finetune_output=args.finetune_output,
                    split_name="train",
                    require_execution_validation=args.require_execution_validation,
                    execution_validator_command=args.execution_validator_command,
                    execution_validator_timeout=args.execution_validator_timeout,
                    dry_run=False,
                    dataset="cwq",
                    decompose_source=args.decompose_source,
                    preserve_source_decomposition=args.preserve_source_decomposition,
                )
                accepted_status = "accepted"
            except (FileNotFoundError, ValueError) as exc:
                accepted_status = "failed_accept"
                feedback = _failure_errors(exc)
                attempt_summary["accept_errors"] = feedback
                attempt_summary["status"] = "failed_accept"
                attempt_summary["errors"] = feedback
                attempt_summary["feedback_after"] = list(feedback)
                _print_feedback("accept errors", feedback)
        return _result_record(
            record,
            status=accepted_status,
            errors=feedback if accepted_status == "failed_accept" else [],
            correction_path=final_path,
            attempts=attempt_summaries,
            accept_result=accept_result,
        )

    print(
        f"  all {args.max_attempts} attempts failed for compose_source_index="
        f"{pair.compose_index}",
        file=sys.stderr,
        flush=True,
    )
    _print_feedback("final errors", feedback)
    return _result_record(
        record,
        status="failed",
        errors=feedback,
        correction_path=None,
        attempts=attempt_summaries,
    )


def _print_attempt_header(pair: SourcePair, attempt: int, max_attempts: int) -> None:
    print(
        f"  attempt {attempt}/{max_attempts}: correcting compose_source_index="
        f"{pair.compose_index}",
        file=sys.stderr,
        flush=True,
    )


def _print_feedback(label: str, feedback: Sequence[str]) -> None:
    if not feedback:
        print(f"    {label}: none", file=sys.stderr, flush=True)
        return
    print(f"    {label}:", file=sys.stderr, flush=True)
    for error in feedback:
        print(f"      - {error}", file=sys.stderr, flush=True)


def _validate_candidate(
    correction: dict[str, Any],
    *,
    pair: SourcePair,
    args: argparse.Namespace,
    attempt: int = 0,
    semantic_expected: list[dict[str, Any]] | None = None,
    compose_expected: list[dict[str, Any]] | None = None,
) -> list[str]:
    try:
        _validate_golden_metadata(correction, pair)
        decomposition = _correction_decomposition(
            correction,
            pair=pair,
            required=args.preserve_source_decomposition,
        )
    except (FileNotFoundError, ValueError) as exc:
        return _failure_errors(exc)

    source_contract_errors = (
        _validate_source_path_contract(
            correction["semantic_graph"],
            pair.semantic_output,
        )
        if args.preserve_source_decomposition
        else []
    )
    semantic_errors = validate_semantic_graph(
        correction["semantic_graph"],
        expected=(
            pair.expected_operators
            if semantic_expected is None
            else semantic_expected
            or list(correction["semantic_graph"].get("operators", []) or [])
        ),
        decomposition=(
            list(decomposition.get("decomposition", []) or [])
            if decomposition is not None
            else None
        ),
        source_graph=None,
    )
    compose_errors = validate_compose_program(
        correction["compose_program"],
        semantic_graph=correction["semantic_graph"],
        expected=(
            pair.compose_expected_operators
            if compose_expected is None
            else compose_expected
            or list(correction["compose_program"].get("operators", []) or [])
        ),
        source_graph=None,
        source_program=None,
    )
    errors = [
        *(f"semantic: {error}" for error in source_contract_errors),
        *(f"semantic: {error}" for error in semantic_errors),
        *(f"compose: {error}" for error in compose_errors),
    ]
    if errors or not args.require_execution_validation:
        return errors

    validator_errors = run_external_validator(
        args.execution_validator_command,
        {
            "source_id": pair.source_id,
            "source_index": pair.index,
            "attempt": attempt,
            "question": pair.question,
            "semantic_graph": correction["semantic_graph"],
            "compose_program": correction["compose_program"],
            "source_semantic_graph": pair.semantic_output,
            "source_compose_program": pair.compose_output,
            "webqsp_source": None,
            "cwq_source": pair.cwq_source.as_context() if pair.cwq_source else None,
        },
        timeout=args.execution_validator_timeout,
    )
    return [f"execution: {error}" for error in validator_errors]


def _repair_candidate(
    correction: dict[str, Any],
    pair: SourcePair,
    *,
    compose_strategy: str = "validate_then_compile",
    semantic_expected: list[dict[str, Any]] | None = None,
    compose_expected: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize semantic output and compile Compose only when necessary."""
    semantic_graph, semantic_variable_notes = repair_semantic_graph_cross_path_variables(
        correction["semantic_graph"]
    )
    effective_semantic_expected = (
        pair.expected_operators if semantic_expected is None else semantic_expected
    )
    semantic_operator_notes: list[str] = []
    if effective_semantic_expected:
        semantic_graph, semantic_operator_notes = repair_operator_fidelity(
            semantic_graph,
            effective_semantic_expected,
            location="semantic_graph",
        )
    compose_program = correction["compose_program"]
    compile_errors = validate_compose_program(
        compose_program,
        semantic_graph=semantic_graph,
        expected=list(compose_program.get("operators", []) or []),
        source_graph=None,
        source_program=None,
    )
    should_compile = compose_strategy == "compile" or (
        compose_strategy == "validate_then_compile" and bool(compile_errors)
    )
    compile_notes: list[str] = []
    if should_compile:
        compose_program = compile_compose_program(
            semantic_graph,
            answer_path_ids=getattr(pair, "answer_path_ids", None),
            source_graph=pair.semantic_output,
            source_program=pair.compose_output,
        )
        compile_notes.append(
            "rebuilt compose_program because model Compose failed static mapping: "
            + "; ".join(compile_errors[:3])
            if compile_errors
            else "rebuilt compose_program by configured compile strategy"
        )
    compose_program, compose_surface_notes = repair_compose_entity_surfaces(
        compose_program,
        semantic_graph,
    )
    effective_compose_expected = (
        pair.compose_expected_operators if compose_expected is None else compose_expected
    )
    compose_operator_notes: list[str] = []
    if effective_compose_expected:
        compose_program, compose_operator_notes = repair_operator_fidelity(
            compose_program,
            effective_compose_expected,
            location="compose_program",
        )
    exclusions = pair.cwq_source.no_equal_exclusions if pair.cwq_source else []
    semantic_graph, compose_program, exclusion_notes = repair_answer_exclusion_scopes(
        semantic_graph,
        compose_program,
        exclusions,
        source_graph=pair.semantic_output,
        source_program=pair.compose_output,
    )
    repaired = dict(correction)
    repaired["semantic_graph"] = semantic_graph
    repaired["compose_program"] = compose_program
    notes = [
        *semantic_variable_notes,
        *semantic_operator_notes,
        *compile_notes,
        *compose_surface_notes,
        *compose_operator_notes,
        *exclusion_notes,
    ]
    if notes:
        repaired["notes"] = " ".join(
            value for value in (str(correction.get("notes", "")), *notes) if value
        )
    return repaired, notes


def _correction_prompt(record: dict[str, Any]) -> str:
    category = str(record.get("error_category", "")).strip()
    return (
        CORRECTION_PROMPT
        + "\n\nCASE-SPECIFIC REPAIR POLICY\n"
        + f"Error category: {category or 'unclassified'}\n"
        + CATEGORY_PROMPTS.get(category, DEFAULT_CATEGORY_PROMPT)
    )


def _repair_operator_contract(
    record: dict[str, Any], pair: SourcePair
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Do not force rejected source operators onto gold-first graph repairs.

    Execution validation remains the authority for these categories. Operator-specific
    queues retain the source-derived contract because their task is mechanical operator repair.
    Gold-derived NO_EQUAL exclusions are applied separately by
    repair_answer_exclusion_scopes.
    """
    category = str(record.get("error_category", "")).strip()
    if category in {
        "01_execution_answer_mismatch",
        "03_semantic_compose_mapping",
        "04_freebase_grounding",
    }:
        return [], []
    return pair.expected_operators, pair.compose_expected_operators


def _initial_correction(record: dict[str, Any], pair: SourcePair) -> dict[str, Any]:
    semantic = record.get("last_semantic_candidate")
    compose = record.get("last_compose_candidate")
    decomposition = record.get("last_decomposition_candidate")
    if not isinstance(semantic, dict) or not semantic:
        semantic = pair.semantic_output
    if not isinstance(compose, dict) or not compose:
        compose = pair.compose_output
    if not isinstance(decomposition, dict) or not decomposition:
        decomposition = pair.decompose_output or {"decomposition": []}
    return {
        "compose_source_index": pair.compose_index,
        "source_id": pair.source_id,
        "golden_source_id": pair.cwq_source.source_id if pair.cwq_source else "",
        "golden_parse_id": "",
        "golden_sparql": pair.cwq_source.sparql if pair.cwq_source else "",
        "decomposition": decomposition,
        "semantic_graph": semantic,
        "compose_program": compose,
        "notes": "Initial rejected candidate; must be corrected against golden_sparql.",
    }


def _prompt_payload(
    record: dict[str, Any],
    pair: SourcePair,
    *,
    current: dict[str, Any],
    feedback: list[str],
    attempt: int,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "question": pair.question,
        "decomposition": current.get("decomposition"),
        "semantic_graph": current.get("semantic_graph"),
        "compose_program": current.get("compose_program"),
        "golden_source_id": pair.cwq_source.source_id if pair.cwq_source else "",
        "golden_parse_id": "",
        "golden_sparql": pair.cwq_source.sparql if pair.cwq_source else "",
        "gold_answers": pair.cwq_source.answers if pair.cwq_source else [],
        "composition_answer": pair.cwq_source.composition_answer if pair.cwq_source else "",
        "previous_attempt_errors": feedback[-24:],
        "validator_errors": feedback[-24:],
        "original_rejected_errors": record.get("errors", []),
        "source_semantic_graph_reference": pair.semantic_output,
        "source_compose_program_reference": pair.compose_output,
        "execution_source_context": (
            pair.cwq_source.as_context() if pair.cwq_source else None
        ),
        "latest_execution_diagnostics": feedback[-24:],
        "instruction": (
            "Reference graphs are not authoritative labels. Use golden_sparql and the "
            "validator diagnostics as the authority, and output only the correction object."
        ),
    }


def _normalize_model_correction(
    value: Any,
    record: dict[str, Any],
    pair: SourcePair,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model output must be an object")
    required = ("decomposition", "semantic_graph", "compose_program")
    missing = [field for field in required if not isinstance(value.get(field), dict)]
    if missing:
        raise ValueError("model correction is missing object fields: " + ", ".join(missing))
    golden_source_id = pair.cwq_source.source_id if pair.cwq_source else ""
    golden_sparql = pair.cwq_source.sparql if pair.cwq_source else ""
    return {
        "compose_source_index": pair.compose_index,
        "source_id": pair.source_id,
        "golden_source_id": golden_source_id,
        "golden_parse_id": "",
        "golden_sparql": golden_sparql,
        "decomposition": value["decomposition"],
        "semantic_graph": value["semantic_graph"],
        "compose_program": value["compose_program"],
        "notes": str(value.get("notes", "GLM automatic correction")),
    }


def _result_record(
    record: dict[str, Any],
    *,
    status: str,
    errors: list[str],
    correction_path: Path | None = None,
    attempts: list[dict[str, Any]] | None = None,
    accept_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "source_id": record.get("source_id", ""),
        "compose_source_index": record.get("compose_source_index"),
        "question": record.get("question", ""),
        "error_category": record.get("error_category", ""),
        "errors": errors,
        "correction_path": str(correction_path) if correction_path else "",
        "attempts": attempts or [],
        "accept_result": accept_result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _failure_errors(exc: Exception) -> list[str]:
    text = str(exc)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(payload, dict) and isinstance(payload.get("errors"), list):
        return [str(value) for value in payload["errors"]]
    return [text]


def _is_source_sparql_failure(record: dict[str, Any]) -> bool:
    """Identify rows that cannot pass execution gating because gold is empty."""
    if str(record.get("error_category", "")).strip() == SOURCE_SPARQL_NO_ANSWERS_CATEGORY:
        return True
    return any(
        SOURCE_SPARQL_NO_ANSWERS_ERROR in str(error)
        for error in record.get("errors", []) or []
    )


def _completed_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        key = _result_key(record)
        if key:
            output[key] = record
    return output


def _result_key(record: dict[str, Any]) -> str:
    """Prefer the candidate index because one CWQ source may have multiple candidates."""
    compose_index = record.get("compose_source_index")
    if compose_index is not None:
        return f"compose:{compose_index}"
    source_id = str(record.get("source_id", "")).strip()
    return f"source:{source_id}" if source_id else ""


def _current_state_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter(
        str(record.get("status", "unknown"))
        for record in _completed_results(path).values()
    )
    return dict(sorted(counts.items()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {path}")
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            output.append(value)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
