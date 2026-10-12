from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Sequence

from .openai_client import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_GLM_BASE_URL,
    DEFAULT_RESPONSES_URL,
    ExternalResponsesClient,
    ResponsesClient,
)
from .pipeline import (
    Flywheel,
    FlywheelConfig,
    load_source_pairs,
    materialize,
    resolve_webqsp_path,
    source_summary,
    resolve_cwq_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(
    os.environ.get("GPT54_FLYWHEEL_INPUT_ROOT", PROJECT_ROOT / "inputs")
).expanduser()
DEFAULT_SEMANTIC = INPUT_ROOT / "webqsp/semantic_path_train.json"
DEFAULT_COMPOSE = INPUT_ROOT / "webqsp/compose_train.json"
DEFAULT_WEBQSP_SOURCE = (
    INPUT_ROOT / "webqsp/WebQSP.train.json"
)
DEFAULT_CWQ_DECOMPOSE = INPUT_ROOT / "cwq/decompose_train.json"
DEFAULT_CWQ_SEMANTIC = INPUT_ROOT / "cwq/semantic_path_train.json"
DEFAULT_CWQ_COMPOSE = INPUT_ROOT / "cwq/compose_train.json"
DEFAULT_CWQ_SOURCE = INPUT_ROOT / "cwq/ComplexWebQuestions_train.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "work/generated"
DEFAULT_EXECUTION_VALIDATOR = (
    f"{shlex.quote(sys.executable)} "
    f"{shlex.quote(str(PROJECT_ROOT / 'scripts' / 'cwq_execution_validator.py'))}"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt54-flywheel",
        description="Construct and verify CWQ TRAIN stage-aligned data with GLM-5.2.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="pair source rows without calling the API")
    preflight.add_argument("--dataset", choices=("cwq",), default="cwq")
    _add_source_arguments(preflight)
    preflight.add_argument("--split-name", default="train")

    generate = subparsers.add_parser("generate", help="run the verified generation loop")
    generate.add_argument("--dataset", choices=("cwq",), default="cwq")
    _add_source_arguments(generate)
    generate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    generate.add_argument("--split-name", default="train")
    generate.add_argument(
        "--model",
        default=None,
    )
    generate.add_argument(
        "--responses-url",
        "--base-url",
        dest="responses_url",
        default=None,
        help=(
            "provider base URL or full Responses API endpoint; chat mode derives "
            "/chat/completions and --base-url remains a compatibility alias"
        ),
    )
    generate.add_argument(
        "--reasoning-effort",
        choices=_EFFORTS,
        default=None,
    )
    generate.add_argument("--verifier-reasoning-effort", choices=_EFFORTS, default="high")
    generate.add_argument("--max-attempts", type=int, default=5)
    generate.add_argument("--network-retries", type=int, default=5)
    generate.add_argument("--timeout", type=float, default=180.0)
    generate.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help=(
            "optional output cap; 0 omits it for Responses and uses the external "
            "Chat default of 2048"
        ),
    )
    generate.add_argument(
        "--json-mode",
        choices=("auto", "schema", "prompt"),
        default="auto",
        help="structured-output mode; prompt avoids gateways that fail on text.format.json_schema",
    )
    generate.add_argument(
        "--api-mode",
        choices=("chat",),
        default="chat",
        help="API transport; chat is the OpenAI-compatible chat interface",
    )
    generate.add_argument(
        "--parallel-tool-calls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "set the provider parallel_tool_calls request field; DeepSeek v4 Flash "
            "defaults to false"
        ),
    )
    generate.add_argument(
        "--pipeline-profile",
        choices=("auto", "standard", "lean", "micro"),
        default="auto",
        help=(
            "auto uses lean for the External proxy, micro for chat, and standard for "
            "direct Responses"
        ),
    )
    generate.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "rebuild semantic_path/compose JSON after this many processed rows; "
            "default 1 writes after every question"
        ),
    )
    generate.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show per-row semantic/compose/verifier progress (disable with --no-progress)",
    )
    generate.add_argument("--start", type=int, default=0)
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument(
        "--skip-source-ids",
        type=Path,
        default=None,
        help=(
            "JSON array or JSONL file of source IDs (or objects containing source_id) "
            "that must be skipped even when they are not accepted"
        ),
    )
    generate.add_argument(
        "--skip-rejected-attempts",
        type=int,
        default=None,
        help=(
            "when --skip-source-ids contains rejected record objects, skip only "
            "records whose attempts equals this value"
        ),
    )
    generate.add_argument(
        "--reverse",
        action="store_true",
        help="process selected source rows from the last row toward the first",
    )
    generate.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of samples to process concurrently; each sample keeps its stage order",
    )
    generate.add_argument(
        "--operator-only",
        action="store_true",
        help="process rows with explicit semantic AND or an executable source operator",
    )
    generate.add_argument(
        "--external-validator-command",
        default=None,
        help="command reading a candidate JSON object on stdin and returning {\"valid\": true|false}",
    )
    generate.add_argument("--external-validator-timeout", type=float, default=120.0)
    generate.add_argument(
        "--require-execution-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fail closed unless the external execution validator approves the candidate",
    )
    generate.add_argument(
        "--preserve-source-decomposition",
        action="store_true",
        help=(
            "for CWQ, keep the supplied CWQ path-level decomposition and start at "
            "semantic generation; default rewrites decomposition as before"
        ),
    )
    generate.add_argument(
        "--strict-decomposition-rewrite",
        action="store_true",
        help=(
            "rewrite CWQ decomposition with deterministic natural-language and "
            "cross-stage alignment gates; mutually exclusive with "
            "--preserve-source-decomposition"
        ),
    )
    generate.add_argument(
        "--retain-rejection-trace",
        action="store_true",
        help=(
            "write full rejected traces to rejected.jsonl; use only for small "
            "debug runs because the audit file grows substantially"
        ),
    )

    render = subparsers.add_parser("materialize", help="rebuild JSON arrays from accepted.jsonl")
    render.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    render.add_argument("--split-name", default="train")
    return parser


_EFFORTS = ("none", "low", "medium", "high", "xhigh")


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decompose", type=Path, default=None, help="CWQ decompose source JSON")
    parser.add_argument("--semantic", type=Path, default=None)
    parser.add_argument("--compose", type=Path, default=None)
    parser.add_argument(
        "--cwq-source", dest="webqsp_source",
        type=Path,
        default=None,
        help=(
            "WebQSP source JSON or split directory; use a file with non-empty Sparql "
            "fields, not the converted OpenDataLab copy that blanks them out"
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        manifest = materialize(args.output, split_name=args.split_name)
        print(json.dumps(manifest.get("counts", {}), ensure_ascii=False, sort_keys=True))
        return

    defaults = _dataset_defaults(args.dataset, args.split_name)
    semantic_path = args.semantic or defaults["semantic"]
    compose_path = args.compose or defaults["compose"]
    decompose_path = args.decompose or defaults.get("decompose")
    source_path = args.webqsp_source or defaults["source"]
    requested_api_mode = getattr(args, "api_mode", None) or os.environ.get("GLM_API_MODE")
    deepseek_model_env = os.environ.get("DEEPSEEK_MODEL")
    model_env = deepseek_model_env or (
        os.environ.get("GLM_MODEL")
        if args.dataset == "cwq" or requested_api_mode in {"chat", "responses"}
        else os.environ.get("WEBQSP_MODEL")
    )
    model_hint = (
        getattr(args, "model", None) or model_env or os.environ.get("OPENAI_MODEL") or ""
    )
    is_deepseek = model_hint.casefold().startswith("deepseek")
    api_mode = requested_api_mode or (
        "chat" if args.dataset == "cwq" or is_deepseek else "external"
    )
    if is_deepseek and api_mode != "chat":
        raise SystemExit("deepseek-v4-flash requires --api-mode chat")
    is_glm = api_mode in {"chat", "responses"} and (
        (
            not is_deepseek
            and model_hint.casefold().startswith("glm")
        )
        or (args.dataset == "cwq" and not model_hint)
    )
    model = model_hint or ("glm-5.2" if is_glm else "gpt-5.4")
    if is_deepseek:
        url_env = (
            os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("DEEPSEEK_RESPONSES_URL")
            or os.environ.get("CSTCLOUD_BASE_URL")
        )
    elif is_glm:
        url_env = (
            os.environ.get("GLM_RESPONSES_URL")
            or os.environ.get("GLM_BASE_URL")
        )
    else:
        url_env = (
            os.environ.get("WEBQSP_RESPONSES_URL")
            or os.environ.get("OPENAI_RESPONSES_URL")
        )
    responses_url = getattr(args, "responses_url", None) or (
        url_env
        or (
            DEFAULT_DEEPSEEK_BASE_URL
            if is_deepseek
            else (DEFAULT_GLM_BASE_URL if is_glm else DEFAULT_RESPONSES_URL)
        )
    )
    requested_execution_validation = getattr(args, "require_execution_validation", None)
    require_execution_validation = (
        requested_execution_validation
        if requested_execution_validation is not None
        else args.dataset == "webqsp"
    )
    require_sparql = (
        args.command == "preflight"
        or require_execution_validation
        or args.dataset == "cwq"
    )
    source_resolver = resolve_cwq_path if args.dataset == "cwq" else resolve_webqsp_path
    webqsp_source = source_resolver(source_path, split_name=args.split_name)
    try:
        pairs = load_source_pairs(
            semantic_path,
            compose_path,
            webqsp_path=webqsp_source,
            require_sparql=require_sparql,
            split_name=args.split_name,
            decompose_path=decompose_path,
            dataset=args.dataset,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    summary = source_summary(pairs)
    summary["dataset_source"] = str(webqsp_source.resolve())
    if args.command == "preflight":
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.start < 0 or args.limit < 0:
        raise SystemExit("--start and --limit must be non-negative")
    if args.preserve_source_decomposition and args.strict_decomposition_rewrite:
        raise SystemExit(
            "--strict-decomposition-rewrite cannot be combined with "
            "--preserve-source-decomposition"
        )
    if args.strict_decomposition_rewrite and args.dataset != "cwq":
        raise SystemExit("--strict-decomposition-rewrite is only supported for CWQ")
    selected = list(reversed(pairs)) if args.reverse else pairs
    if args.operator_only:
        selected = [pair for pair in selected if pair.has_semantic_operator]
    selected = selected[args.start :]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("no source rows selected")

    try:
        skip_source_ids = _load_skip_source_ids(
            args.skip_source_ids,
            attempts=args.skip_rejected_attempts,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    reasoning_effort = getattr(args, "reasoning_effort", None) or (
        os.environ.get(
            "DEEPSEEK_REASONING_EFFORT"
            if is_deepseek
            else ("GLM_REASONING_EFFORT" if is_glm else "WEBQSP_REASONING_EFFORT")
        )
        or os.environ.get("OPENAI_REASONING_EFFORT") or "low"
    )
    if is_deepseek:
        key_names = ("DEEPSEEK_API_KEY", "CSTCLOUD_API_KEY", "OPENAI_API_KEY")
    elif is_glm:
        key_names = ("GLM_API_KEY", "BIGMODEL_API_KEY")
    else:
        key_names = ("WEBQSP_API_KEY", "OPENAI_API_KEY", "GPT54_API_KEY")
    api_key = next((os.environ.get(name, "") for name in key_names if os.environ.get(name)), "")
    if not api_key:
        raise SystemExit(f"{', '.join(key_names)} is required for generate")
    external_validator_command = args.external_validator_command
    if external_validator_command is None:
        validator_env = (
            "WEBQSP_EXECUTION_VALIDATOR_COMMAND"
            if args.dataset == "webqsp"
            else "CWQ_EXECUTION_VALIDATOR_COMMAND"
        )
        external_validator_command = os.environ.get(validator_env) or (
            DEFAULT_EXECUTION_VALIDATOR if require_execution_validation else ""
        )
    if api_mode == "external":
        client = ExternalResponsesClient(
            api_key=api_key,
            model=model,
            responses_url=responses_url,
            timeout=args.timeout,
            max_network_retries=args.network_retries,
            reasoning_effort=reasoning_effort,
        )
    else:
        client = ResponsesClient(
            api_key=api_key,
            model=model,
            responses_url=responses_url,
            timeout=args.timeout,
            max_network_retries=args.network_retries,
            reasoning_effort=reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            json_mode=args.json_mode,
            api_mode=api_mode,
            parallel_tool_calls=(
                args.parallel_tool_calls
                if args.parallel_tool_calls is not None
                else (False if is_deepseek else None)
            ),
        )
    pipeline_profile = args.pipeline_profile
    if pipeline_profile == "auto":
        if api_mode == "chat" and model.strip().casefold().startswith("glm"):
            pipeline_profile = "standard"
        else:
            pipeline_profile = {
                "external": "lean",
                "chat": "micro",
                "responses": "standard",
            }[api_mode]
    flywheel = Flywheel(
        client=client,
        config=FlywheelConfig(
            output_dir=args.output,
            split_name=args.split_name,
            dataset=args.dataset,
            max_attempts=args.max_attempts,
            checkpoint_every=args.checkpoint_every,
            verifier_reasoning_effort=args.verifier_reasoning_effort,
            external_validator_command=external_validator_command,
            external_validator_timeout=args.external_validator_timeout,
            require_execution_validation=require_execution_validation,
            pipeline_profile=pipeline_profile,
            show_progress=args.progress,
            workers=args.workers,
            preserve_source_decomposition=args.preserve_source_decomposition,
            strict_decomposition_rewrite=args.strict_decomposition_rewrite,
            skip_source_ids=frozenset(skip_source_ids),
            retain_rejection_trace=args.retain_rejection_trace,
        ),
        semantic_source=semantic_path,
        compose_source=compose_path,
        webqsp_source=webqsp_source,
        decompose_source=decompose_path,
    )
    print(
        json.dumps(
            {
                "model": model,
                "api_mode": client.api_mode,
                "api_url": client.api_url,
                "json_mode": client.json_mode,
                "pipeline_profile": pipeline_profile,
                "responses_url": client.responses_url,
                "execution_validator": external_validator_command,
                "require_execution_validation": require_execution_validation,
                "preserve_source_decomposition": args.preserve_source_decomposition,
                "strict_decomposition_rewrite": args.strict_decomposition_rewrite,
                "selected_rows": len(selected),
                "skip_source_ids": len(skip_source_ids),
                "retain_rejection_trace": args.retain_rejection_trace,
                "workers": args.workers,
                "source_summary": summary,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    result = flywheel.run(selected)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def _load_skip_source_ids(path: Path | None, *, attempts: int | None = None) -> set[str]:
    if path is None:
        return set()
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"skip source IDs file does not exist: {resolved}")
    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        return set()

    values: list[object]
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        values = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{resolved}:{line_number} contains invalid JSON: {exc}"
                ) from exc
    else:
        if not isinstance(decoded, list):
            raise ValueError(f"{resolved} must contain a JSON array or JSONL records")
        values = decoded

    source_ids: set[str] = set()
    for index, value in enumerate(values, 1):
        if isinstance(value, str):
            if attempts is not None:
                raise ValueError(
                    f"{resolved}: --skip-rejected-attempts requires record objects "
                    "containing source_id and attempts"
                )
            source_id = value.strip()
        elif isinstance(value, dict):
            if attempts is not None:
                try:
                    record_attempts = int(value.get("attempts", -1))
                except (TypeError, ValueError):
                    record_attempts = -1
                if record_attempts != attempts:
                    continue
            source_id = str(value.get("source_id", "")).strip()
        else:
            source_id = ""
        if not source_id:
            raise ValueError(
                f"{resolved}: item {index} must be a non-empty source ID or an object "
                "containing source_id"
            )
        source_ids.add(source_id)
    return source_ids


def _dataset_defaults(dataset: str, split_name: str) -> dict[str, Path | None]:
    if dataset == "cwq":
        root = INPUT_ROOT / "cwq"
        return {
            "decompose": root / f"decompose_{split_name}.json",
            "semantic": root / f"semantic_path_{split_name}.json",
            "compose": root / f"compose_{split_name}.json",
            "source": DEFAULT_CWQ_SOURCE,
        }
    return {
        "decompose": None,
        "semantic": DEFAULT_SEMANTIC,
        "compose": DEFAULT_COMPOSE,
        "source": Path(os.environ.get("WEBQSP_SOURCE") or DEFAULT_WEBQSP_SOURCE),
    }


if __name__ == "__main__":
    main()
