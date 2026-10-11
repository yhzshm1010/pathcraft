from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Protocol

from .openai_client import Generation
from .prompts import (
    COMPOSE_INSTRUCTIONS,
    COMPOSE_SFT_INSTRUCTION,
    CWQ_GATEWAY_DECOMPOSE_INSTRUCTIONS,
    CWQ_GATEWAY_DECOMPOSE_VERIFY_INSTRUCTIONS,
    DECOMPOSE_INSTRUCTIONS,
    DECOMPOSE_SFT_INSTRUCTION,
    DECOMPOSE_VERIFY_INSTRUCTIONS,
    GATEWAY_COMPOSE_INSTRUCTIONS,
    GATEWAY_SEMANTIC_INSTRUCTIONS,
    GATEWAY_VERIFY_INSTRUCTIONS,
    KAEDE_DECOMPOSE_SFT_INSTRUCTION,
    KAEDE_SEMANTIC_SFT_INSTRUCTION,
    KAEDE_SOURCE_DECOMPOSITION_ALIGNMENT,
    LEAN_COMPOSE_INSTRUCTIONS,
    LEAN_SEMANTIC_INSTRUCTIONS,
    LEAN_VERIFY_INSTRUCTIONS,
    MICRO_COMPOSE_INSTRUCTIONS,
    MICRO_SEMANTIC_INSTRUCTIONS,
    MICRO_VERIFY_INSTRUCTIONS,
    SEMANTIC_INSTRUCTIONS,
    SEMANTIC_SFT_INSTRUCTION,
    STRICT_DECOMPOSE_SFT_INSTRUCTION,
    STRICT_DECOMPOSITION_REWRITE_INSTRUCTIONS,
    VERIFY_INSTRUCTIONS,
)
from .schemas import (
    COMPOSE_PROGRAM_SCHEMA,
    DECOMPOSITION_SCHEMA,
    DECOMPOSITION_VERIFICATION_SCHEMA,
    LIGHT_DECOMPOSITION_VERIFICATION_SCHEMA,
    LIGHT_VERIFICATION_SCHEMA,
    SEMANTIC_GRAPH_SCHEMA,
    VERIFICATION_SCHEMA,
)
from .validation import (
    compile_compose_program,
    expected_operators,
    operator_signature,
    project_program_operators_to_semantic,
    repair_answer_exclusion_scopes,
    repair_compose_literal_equalities,
    repair_compose_entity_surfaces,
    repair_operator_fidelity,
    repair_semantic_graph_directions_to_compose,
    repair_semantic_graph_cross_path_variables,
    repair_semantic_path_output_variables,
    repair_source_structural_and,
    repair_unscoped_semantic_operator_variables,
    repair_superfluous_semantic_and,
    run_external_validator,
    semantic_answer_variables,
    validate_compose_program,
    validate_decomposition,
    validate_semantic_graph,
)


_SELECT_VARIABLE_RE = re.compile(
    r"\bSELECT\s+(?:DISTINCT\s+)?(?P<variable>\?[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_ANSWER_EXCLUSION_RE = re.compile(
    r"FILTER\s*\(\s*(?P<variable>\?[A-Za-z_][A-Za-z0-9_]*)\s*!=\s*"
    r"(?:ns:(?P<compact>[mg]\.[A-Za-z0-9_]+)|"
    r"<http://rdf\.freebase\.com/ns/(?P<uri>[mg]\.[A-Za-z0-9_]+)>)\s*\)",
    re.IGNORECASE,
)
_VARIABLE_EXCLUSION_RE = re.compile(
    r"FILTER\s*\(\s*(?P<answer>\?[A-Za-z_][A-Za-z0-9_]*)\s*!=\s*"
    r"(?P<excluded>\?[A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.IGNORECASE,
)
_SOURCE_SPARQL_NO_ANSWERS = "source SPARQL executed but returned no answers"


def _source_sparql_has_no_answers(errors: list[str]) -> bool:
    """Whether execution established an unfixable empty gold-answer set."""
    return any(_SOURCE_SPARQL_NO_ANSWERS in error for error in errors)
_STRING_EQUAL_FILTER_RE = re.compile(
    r"FILTER\s*\(\s*str\s*\(\s*(?P<attribute>\?[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*"
    r"=\s*\"(?P<value>(?:\\.|[^\"\\])*)\"\s*\)",
    re.IGNORECASE,
)
_VARIABLE_RELATION_TRIPLE_RE = re.compile(
    r"(?P<input>\?[A-Za-z_][A-Za-z0-9_]*)\s+ns:"
    r"(?P<relation>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)\s+"
    r"(?P<attribute>\?[A-Za-z_][A-Za-z0-9_]*)\s*\.",
    re.IGNORECASE,
)


class JsonGenerator(Protocol):
    model: str

    def generate_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        payload: dict[str, Any],
        reasoning_effort: str | None = None,
        gateway_instructions: str | None = None,
        output_tokens: int | None = None,
    ) -> Generation: ...


@dataclass(frozen=True)
class WebQSPSource:
    question_id: str
    parse_id: str
    sparql: str
    topic_entities: list[dict[str, str]]
    answers: list[dict[str, str]]
    inferential_chain: list[str]
    constraints: list[dict[str, Any]]

    def as_context(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "parse_id": self.parse_id,
            "sparql": self.sparql,
            "topic_entities": self.topic_entities,
            "answers": self.answers,
            "inferential_chain": self.inferential_chain,
            "constraints": self.constraints,
        }

    def as_generation_context(self) -> dict[str, Any]:
        context = self.as_context()
        context["no_equal_exclusions"] = self.no_equal_exclusions
        context["equal_constraints"] = self.equal_constraints
        return context

    @property
    def equal_constraints(self) -> list[dict[str, Any]]:
        """Return literal equality constraints in the public operator vocabulary."""
        output: list[dict[str, Any]] = []
        for item in self.constraints:
            if str(item.get("Operator", "")).casefold() != "equal":
                continue
            if str(item.get("ArgumentType", "")).casefold() != "value":
                continue
            relation_label = [
                segment.replace("_", " ")
                for segment in str(item.get("NodePredicate", "")).split(".")
                if segment
            ]
            value = str(item.get("Argument", "")).strip()
            if len(relation_label) < 2 or not value:
                continue
            output.append(
                {
                    "attribute_relation_label": relation_label,
                    "value": value,
                    "value_type": str(item.get("ValueType", "") or "string").casefold(),
                }
            )
        return output

    @property
    def no_equal_exclusions(self) -> list[str]:
        """Return named entities explicitly excluded from the Gold answer variable."""
        select_match = _SELECT_VARIABLE_RE.search(self.sparql)
        if select_match is None:
            return []
        answer_variable = select_match.group("variable").casefold()
        names_by_id = {
            str(item.get("id", "")).strip(): str(item.get("name", "")).strip()
            for item in self.topic_entities
            if str(item.get("id", "")).strip() and str(item.get("name", "")).strip()
        }
        for item in self.constraints:
            entity_id = str(item.get("Argument", "")).strip()
            entity_name = str(item.get("EntityName", "")).strip()
            if entity_id and entity_name:
                names_by_id.setdefault(entity_id, entity_name)
        output: list[str] = []
        for match in _ANSWER_EXCLUSION_RE.finditer(self.sparql):
            if match.group("variable").casefold() != answer_variable:
                continue
            entity_id = match.group("compact") or match.group("uri") or ""
            surface = names_by_id.get(entity_id, "")
            if surface and surface not in output:
                output.append(surface)
        return output


@dataclass(frozen=True)
class CWQSource:
    source_id: str
    question: str
    sparql: str
    answers: list[dict[str, Any]]
    composition_answer: str
    compositionality_type: str
    machine_question: str
    webqsp_id: str

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "question": self.question,
            "sparql": self.sparql,
            "answers": self.answers,
            "composition_answer": self.composition_answer,
            "compositionality_type": self.compositionality_type,
            "machine_question": self.machine_question,
            "webqsp_ID": self.webqsp_id,
        }

    def as_generation_context(self) -> dict[str, Any]:
        context = self.as_context()
        context["no_equal_exclusions"] = self.no_equal_exclusions
        context["equal_constraints"] = [
            {
                "attribute_relation_label": item["attribute_relation_label"],
                "value": item["value"],
                "value_type": item["value_type"],
            }
            for item in self.equal_constraints
        ]
        return context

    def as_decomposition_context(self) -> dict[str, Any]:
        """Return sanitized decomposition evidence without executable KB syntax."""
        return {
            "question": self.question,
            "machine_question": self.machine_question,
            "composition_answer": self.composition_answer,
            "compositionality_type": self.compositionality_type,
            "no_equal_exclusions": self.no_equal_exclusions,
            "equal_constraints": [
                {
                    "attribute_relation_label": item["attribute_relation_label"],
                    "value": item["value"],
                    "value_type": item["value_type"],
                }
                for item in self.equal_constraints
            ],
        }

    @property
    def equal_constraints(self) -> list[dict[str, Any]]:
        """Extract string equality filters and their owning Freebase relation."""
        triples = list(_VARIABLE_RELATION_TRIPLE_RE.finditer(self.sparql))
        output: list[dict[str, Any]] = []
        for match in _STRING_EQUAL_FILTER_RE.finditer(self.sparql):
            attribute = match.group("attribute").casefold()
            owner = next(
                (
                    triple
                    for triple in triples
                    if triple.group("attribute").casefold() == attribute
                ),
                None,
            )
            if owner is None:
                continue
            raw_value = match.group("value")
            try:
                value = json.loads(f'"{raw_value}"')
            except json.JSONDecodeError:
                value = raw_value
            output.append(
                {
                    "attribute_relation_label": [
                        segment.replace("_", " ")
                        for segment in owner.group("relation").split(".")
                    ],
                    "value": str(value),
                    "value_type": "string",
                    "input_sparql_var": owner.group("input"),
                }
            )
        return output

    @property
    def no_equal_exclusions(self) -> list[str]:
        """Return named CWQ entities excluded from the answer variable."""
        select_match = _SELECT_VARIABLE_RE.search(self.sparql)
        if select_match is None or not self.composition_answer:
            return []
        answer_variable = select_match.group("variable").casefold()
        if not any(
            match.group("answer").casefold() == answer_variable
            for match in _VARIABLE_EXCLUSION_RE.finditer(self.sparql)
        ):
            return []
        return [self.composition_answer]

@dataclass(frozen=True)
class SourcePair:
    source_id: str
    index: int
    semantic_index: int
    compose_index: int
    question: str
    semantic_input: dict[str, Any]
    semantic_output: dict[str, Any]
    compose_input: dict[str, Any]
    compose_output: dict[str, Any]
    webqsp_source: WebQSPSource | None = None
    decompose_input: dict[str, Any] | None = None
    decompose_output: dict[str, Any] | None = None
    cwq_source: CWQSource | None = None

    @property
    def compose_expected_operators(self) -> list[dict[str, Any]]:
        output = expected_operators(self.compose_output)
        if self.cwq_source is not None:
            existing_signatures = {operator_signature(item) for item in output}
            select_match = _SELECT_VARIABLE_RE.search(self.cwq_source.sparql)
            answer_sparql_var = select_match.group("variable") if select_match else ""
            for constraint in self.cwq_source.equal_constraints:
                input_var = ""
                for triple in self.compose_output.get("triples", []) or []:
                    if not isinstance(triple, dict):
                        continue
                    if [str(value) for value in triple.get("relation_label", []) or []] != constraint["attribute_relation_label"]:
                        continue
                    literal = _program_literal(str(triple.get("object", "")))
                    if literal == constraint["value"]:
                        input_var = str(triple.get("subject", ""))
                        break
                if (
                    not input_var
                    and constraint.get("input_sparql_var", "").casefold()
                    == answer_sparql_var.casefold()
                ):
                    input_var = str(self.compose_output.get("answer_var", ""))
                operator = {
                    "type": "EQUAL",
                    "inputs": [],
                    "input_var": input_var,
                    "attribute_relation_label": constraint["attribute_relation_label"],
                    "attribute_relation_labels": [],
                    "value": constraint["value"],
                    "value_type": constraint["value_type"],
                }
                if operator_signature(operator) not in existing_signatures:
                    output.append(operator)
                    existing_signatures.add(operator_signature(operator))
            exclusions = self.cwq_source.no_equal_exclusions
            existing = {
                (str(operator.get("type", "")).upper(), str(operator.get("value", "")).casefold())
                for operator in output
            }
            # Constant MID exclusions are normalized by the execution
            # validator. Only variable exclusions need a generated NO_EQUAL
            # operator because the excluded entity must be represented in the
            # candidate graph itself.
            for surface in exclusions:
                key = ("NO_EQUAL", surface.casefold())
                if key in existing:
                    continue
                output.append(
                    {
                        "type": "NO_EQUAL",
                        "inputs": [],
                        "input_var": str(self.compose_output.get("answer_var", "")),
                        "attribute_relation_label": [],
                        "attribute_relation_labels": [],
                        "value": surface,
                        "value_type": "entity",
                    }
                )
                existing.add(key)
        return output

    @property
    def expected_operators(self) -> list[dict[str, Any]]:
        """Return requirements in the semantic path-local variable namespace."""
        mapping_program, _ = repair_compose_entity_surfaces(
            self.compose_output,
            self.semantic_output,
        )
        return project_program_operators_to_semantic(
            self.semantic_output,
            mapping_program,
            self.compose_expected_operators,
        )

    @property
    def answer_path_ids(self) -> list[str]:
        mapping_program, _ = repair_compose_entity_surfaces(
            self.compose_output,
            self.semantic_output,
        )
        answer_variables = set(
            semantic_answer_variables(self.semantic_output, mapping_program)
        )
        return [
            str(path.get("id", ""))
            for path in self.semantic_output.get("semantic_paths", []) or []
            if isinstance(path, dict)
            and str(path.get("path_output_var", "")) in answer_variables
        ]

    @property
    def has_structural_and(self) -> bool:
        return any(
            isinstance(operator, dict) and str(operator.get("type", "")).upper() == "AND"
            for operator in self.semantic_output.get("operators", []) or []
        )

    @property
    def has_semantic_operator(self) -> bool:
        return self.has_structural_and or bool(self.expected_operators)


@dataclass(frozen=True)
class FlywheelConfig:
    output_dir: Path
    split_name: str = "train"
    max_attempts: int = 5
    checkpoint_every: int = 1
    verifier_reasoning_effort: str = "high"
    external_validator_command: str = ""
    external_validator_timeout: float = 120.0
    require_execution_validation: bool = False
    pipeline_profile: str = "standard"
    show_progress: bool = False
    dataset: str = "webqsp"
    workers: int = 1
    preserve_source_decomposition: bool = False
    strict_decomposition_rewrite: bool = False
    skip_source_ids: frozenset[str] = frozenset()
    retain_rejection_trace: bool = False


class _ProgressReporter:
    def __init__(self, *, total: int, enabled: bool, skipped: int = 0) -> None:
        self.total = max(1, total)
        self.enabled = enabled
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.started = time.monotonic()
        self.accepted = 0
        self.rejected = 0
        self.skipped = skipped
        self._last_width = 0
        self._open_line = False
        self._lock = threading.Lock()

    def stage(
        self,
        *,
        position: int,
        attempt: int,
        max_attempts: int,
        stage: str,
        question: str,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._write(
                completed=position - 1,
                position=position,
                status=f"{stage} | attempt {attempt}/{max_attempts}",
                question=question,
                final=False,
            )

    def finish(self, *, position: int, status: str, question: str) -> None:
        with self._lock:
            if status == "accepted":
                self.accepted += 1
            elif status == "rejected":
                self.rejected += 1
            elif status == "skipped":
                self.skipped += 1
            if not self.enabled:
                return
            self._write(
                completed=position,
                position=position,
                status=status,
                question=question,
                final=True,
            )

    def close(self) -> None:
        with self._lock:
            if self.enabled and self.is_tty and self._open_line:
                print(file=sys.stderr, flush=True)
                self._open_line = False

    def _write(
        self,
        *,
        completed: int,
        position: int,
        status: str,
        question: str,
        final: bool,
    ) -> None:
        width = 24
        ratio = min(1.0, max(0.0, completed / self.total))
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started
        eta = "--:--"
        if completed > 0 and completed < self.total:
            eta = _format_duration((elapsed / completed) * (self.total - completed))
        elif completed >= self.total:
            eta = "00:00"
        counts = f"A/R/S {self.accepted}/{self.rejected}/{self.skipped}"
        current = f"row {min(position, self.total)}/{self.total}"
        text = (
            f"[{bar}] {completed}/{self.total} {ratio * 100:5.1f}% | {current} | "
            f"{status} | {counts} | elapsed {_format_duration(elapsed)} eta {eta} | "
            f"{_short_question(question)}"
        )
        if self.is_tty:
            padding = " " * max(0, self._last_width - len(text))
            print(f"\r{text}{padding}", end="\n" if final else "", file=sys.stderr, flush=True)
            self._last_width = 0 if final else len(text)
            self._open_line = not final
        else:
            print(text, file=sys.stderr, flush=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _short_question(question: str, limit: int = 64) -> str:
    compact = " ".join(question.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _align_semantic_graph_goals(
    graph: Any,
    decomposition: list[str],
) -> tuple[Any, list[str]]:
    """Copy path-level KaeDe decomposition text into semantic path goals."""
    if not isinstance(graph, dict):
        return graph, []
    paths = graph.get("semantic_paths")
    if not isinstance(paths, list) or len(paths) != len(decomposition):
        return graph, []
    notes: list[str] = []
    for index, (path, item) in enumerate(zip(paths, decomposition)):
        if not isinstance(path, dict):
            continue
        desired = str(item)
        if path.get("goal") != desired:
            path["goal"] = desired
            notes.append(f"copied decomposition[{index}] into semantic_paths[{index}].goal")
    return graph, notes


def _exact_decomposition_semantic_errors(
    graph: Any,
    decomposition: dict[str, Any] | None,
) -> list[str]:
    """Check the exact, position-preserving decomposition/semantic contract."""
    if decomposition is None:
        return []
    if not isinstance(decomposition, dict):
        return ["final decomposition must be an object"]
    items = decomposition.get("decomposition")
    if not isinstance(items, list):
        return ["final decomposition.decomposition must be an array"]
    if not isinstance(graph, dict):
        return ["final semantic graph must be an object"]
    paths = graph.get("semantic_paths")
    if not isinstance(paths, list):
        return ["final semantic_paths must be an array"]

    errors: list[str] = []
    if len(paths) != len(items):
        errors.append(
            "final semantic_paths count must equal decomposition count "
            f"({len(paths)} != {len(items)})"
        )
    for index, (path, item) in enumerate(zip(paths, items)):
        if not isinstance(path, dict):
            errors.append(f"final semantic_paths[{index}] must be an object")
            continue
        if path.get("goal") != item:
            errors.append(
                f"final semantic_paths[{index}].goal must exactly equal "
                f"decomposition[{index}]"
            )
    return errors


def load_source_pairs(
    semantic_path: Path,
    compose_path: Path,
    webqsp_path: Path | None = None,
    *,
    require_sparql: bool = False,
    split_name: str = "train",
    decompose_path: Path | None = None,
    dataset: str = "webqsp",
) -> list[SourcePair]:
    semantic_rows = _load_rows(semantic_path)
    compose_rows = _load_rows(compose_path)
    decompose_rows = _load_rows(decompose_path) if decompose_path else []
    decompose_by_question: dict[str, deque[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(deque)
    for index, row in enumerate(decompose_rows):
        decompose_input = _row_json(row, "input", decompose_path, index)
        decompose_output = _row_json(row, "output", decompose_path, index)
        question = str(decompose_input.get("question", "")).strip()
        if not question:
            raise ValueError(f"{decompose_path}: row {index} has no question")
        decompose_by_question[_normalize_question(question)].append(
            (index, decompose_input, decompose_output)
        )
    compose_by_question: dict[str, deque[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]]] = (
        defaultdict(deque)
    )
    for index, row in enumerate(compose_rows):
        compose_input = _row_json(row, "input", compose_path, index)
        compose_output = _row_json(row, "output", compose_path, index)
        question = str(compose_input.get("question", "")).strip()
        if not question:
            raise ValueError(f"{compose_path}: row {index} has no question")
        compose_by_question[question].append((index, row, compose_input, compose_output))

    webqsp_sources = (
        _load_webqsp_sources(webqsp_path, split_name=split_name, require_sparql=require_sparql)
        if webqsp_path and dataset == "webqsp" else {}
    )
    cwq_sources = (
        _load_cwq_sources(webqsp_path, split_name=split_name, require_sparql=require_sparql)
        if webqsp_path and dataset == "cwq" else {}
    )
    pairs: list[SourcePair] = []
    for semantic_index, semantic_row in enumerate(semantic_rows):
        semantic_input = _row_json(semantic_row, "input", semantic_path, semantic_index)
        semantic_output = _row_json(semantic_row, "output", semantic_path, semantic_index)
        question = str(semantic_input.get("question", "")).strip()
        if not question:
            raise ValueError(f"{semantic_path}: row {semantic_index} has no question")
        if not compose_by_question[question]:
            raise ValueError(
                f"no compose row matches semantic row {semantic_index} question {question!r}"
            )
        compose_index, _, compose_input, compose_output = compose_by_question[question].popleft()
        decompose_input = None
        decompose_output = None
        if decompose_path:
            queue = decompose_by_question.get(_normalize_question(question), deque())
            if not queue:
                raise ValueError(f"no decompose row matches question {question!r}")
            _, decompose_input, decompose_output = queue.popleft()
        webqsp_source = None
        cwq_source = None
        if webqsp_path:
            source_map = webqsp_sources if dataset == "webqsp" else cwq_sources
            candidates = source_map.get(_normalize_question(question), [])
            if candidates:
                source = candidates.pop(0)
                if dataset == "webqsp":
                    webqsp_source = source
                else:
                    cwq_source = source
            elif require_sparql:
                raise ValueError(f"{webqsp_path}: no {dataset.upper()} source row matches question {question!r}")
        source_sparql = (
            webqsp_source.sparql if webqsp_source else cwq_source.sparql if cwq_source else ""
        )
        if require_sparql and not source_sparql.strip():
            raise ValueError(
                f"{webqsp_path}: question {question!r} has no non-empty source SPARQL"
            )
        identity = {
            "semantic_input": semantic_input,
            "semantic_output": semantic_output,
            "compose_output": compose_output,
            "webqsp_source": webqsp_source.as_context() if webqsp_source else None,
            "decompose_output": decompose_output,
            "cwq_source": cwq_source.as_context() if cwq_source else None,
        }
        digest = hashlib.sha256(_json_text(identity).encode("utf-8")).hexdigest()[:20]
        pairs.append(
            SourcePair(
                source_id=digest,
                index=len(pairs),
                semantic_index=semantic_index,
                compose_index=compose_index,
                question=question,
                semantic_input=semantic_input,
                semantic_output=semantic_output,
                compose_input=compose_input,
                compose_output=compose_output,
                webqsp_source=webqsp_source,
                decompose_input=decompose_input,
                decompose_output=decompose_output,
                cwq_source=cwq_source,
            )
        )

    leftovers = sum(len(queue) for queue in compose_by_question.values())
    if leftovers:
        raise ValueError(f"{compose_path} contains {leftovers} unmatched compose rows")
    if decompose_path:
        leftovers = sum(len(queue) for queue in decompose_by_question.values())
        if leftovers:
            raise ValueError(f"{decompose_path} contains {leftovers} unmatched decompose rows")
    return pairs


def resolve_webqsp_path(path: Path, *, split_name: str = "train") -> Path:
    """Resolve either a WebQSP JSON file or its split directory."""
    candidate = path.expanduser()
    if candidate.is_dir():
        for name in (f"WebQSP.{split_name}.json", f"webqsp_{split_name}.json", f"{split_name}.json"):
            split_path = candidate / name
            if split_path.exists():
                return split_path.resolve()
        raise FileNotFoundError(f"could not find WebQSP {split_name} JSON under {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"WebQSP source does not exist: {candidate}")
    return candidate.resolve()


def resolve_cwq_path(path: Path, *, split_name: str = "train") -> Path:
    """Resolve a CWQ JSON file or the directory containing its official split files."""
    candidate = path.expanduser()
    if candidate.is_dir():
        names = (
            f"ComplexWebQuestions_{split_name}.json",
            f"CWQ_{split_name}.json",
            f"{split_name}.json",
        )
        for name in names:
            split_path = candidate / name
            if split_path.exists():
                return split_path.resolve()
        raise FileNotFoundError(f"could not find CWQ {split_name} JSON under {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"CWQ source does not exist: {candidate}")
    return candidate.resolve()


def _load_webqsp_sources(
    path: Path | None,
    *,
    split_name: str = "train",
    require_sparql: bool = False,
) -> dict[str, list[WebQSPSource]]:
    if path is None:
        return {}
    source_path = resolve_webqsp_path(path, split_name=split_name)
    if require_sparql:
        _raise_if_known_blank_sparql_source(source_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = payload.get("Questions", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{source_path}: expected a Questions array or a JSON array")
    by_question: dict[str, list[WebQSPSource]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question_id = str(
            row.get("QuestionId") or row.get("question_id") or row.get("qid") or ""
        ).strip()
        question = str(
            row.get("ProcessedQuestion") or row.get("RawQuestion") or row.get("question") or ""
        ).strip()
        parses = [item for item in row.get("Parses", []) or [] if isinstance(item, dict)]
        parse_index, parse = next(
            ((index, item) for index, item in enumerate(parses) if str(item.get("Sparql", "")).strip()),
            (0, parses[0] if parses else {}),
        )
        sparql = str(parse.get("Sparql") or row.get("Sparql") or row.get("sparql_query") or "").strip()
        parse_id = str(parse.get("ParseId") or f"{question_id}.P{parse_index}").strip()
        topic_entities = _webqsp_topic_entities(row, parses)
        answers = _webqsp_answers(row, parses)
        inferential_chain = [str(value).strip() for value in parse.get("InferentialChain", []) or [] if str(value).strip()]
        constraints = [value for value in parse.get("Constraints", []) or [] if isinstance(value, dict)]
        source = WebQSPSource(
            question_id=question_id or f"row-{row_index}",
            parse_id=parse_id,
            sparql=sparql,
            topic_entities=topic_entities,
            answers=answers,
            inferential_chain=inferential_chain,
            constraints=constraints,
        )
        if question:
            by_question[_normalize_question(question)].append(source)
    return by_question


def _load_cwq_sources(
    path: Path | None,
    *,
    split_name: str = "train",
    require_sparql: bool = False,
) -> dict[str, list[CWQSource]]:
    if path is None:
        return {}
    source_path = resolve_cwq_path(path, split_name=split_name)
    source_paths = [source_path]
    # CWQ_initial's dev/test subsets are derived from the WebQSP questions and do
    # not always use the official CWQ split filename. Fall back to sibling official
    # splits while preserving the requested split's rows and ordering first.
    if source_path.name.startswith("ComplexWebQuestions_"):
        for sibling_split in ("train", "dev", "test"):
            sibling = source_path.parent / f"ComplexWebQuestions_{sibling_split}.json"
            if sibling.exists() and sibling.resolve() not in {item.resolve() for item in source_paths}:
                source_paths.append(sibling.resolve())
    by_question: dict[str, list[CWQSource]] = defaultdict(list)
    for candidate_path in source_paths:
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        rows = payload.get("Questions", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"{candidate_path}: expected a JSON array")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            question = str(row.get("question") or row.get("Question") or "").strip()
            if not question:
                continue
            sparql = str(row.get("sparql") or row.get("Sparql") or "").strip()
            if require_sparql and not sparql:
                raise ValueError(f"{candidate_path}: row {index} has empty sparql")
            source = CWQSource(
                source_id=str(row.get("ID") or row.get("id") or f"cwq-{index}").strip(),
                question=question,
                sparql=sparql,
                answers=[item for item in row.get("answers", []) or [] if isinstance(item, dict)],
                composition_answer=str(row.get("composition_answer") or "").strip(),
                compositionality_type=str(row.get("compositionality_type") or "").strip(),
                machine_question=str(row.get("machine_question") or "").strip(),
                webqsp_id=str(row.get("webqsp_ID") or "").strip(),
            )
            by_question[_normalize_question(question)].append(source)
    return by_question


def _webqsp_topic_entities(row: dict[str, Any], parses: list[dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for parse in parses:
        mid = str(parse.get("TopicEntityMid", "")).strip()
        name = str(parse.get("TopicEntityName", "") or mid).strip()
        if not mid and not name:
            continue
        key = (mid, name)
        if key not in seen:
            seen.add(key)
            output.append({"id": mid, "name": name})
    if not output:
        for item in row.get("topic_entities", []) or []:
            if isinstance(item, dict):
                mid = str(item.get("id") or item.get("kb_id") or "").strip()
                name = str(item.get("name") or item.get("label") or mid).strip()
                if mid or name:
                    output.append({"id": mid, "name": name})
    return output


def _webqsp_answers(row: dict[str, Any], parses: list[dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    raw_answers: list[Any] = list(row.get("Answers", []) or row.get("answers", []) or [])
    for parse in parses:
        raw_answers.extend(parse.get("Answers", []) or [])
    for answer in raw_answers:
        if not isinstance(answer, dict):
            continue
        answer_id = str(answer.get("AnswerArgument") or answer.get("answer_argument") or answer.get("id") or "").strip()
        name = str(answer.get("EntityName") or answer.get("entity_name") or answer.get("label") or answer_id).strip()
        if not answer_id and not name:
            continue
        key = (answer_id, name)
        if key not in seen:
            seen.add(key)
            output.append({"id": answer_id, "name": name, "type": str(answer.get("AnswerType") or answer.get("answer_type") or "")})
    return output


def _normalize_question(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", str(value).casefold()))


def _raise_if_known_blank_sparql_source(source_path: Path) -> None:
    metadata_path = source_path.parent / "conversion_metadata.json"
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    source = str(metadata.get("source", "")).casefold() if isinstance(metadata, dict) else ""
    if "ml1996/webqsp parquet" in source:
        raise ValueError(
            f"{source_path}: conversion_metadata shows this is the OpenDataLab/HF "
            "parquet conversion, whose Sparql fields are empty. Use the original "
            "WebQSP JSON with non-empty Sparql fields."
        )


def source_summary(pairs: list[SourcePair]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    operator_rows = 0
    executable_operator_rows = 0
    webqsp_rows = 0
    cwq_rows = 0
    decompose_rows = 0
    for pair in pairs:
        requirements = pair.expected_operators
        if pair.has_semantic_operator:
            operator_rows += 1
        if requirements:
            executable_operator_rows += 1
        if pair.webqsp_source is not None and pair.webqsp_source.sparql.strip():
            webqsp_rows += 1
        if pair.cwq_source is not None and pair.cwq_source.sparql.strip():
            cwq_rows += 1
        if pair.decompose_output is not None:
            decompose_rows += 1
        if pair.has_structural_and:
            counts["AND"] += 1
        counts.update(str(operator.get("type", "")) for operator in requirements)
    return {
        "pairs": len(pairs),
        "rows_with_operators": operator_rows,
        "rows_with_executable_operators": executable_operator_rows,
        "rows_with_webqsp_sparql": webqsp_rows,
        "rows_with_cwq_sparql": cwq_rows,
        "rows_with_decomposition": decompose_rows,
        "operator_counts": dict(sorted(counts.items())),
    }


def _retry_stage_for_compose_errors(errors: list[str]) -> str:
    """Choose the earliest stage that can repair a Compose failure."""
    text = " ".join(errors).casefold()
    if any(
        marker in text
        for marker in (
            "semantic path",
            "semantic graph",
            "path-to-program",
            "operator variables",
            "and lowering",
        )
    ):
        return "semantic"
    return "compose"


class Flywheel:
    def __init__(
        self,
        *,
        client: JsonGenerator,
        config: FlywheelConfig,
        semantic_source: Path,
        compose_source: Path,
        webqsp_source: Path | None = None,
        decompose_source: Path | None = None,
    ) -> None:
        if config.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if config.workers < 1:
            raise ValueError("workers must be at least 1")
        if config.pipeline_profile not in {"standard", "lean", "micro"}:
            raise ValueError("pipeline_profile must be standard, lean, or micro")
        if config.dataset.casefold() not in {"webqsp", "cwq"}:
            raise ValueError("dataset must be webqsp or cwq")
        self.client = client
        self.config = config
        self.semantic_source = semantic_source.resolve()
        self.compose_source = compose_source.resolve()
        self.dataset = config.dataset.casefold()
        if config.preserve_source_decomposition and self.dataset != "cwq":
            raise ValueError("preserve_source_decomposition is only supported for CWQ")
        if config.strict_decomposition_rewrite and self.dataset != "cwq":
            raise ValueError("strict_decomposition_rewrite is only supported for CWQ")
        if config.strict_decomposition_rewrite and config.preserve_source_decomposition:
            raise ValueError(
                "strict_decomposition_rewrite cannot be combined with "
                "preserve_source_decomposition"
            )
        if config.preserve_source_decomposition and decompose_source is None:
            raise ValueError(
                "preserve_source_decomposition=True requires a KaeDe decomposition source"
            )
        if config.strict_decomposition_rewrite and decompose_source is None:
            raise ValueError(
                "strict_decomposition_rewrite=True requires a CWQ decomposition source"
            )
        self.webqsp_source = (
            (resolve_cwq_path if self.dataset == "cwq" else resolve_webqsp_path)(
                webqsp_source, split_name=config.split_name
            )
            if webqsp_source is not None else None
        )
        self.decompose_source = decompose_source.resolve() if decompose_source else None
        self.output_dir = config.output_dir.resolve()
        if config.require_execution_validation and not config.external_validator_command.strip():
            raise ValueError(
                "require_execution_validation=True needs external_validator_command; "
                "use scripts/cwq_execution_validator.py or provide your validator"
            )
        self.accepted_path = self.output_dir / "accepted.jsonl"
        self.rejected_path = self.output_dir / "rejected.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_manifest()

    def _source_decomposition_fallback(
        self,
        pair: SourcePair,
    ) -> dict[str, Any] | None:
        candidate = pair.decompose_output
        if not isinstance(candidate, dict):
            return None
        errors = validate_decomposition(
            candidate,
            question=pair.question,
            strict_natural_language=self.config.strict_decomposition_rewrite,
        )
        errors.extend(self._strict_rewrite_candidate_errors(pair, candidate))
        if errors:
            return None
        return {"decomposition": list(candidate.get("decomposition", []) or [])}

    def _enforce_decomposition_alignment(self) -> bool:
        """Whether decomposition is frozen as the semantic path contract."""
        return bool(
            self.config.preserve_source_decomposition
            or self.config.strict_decomposition_rewrite
        )

    def _decomposition_source_context(self, pair: SourcePair) -> dict[str, Any] | None:
        if pair.cwq_source is None:
            return None
        if not self.config.strict_decomposition_rewrite:
            return pair.cwq_source.as_generation_context()
        return pair.cwq_source.as_decomposition_context()

    def _decomposition_requirements(self, pair: SourcePair) -> dict[str, Any]:
        if not self.config.strict_decomposition_rewrite:
            return {}
        anchor_surfaces = [
            str(anchor.get("surface", "")).strip()
            for anchor in pair.semantic_output.get("anchors", []) or []
            if isinstance(anchor, dict) and str(anchor.get("surface", "")).strip()
        ]
        operators = []
        for raw in pair.expected_operators:
            if not isinstance(raw, dict):
                continue
            input_var = str(raw.get("input_var", ""))
            path_match = re.fullmatch(r"P(?P<index>\d+)\.V\d+", input_var)
            path_index = int(path_match.group("index")) if path_match else None
            source_paths = pair.semantic_output.get("semantic_paths", []) or []
            if path_index is None and len(source_paths) == 1:
                path_index = 0
            operators.append(
                {
                    "type": str(raw.get("type", "")),
                    "path_index": path_index,
                    "attribute_relation_label": list(
                        raw.get("attribute_relation_label", []) or []
                    ),
                    "attribute_relation_labels": list(
                        raw.get("attribute_relation_labels", []) or []
                    ),
                    "value": str(raw.get("value", "")),
                    "value_type": str(raw.get("value_type", "")),
                }
            )
        return {
            "anchor_surfaces": anchor_surfaces,
            "source_path_count": len(
                (pair.decompose_output or {}).get("decomposition", []) or []
            ),
            "required_operators": operators,
        }

    def _strict_rewrite_candidate_errors(
        self,
        pair: SourcePair,
        candidate: dict[str, Any],
    ) -> list[str]:
        """Validate source path order and path-local constraint attachment."""
        if not self.config.strict_decomposition_rewrite:
            return []
        items = candidate.get("decomposition")
        source = pair.decompose_output if isinstance(pair.decompose_output, dict) else {}
        source_items = source.get("decomposition")
        if not isinstance(items, list) or not isinstance(source_items, list):
            return []

        errors: list[str] = []
        if len(items) != len(source_items):
            errors.append(
                "strict decomposition must preserve source path count "
                f"({len(items)} != {len(source_items)})"
            )
            return errors

        anchors_by_id = {
            str(anchor.get("id", "")): str(anchor.get("surface", "")).strip()
            for anchor in pair.semantic_output.get("anchors", []) or []
            if isinstance(anchor, dict)
        }
        source_paths = pair.semantic_output.get("semantic_paths", []) or []
        for index, (source_item, candidate_item) in enumerate(zip(source_items, items)):
            if index >= len(source_paths) or not isinstance(source_paths[index], dict):
                continue
            surface = anchors_by_id.get(
                str(source_paths[index].get("anchor_ref", "")),
                "",
            )
            source_text = " ".join(str(source_item).split()).casefold()
            candidate_text = " ".join(str(candidate_item).split()).casefold()
            if surface and surface.casefold() in source_text and surface.casefold() not in candidate_text:
                errors.append(
                    f"decomposition[{index}] must preserve source anchor surface {surface!r}"
                )

        greater = re.compile(r"\b(?:greater|more|higher|larger|above|over|exceed)\w*\b", re.I)
        less = re.compile(r"\b(?:less|fewer|lower|smaller|below|under)\w*\b", re.I)
        maximum = re.compile(r"\b(?:most|highest|largest|greatest|latest|newest|maximum)\b", re.I)
        minimum = re.compile(r"\b(?:least|lowest|smallest|earliest|oldest|first|minimum)\b", re.I)
        current = re.compile(r"\b(?:current|currently|now|present|ongoing|incumbent)\b", re.I)
        for requirement in self._decomposition_requirements(pair).get(
            "required_operators", []
        ):
            path_index = requirement.get("path_index")
            if not isinstance(path_index, int) or not (0 <= path_index < len(items)):
                continue
            item_text = str(items[path_index])
            normalized_item = item_text.casefold().replace(",", "").replace("，", "")
            operator_type = str(requirement.get("type", "")).upper()
            value = str(requirement.get("value", "")).strip()
            normalized_value = value.casefold().replace(",", "").replace("，", "")
            if normalized_value and (
                re.search(r"\d", normalized_value) or operator_type == "TC"
            ):
                value_present = normalized_value in normalized_item
                if operator_type == "TC" and normalized_value == "now":
                    value_present = bool(current.search(item_text))
                if not value_present:
                    errors.append(
                        f"decomposition[{path_index}] is missing {operator_type} "
                        f"constraint value {value!r}"
                    )
            cue_pattern = {
                "GREATER_THAN": greater,
                "GREATER_OR_EQUAL": greater,
                "LESS_THAN": less,
                "LESS_OR_EQUAL": less,
                "ARGMAX": maximum,
                "ARGMIN": minimum,
                "TC": current if normalized_value == "now" else None,
            }.get(operator_type)
            if cue_pattern is not None and not cue_pattern.search(item_text):
                errors.append(
                    f"decomposition[{path_index}] is missing {operator_type} constraint cue"
                )
        return errors

    def _decomposition_errors_can_be_verified(self, errors: list[str]) -> bool:
        """Whether a natural candidate may be sent to the repair verifier."""
        if not self.config.strict_decomposition_rewrite or not errors:
            return False
        repairable_markers = (
            "decomposition is missing ",
            " is missing ",
            "strict decomposition must preserve source path count",
            "must preserve source anchor surface",
        )
        return all(any(marker in error for marker in repairable_markers) for error in errors)

    def _semantic_decomposition(
        self,
        decomposition: dict[str, Any] | None,
    ) -> list[str] | None:
        if not self._enforce_decomposition_alignment():
            return None
        if not isinstance(decomposition, dict):
            return []
        items = decomposition.get("decomposition", [])
        return list(items) if isinstance(items, list) else []

    def _validate_semantic_graph(
        self,
        graph: Any,
        *,
        expected: list[dict[str, Any]],
        decomposition: dict[str, Any] | None,
        source_graph: dict[str, Any] | None = None,
    ) -> list[str]:
        return validate_semantic_graph(
            graph,
            expected=expected,
            decomposition=self._semantic_decomposition(decomposition),
            source_graph=(
                source_graph
                if self.config.preserve_source_decomposition
                or self.config.strict_decomposition_rewrite
                else None
            ),
        )

    def _validated_source_fallback(
        self,
        pair: SourcePair,
        rejected: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Accept a deterministic source lowering only after execution equivalence."""
        if self.config.strict_decomposition_rewrite:
            # Strict rewrite must never silently replace a rejected generated
            # decomposition with the old source decomposition/semantic pair.
            return None
        if not self.config.require_execution_validation:
            return None
        semantic_requirements = pair.expected_operators
        compose_requirements = pair.compose_expected_operators
        reference_compose, surface_notes = repair_compose_entity_surfaces(
            pair.compose_output,
            pair.semantic_output,
        )
        exclusions = (
            pair.cwq_source.no_equal_exclusions
            if pair.cwq_source is not None
            else []
        )
        semantic, operator_notes = repair_operator_fidelity(
            pair.semantic_output,
            semantic_requirements,
            location="semantic_graph",
        )
        semantic, _, exclusion_notes = repair_answer_exclusion_scopes(
            semantic,
            reference_compose,
            exclusions,
            source_graph=pair.semantic_output,
            source_program=reference_compose,
            repair_program=False,
        )
        if self.config.preserve_source_decomposition:
            and_notes = []
        else:
            semantic, and_notes = repair_superfluous_semantic_and(
                semantic,
                reference_compose,
            )
        compose = compile_compose_program(
            semantic,
            answer_path_ids=pair.answer_path_ids,
            source_graph=pair.semantic_output,
            source_program=reference_compose,
        )
        compose, compose_operator_notes = repair_operator_fidelity(
            compose,
            compose_requirements,
            location="compose_program",
        )
        semantic, compose, compose_exclusion_notes = repair_answer_exclusion_scopes(
            semantic,
            compose,
            exclusions,
            source_graph=pair.semantic_output,
            source_program=reference_compose,
        )
        if self._enforce_decomposition_alignment():
            # This fallback lowers the source semantic graph, so it must use the
            # source decomposition too. Mixing a generated decomposition with
            # source semantics can silently materialize contradictory SFT rows.
            decomposition = self._source_decomposition_fallback(pair)
        else:
            decomposition = rejected.get("last_decomposition_candidate")
            if not isinstance(decomposition, dict) or validate_decomposition(
                decomposition,
                question=pair.question,
                strict_natural_language=self.config.strict_decomposition_rewrite,
            ):
                decomposition = self._source_decomposition_fallback(pair)
        goal_repairs: list[str] = []
        if self.config.preserve_source_decomposition and decomposition is not None:
            semantic, goal_repairs = _align_semantic_graph_goals(
                semantic,
                self._semantic_decomposition(decomposition) or [],
            )
        errors = self._validate_semantic_graph(
            semantic,
            expected=semantic_requirements,
            decomposition=decomposition,
            source_graph=pair.semantic_output,
        )
        errors.extend(
            validate_compose_program(
                compose,
                semantic_graph=semantic,
                expected=compose_requirements,
                source_graph=pair.semantic_output,
                source_program=reference_compose,
            )
        )
        if not errors:
            errors.extend(self._external_errors(pair, semantic, compose, 0))
        fallback_trace = {
            "stage": "validated_source_fallback",
            "static_and_execution_errors": errors,
            "repairs": [
                *surface_notes,
                *operator_notes,
                *exclusion_notes,
                *and_notes,
                *compose_operator_notes,
                *compose_exclusion_notes,
                *goal_repairs,
            ],
        }
        if errors:
            rejected.setdefault("trace", []).append(fallback_trace)
            return None
        trace = [*list(rejected.get("trace", []) or []), fallback_trace]
        return self._accepted_record(
            pair,
            max(1, int(rejected.get("attempts", 0) or 0)),
            semantic,
            compose,
            ["accepted deterministic source fallback after execution equivalence"],
            trace,
            dict(rejected.get("usage", {}) or {}),
            decomposition=decomposition,
        )

    def run(self, pairs: list[SourcePair]) -> dict[str, int]:
        accepted_ids = set(_accepted_records(self.accepted_path))
        accepted_now = 0
        rejected_now = 0
        processed_now = 0
        pending_pairs = [
            pair
            for pair in pairs
            if pair.source_id not in accepted_ids
            and pair.source_id not in self.config.skip_source_ids
        ]
        skipped = len(pairs) - len(pending_pairs)
        pending = list(enumerate(pending_pairs, start=1))
        progress = _ProgressReporter(
            total=len(pending),
            enabled=self.config.show_progress,
            skipped=skipped,
        )
        try:
            if self.config.workers == 1:
                results = (
                    self._run_pair(pair, progress=progress, position=position)
                    for position, pair in pending
                )
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                executor = ThreadPoolExecutor(max_workers=self.config.workers)
                futures = {
                    executor.submit(
                        self._run_pair,
                        pair,
                        progress=progress,
                        position=position,
                    ): (position, pair)
                    for position, pair in pending
                }

                def completed_results():
                    try:
                        for future in as_completed(futures):
                            yield future.result()
                    finally:
                        executor.shutdown(wait=True)

                results = completed_results()

            for position, pair, record in results:
                if record["status"] == "accepted":
                    _append_jsonl(self.accepted_path, record)
                    accepted_ids.add(pair.source_id)
                    accepted_now += 1
                    progress.finish(position=position, status="accepted", question=pair.question)
                else:
                    _append_jsonl(
                        self.rejected_path,
                        record
                        if self.config.retain_rejection_trace
                        else _compact_rejected_record(record),
                    )
                    rejected_now += 1
                    progress.finish(position=position, status="rejected", question=pair.question)
                processed_now += 1
                if processed_now % max(1, self.config.checkpoint_every) == 0:
                    materialize(self.output_dir, split_name=self.config.split_name)
        finally:
            progress.close()
            materialize(self.output_dir, split_name=self.config.split_name)
        return {"accepted": accepted_now, "rejected": rejected_now, "skipped": skipped}

    def _run_pair(
        self,
        pair: SourcePair,
        *,
        progress: _ProgressReporter,
        position: int,
    ) -> tuple[int, SourcePair, dict[str, Any]]:
        try:
            record = self._process_pair(
                pair,
                progress=progress,
                position=position,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one failed model row
            # Transport failures and malformed model JSON remain auditable;
            # the next invocation retries this source_id.
            record = self._exception_record(pair, exc)
        if (
            record.get("status") == "rejected"
            and not _source_sparql_has_no_answers(list(record.get("errors", []) or []))
        ):
            fallback = self._validated_source_fallback(pair, record)
            if fallback is not None:
                record = fallback
        return position, pair, record

    def _exception_record(self, pair: SourcePair, exc: Exception) -> dict[str, Any]:
        return {
            "status": "rejected",
            "source_id": pair.source_id,
            "source_index": pair.index,
            "semantic_source_index": pair.semantic_index,
            "compose_source_index": pair.compose_index,
            "question": pair.question,
            "dataset": self.dataset,
            "attempts": 0,
            "errors": [f"generation exception: {type(exc).__name__}: {exc}"],
            "trace": [{"stage": "generation", "exception_type": type(exc).__name__}],
            "usage": {},
            "webqsp_source": pair.webqsp_source.as_context() if pair.webqsp_source else None,
            "cwq_source": pair.cwq_source.as_context() if pair.cwq_source else None,
            "created_at": _utc_now(),
        }

    def _process_pair(
        self,
        pair: SourcePair,
        *,
        progress: _ProgressReporter,
        position: int,
    ) -> dict[str, Any]:
        lean = self.config.pipeline_profile == "lean"
        micro = self.config.pipeline_profile == "micro"
        constrained = lean or micro
        semantic_requirements = pair.expected_operators
        compose_requirements = pair.compose_expected_operators
        reference_compose, _ = repair_compose_entity_surfaces(
            pair.compose_output,
            pair.semantic_output,
        )
        source_semantic_contract = pair.semantic_output
        source_contract_repairs: list[str] = []
        if self.config.strict_decomposition_rewrite:
            source_semantic_contract, source_contract_repairs = (
                repair_superfluous_semantic_and(
                    source_semantic_contract,
                    reference_compose,
                )
            )
        answer_exclusions = (
            pair.cwq_source.no_equal_exclusions
            if pair.cwq_source is not None
            else []
        )
        feedback: list[str] = []
        trace: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        last_semantic: dict[str, Any] = {}
        last_compose: dict[str, Any] = {}
        last_decomposition: dict[str, Any] = {
            "decomposition": list(
                (pair.decompose_output or {}).get(
                    "decomposition",
                    pair.semantic_input.get("decomposition", []),
                )
                or []
            )
        }
        if self.config.strict_decomposition_rewrite:
            source_decomposition_errors = validate_decomposition(
                last_decomposition,
                strict_natural_language=True,
            )
            if source_decomposition_errors:
                feedback = [
                    "source decomposition was withheld from the model because it failed "
                    "the strict natural-language gate",
                    *source_decomposition_errors,
                ]
                last_decomposition = {"decomposition": []}
        has_decomposition = self.dataset == "cwq" or pair.decompose_output is not None
        retry_stage = (
            "semantic"
            if self.config.preserve_source_decomposition
            else ("decompose" if has_decomposition else "semantic")
        )

        for attempt in range(1, self.config.max_attempts + 1):
            attempt_trace: dict[str, Any] = {
                "attempt": attempt,
                "retry_from_stage": retry_stage,
            }
            if has_decomposition and retry_stage == "decompose":
                progress.stage(
                    position=position,
                    attempt=attempt,
                    max_attempts=self.config.max_attempts,
                    stage="decompose 1/5: waiting for model",
                    question=pair.question,
                )
                decompose_instructions = DECOMPOSE_INSTRUCTIONS
                decompose_gateway_instructions = CWQ_GATEWAY_DECOMPOSE_INSTRUCTIONS
                if self.config.strict_decomposition_rewrite:
                    decompose_instructions += (
                        "\n" + STRICT_DECOMPOSITION_REWRITE_INSTRUCTIONS
                    )
                    decompose_gateway_instructions += (
                        "\n" + STRICT_DECOMPOSITION_REWRITE_INSTRUCTIONS
                    )
                decompose_generation = self.client.generate_json(
                    schema_name="rewritten_decomposition",
                    schema=DECOMPOSITION_SCHEMA,
                    instructions=decompose_instructions,
                    payload={
                        "question": pair.question,
                        "source_decomposition": last_decomposition,
                        "previous_decomposition_candidate": last_decomposition,
                        "validation_feedback": feedback[-2:] if micro else (feedback[-4:] if lean else feedback[-12:]),
                        "decomposition_requirements": self._decomposition_requirements(pair),
                        "cwq_source": self._decomposition_source_context(pair),
                    },
                    gateway_instructions=decompose_gateway_instructions,
                    output_tokens=384 if constrained else 768,
                )
                _merge_usage(usage, decompose_generation.usage)
                decompose_candidate = decompose_generation.value
                decompose_errors = validate_decomposition(
                    decompose_candidate,
                    question=pair.question,
                    strict_natural_language=self.config.strict_decomposition_rewrite,
                )
                if isinstance(decompose_candidate, dict):
                    decompose_errors.extend(
                        self._strict_rewrite_candidate_errors(pair, decompose_candidate)
                    )
                decompose_trace: dict[str, Any] = {
                    "attempt": attempt,
                    "retry_from_stage": "decompose",
                    "decompose_response_id": decompose_generation.response_id,
                    "decompose_transport_mode": decompose_generation.transport_mode,
                    "decompose_errors": decompose_errors,
                }
                if decompose_errors:
                    feedback = decompose_errors
                    if not self._decomposition_errors_can_be_verified(decompose_errors):
                        retry_stage = "decompose"
                        trace.append(decompose_trace)
                        continue
                decompose_verify_instructions = DECOMPOSE_VERIFY_INSTRUCTIONS
                decompose_verify_gateway_instructions = (
                    CWQ_GATEWAY_DECOMPOSE_VERIFY_INSTRUCTIONS
                )
                if self.config.strict_decomposition_rewrite:
                    decompose_verify_instructions += (
                        "\n" + STRICT_DECOMPOSITION_REWRITE_INSTRUCTIONS
                    )
                    decompose_verify_gateway_instructions += (
                        "\n" + STRICT_DECOMPOSITION_REWRITE_INSTRUCTIONS
                    )
                decompose_verification = self.client.generate_json(
                    schema_name=(
                        "decomposition_verification_light"
                        if constrained else "decomposition_verification"
                    ),
                    schema=(
                        LIGHT_DECOMPOSITION_VERIFICATION_SCHEMA
                        if constrained else DECOMPOSITION_VERIFICATION_SCHEMA
                    ),
                    instructions=decompose_verify_instructions,
                    payload={
                        "question": pair.question,
                        "source_decomposition": last_decomposition,
                        "candidate_decomposition": decompose_candidate,
                        "previous_decomposition_candidate": last_decomposition,
                        "validation_feedback": feedback[-2:] if micro else (feedback[-4:] if lean else feedback[-12:]),
                        "decomposition_requirements": self._decomposition_requirements(pair),
                        "cwq_source": self._decomposition_source_context(pair),
                    },
                    gateway_instructions=decompose_verify_gateway_instructions,
                    output_tokens=96 if constrained else 384,
                    reasoning_effort=self.config.verifier_reasoning_effort,
                )
                _merge_usage(usage, decompose_verification.usage)
                decompose_verdict = decompose_verification.value
                decompose_status = str(decompose_verdict.get("status", "reject"))
                decompose_issues = [
                    str(value) for value in decompose_verdict.get("issues", []) or []
                ]
                decompose_trace.update(
                    {
                        "decompose_verify_response_id": decompose_verification.response_id,
                        "decompose_verify_transport_mode": decompose_verification.transport_mode,
                        "decompose_verify_status": decompose_status,
                        "decompose_verify_issues": decompose_issues,
                    }
                )
                if decompose_status == "corrected" and not constrained:
                    corrected = decompose_verdict.get("decomposition")
                    corrected_errors = validate_decomposition(
                        {"decomposition": corrected},
                        question=pair.question,
                        strict_natural_language=self.config.strict_decomposition_rewrite,
                    )
                    corrected_candidate = {"decomposition": corrected}
                    corrected_errors.extend(
                        self._strict_rewrite_candidate_errors(pair, corrected_candidate)
                    )
                    decompose_trace["decompose_post_verify_errors"] = corrected_errors
                    if not corrected_errors:
                        decompose_candidate = {"decomposition": corrected}
                    elif self.config.strict_decomposition_rewrite:
                        decompose_trace["invalid_verifier_correction_ignored"] = (
                            "kept the original candidate that passed deterministic strict gates"
                        )
                    else:
                        feedback = [*decompose_issues, *corrected_errors]
                        retry_stage = "decompose"
                        trace.append(decompose_trace)
                        continue
                elif decompose_status == "pass" and not constrained:
                    verified = decompose_verdict.get("decomposition")
                    if verified != decompose_candidate.get("decomposition"):
                        verified_candidate = {"decomposition": verified}
                        verified_errors = validate_decomposition(
                            verified_candidate,
                            question=pair.question,
                            strict_natural_language=self.config.strict_decomposition_rewrite,
                        )
                        verified_errors.extend(
                            self._strict_rewrite_candidate_errors(pair, verified_candidate)
                        )
                        if not verified_errors:
                            decompose_candidate = verified_candidate
                            decompose_trace["decompose_pass_normalized"] = (
                                "accepted verifier-returned decomposition correction"
                            )
                        elif self.config.strict_decomposition_rewrite:
                            decompose_trace["invalid_verifier_correction_ignored"] = (
                                "kept the original candidate that passed deterministic strict gates"
                            )
                        else:
                            feedback = [
                                "decomposition verifier changed a candidate while claiming status=pass",
                                *verified_errors,
                            ]
                            decompose_trace["decompose_post_verify_errors"] = feedback
                            retry_stage = "decompose"
                            trace.append(decompose_trace)
                            continue
                elif decompose_status != "pass":
                    if self.config.strict_decomposition_rewrite:
                        feedback = decompose_issues or ["decomposition verifier rejected candidate"]
                        retry_stage = "decompose"
                        trace.append(decompose_trace)
                        continue
                    fallback = self._source_decomposition_fallback(pair)
                    if fallback is None:
                        feedback = decompose_issues or ["decomposition verifier rejected candidate"]
                        retry_stage = "decompose"
                        trace.append(decompose_trace)
                        continue
                    decompose_candidate = fallback
                    decompose_trace["decompose_fallback"] = (
                        "validated source decomposition retained after verifier rejection"
                    )
                final_decompose_errors = validate_decomposition(
                    decompose_candidate,
                    question=pair.question,
                    strict_natural_language=self.config.strict_decomposition_rewrite,
                )
                final_decompose_errors.extend(
                    self._strict_rewrite_candidate_errors(pair, decompose_candidate)
                )
                if final_decompose_errors:
                    decompose_trace["decompose_final_errors"] = final_decompose_errors
                    feedback = [*decompose_issues, *final_decompose_errors]
                    retry_stage = "decompose"
                    trace.append(decompose_trace)
                    continue
                trace.append(decompose_trace)
                last_decomposition = decompose_candidate
                retry_stage = "semantic"

            if retry_stage in {"decompose", "semantic"}:
                progress.stage(
                    position=position,
                    attempt=attempt,
                    max_attempts=self.config.max_attempts,
                    stage=(
                        "semantic 1/3"
                        if self.config.preserve_source_decomposition or self.dataset != "cwq"
                        else "semantic 2/5"
                    ) + ": waiting for model",
                    question=pair.question,
                )
                semantic_payload: dict[str, Any] = {
                    "question": pair.question,
                    "decomposition": last_decomposition.get("decomposition", []),
                    "source_semantic_graph": source_semantic_contract,
                    "required_operators": semantic_requirements,
                    "validation_feedback": feedback[-2:] if micro else (feedback[-4:] if lean else feedback[-12:]),
                }
                if pair.webqsp_source is not None:
                    semantic_payload["webqsp_source"] = pair.webqsp_source.as_generation_context()
                if pair.cwq_source is not None:
                    semantic_payload["cwq_source"] = pair.cwq_source.as_generation_context()
                if not constrained:
                    semantic_payload["source_compose_program"] = pair.compose_output
                if last_semantic and not constrained:
                    semantic_payload["previous_semantic_candidate"] = last_semantic
                semantic_instructions = (
                    MICRO_SEMANTIC_INSTRUCTIONS
                    if micro
                    else (LEAN_SEMANTIC_INSTRUCTIONS if lean else GATEWAY_SEMANTIC_INSTRUCTIONS)
                )
                semantic_model_instructions = SEMANTIC_INSTRUCTIONS
                if self._enforce_decomposition_alignment():
                    semantic_model_instructions += "\n" + KAEDE_SOURCE_DECOMPOSITION_ALIGNMENT
                    semantic_instructions += "\n" + KAEDE_SOURCE_DECOMPOSITION_ALIGNMENT
                semantic_generation = self.client.generate_json(
                    schema_name="rewritten_semantic_graph",
                    schema=SEMANTIC_GRAPH_SCHEMA,
                    instructions=semantic_model_instructions,
                    payload=semantic_payload,
                    gateway_instructions=semantic_instructions,
                    output_tokens=384 if micro else (1024 if lean else None),
                )
                _merge_usage(usage, semantic_generation.usage)
                last_semantic, semantic_repairs = repair_semantic_graph_cross_path_variables(
                    semantic_generation.value
                )
                path_output_repairs: list[str] = []
                if self.config.strict_decomposition_rewrite:
                    last_semantic, path_output_repairs = (
                        repair_semantic_path_output_variables(last_semantic)
                    )
                last_semantic, unscoped_variable_repairs = (
                    repair_unscoped_semantic_operator_variables(last_semantic)
                )
                last_semantic, operator_repairs = repair_operator_fidelity(
                    last_semantic,
                    semantic_requirements,
                    location="semantic_graph",
                )
                source_and_restores: list[str] = []
                strict_and_repairs: list[str] = []
                if self.config.strict_decomposition_rewrite:
                    last_semantic, strict_and_repairs = repair_superfluous_semantic_and(
                        last_semantic,
                        reference_compose,
                    )
                    last_semantic, source_and_restores = repair_source_structural_and(
                        last_semantic,
                        source_semantic_contract,
                    )
                last_semantic, _, answer_exclusion_repairs = repair_answer_exclusion_scopes(
                    last_semantic,
                    reference_compose,
                    answer_exclusions,
                    source_graph=source_semantic_contract,
                    source_program=reference_compose,
                    repair_program=False,
                )
                goal_repairs: list[str] = []
                if self.config.preserve_source_decomposition:
                    last_semantic, goal_repairs = _align_semantic_graph_goals(
                        last_semantic,
                        self._semantic_decomposition(last_decomposition) or [],
                    )
                semantic_errors = self._validate_semantic_graph(
                    last_semantic,
                    expected=semantic_requirements,
                    decomposition=last_decomposition,
                    source_graph=source_semantic_contract,
                )
                attempt_trace.update(
                    {
                        "semantic_response_id": semantic_generation.response_id,
                        "semantic_transport_mode": semantic_generation.transport_mode,
                        "semantic_errors": semantic_errors,
                    }
                )
                if semantic_repairs:
                    attempt_trace["semantic_repairs"] = semantic_repairs
                if path_output_repairs:
                    attempt_trace["semantic_path_output_repairs"] = path_output_repairs
                if unscoped_variable_repairs:
                    attempt_trace["semantic_unscoped_variable_repairs"] = (
                        unscoped_variable_repairs
                    )
                if operator_repairs:
                    attempt_trace["semantic_operator_repairs"] = operator_repairs
                if source_and_restores:
                    attempt_trace["semantic_source_and_repairs"] = source_and_restores
                if source_contract_repairs or strict_and_repairs:
                    attempt_trace["semantic_source_contract_repairs"] = [
                        *source_contract_repairs,
                        *strict_and_repairs,
                    ]
                if answer_exclusion_repairs:
                    attempt_trace["semantic_answer_exclusion_repairs"] = (
                        answer_exclusion_repairs
                    )
                if goal_repairs:
                    attempt_trace["semantic_goal_repairs"] = goal_repairs
                if semantic_errors:
                    feedback = semantic_errors
                    retry_stage = "semantic"
                    trace.append(attempt_trace)
                    continue
                retry_stage = "compose"

            progress.stage(
                position=position,
                attempt=attempt,
                max_attempts=self.config.max_attempts,
                stage=(
                    "compose 2/3"
                    if self.config.preserve_source_decomposition or self.dataset != "cwq"
                    else "compose 3/5"
                ) + ": waiting for model",
                question=pair.question,
            )
            compose_payload: dict[str, Any] = {
                "question": pair.question,
                "semantic_graph": last_semantic,
                "source_compose_program": pair.compose_output,
                "validation_feedback": feedback[-2:] if micro else (feedback[-4:] if lean else feedback[-12:]),
            }
            if pair.webqsp_source is not None:
                compose_payload["webqsp_source"] = pair.webqsp_source.as_generation_context()
            if pair.cwq_source is not None:
                compose_payload["cwq_source"] = pair.cwq_source.as_generation_context()
            if last_compose and not constrained:
                compose_payload["previous_compose_candidate"] = last_compose
            compose_generation = self.client.generate_json(
                schema_name="rebuilt_compose_program",
                schema=COMPOSE_PROGRAM_SCHEMA,
                instructions=COMPOSE_INSTRUCTIONS,
                payload=compose_payload,
                gateway_instructions=(
                    MICRO_COMPOSE_INSTRUCTIONS
                    if micro
                    else (LEAN_COMPOSE_INSTRUCTIONS if lean else GATEWAY_COMPOSE_INSTRUCTIONS)
                ),
                output_tokens=384 if micro else (1024 if lean else None),
            )
            _merge_usage(usage, compose_generation.usage)
            last_compose = compose_generation.value
            last_compose, entity_surface_repairs = repair_compose_entity_surfaces(
                last_compose,
                last_semantic,
            )
            last_compose, literal_equality_repairs = repair_compose_literal_equalities(
                last_compose,
                compose_requirements,
            )
            last_compose, compose_operator_repairs = repair_operator_fidelity(
                last_compose,
                compose_requirements,
                location="compose_program",
            )
            last_semantic, last_compose, answer_exclusion_repairs = (
                repair_answer_exclusion_scopes(
                    last_semantic,
                    last_compose,
                    answer_exclusions,
                    source_graph=source_semantic_contract,
                    source_program=reference_compose,
                )
            )
            source_and_repairs: list[str] = []
            if not self._enforce_decomposition_alignment():
                last_semantic, source_and_repairs = repair_superfluous_semantic_and(
                    last_semantic,
                    reference_compose,
                )
            if compose_operator_repairs:
                attempt_trace["compose_operator_repairs"] = compose_operator_repairs
            if literal_equality_repairs:
                attempt_trace["compose_literal_equality_repairs"] = literal_equality_repairs
            if entity_surface_repairs:
                attempt_trace["compose_entity_surface_repairs"] = entity_surface_repairs
            if answer_exclusion_repairs:
                attempt_trace["compose_answer_exclusion_repairs"] = (
                    answer_exclusion_repairs
                )
            if source_and_repairs:
                attempt_trace["semantic_and_repairs"] = source_and_repairs
            compose_errors = validate_compose_program(
                last_compose,
                semantic_graph=last_semantic,
                expected=compose_requirements,
                source_graph=source_semantic_contract,
                source_program=reference_compose,
            )
            attempt_trace["compose_response_id"] = compose_generation.response_id
            attempt_trace["compose_transport_mode"] = compose_generation.transport_mode
            attempt_trace["compose_errors"] = compose_errors
            if compose_errors:
                mapping_error = any(
                    marker in " ".join(compose_errors).casefold()
                    for marker in (
                        "semantic paths cannot be mapped",
                        "semantic operator variables",
                        "path-to-program mapping",
                        "and lowering",
                    )
                )
                if not mapping_error:
                    feedback = compose_errors
                    retry_stage = _retry_stage_for_compose_errors(compose_errors)
                    trace.append(attempt_trace)
                    continue
                deterministic_compose = compile_compose_program(
                    last_semantic,
                    answer_path_ids=pair.answer_path_ids,
                    source_graph=source_semantic_contract,
                    source_program=reference_compose,
                )
                deterministic_compose, _ = repair_operator_fidelity(
                    deterministic_compose,
                    compose_requirements,
                    location="compose_program",
                )
                deterministic_compose_errors = validate_compose_program(
                    deterministic_compose,
                    semantic_graph=last_semantic,
                    expected=compose_requirements,
                    source_graph=source_semantic_contract,
                    source_program=reference_compose,
                )
                attempt_trace["deterministic_compose_errors"] = (
                    deterministic_compose_errors
                )
                if deterministic_compose_errors:
                    feedback = compose_errors
                    retry_stage = _retry_stage_for_compose_errors(compose_errors)
                    trace.append(attempt_trace)
                    continue
                last_compose = deterministic_compose
                attempt_trace["compose_fallback"] = "deterministically compiled from semantic graph"

            external_errors = self._external_errors(pair, last_semantic, last_compose, attempt)
            attempt_trace["external_errors"] = external_errors
            if _source_sparql_has_no_answers(external_errors):
                trace.append(attempt_trace)
                return self._rejected_record(
                    pair,
                    attempts=self.config.max_attempts,
                    errors=external_errors,
                    semantic=last_semantic,
                    compose=last_compose,
                    decomposition=last_decomposition,
                    trace=trace,
                    usage=usage,
                )
            execution_failed = False
            if external_errors:
                source_aligned, direction_notes = repair_semantic_graph_directions_to_compose(
                    last_semantic,
                    reference_compose,
                )
                if self._enforce_decomposition_alignment():
                    and_notes = []
                else:
                    source_aligned, and_notes = repair_superfluous_semantic_and(
                        source_aligned,
                        reference_compose,
                    )
                if direction_notes or and_notes:
                    fallback_compose = compile_compose_program(
                        source_aligned,
                        answer_path_ids=pair.answer_path_ids,
                        source_graph=source_semantic_contract,
                        source_program=reference_compose,
                    )
                    fallback_compose, _ = repair_operator_fidelity(
                        fallback_compose,
                        compose_requirements,
                        location="compose_program",
                    )
                    fallback_goal_repairs: list[str] = []
                    if self.config.preserve_source_decomposition:
                        source_aligned, fallback_goal_repairs = _align_semantic_graph_goals(
                            source_aligned,
                            self._semantic_decomposition(last_decomposition) or [],
                        )
                    fallback_errors = self._validate_semantic_graph(
                        source_aligned,
                        expected=semantic_requirements,
                        decomposition=last_decomposition,
                        source_graph=source_semantic_contract,
                    )
                    fallback_errors.extend(
                        validate_compose_program(
                            fallback_compose,
                            semantic_graph=source_aligned,
                            expected=compose_requirements,
                            source_graph=source_semantic_contract,
                            source_program=reference_compose,
                        )
                    )
                    if not fallback_errors:
                        fallback_errors.extend(
                            self._external_errors(
                                pair,
                                source_aligned,
                                fallback_compose,
                                attempt,
                            )
                    )
                    attempt_trace["source_aligned_fallback_errors"] = fallback_errors
                    if fallback_goal_repairs:
                        attempt_trace["source_aligned_goal_repairs"] = fallback_goal_repairs
                    if not fallback_errors:
                        last_semantic = source_aligned
                        last_compose = fallback_compose
                        attempt_trace["semantic_repairs"] = [*direction_notes, *and_notes]
                        external_errors = []
                if external_errors:
                    feedback = external_errors
                    execution_failed = True
                    # Let the verifier inspect the concrete execution diagnostics
                    # before the next attempt regenerates the semantic graph.
                    retry_stage = "semantic"

            progress.stage(
                position=position,
                attempt=attempt,
                max_attempts=self.config.max_attempts,
                stage=(
                    "verifier 3/3"
                    if self.config.preserve_source_decomposition or self.dataset != "cwq"
                    else "verifier 5/5"
                ) + ": waiting for model",
                question=pair.question,
            )
            verify_instructions = VERIFY_INSTRUCTIONS
            verify_gateway_instructions = (
                MICRO_VERIFY_INSTRUCTIONS
                if micro
                else (LEAN_VERIFY_INSTRUCTIONS if lean else GATEWAY_VERIFY_INSTRUCTIONS)
            )
            if self._enforce_decomposition_alignment():
                verify_instructions += "\n" + KAEDE_SOURCE_DECOMPOSITION_ALIGNMENT
                verify_gateway_instructions += "\n" + KAEDE_SOURCE_DECOMPOSITION_ALIGNMENT
            verify_generation = self.client.generate_json(
                schema_name=(
                    "flywheel_verification_light" if constrained else "flywheel_verification"
                ),
                schema=(LIGHT_VERIFICATION_SCHEMA if constrained else VERIFICATION_SCHEMA),
                instructions=verify_instructions,
                payload={
                    "question": pair.question,
                    "expected_operators": semantic_requirements,
                    "semantic_candidate": last_semantic,
                    "compose_candidate": last_compose,
                    **(
                        {"decomposition": self._semantic_decomposition(last_decomposition)}
                        if self._enforce_decomposition_alignment()
                        else {}
                    ),
                    "validation_feedback": feedback[-2:] if micro else (feedback[-4:] if lean else feedback[-12:]),
                    **(
                        {"webqsp_source": pair.webqsp_source.as_generation_context()}
                        if pair.webqsp_source is not None
                        else {}
                    ),
                    **(
                        {"cwq_source": pair.cwq_source.as_generation_context()}
                        if pair.cwq_source is not None
                        else {}
                    ),
                },
                reasoning_effort=self.config.verifier_reasoning_effort,
                gateway_instructions=verify_gateway_instructions,
                output_tokens=96 if micro else (256 if lean else None),
            )
            _merge_usage(usage, verify_generation.usage)
            verdict = verify_generation.value
            status = str(verdict.get("status", "reject"))
            issues = _normalize_messages(verdict.get("issues"))
            attempt_trace.update(
                {
                    "verify_response_id": verify_generation.response_id,
                    "verify_transport_mode": verify_generation.transport_mode,
                    "verify_status": status,
                    "verify_issues": issues,
                }
            )

            if status == "pass":
                verifier_changed_candidate = not constrained and (
                    verdict.get("semantic_graph") != last_semantic
                    or verdict.get("compose_program") != last_compose
                )
                if execution_failed and not verifier_changed_candidate:
                    feedback = [
                        *issues,
                        "verifier passed a candidate that still fails execution validation",
                        *external_errors,
                    ]
                    retry_stage = "semantic"
                    trace.append(attempt_trace)
                    continue
                if verifier_changed_candidate:
                    verified_semantic = verdict.get("semantic_graph")
                    verified_compose = verdict.get("compose_program")
                    verified_semantic, verified_compose, answer_exclusion_repairs = (
                        repair_answer_exclusion_scopes(
                            verified_semantic,
                            verified_compose,
                            answer_exclusions,
                            source_graph=source_semantic_contract,
                            source_program=reference_compose,
                        )
                    )
                    if answer_exclusion_repairs:
                        attempt_trace["post_verify_answer_exclusion_repairs"] = (
                            answer_exclusion_repairs
                        )
                    post_path_output_repairs: list[str] = []
                    if self.config.strict_decomposition_rewrite:
                        verified_semantic, post_path_output_repairs = (
                            repair_semantic_path_output_variables(verified_semantic)
                        )
                        if post_path_output_repairs:
                            attempt_trace["post_verify_path_output_repairs"] = (
                                post_path_output_repairs
                            )
                        verified_semantic, post_strict_and_repairs = (
                            repair_superfluous_semantic_and(
                                verified_semantic,
                                reference_compose,
                            )
                        )
                        if post_strict_and_repairs:
                            attempt_trace["post_verify_source_contract_repairs"] = (
                                post_strict_and_repairs
                            )
                        verified_semantic, post_source_and_repairs = (
                            repair_source_structural_and(
                                verified_semantic,
                                source_semantic_contract,
                            )
                        )
                        if post_source_and_repairs:
                            attempt_trace["post_verify_source_and_repairs"] = (
                                post_source_and_repairs
                            )
                    post_goal_repairs: list[str] = []
                    if self.config.preserve_source_decomposition:
                        verified_semantic, post_goal_repairs = _align_semantic_graph_goals(
                            verified_semantic,
                            self._semantic_decomposition(last_decomposition) or [],
                        )
                    post_errors = self._validate_semantic_graph(
                        verified_semantic,
                        expected=semantic_requirements,
                        decomposition=last_decomposition,
                        source_graph=source_semantic_contract,
                    )
                    if isinstance(verified_semantic, dict):
                        post_errors.extend(
                            validate_compose_program(
                                verified_compose,
                                semantic_graph=verified_semantic,
                                expected=compose_requirements,
                                source_graph=source_semantic_contract,
                                source_program=reference_compose,
                            )
                        )
                    else:
                        post_errors.append("verifier pass correction has no semantic graph")
                    if not post_errors and isinstance(verified_semantic, dict) and isinstance(verified_compose, dict):
                        post_errors.extend(
                            self._external_errors(
                                pair,
                                verified_semantic,
                                verified_compose,
                                attempt,
                            )
                        )
                    attempt_trace["post_verify_errors"] = post_errors
                    if post_goal_repairs:
                        attempt_trace["post_verify_goal_repairs"] = post_goal_repairs
                    if not post_errors and isinstance(verified_semantic, dict) and isinstance(verified_compose, dict):
                        trace.append(attempt_trace)
                        return self._accepted_record(
                            pair,
                            attempt,
                            verified_semantic,
                            verified_compose,
                            issues,
                            trace,
                            usage,
                            decomposition=last_decomposition,
                        )
                    feedback = [*issues, *post_errors]
                    last_semantic = verified_semantic if isinstance(verified_semantic, dict) else last_semantic
                    last_compose = verified_compose if isinstance(verified_compose, dict) else last_compose
                    retry_stage = "semantic"
                    trace.append(attempt_trace)
                    continue
                trace.append(attempt_trace)
                return self._accepted_record(
                    pair,
                    attempt,
                    last_semantic,
                    last_compose,
                    issues,
                    trace,
                    usage,
                    decomposition=last_decomposition,
                )

            if status == "corrected" and not constrained:
                corrected_semantic = verdict.get("semantic_graph")
                corrected_compose = verdict.get("compose_program")
                corrected_semantic, corrected_compose, answer_exclusion_repairs = (
                    repair_answer_exclusion_scopes(
                        corrected_semantic,
                        corrected_compose,
                        answer_exclusions,
                        source_graph=source_semantic_contract,
                        source_program=reference_compose,
                    )
                )
                if answer_exclusion_repairs:
                    attempt_trace["post_verify_answer_exclusion_repairs"] = (
                        answer_exclusion_repairs
                    )
                corrected_path_output_repairs: list[str] = []
                if self.config.strict_decomposition_rewrite:
                    corrected_semantic, corrected_path_output_repairs = (
                        repair_semantic_path_output_variables(corrected_semantic)
                    )
                    if corrected_path_output_repairs:
                        attempt_trace["post_verify_path_output_repairs"] = (
                            corrected_path_output_repairs
                        )
                    corrected_semantic, corrected_strict_and_repairs = (
                        repair_superfluous_semantic_and(
                            corrected_semantic,
                            reference_compose,
                        )
                    )
                    if corrected_strict_and_repairs:
                        attempt_trace["post_verify_source_contract_repairs"] = (
                            corrected_strict_and_repairs
                        )
                    corrected_semantic, corrected_source_and_repairs = (
                        repair_source_structural_and(
                            corrected_semantic,
                            source_semantic_contract,
                        )
                    )
                    if corrected_source_and_repairs:
                        attempt_trace["post_verify_source_and_repairs"] = (
                            corrected_source_and_repairs
                        )
                corrected_goal_repairs: list[str] = []
                if self.config.preserve_source_decomposition:
                    corrected_semantic, corrected_goal_repairs = _align_semantic_graph_goals(
                        corrected_semantic,
                        self._semantic_decomposition(last_decomposition) or [],
                    )
                post_errors = self._validate_semantic_graph(
                    corrected_semantic,
                    expected=semantic_requirements,
                    decomposition=last_decomposition,
                    source_graph=source_semantic_contract,
                )
                if isinstance(corrected_semantic, dict):
                    post_errors.extend(
                        validate_compose_program(
                            corrected_compose,
                            semantic_graph=corrected_semantic,
                            expected=compose_requirements,
                            source_graph=source_semantic_contract,
                            source_program=reference_compose,
                        )
                    )
                else:
                    post_errors.append("verifier correction has no semantic graph")
                if not post_errors and isinstance(corrected_semantic, dict) and isinstance(corrected_compose, dict):
                    post_errors.extend(
                        self._external_errors(
                            pair,
                            corrected_semantic,
                            corrected_compose,
                            attempt,
                        )
                    )
                attempt_trace["post_verify_errors"] = post_errors
                if corrected_goal_repairs:
                    attempt_trace["post_verify_goal_repairs"] = corrected_goal_repairs
                trace.append(attempt_trace)
                if not post_errors and isinstance(corrected_semantic, dict) and isinstance(corrected_compose, dict):
                    return self._accepted_record(
                        pair,
                        attempt,
                        corrected_semantic,
                        corrected_compose,
                        issues,
                        trace,
                        usage,
                        decomposition=last_decomposition,
                    )
                feedback = [*issues, *post_errors]
                last_semantic = corrected_semantic if isinstance(corrected_semantic, dict) else last_semantic
                last_compose = corrected_compose if isinstance(corrected_compose, dict) else last_compose
                retry_stage = "semantic"
                continue

            feedback = issues or ["independent verifier rejected the candidate"]
            retry_stage = "semantic"
            trace.append(attempt_trace)

        return self._rejected_record(
            pair,
            attempts=self.config.max_attempts,
            errors=feedback,
            semantic=last_semantic,
            compose=last_compose,
            decomposition=last_decomposition,
            trace=trace,
            usage=usage,
        )

    def _rejected_record(
        self,
        pair: SourcePair,
        *,
        attempts: int,
        errors: list[str],
        semantic: dict[str, Any],
        compose: dict[str, Any],
        decomposition: dict[str, Any] | None,
        trace: list[dict[str, Any]],
        usage: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "status": "rejected",
            "source_id": pair.source_id,
            "source_index": pair.index,
            "semantic_source_index": pair.semantic_index,
            "compose_source_index": pair.compose_index,
            "question": pair.question,
            "dataset": self.dataset,
            "attempts": attempts,
            "errors": errors,
            "last_semantic_candidate": semantic,
            "last_compose_candidate": compose,
            "last_decomposition_candidate": decomposition,
            "trace": trace,
            "usage": usage,
            "webqsp_source": pair.webqsp_source.as_context() if pair.webqsp_source else None,
            "cwq_source": pair.cwq_source.as_context() if pair.cwq_source else None,
            "created_at": _utc_now(),
        }

    def _accepted_record(
        self,
        pair: SourcePair,
        attempt: int,
        semantic_graph: dict[str, Any],
        compose_program: dict[str, Any],
        issues: list[str],
        trace: list[dict[str, Any]],
        usage: dict[str, int],
        decomposition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.dataset != "cwq" and pair.decompose_output is None:
            # WebQSP 0.1.0 keeps its original two-task materialization contract.
            decomposition = None
        if self._enforce_decomposition_alignment():
            final_errors = _exact_decomposition_semantic_errors(
                semantic_graph,
                decomposition,
            )
            if self.config.strict_decomposition_rewrite:
                final_errors = [
                    *validate_decomposition(
                        decomposition,
                        question=pair.question,
                        strict_natural_language=True,
                    ),
                    *(
                        self._strict_rewrite_candidate_errors(pair, decomposition)
                        if isinstance(decomposition, dict)
                        else []
                    ),
                    *final_errors,
                ]
            if final_errors:
                final_trace = [
                    *trace,
                    {
                        "stage": "final_decomposition_semantic_alignment",
                        "errors": final_errors,
                    },
                ]
                return self._rejected_record(
                    pair,
                    attempts=attempt,
                    errors=final_errors,
                    semantic=semantic_graph,
                    compose=compose_program,
                    decomposition=decomposition,
                    trace=final_trace,
                    usage=usage,
                )
        semantic_input = dict(pair.semantic_input)
        if decomposition is not None:
            semantic_input["decomposition"] = list(
                decomposition.get("decomposition", []) or []
            )
        compose_input = {
            "question": pair.question,
            "anchors": semantic_graph["anchors"],
            "semantic_paths": semantic_graph["semantic_paths"],
            "operators": semantic_graph["operators"],
        }
        return {
            "status": "accepted",
            "source_id": pair.source_id,
            "source_index": pair.index,
            "semantic_source_index": pair.semantic_index,
            "compose_source_index": pair.compose_index,
            "question": pair.question,
            "dataset": self.dataset,
            "attempts": attempt,
            "verifier_issues": issues,
            "webqsp_source": pair.webqsp_source.as_context() if pair.webqsp_source else None,
            "cwq_source": pair.cwq_source.as_context() if pair.cwq_source else None,
            "decompose_row": (
                {
                    "instruction": (
                        KAEDE_DECOMPOSE_SFT_INSTRUCTION
                        if self.config.preserve_source_decomposition
                        else (
                            STRICT_DECOMPOSE_SFT_INSTRUCTION
                            if self.config.strict_decomposition_rewrite
                            else DECOMPOSE_SFT_INSTRUCTION
                        )
                    ),
                    "input": _json_text({"question": pair.question}),
                    "output": _json_text(decomposition or {"decomposition": []}),
                    "history": [],
                }
                if decomposition is not None else None
            ),
            "semantic_row": {
                "instruction": (
                    KAEDE_SEMANTIC_SFT_INSTRUCTION
                    if self.config.preserve_source_decomposition
                    or self.config.strict_decomposition_rewrite
                    else SEMANTIC_SFT_INSTRUCTION
                ),
                "input": _json_text(semantic_input),
                "output": _json_text(semantic_graph),
                "history": [],
            },
            "compose_row": {
                "instruction": COMPOSE_SFT_INSTRUCTION,
                "input": _json_text(compose_input),
                "output": _json_text(compose_program),
                "history": [],
            },
            "trace": trace,
            "usage": usage,
            "created_at": _utc_now(),
        }

    def _external_errors(
        self,
        pair: SourcePair,
        semantic_graph: dict[str, Any],
        compose_program: dict[str, Any],
        attempt: int,
    ) -> list[str]:
        return run_external_validator(
            self.config.external_validator_command,
            {
                "source_id": pair.source_id,
                "source_index": pair.index,
                "attempt": attempt,
                "question": pair.question,
                "semantic_graph": semantic_graph,
                "compose_program": compose_program,
                "source_semantic_graph": pair.semantic_output,
                "source_compose_program": pair.compose_output,
                "webqsp_source": pair.webqsp_source.as_context() if pair.webqsp_source else None,
                "cwq_source": pair.cwq_source.as_context() if pair.cwq_source else None,
            },
            timeout=self.config.external_validator_timeout,
        )

    def _ensure_manifest(self) -> None:
        configuration = {
            "contract_version": (
                "operator-v7-strict-natural-decomposition-alignment"
                if self.config.strict_decomposition_rewrite
                else "operator-v6-source-grounded-repair"
            ),
            "semantic_source": str(self.semantic_source),
            "semantic_source_sha256": _file_sha256(self.semantic_source),
            "compose_source": str(self.compose_source),
            "compose_source_sha256": _file_sha256(self.compose_source),
            "webqsp_source": str(self.webqsp_source) if self.webqsp_source else "",
            "webqsp_source_sha256": _file_sha256(self.webqsp_source) if self.webqsp_source else "",
            "decompose_source": str(self.decompose_source) if self.decompose_source else "",
            "decompose_source_sha256": _file_sha256(self.decompose_source) if self.decompose_source else "",
            "dataset": self.dataset,
            "require_execution_validation": self.config.require_execution_validation,
            "model": self.client.model,
            "api_mode": str(getattr(self.client, "api_mode", "unspecified")),
            "api_url": str(getattr(self.client, "api_url", "unspecified")),
            "json_mode": str(getattr(self.client, "json_mode", "unspecified")),
            "responses_url": str(getattr(self.client, "responses_url", "unspecified")),
            "generator_reasoning_effort": str(
                getattr(self.client, "reasoning_effort", "unspecified")
            ),
            "verifier_reasoning_effort": self.config.verifier_reasoning_effort,
            "external_validator_command": self.config.external_validator_command,
            "pipeline_profile": self.config.pipeline_profile,
            "preserve_source_decomposition": self.config.preserve_source_decomposition,
            "strict_decomposition_rewrite": self.config.strict_decomposition_rewrite,
            "split_name": self.config.split_name,
        }
        if self.manifest_path.exists():
            current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            existing = dict(current.get("configuration", {}) or {})
            # Manifests written before the optional KaeDe-preservation mode used
            # no flag; treat them as the default rewriting mode for resumption.
            existing.setdefault("preserve_source_decomposition", False)
            existing.setdefault("strict_decomposition_rewrite", False)
            if existing != configuration:
                counts = current.get("counts", {})
                has_records = any(
                    path.exists() and path.stat().st_size > 0
                    for path in (self.accepted_path, self.rejected_path)
                )
                if (
                    not has_records
                    and int(counts.get("accepted", 0) or 0) == 0
                    and int(counts.get("rejected_events", 0) or 0) == 0
                ):
                    current["configuration"] = configuration
                    current["updated_at"] = _utc_now()
                    _atomic_write_json(self.manifest_path, current)
                    return
                raise ValueError(
                    "output directory belongs to a different source/model configuration; "
                    "choose another --output directory"
                )
            return
        if self.config.strict_decomposition_rewrite and any(
            path.exists() and path.stat().st_size > 0
            for path in (self.accepted_path, self.rejected_path)
        ):
            raise ValueError(
                "strict decomposition rewrite requires an empty output directory when "
                "manifest.json is absent; do not seed it with legacy accepted/rejected ledgers"
            )
        _atomic_write_json(
            self.manifest_path,
            {
                "format_version": 1,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "configuration": configuration,
                "counts": {"accepted": 0, "rejected_events": 0},
            },
        )


def materialize(output_dir: Path, *, split_name: str = "train") -> dict[str, Any]:
    root = output_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"format_version": 1, "created_at": _utc_now(), "configuration": {}}
    )
    accepted = list(_accepted_records(root / "accepted.jsonl").values())
    accepted.sort(key=lambda record: int(record.get("source_index", 0)))
    semantic_rows = [record["semantic_row"] for record in accepted]
    compose_rows = [record["compose_row"] for record in accepted]
    decompose_rows = [
        record["decompose_row"]
        for record in accepted
        if isinstance(record.get("decompose_row"), dict)
    ]
    semantic_name = f"semantic_path_{split_name}.json"
    compose_name = f"compose_{split_name}.json"
    decompose_name = f"decompose_{split_name}.json"
    _atomic_write_json(root / semantic_name, semantic_rows)
    _atomic_write_json(root / compose_name, compose_rows)
    dataset_info = {
        f"semantic_path_{split_name}": _dataset_entry(semantic_name),
        f"compose_{split_name}": _dataset_entry(compose_name),
    }
    configuration = manifest.get("configuration", {})
    is_cwq = (
        isinstance(configuration, dict)
        and str(configuration.get("dataset", "")).casefold() == "cwq"
    )
    if decompose_rows or is_cwq:
        _atomic_write_json(root / decompose_name, decompose_rows)
        dataset_info[f"decompose_{split_name}"] = _dataset_entry(decompose_name)
    _atomic_write_json(root / "dataset_info.json", dataset_info)
    error_samples = _active_error_samples(
        root / "rejected.jsonl",
        accepted_ids={str(record.get("source_id", "")) for record in accepted},
    )
    _atomic_write_json(root / "error_samples.json", error_samples)

    operator_counts: Counter[str] = Counter()
    for row in semantic_rows:
        output = json.loads(row["output"])
        operator_counts.update(str(value.get("type", "")) for value in output.get("operators", []))
    manifest.update(
        {
            "updated_at": _utc_now(),
            "counts": {
                "accepted": len(accepted),
                "rejected_events": _line_count(root / "rejected.jsonl"),
                "current_errors": len(error_samples),
                "operators": dict(sorted(operator_counts.items())),
            },
            "outputs": {
                "semantic_path": semantic_name,
                "compose": compose_name,
                "dataset_info": "dataset_info.json",
                "accepted_checkpoint": "accepted.jsonl",
                "rejected_audit": "rejected.jsonl",
                "error_samples": "error_samples.json",
            },
        }
    )
    if decompose_rows or is_cwq:
        manifest["outputs"]["decompose"] = decompose_name
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _active_error_samples(path: Path, *, accepted_ids: set[str]) -> list[dict[str, Any]]:
    """Return the latest rejected record for each source not currently accepted."""
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    continue
                source_id = str(record.get("source_id", "")).strip()
                if source_id:
                    latest[source_id] = record
    active = [
        record for source_id, record in latest.items()
        if source_id not in accepted_ids
    ]
    active.sort(
        key=lambda record: (
            int(record.get("source_index", 0)),
            str(record.get("source_id", "")),
        )
    )
    return active


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain one JSON array")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} contains a non-object row")
    return value


def _row_json(row: dict[str, Any], field: str, path: Path, index: int) -> dict[str, Any]:
    raw = row.get(field)
    if not isinstance(raw, str):
        raise ValueError(f"{path}: row {index} field {field!r} must be a JSON string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: row {index} field {field!r} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: row {index} field {field!r} must decode to an object")
    return value


def _accepted_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
            source_id = str(record.get("source_id", ""))
            if source_id and record.get("status") == "accepted":
                records[source_id] = record
    return records


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_text(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _normalize_messages(value: Any) -> list[str]:
    """Normalize model diagnostics without splitting a string into characters."""
    if value is None:
        return []
    if isinstance(value, str):
        message = value.strip()
        return [message] if message else []
    if isinstance(value, (list, tuple)):
        return [message for item in value if (message := str(item).strip())]
    message = str(value).strip()
    return [message] if message else []


def _compact_rejected_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep the latest failure summary and candidates needed for diagnosis."""
    error_keys = (
        "static_and_execution_errors",
        "post_verify_errors",
        "source_aligned_fallback_errors",
        "external_errors",
        "compose_errors",
        "semantic_errors",
        "decompose_post_verify_errors",
        "decompose_errors",
    )
    final_errors: list[str] = []
    trace = record.get("trace")
    if isinstance(trace, list):
        for event in reversed(trace):
            if not isinstance(event, dict):
                continue
            for key in error_keys:
                final_errors = _normalize_messages(event.get(key))
                if final_errors:
                    break
            if final_errors:
                break
    if not final_errors:
        final_errors = _normalize_messages(record.get("errors"))

    compact = {
        key: record[key]
        for key in (
            "status",
            "source_id",
            "source_index",
            "semantic_source_index",
            "compose_source_index",
            "question",
            "dataset",
            "attempts",
            "created_at",
        )
        if key in record
    }
    compact["errors"] = [final_errors[0]] if final_errors else ["unknown rejection"]
    for key in (
        "last_decomposition_candidate",
        "last_semantic_candidate",
        "last_compose_candidate",
    ):
        candidate = record.get(key)
        if isinstance(candidate, dict):
            compact[key] = candidate
    return compact


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _merge_usage(target: dict[str, int], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _program_literal(value: str) -> str | None:
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, str) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
