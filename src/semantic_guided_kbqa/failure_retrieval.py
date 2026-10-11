"""Consolidated failure retrieval; original rules and budgets retained."""
from __future__ import annotations


# ---- failure_path_retrieval ----
"""Bounded endpoint path retrieval for an already-failed pipeline row.

This module is deliberately not wired into :mod:`pipeline`.  A caller may use
it after normal grounding/execution has failed, without making another chat
model call.  It enumerates real one- and two-hop paths from the supplied
anchor bindings, rejects schema-inconsistent joins, and retains a small ranked
beam.

There is intentionally no dataset index, Gold answer, question pattern, or
relation allow-list in this implementation.  The only persistent state is an
endpoint-result cache keyed by ``(entity_id, directions)``.
"""



from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Mapping, Sequence

from .clients import EmbeddingRanker, KnowledgeGraph
from .contracts import EntityCandidate as _failure_path_retrieval_EntityCandidate
from .ontology import FreebaseOntology as _failure_path_retrieval_FreebaseOntology





_RELATION_ID_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+")
_failure_path_retrieval_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIRECTION_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("forward",),
    ("backward",),
    ("forward", "forward"),
    ("forward", "backward"),
    ("backward", "forward"),
    ("backward", "backward"),
)
@dataclass(frozen=True, slots=True)
class _failure_path_retrieval_FailurePathCandidate:
    """One endpoint-verified relation sequence for a bound anchor."""

    anchor_ref: str
    entity_id: str
    relation_ids: tuple[str, ...]
    directions: tuple[str, ...]
    score: float
    schema_status: str
    intermediate_type: str = ""
    cvt_continuity: bool = False

    @property
    def depth(self) -> int:
        return len(self.relation_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_ref": self.anchor_ref,
            "entity_id": self.entity_id,
            "relation_ids": list(self.relation_ids),
            "directions": list(self.directions),
            "depth": self.depth,
            "score": self.score,
            "schema_status": self.schema_status,
            "intermediate_type": self.intermediate_type,
            "cvt_continuity": self.cvt_continuity,
        }


@dataclass(frozen=True, slots=True)
class _SchemaCheck:
    accepted: bool
    status: str
    intermediate_type: str = ""
    cvt_continuity: bool = False
    score_adjustment: float = 0.0


@dataclass(frozen=True, slots=True)
class _CachedPathRows:
    rows: tuple[tuple[str, ...], ...]
    error: str = ""


@dataclass(frozen=True, slots=True)
class _UnrankedPath:
    anchor_ref: str
    entity_id: str
    relation_ids: tuple[str, ...]
    directions: tuple[str, ...]
    schema: _SchemaCheck


class _failure_path_retrieval_FailurePathRetriever:
    """Retrieve a bounded local path beam after normal retrieval has failed.

    ``retrieve`` explores all direction patterns through ``max_depth`` (one
    or two).  The endpoint is therefore the authority on path existence;
    ontology metadata is used only for type/CVT continuity and ranking.  At
    most ``top_k`` paths across the enabled depths are retained for each
    anchor.  ``top_k`` is capped at 24 so accidental integration cannot
    silently create an unbounded graph beam.

    The class performs no failure detection itself.  Keeping that policy at
    the caller makes it impossible for this isolated module to alter a normal
    successful lane before it is explicitly integrated.
    """

    def __init__(
        self,
        *,
        ontology: _failure_path_retrieval_FreebaseOntology,
        kg: KnowledgeGraph,
        ranker: EmbeddingRanker,
        top_k: int = 24,
        query_limit: int = 10_000,
        max_depth: int = 2,
    ) -> None:
        self.ontology = ontology
        self.kg = kg
        self.ranker = ranker
        self.top_k = min(24, max(1, int(top_k)))
        self.query_limit = min(10_000, max(self.top_k, int(query_limit)))
        self.max_depth = min(2, max(1, int(max_depth)))
        self.direction_patterns = tuple(
            pattern
            for pattern in _DIRECTION_PATTERNS
            if len(pattern) <= self.max_depth
        )
        self._cache: dict[tuple[str, tuple[str, ...]], _CachedPathRows] = {}
        self._cache_lock = threading.Lock()
        self._inflight: dict[
            tuple[str, tuple[str, ...]],
            threading.Event,
        ] = {}

    def clear_cache(self) -> None:
        """Discard endpoint path rows cached by this retriever instance."""

        with self._cache_lock:
            self._cache.clear()

    def retrieve(
        self,
        *,
        question: str,
        decompositions: Sequence[str],
        anchor_bindings: Mapping[str, Any],
    ) -> tuple[list[_failure_path_retrieval_FailurePathCandidate], dict[str, Any]]:
        started = time.monotonic()
        context = " ".join(
            value
            for value in (
                str(question).strip(),
                *(str(item).strip() for item in decompositions),
            )
            if value
        )
        anchors = _normalize_anchor_bindings(anchor_bindings)
        diagnostics: dict[str, Any] = {
            "strategy": "failure_endpoint_path_retrieval",
            "model_calls": 0,
            "llm_calls": 0,
            "ranker_call_count": 0,
            "top_k": self.top_k,
            "query_limit": self.query_limit,
            "anchor_count": len(anchors),
            "max_depth": self.max_depth,
            "direction_pattern_count": len(self.direction_patterns),
            "query_count": 0,
            "cache_hits": 0,
            "query_elapsed_seconds": 0.0,
            "ranking_elapsed_seconds": 0.0,
            "raw_candidate_count": 0,
            "deduplicated_candidate_count": 0,
            "schema_rejected_count": 0,
            "retained_candidate_count": 0,
            "errors": [],
            "anchors": [],
        }
        if not anchors:
            diagnostics["code"] = "NO_ANCHOR_BINDINGS"
            diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
            return [], diagnostics

        unranked: list[_UnrankedPath] = []
        seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
        for anchor_ref, entity_id in anchors:
            anchor_diag: dict[str, Any] = {
                "anchor_ref": anchor_ref,
                "entity_id": entity_id,
                "queries": [],
            }
            for directions in self.direction_patterns:
                rows, query_diag = self._path_rows(entity_id, directions)
                diagnostics["query_count"] += int(not query_diag["cache_hit"])
                diagnostics["cache_hits"] += int(query_diag["cache_hit"])
                diagnostics["query_elapsed_seconds"] += float(
                    query_diag["elapsed_seconds"]
                )
                diagnostics["raw_candidate_count"] += len(rows)
                anchor_diag["queries"].append(query_diag)
                if query_diag.get("error"):
                    diagnostics["errors"].append(
                        {
                            "anchor_ref": anchor_ref,
                            "entity_id": entity_id,
                            "directions": list(directions),
                            "error": query_diag["error"],
                        }
                    )
                for relation_ids in rows:
                    key = (anchor_ref, entity_id, directions, relation_ids)
                    if key in seen:
                        continue
                    seen.add(key)
                    schema = self._schema_check(relation_ids, directions)
                    if not schema.accepted:
                        diagnostics["schema_rejected_count"] += 1
                        continue
                    unranked.append(
                        _UnrankedPath(
                            anchor_ref=anchor_ref,
                            entity_id=entity_id,
                            relation_ids=relation_ids,
                            directions=directions,
                            schema=schema,
                        )
                    )
            diagnostics["anchors"].append(anchor_diag)

        diagnostics["deduplicated_candidate_count"] = len(unranked)
        ranking_started = time.monotonic()
        diagnostics["ranker_call_count"] = int(bool(unranked))
        candidates, ranker_error = self._rank(context, unranked)
        diagnostics["ranking_elapsed_seconds"] = round(
            time.monotonic() - ranking_started,
            6,
        )
        if ranker_error:
            diagnostics["errors"].append(
                {"stage": "embedding_ranking", "error": ranker_error}
            )

        # Bound the complete one/two-hop beam per anchor.  Anchors remain
        # separate because starving a secondary constraint anchor would make
        # a later deterministic intersection impossible.
        retained: list[_failure_path_retrieval_FailurePathCandidate] = []
        grouped: dict[str, list[_failure_path_retrieval_FailurePathCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.anchor_ref, []).append(candidate)
        for key in sorted(grouped):
            values = sorted(grouped[key], key=_candidate_sort_key)
            retained.extend(values[: self.top_k])
        retained.sort(key=_candidate_sort_key)

        diagnostics["retained_candidate_count"] = len(retained)
        diagnostics["retained_by_depth"] = {
            str(depth): sum(candidate.depth == depth for candidate in retained)
            for depth in (1, 2)
        }
        diagnostics["query_elapsed_seconds"] = round(
            float(diagnostics["query_elapsed_seconds"]),
            6,
        )
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        diagnostics["code"] = "OK" if retained else "NO_SCHEMA_VALID_PATH"
        return retained, diagnostics

    def _path_rows(
        self,
        entity_id: str,
        directions: tuple[str, ...],
    ) -> tuple[tuple[tuple[str, ...], ...], dict[str, Any]]:
        key = (entity_id, directions)
        while True:
            owner = False
            with self._cache_lock:
                cached = self._cache.get(key)
                if cached is not None:
                    return cached.rows, {
                        "directions": list(directions),
                        "cache_hit": True,
                        "elapsed_seconds": 0.0,
                        "row_count": len(cached.rows),
                        "error": "",
                    }
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    owner = True
            if owner:
                break
            event.wait()

        started = time.monotonic()
        try:
            error = ""
            raw_rows: Any = []
            try:
                raw_rows = self.kg.path_hops(
                    entity_id,
                    list(directions),
                    limit=self.query_limit,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            rows: list[tuple[str, ...]] = []
            seen: set[tuple[str, ...]] = set()
            for raw in raw_rows if isinstance(raw_rows, list) else []:
                if not isinstance(raw, dict):
                    continue
                values = raw.get("relation_ids", raw.get("relations", []))
                if not isinstance(values, (list, tuple)) or len(values) != len(directions):
                    continue
                relation_ids = tuple(str(value).strip() for value in values)
                if not all(_RELATION_ID_RE.fullmatch(value) for value in relation_ids):
                    continue
                if relation_ids in seen:
                    continue
                seen.add(relation_ids)
                rows.append(relation_ids)
            rows.sort()
            result = _CachedPathRows(tuple(rows), "")
            # Endpoint failures are transient and must never become a durable
            # empty-path fact for later questions or worker threads.
            if not error:
                with self._cache_lock:
                    self._cache[key] = result
            elapsed = time.monotonic() - started
            return result.rows, {
                "directions": list(directions),
                "cache_hit": False,
                "elapsed_seconds": round(elapsed, 6),
                "row_count": len(result.rows),
                "error": error,
            }
        finally:
            with self._cache_lock:
                event = self._inflight.pop(key, None)
                if event is not None:
                    event.set()

    def _schema_check(
        self,
        relation_ids: tuple[str, ...],
        directions: tuple[str, ...],
    ) -> _SchemaCheck:
        if len(relation_ids) == 1:
            relation_id = relation_ids[0]
            known = bool(
                self.ontology.domain_for_relation(relation_id)
                or self.ontology.range_for_relation(relation_id)
            )
            return _SchemaCheck(
                True,
                "known_one_hop" if known else "unknown_one_hop",
                score_adjustment=0.01 if known else -0.04,
            )

        left_output = _output_type(
            self.ontology,
            relation_ids[0],
            directions[0],
        )
        right_input = _input_type(
            self.ontology,
            relation_ids[1],
            directions[1],
        )
        if not left_output or not right_input:
            return _SchemaCheck(
                True,
                "unknown_bridge",
                intermediate_type=left_output or right_input,
                score_adjustment=-0.06,
            )

        cvt_like = _is_cvt_like(self.ontology, left_output) or _is_cvt_like(
            self.ontology,
            right_input,
        )
        if left_output == right_input:
            return _SchemaCheck(
                True,
                "cvt_exact" if cvt_like else "exact_bridge",
                intermediate_type=left_output,
                cvt_continuity=cvt_like,
                score_adjustment=0.05 if cvt_like else 0.03,
            )

        # A mediator/CVT must join on the exact anonymous-record type.  Using
        # a broad ancestor here creates the familiar parallel-CVT false join.
        if cvt_like:
            return _SchemaCheck(
                False,
                "cvt_type_mismatch",
                intermediate_type=f"{left_output}!={right_input}",
            )

        if _failure_path_retrieval_types_compatible(self.ontology, left_output, right_input):
            return _SchemaCheck(
                True,
                "subtype_bridge",
                intermediate_type=f"{left_output}~{right_input}",
                score_adjustment=0.015,
            )
        return _SchemaCheck(
            False,
            "type_mismatch",
            intermediate_type=f"{left_output}!={right_input}",
        )

    def _rank(
        self,
        context: str,
        paths: list[_UnrankedPath],
    ) -> tuple[list[_failure_path_retrieval_FailurePathCandidate], str]:
        if not paths:
            return [], ""
        relation_ids = sorted(
            {relation_id for path in paths for relation_id in path.relation_ids}
        )
        relation_texts = [_failure_path_retrieval_relation_text(value) for value in relation_ids]
        ranker_error = ""
        try:
            embedding_scores = self.ranker.score(context, relation_texts)
            if len(embedding_scores) != len(relation_ids):
                raise ValueError("ranker returned a score vector with the wrong length")
            embedding_by_relation = {
                relation_id: float(score)
                for relation_id, score in zip(relation_ids, embedding_scores)
            }
        except (RuntimeError, TypeError, ValueError) as exc:
            ranker_error = f"{type(exc).__name__}: {exc}"
            embedding_by_relation = {relation_id: 0.0 for relation_id in relation_ids}

        lexical_by_relation = {
            relation_id: _lexical_coverage(context, _failure_path_retrieval_relation_text(relation_id))
            for relation_id in relation_ids
        }
        output: list[_failure_path_retrieval_FailurePathCandidate] = []
        for path in paths:
            hop_scores = [
                (0.7 * embedding_by_relation.get(relation_id, 0.0))
                + (0.3 * lexical_by_relation.get(relation_id, 0.0))
                for relation_id in path.relation_ids
            ]
            score = (
                sum(hop_scores) / max(1, len(hop_scores))
                + path.schema.score_adjustment
            )
            output.append(
                _failure_path_retrieval_FailurePathCandidate(
                    anchor_ref=path.anchor_ref,
                    entity_id=path.entity_id,
                    relation_ids=path.relation_ids,
                    directions=path.directions,
                    score=score,
                    schema_status=path.schema.status,
                    intermediate_type=path.schema.intermediate_type,
                    cvt_continuity=path.schema.cvt_continuity,
                )
            )
        return output, ranker_error


def _normalize_anchor_bindings(
    values: Mapping[str, Any],
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_anchor_ref, raw_value in values.items():
        anchor_ref = str(raw_anchor_ref).strip()
        entity_id = ""
        if isinstance(raw_value, _failure_path_retrieval_EntityCandidate):
            entity_id = str(raw_value.entity_id).strip()
        elif isinstance(raw_value, Mapping):
            entity_id = str(
                raw_value.get(
                    "entity_id",
                    raw_value.get("id", raw_value.get("value", "")),
                )
            ).strip()
        else:
            entity_id = str(raw_value).strip()
        key = (anchor_ref, entity_id)
        if not anchor_ref or not entity_id or key in seen:
            continue
        seen.add(key)
        output.append(key)
    output.sort()
    return output


def _input_type(
    ontology: _failure_path_retrieval_FreebaseOntology,
    relation_id: str,
    direction: str,
) -> str:
    if direction == "forward":
        return str(ontology.domain_for_relation(relation_id))
    return str(ontology.range_for_relation(relation_id))


def _output_type(
    ontology: _failure_path_retrieval_FreebaseOntology,
    relation_id: str,
    direction: str,
) -> str:
    if direction == "forward":
        return str(ontology.range_for_relation(relation_id))
    return str(ontology.domain_for_relation(relation_id))


def _failure_path_retrieval_types_compatible(
    ontology: _failure_path_retrieval_FreebaseOntology,
    left: str,
    right: str,
) -> bool:
    if left == right:
        return True
    left_parents = set(ontology.supertypes(left))
    right_parents = set(ontology.supertypes(right))
    return right in left_parents or left in right_parents


def _is_cvt_like(ontology: _failure_path_retrieval_FreebaseOntology, type_id: str) -> bool:
    """Infer mediator-like schema types without a relation/type allow-list."""

    explicit = getattr(ontology, "is_cvt_type", None)
    if callable(explicit):
        return bool(explicit(type_id))
    # Primitive/value types use Freebase's ``type.*`` schema namespace.  This
    # namespace-level rule avoids maintaining a hand-authored type list.
    if not type_id or type_id.startswith("type."):
        return False
    # Ordinary named entities inherit ``common.topic`` in fb_types; anonymous
    # mediator/value records generally do not.  This is intentionally a
    # schema-derived fallback rather than a namespace list.
    return "common.topic" not in set(ontology.supertypes(type_id))


def _failure_path_retrieval_relation_text(relation_id: str) -> str:
    return " ".join(
        segment.replace("_", " ")
        for segment in str(relation_id).split(".")
        if segment
    )


def _lexical_coverage(query: str, candidate: str) -> float:
    query_tokens = set(_failure_path_retrieval_TOKEN_RE.findall(str(query).casefold()))
    candidate_tokens = set(_failure_path_retrieval_TOKEN_RE.findall(str(candidate).casefold()))
    if not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(candidate_tokens)


def _candidate_sort_key(candidate: _failure_path_retrieval_FailurePathCandidate) -> tuple[Any, ...]:
    return (
        -candidate.score,
        candidate.anchor_ref,
        candidate.depth,
        candidate.directions,
        candidate.relation_ids,
        candidate.entity_id,
    )



# ---- failure_retrieval_plugin ----
"""Gold-blind failure-only bridge from training templates to endpoint paths."""



from copy import deepcopy
from itertools import islice, permutations, product
import json
import re
from typing import Any

from .contracts import EntityCandidate as _failure_retrieval_plugin_EntityCandidate, GroundedSemanticCandidate as _failure_retrieval_plugin_GroundedSemanticCandidate
from .graph_reconstruction import reconstruct_failure_query_graphs
from .ontology import relation_id_from_label as _failure_retrieval_plugin_relation_id_from_label, relation_label_from_id as _failure_retrieval_plugin_relation_label_from_id
from .template_path_retrieval import TemplatePathRetriever


_ANCHOR_COVERAGE_SCORE_FLOOR = 0.5
_SINGLETON_REFINEMENT_SCORE_FLOOR = 0.8
_SINGLETON_REFINEMENT_MIN_BROAD_ANSWERS = 32
_DIVERSITY_TEMPLATE_TOP_K = 24
_DIVERSITY_PER_CANDIDATE_LIMIT = 1
_CONTINUATION_DIVERSITY_SCORE_FLOOR = 40.0
_CONTINUATION_DIVERSITY_GRAPH_BUDGET = 8
_NOUN_QUESTION_RE = re.compile(r"\b(?:what|which)\s+(.+)", re.IGNORECASE)
_TYPE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_EXPLICIT_EXTREMA_RE = re.compile(
    r"\b(?:latest|last|earliest|first|largest|smallest|highest|lowest|"
    r"maximum|minimum|most recent)\b",
    re.IGNORECASE,
)


def _failure_retrieval_plugin_question_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _expects_entity_answer(question: str) -> bool:
    """Conservatively identify interrogatives that cannot return schema IDs."""

    text = " ".join(str(question).strip().casefold().split())
    if re.match(r"^(?:who|where)\b", text):
        return True
    return bool(
        re.match(r"^which\b", text)
        and not re.match(
            r"^which\s+(?:year|date|number|id|identifier|percentage|age|time)\b",
            text,
        )
    )


def _direct_path_acceptance(
    question: str,
    answer_ids: list[str],
    diagnostics: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Apply the outer confidence gate to one already executed direct path.

    Keeping this calculation in one helper lets the endpoint path retriever
    tell whether its *original* incumbent would be rejected before considering
    a low-confidence best-effort challenger.  The helper uses only runtime
    shape/schema evidence; it never reads Gold answers or dataset identity.
    """

    direct_ids = [str(value) for value in answer_ids]
    full_anchor_consensus = bool(
        int(diagnostics.get("selected_anchor_coverage", -1))
        == int(diagnostics.get("anchor_count", -2))
        and diagnostics.get("all_nonempty_candidates_agree")
    )
    typed_non_cvt_consensus = bool(
        not diagnostics.get("selected_terminal_is_cvt", True)
        and float(diagnostics.get("selected_answer_type_evidence", 0.0)) > 0.0
        and int(diagnostics.get("selected_answer_set_support", 0)) >= 2
    )
    entity_answer_shape_compatible = bool(
        not _expects_entity_answer(question)
        or all(_ENTITY_ID_RE.fullmatch(value) for value in direct_ids)
    )
    accepted = bool(
        len(direct_ids) == 1
        and (full_anchor_consensus or typed_non_cvt_consensus)
        # Direct endpoint paths contain relations only; they cannot implement
        # an explicit ordering program.
        and not _EXPLICIT_EXTREMA_RE.search(question)
        and entity_answer_shape_compatible
    )
    reason = (
        (
            "direct_full_anchor_consensus_singleton"
            if full_anchor_consensus
            else "direct_typed_non_cvt_consensus_singleton"
        )
        if accepted
        else "low_confidence_abstention"
    )
    return accepted, reason, {
        "answer_count": len(direct_ids),
        "full_anchor_consensus": full_anchor_consensus,
        "typed_non_cvt_consensus": typed_non_cvt_consensus,
        "anchor_coverage": diagnostics.get("selected_anchor_coverage"),
        "anchor_count": diagnostics.get("anchor_count"),
        "answer_set_support": diagnostics.get("selected_answer_set_support"),
        "nonempty_candidate_count": diagnostics.get("nonempty_candidate_count"),
        "answer_type_evidence": diagnostics.get("selected_answer_type_evidence"),
        "terminal_is_cvt": diagnostics.get("selected_terminal_is_cvt"),
        "entity_answer_shape_compatible": entity_answer_shape_compatible,
    }


def _failure_retrieval_plugin_decompositions(context: Any) -> list[str]:
    """Use the last reviewed runtime decomposition, then frozen inputs.

    ``pipeline.decompositions`` is the immutable prediction store.  When the
    reviewer rewrites a candidate, the value actually used by the failed run
    exists only in its trace.  Reading the store first silently discarded that
    rewrite during failure recovery.  A trace bundle can contain more than one
    attempt, so only the latest review is authoritative.
    """

    artifact = context.trace_bundle.get("decomposition", {})
    traces = artifact.get("traces", []) if isinstance(artifact, dict) else []
    for trace in reversed(traces if isinstance(traces, list) else []):
        if (
            isinstance(trace, dict)
            and trace.get("stage") == "decomposition_review_and_rewrite"
        ):
            output = trace.get("output")
            candidates = (
                output.get("final_candidates", [])
                if isinstance(output, dict)
                else []
            )
            values = [
                str(item)
                for candidate in candidates
                if isinstance(candidate, dict)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return list(dict.fromkeys(values))
            # The latest review is authoritative even when it produced no
            # usable candidate.  Fall back to the immutable store, not an
            # older review attempt.
            break

    pipeline = context.pipeline
    try:
        candidates = pipeline.decompositions.get(context.question)
    except (AttributeError, KeyError):
        candidates = []
    values = [
        str(item)
        for candidate in candidates
        for item in getattr(candidate, "decomposition", [])
        if str(item).strip()
    ]
    if values:
        return list(dict.fromkeys(values))
    # Retain trace-only replay support when no decomposition store is loaded.
    for trace in reversed(traces if isinstance(traces, list) else []):
        if not isinstance(trace, dict) or trace.get("stage") != "decompose_predictions":
            continue
        candidates = trace.get("output", [])
        for candidate in candidates if isinstance(candidates, list) else []:
            if isinstance(candidate, dict):
                values.extend(
                    str(item) for item in candidate.get("decomposition", [])
                )
        if values:
            break
    return list(dict.fromkeys(value for value in values if value.strip()))


def _current_anchors(context: Any) -> list[dict[str, str]]:
    grounder = context.pipeline.grounder
    entities = getattr(grounder, "gold_entities", {}).get(
        _failure_retrieval_plugin_question_key(context.question),
        {},
    )
    items = [
        (str(entity_id), str(label))
        for entity_id, label in entities.items()
        if str(entity_id).strip() and str(label).strip()
    ]
    normalized = context.question.casefold()
    items.sort(
        key=lambda item: (
            normalized.find(item[1].casefold())
            if item[1].casefold() in normalized
            else len(normalized),
            item[1],
            item[0],
        )
    )
    return [
        {"id": f"A{index}", "surface": label, "entity_id": entity_id}
        for index, (entity_id, label) in enumerate(items)
    ]


def _execution_rank(executed: Any) -> tuple[float, int, str]:
    return (
        float(executed.graph.score),
        -len(executed.answer_ids),
        str(executed.graph.graph_id),
    )


def _anchor_entity_ids(anchors: list[dict[str, str]]) -> set[str]:
    return {
        str(anchor.get("entity_id", ""))
        for anchor in anchors
        if isinstance(anchor, dict) and str(anchor.get("entity_id", "")).strip()
    }


def _anchor_coverage(executed: Any, anchor_ids: set[str]) -> int:
    if not anchor_ids:
        return 0
    graph_nodes = {
        str(node)
        for triple in getattr(executed.graph, "triples", [])
        if isinstance(triple, (list, tuple)) and len(triple) == 3
        for node in (triple[0], triple[2])
    }
    return len(graph_nodes & anchor_ids)


def _type_tokens(type_id: str) -> tuple[str, ...]:
    leaf = str(type_id).rsplit(".", 1)[-1].replace("_", " ").casefold()
    return tuple(_TYPE_TOKEN_RE.findall(leaf))


def _ontology_type_ids(ontology: Any) -> set[str]:
    values: set[str] = set()
    for attribute in ("relation_domains", "relation_ranges"):
        mapping = getattr(ontology, attribute, {})
        if isinstance(mapping, dict):
            values.update(str(value) for value in mapping.values() if str(value))
    parents = getattr(ontology, "type_parents", {})
    if isinstance(parents, dict):
        values.update(str(value) for value in parents if str(value))
        values.update(
            str(value)
            for parent_values in parents.values()
            for value in parent_values
            if str(value)
        )
    return values


def _expected_answer_types(question: str, ontology: Any) -> set[str]:
    """Infer a conservative answer type from interrogative grammar and schema.

    This is deliberately exact and schema-driven: noun phrases must match the
    leading tokens of an ontology type's leaf name.  It therefore cannot route
    on a dataset item, entity, or relation whitelist.
    """

    normalized = " ".join(str(question).casefold().split())
    available = _ontology_type_ids(ontology)
    match = _NOUN_QUESTION_RE.search(normalized)
    if match is None:
        return set()
    question_tokens = tuple(_TYPE_TOKEN_RE.findall(match.group(1)))
    if not question_tokens:
        return set()
    prefix_matches: list[tuple[int, str]] = []
    head_matches: set[str] = set()
    for type_id in available:
        tokens = _type_tokens(type_id)
        if tokens and question_tokens[: len(tokens)] == tokens:
            prefix_matches.append((len(tokens), type_id))
        elif tokens and question_tokens[0] == tokens[-1]:
            # The first post-interrogative noun is commonly the head while a
            # Freebase leaf may qualify it (``sports_team``, ``us_president``).
            # This is derived from ontology spelling, not a type dictionary.
            head_matches.add(type_id)
    if prefix_matches:
        longest = max(length for length, _ in prefix_matches)
        return {
            type_id
            for length, type_id in prefix_matches
            if length == longest
        }
    return head_matches


def _projected_answer_types(executed: Any, ontology: Any) -> set[str]:
    answer_var = str(getattr(executed.graph, "answer_var", ""))
    values: set[str] = set()
    for triple in getattr(executed.graph, "triples", []):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation_id, object_ = (str(value) for value in triple)
        if subject == answer_var:
            value = str(ontology.domain_for_relation(relation_id))
            if value:
                values.add(value)
        if object_ == answer_var:
            value = str(ontology.range_for_relation(relation_id))
            if value:
                values.add(value)
    # Keep ontology leaves without naming any special root type.  A broad
    # ancestor (for example the root inherited by most entities) must not make
    # two otherwise different answer projections look compatible.
    return {
        type_id
        for type_id in values
        if not any(
            type_id != other
            and type_id in set(ontology.supertypes(other))
            for other in values
        )
    }


def _projected_types_are_cvt(types: set[str], ontology: Any) -> bool:
    non_scalar = {type_id for type_id in types if not type_id.startswith("type.")}
    return bool(non_scalar) and all(
        "common.topic" not in set(ontology.supertypes(type_id))
        for type_id in non_scalar
    )


def _project_cvt_answer_neighbor(
    pipeline: Any,
    selected: Any,
    ontology: Any,
) -> tuple[Any, dict[str, Any]]:
    """Project an anonymous mediator answer to one unique entity neighbor."""

    if ontology is None:
        return selected, {"status": "disabled_no_ontology"}
    selected_types = _projected_answer_types(selected, ontology)
    if not _projected_types_are_cvt(selected_types, ontology):
        return selected, {"status": "not_applicable"}
    answer_var = str(selected.graph.answer_var)
    neighbor_vars = {
        str(object_ if str(subject) == answer_var else subject)
        for subject, _, object_ in selected.graph.triples
        if answer_var in {str(subject), str(object_)}
        and str(object_ if str(subject) == answer_var else subject).startswith("V")
    }
    candidates: list[tuple[str, set[str]]] = []
    for neighbor in sorted(neighbor_vars):
        probe = deepcopy(selected)
        probe.graph.answer_var = neighbor
        types = _projected_answer_types(probe, ontology)
        if types and not _projected_types_are_cvt(types, ontology):
            candidates.append((neighbor, types))
    if len(candidates) != 1:
        return selected, {
            "status": "ambiguous_neighbor",
            "source_types": sorted(selected_types),
            "candidate_count": len(candidates),
        }
    neighbor, neighbor_types = candidates[0]
    graph = deepcopy(selected.graph)
    graph.graph_id = f"{graph.graph_id}P"
    graph.answer_var = neighbor
    graph.sparql = ""
    execution, trace = pipeline._execute_graph_candidate(
        graph,
        lookup_labels=False,
    )
    if execution is None or not execution.answer_ids:
        return selected, {
            "status": "projection_empty",
            "source_types": sorted(selected_types),
            "neighbor_types": sorted(neighbor_types),
            "error": trace.get("error", ""),
        }
    return execution, {
        "status": "projected",
        "source_answer_var": answer_var,
        "answer_var": neighbor,
        "source_types": sorted(selected_types),
        "answer_types": sorted(neighbor_types),
        "answer_count": len(execution.answer_ids),
        "model_used": False,
    }


def _failure_retrieval_plugin_types_compatible(expected: set[str], actual: set[str], ontology: Any) -> bool:
    for expected_type in expected:
        parent_mapping = getattr(ontology, "type_parents", {})
        direct_expected_parents = (
            set(parent_mapping.get(expected_type, ()))
            if isinstance(parent_mapping, dict)
            else set()
        )
        if not direct_expected_parents:
            try:
                direct_expected_parents = set(
                    ontology.supertypes(expected_type, max_depth=1)
                ) - {expected_type}
            except TypeError:
                direct_expected_parents = set()
        for actual_type in actual:
            if (
                expected_type == actual_type
                # A direct parent is a useful schema approximation, while a
                # remote root ancestor is too broad to identify an answer.
                or actual_type in direct_expected_parents
                # A projected subtype is always at least as specific as the
                # requested ontology type.
                or expected_type in set(ontology.supertypes(actual_type))
            ):
                return True
    return False


def _has_explicit_singular_type(expected_types: set[str]) -> bool:
    """Return whether ontology-backed target wording has a singular head."""

    return any(
        tokens and not tokens[-1].endswith("s")
        for type_id in expected_types
        if (tokens := _type_tokens(type_id))
    )


def _triple_count(executed: Any) -> int:
    return sum(
        isinstance(triple, (list, tuple)) and len(triple) == 3
        for triple in getattr(executed.graph, "triples", [])
    )


def _refine_broad_selection_to_singleton(
    selected: Any,
    executions: list[Any],
    *,
    anchor_ids: set[str],
    expected_types: set[str],
    ontology: Any,
) -> Any:
    """Narrow a pathological broad result only with convergent local evidence.

    The singleton must already occur inside the selected answer set, preserve
    anchor coverage and projected ontology type, use a strictly smaller graph,
    and retain most of the selected graph's score.  Competing singleton values
    make the evidence ambiguous and therefore disable the refinement.
    """

    if (
        len(selected.answer_ids) < _SINGLETON_REFINEMENT_MIN_BROAD_ANSWERS
        or not _has_explicit_singular_type(expected_types)
    ):
        return selected
    selected_answers = set(selected.answer_ids)
    selected_coverage = _anchor_coverage(selected, anchor_ids)
    selected_types = _projected_answer_types(selected, ontology)
    selected_triples = _triple_count(selected)
    if not selected_types or selected_triples < 2:
        return selected
    candidates = [
        executed
        for executed in executions
        if len(executed.answer_ids) == 1
        and executed.answer_ids[0] in selected_answers
        and float(executed.graph.score)
        >= _SINGLETON_REFINEMENT_SCORE_FLOOR * float(selected.graph.score)
        and _anchor_coverage(executed, anchor_ids) >= selected_coverage
        and _triple_count(executed) < selected_triples
        and bool(
            _projected_answer_types(executed, ontology) & selected_types
        )
    ]
    if len({executed.answer_ids[0] for executed in candidates}) != 1:
        return selected
    return max(candidates, key=_execution_rank)


def select_failure_execution(
    question: str,
    executions: list[Any],
    anchors: list[dict[str, str]],
    ontology: Any,
) -> Any | None:
    """Select an executed failure candidate without Gold data or KG/model calls.

    The ordinary score winner remains the default.  A challenger is considered
    only when it covers more already-linked anchors and retains at least half
    of the winner's graph score.  Among maximum-coverage challengers, a
    confidently inferred ontology answer type is used when available; all
    missing or inconclusive type evidence falls back to the original ranking.
    """

    if not executions:
        return None
    baseline = max(executions, key=_execution_rank)
    anchor_ids = _anchor_entity_ids(anchors)
    baseline_coverage = _anchor_coverage(baseline, anchor_ids)
    minimum_score = _ANCHOR_COVERAGE_SCORE_FLOOR * float(baseline.graph.score)
    challengers = [
        executed
        for executed in executions
        if _anchor_coverage(executed, anchor_ids) > baseline_coverage
        and float(executed.graph.score) >= minimum_score
    ]
    if challengers:
        maximum_coverage = max(
            _anchor_coverage(executed, anchor_ids) for executed in challengers
        )
        pool = [
            executed
            for executed in challengers
            if _anchor_coverage(executed, anchor_ids) == maximum_coverage
        ]
    else:
        # A confidently inferred ontology projection may distinguish two
        # same-coverage graphs.  Keep the same 0.5 score floor so a weak type
        # coincidence cannot displace the ordinary score winner.
        pool = [
            executed
            for executed in executions
            if _anchor_coverage(executed, anchor_ids) == baseline_coverage
            and float(executed.graph.score) >= minimum_score
        ]
    expected_types = _expected_answer_types(question, ontology)
    selected = None
    if expected_types:
        compatible = [
            executed
            for executed in pool
            if _failure_retrieval_plugin_types_compatible(
                expected_types,
                _projected_answer_types(executed, ontology),
                ontology,
            )
        ]
        if compatible:
            selected = max(compatible, key=_execution_rank)
    if selected is None:
        selected = max(pool, key=_execution_rank) if challengers else baseline
    return _refine_broad_selection_to_singleton(
        selected,
        executions,
        anchor_ids=anchor_ids,
        expected_types=expected_types,
        ontology=ontology,
    )


def _semantic_graph(match: dict[str, Any], anchors: list[dict[str, str]], question: str) -> dict[str, Any] | None:
    by_id = {item["id"]: item for item in anchors}
    paths: list[dict[str, Any]] = []
    used: set[str] = set()
    for path_index, raw in enumerate(match.get("paths", [])):
        anchor_ref = str(raw.get("anchor_ref", ""))
        if anchor_ref not in by_id:
            return None
        used.add(anchor_ref)
        path_id = f"P{path_index}"
        steps: list[dict[str, Any]] = []
        source = anchor_ref
        for step_index, raw_step in enumerate(raw.get("steps", [])):
            direction = str(raw_step.get("direction", "")).casefold()
            label = [str(value) for value in raw_step.get("relation_label", []) if str(value)]
            if direction not in {"forward", "backward"} or not label:
                return None
            target = f"{path_id}.V{step_index}"
            steps.append(
                {
                    "id": f"{path_id}.S{step_index}",
                    "relation_label": label,
                    "direction": direction,
                    "from": source,
                    "to": target,
                }
            )
            source = target
        if not steps or len(steps) > 8:
            return None
        paths.append(
            {
                "id": path_id,
                "anchor_ref": anchor_ref,
                "goal": str(question),
                "steps": steps,
                "path_output_var": source,
            }
        )
    if not paths:
        return None
    return {
        "anchors": [
            {"id": item["id"], "surface": item["surface"]}
            for item in anchors
            if item["id"] in used
        ],
        "semantic_paths": paths,
    }


def _template_retriever(pipeline: Any) -> TemplatePathRetriever:
    value = getattr(pipeline, "_failure_template_retriever", None)
    if value is not None:
        return value
    lock = getattr(pipeline, "_failure_template_retriever_lock", None)
    if lock is None:
        value = TemplatePathRetriever.from_contract(pipeline.contract)
        pipeline._failure_template_retriever = value
        return value
    with lock:
        value = getattr(pipeline, "_failure_template_retriever", None)
        if value is None:
            value = TemplatePathRetriever.from_contract(pipeline.contract)
            pipeline._failure_template_retriever = value
    return value


def _compact_execution_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Keep recovery traces useful without persisting every full SPARQL body."""

    return {
        key: trace[key]
        for key in (
            "graph_id",
            "answer_count",
            "row_count",
            "query_elapsed_seconds",
            "label_elapsed_seconds",
            "elapsed_seconds",
            "error",
        )
        if key in trace
    }


def _graph_signature(graph: Any) -> str:
    return json.dumps(
        {
            "triples": getattr(graph, "triples", []),
            "answer_var": getattr(graph, "answer_var", ""),
            "operators": getattr(graph, "operators", []),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _answer_relation_is_explicit(question: str, executed: Any) -> bool:
    """Whether an answer-edge relation is stated lexically in the question."""

    question_tokens = set(_TYPE_TOKEN_RE.findall(str(question).casefold()))
    answer_var = str(getattr(executed.graph, "answer_var", ""))
    for triple in getattr(executed.graph, "triples", []):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation_id, object_ = (str(value) for value in triple)
        if answer_var not in {subject, object_}:
            continue
        leaf = relation_id.rsplit(".", 1)[-1].replace("_", " ")
        relation_tokens = set(_TYPE_TOKEN_RE.findall(leaf.casefold()))
        if relation_tokens and relation_tokens <= question_tokens:
            return True
    return False


def _template_acceptance_reason(
    *,
    question: str,
    selected: Any,
    executions: list[Any],
    lane: str,
    anchors: list[dict[str, str]],
    ontology: Any,
) -> tuple[str, dict[str, Any]]:
    """Return a Gold-blind high-confidence reason or abstain with ``""``.

    Failure rows are excluded from this project's CLI macro denominator.  A
    low-confidence non-empty answer must therefore remain a failure instead of
    silently enlarging that denominator with mostly-wrong predictions.
    """

    answer_count = len(selected.answer_ids)
    selected_set = set(selected.answer_ids)
    support = sum(set(item.answer_ids) == selected_set for item in executions)
    anchor_ids = _anchor_entity_ids(anchors)
    coverage = _anchor_coverage(selected, anchor_ids)
    score = float(selected.graph.score)
    evidence: dict[str, Any] = {
        "lane": lane,
        "answer_count": answer_count,
        "answer_set_support": support,
        "nonempty_candidate_count": len(executions),
        "graph_score": score,
        "anchor_coverage": coverage,
        "anchor_count": len(anchor_ids),
    }
    projected_types = (
        _projected_answer_types(selected, ontology)
        if ontology is not None
        else set()
    )
    projected_non_scalar = {
        type_id for type_id in projected_types if not type_id.startswith("type.")
    }
    projected_answer_is_cvt = bool(projected_non_scalar) and all(
        "common.topic" not in set(ontology.supertypes(type_id))
        for type_id in projected_non_scalar
    )
    evidence["projected_answer_types"] = sorted(projected_types)
    evidence["projected_answer_is_cvt"] = projected_answer_is_cvt
    if projected_answer_is_cvt:
        # Anonymous mediator records are implementation details, not final
        # answers.  Their labeled entity projection must be recovered by a
        # graph rather than accepted as a high-score singleton.
        return "", evidence

    if lane == "diversity_template":
        if answer_count != 1:
            return "", evidence
        if score >= 20.0:
            evidence["nested_singleton_refinement"] = False
            return "diversity_singleton_score", evidence
        nested = False
        selected_types = (
            _projected_answer_types(selected, ontology)
            if ontology is not None
            else set()
        )
        selected_triples = _triple_count(selected)
        if selected_types:
            for broad in executions:
                if (
                    broad is not selected
                    and len(broad.answer_ids)
                    >= _SINGLETON_REFINEMENT_MIN_BROAD_ANSWERS
                    and selected.answer_ids[0] in set(broad.answer_ids)
                    and float(broad.graph.score) > score
                    and score
                    >= _SINGLETON_REFINEMENT_SCORE_FLOOR
                    * float(broad.graph.score)
                    and coverage >= _anchor_coverage(broad, anchor_ids)
                    and selected_triples < _triple_count(broad)
                    and bool(
                        selected_types
                        & _projected_answer_types(broad, ontology)
                    )
                ):
                    nested = True
                    break
        evidence["nested_singleton_refinement"] = nested
        if nested:
            return "diversity_nested_singleton", evidence
        return "", evidence

    if lane != "ordinary_template":
        return "", evidence
    coverage_ok = bool(anchor_ids) and coverage * 3 >= len(anchor_ids) * 2
    evidence["two_thirds_anchor_coverage"] = coverage_ok
    if score < 20.0 or not coverage_ok:
        return "", evidence
    if answer_count == 1:
        return "ordinary_singleton", evidence
    if answer_count <= 5 and support >= 5:
        return "ordinary_answer_set_consensus", evidence
    if (
        2 <= answer_count <= 10
        and len(executions) == 1
        and re.match(r"\s*(?:what|which)\b", str(question), re.IGNORECASE)
        and _answer_relation_is_explicit(question, selected)
    ):
        return "ordinary_explicit_answer_relation", evidence
    return "", evidence


def _execute_failure_graphs(
    *,
    pipeline: Any,
    graphs: list[Any],
    question: str,
    decompositions: list[str],
    graph_budget: int,
    seen_signatures: set[str],
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Execute a score-ordered lane with deduplication and a safe early stop."""

    executions: list[Any] = []
    traces: list[dict[str, Any]] = []
    first_nonempty_score: float | None = None
    stopped_below_score_floor = False
    duplicates_skipped = 0
    for graph in graphs:
        signature = _graph_signature(graph)
        if signature in seen_signatures:
            duplicates_skipped += 1
            continue
        if (
            first_nonempty_score is not None
            and first_nonempty_score > 0.0
            and float(graph.score)
            < _ANCHOR_COVERAGE_SCORE_FLOOR * first_nonempty_score
        ):
            stopped_below_score_floor = True
            break
        seen_signatures.add(signature)
        graph.provenance["original_question"] = question
        graph.provenance["decomposition"] = list(decompositions)
        executed, trace = pipeline._execute_graph_candidate(
            graph,
            lookup_labels=False,
        )
        traces.append(_compact_execution_trace(trace))
        if executed is not None and executed.answer_ids:
            if first_nonempty_score is None:
                first_nonempty_score = float(executed.graph.score)
            executions.append(executed)
    return executions, traces, {
        "maximum_graph_budget": graph_budget,
        "executed_graph_count": len(traces),
        "duplicate_graphs_skipped": duplicates_skipped,
        "score_floor_ratio": _ANCHOR_COVERAGE_SCORE_FLOOR,
        "stopped_below_score_floor": stopped_below_score_floor,
        "candidate_label_queries": 0,
    }


def _ground_templates(
    *,
    matches: list[dict[str, Any]],
    anchors: list[dict[str, str]],
    question: str,
    ontology: Any,
    endpoint: Any,
) -> tuple[list[_failure_retrieval_plugin_GroundedSemanticCandidate], dict[str, Any]]:
    """Verify all template paths with one VALUES query per direction group."""
    anchors_by_ref = {item["id"]: item for item in anchors}
    prepared: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    groups: dict[tuple[str, tuple[str, ...]], list[set[str]]] = {}
    for match in matches:
        raw_paths = [path for path in match.get("paths", []) if isinstance(path, dict)]
        slots = sorted({int(path.get("anchor_slot", 0)) for path in raw_paths})
        if not raw_paths or len(slots) > len(anchors):
            continue
        for assigned in permutations(sorted(anchors_by_ref), len(slots)):
            slot_mapping = dict(zip(slots, assigned))
            paths: list[dict[str, Any]] = []
            valid = True
            for raw_path in raw_paths:
                anchor_ref = slot_mapping[int(raw_path.get("anchor_slot", 0))]
                directions: list[str] = []
                allowed: list[set[str]] = []
                for raw_step in raw_path.get("steps", []):
                    direction = str(raw_step.get("direction", "")).casefold()
                    label = raw_step.get("relation_label", [])
                    if direction not in {"forward", "backward"}:
                        valid = False
                        break
                    candidates = set(
                        ontology.candidates_for_label(label, limit=4)
                    )
                    direct = _failure_retrieval_plugin_relation_id_from_label(label)
                    if direct:
                        candidates.add(direct)
                    if not candidates:
                        valid = False
                        break
                    directions.append(direction)
                    allowed.append(candidates)
                if not valid or not directions or len(directions) > 8:
                    valid = False
                    break
                key = (anchor_ref, tuple(directions))
                union_sets = groups.setdefault(
                    key,
                    [set() for _ in directions],
                )
                for index, values in enumerate(allowed):
                    union_sets[index].update(values)
                paths.append(
                    {
                        "anchor_ref": anchor_ref,
                        "directions": tuple(directions),
                        "allowed": tuple(frozenset(values) for values in allowed),
                    }
                )
            if valid and paths:
                prepared.append((match, paths))

    verified: dict[tuple[str, tuple[str, ...]], list[tuple[str, ...]]] = {}
    query_diagnostics: list[dict[str, Any]] = []
    for (anchor_ref, directions), candidate_sets in sorted(groups.items()):
        anchor = anchors_by_ref[anchor_ref]
        error = ""
        rows: list[dict[str, Any]] = []
        try:
            rows = endpoint.path_hops(
                anchor["entity_id"],
                list(directions),
                limit=10_000,
                relation_candidates=[sorted(values) for values in candidate_sets],
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}:{exc}"
        sequences = sorted(
            {
                tuple(str(value) for value in row.get("relation_ids", []))
                for row in rows
                if isinstance(row, dict)
                and len(row.get("relation_ids", [])) == len(directions)
            }
        )
        verified[(anchor_ref, directions)] = sequences
        query_diagnostics.append(
            {
                "anchor_ref": anchor_ref,
                "directions": list(directions),
                "candidate_widths": [len(values) for values in candidate_sets],
                "sequence_count": len(sequences),
                "error": error,
            }
        )

    grounded: list[_failure_retrieval_plugin_GroundedSemanticCandidate] = []
    seen: set[str] = set()
    for match, paths in prepared:
        choices: list[list[tuple[str, ...]]] = []
        for path in paths:
            sequences = verified.get(
                (path["anchor_ref"], path["directions"]),
                [],
            )
            matching = [
                sequence
                for sequence in sequences
                if all(
                    relation_id in path["allowed"][index]
                    for index, relation_id in enumerate(sequence)
                )
            ]
            choices.append(matching[:2])
        if any(not values for values in choices):
            continue
        for combination in islice(product(*choices), 16):
            semantic_paths: list[dict[str, Any]] = []
            relation_bindings: dict[str, str] = {}
            used_anchors: set[str] = set()
            for path_index, (path, sequence) in enumerate(zip(paths, combination)):
                path_id = f"P{path_index}"
                anchor_ref = path["anchor_ref"]
                used_anchors.add(anchor_ref)
                source = anchor_ref
                steps: list[dict[str, Any]] = []
                for step_index, (relation_id, direction) in enumerate(
                    zip(sequence, path["directions"])
                ):
                    step_id = f"{path_id}.S{step_index}"
                    target = f"{path_id}.V{step_index}"
                    steps.append(
                        {
                            "id": step_id,
                            "relation_label": _failure_retrieval_plugin_relation_label_from_id(relation_id),
                            "direction": direction,
                            "from": source,
                            "to": target,
                        }
                    )
                    relation_bindings[step_id] = relation_id
                    source = target
                semantic_paths.append(
                    {
                        "id": path_id,
                        "anchor_ref": anchor_ref,
                        "goal": question,
                        "steps": steps,
                        "path_output_var": source,
                    }
                )
            signature = json.dumps(
                {
                    "anchors": sorted(used_anchors),
                    "bindings": sorted(relation_bindings.items()),
                },
                sort_keys=True,
            )
            if signature in seen:
                continue
            seen.add(signature)
            anchor_bindings = {
                anchor_ref: _failure_retrieval_plugin_EntityCandidate(
                    anchors_by_ref[anchor_ref]["entity_id"],
                    anchors_by_ref[anchor_ref]["surface"],
                    1.0,
                    "failure_template",
                )
                for anchor_ref in used_anchors
            }
            grounded.append(
                _failure_retrieval_plugin_GroundedSemanticCandidate(
                    compose_input={
                        "question": question,
                        "anchors": [
                            {
                                "id": anchor_ref,
                                "surface": anchors_by_ref[anchor_ref]["surface"],
                            }
                            for anchor_ref in sorted(used_anchors)
                        ],
                        "semantic_paths": semantic_paths,
                    },
                    anchor_bindings=anchor_bindings,
                    relation_bindings=relation_bindings,
                    score=1.0 / max(1, int(match.get("rank", 1))),
                    provenance={
                        "failure_template_retrieval": {
                            "template_rank": int(match.get("rank", 0)),
                            "source_example_index": int(
                                match.get("source_example_index", -1)
                            ),
                            "model_used": False,
                            "uses_gold": False,
                        }
                    },
                )
            )
    grounded.sort(key=lambda item: -item.score)
    return grounded[:80], {
        "group_query_count": len(query_diagnostics),
        "groups": query_diagnostics,
        "prepared_template_count": len(prepared),
        "grounded_candidate_count": len(grounded),
    }


def _direct_template_candidates(
    *,
    matches: list[dict[str, Any]],
    anchors: list[dict[str, str]],
    question: str,
) -> tuple[list[_failure_retrieval_plugin_GroundedSemanticCandidate], dict[str, Any]]:
    """Compile training templates directly; final SPARQL verifies existence."""
    anchors_by_ref = {item["id"]: item for item in anchors}
    output: list[_failure_retrieval_plugin_GroundedSemanticCandidate] = []
    seen: set[str] = set()
    for match in matches:
        raw_paths = [path for path in match.get("paths", []) if isinstance(path, dict)]
        slots = sorted({int(path.get("anchor_slot", 0)) for path in raw_paths})
        if not raw_paths or len(slots) > len(anchors):
            continue
        expected_mapping = {
            int(slot): str(anchor_ref)
            for slot, anchor_ref in (match.get("anchor_mapping") or {}).items()
        }
        for assigned in permutations(sorted(anchors_by_ref), len(slots)):
            slot_mapping = dict(zip(slots, assigned))
            semantic_paths: list[dict[str, Any]] = []
            relation_bindings: dict[str, str] = {}
            valid = True
            for path_index, raw_path in enumerate(raw_paths):
                path_id = f"P{path_index}"
                anchor_ref = slot_mapping[int(raw_path.get("anchor_slot", 0))]
                source = anchor_ref
                steps: list[dict[str, Any]] = []
                for step_index, raw_step in enumerate(raw_path.get("steps", [])):
                    direction = str(raw_step.get("direction", "")).casefold()
                    relation_id = _failure_retrieval_plugin_relation_id_from_label(
                        raw_step.get("relation_label", [])
                    )
                    if direction not in {"forward", "backward"} or not relation_id:
                        valid = False
                        break
                    step_id = f"{path_id}.S{step_index}"
                    target = f"{path_id}.V{step_index}"
                    steps.append(
                        {
                            "id": step_id,
                            "relation_label": _failure_retrieval_plugin_relation_label_from_id(relation_id),
                            "direction": direction,
                            "from": source,
                            "to": target,
                        }
                    )
                    relation_bindings[step_id] = relation_id
                    source = target
                if not valid or not steps or len(steps) > 8:
                    valid = False
                    break
                semantic_paths.append(
                    {
                        "id": path_id,
                        "anchor_ref": anchor_ref,
                        "goal": question,
                        "steps": steps,
                        "path_output_var": source,
                    }
                )
            if not valid or not semantic_paths:
                continue
            signature = json.dumps(
                {
                    "mapping": slot_mapping,
                    "bindings": sorted(relation_bindings.items()),
                },
                sort_keys=True,
            )
            if signature in seen:
                continue
            seen.add(signature)
            used = {path["anchor_ref"] for path in semantic_paths}
            mapping_bonus = 0.1 * sum(
                expected_mapping.get(slot) == anchor_ref
                for slot, anchor_ref in slot_mapping.items()
            )
            output.append(
                _failure_retrieval_plugin_GroundedSemanticCandidate(
                    compose_input={
                        "question": question,
                        "anchors": [
                            {
                                "id": anchor_ref,
                                "surface": anchors_by_ref[anchor_ref]["surface"],
                            }
                            for anchor_ref in sorted(used)
                        ],
                        "semantic_paths": semantic_paths,
                    },
                    anchor_bindings={
                        anchor_ref: _failure_retrieval_plugin_EntityCandidate(
                            anchors_by_ref[anchor_ref]["entity_id"],
                            anchors_by_ref[anchor_ref]["surface"],
                            1.0,
                            "failure_template",
                        )
                        for anchor_ref in used
                    },
                    relation_bindings=relation_bindings,
                    # Keep BM25 template order dominant over the many graph
                    # merge variants produced downstream. Endpoint execution
                    # remains the factual verifier.
                    score=(10.0 / max(1, int(match.get("rank", 1)))) + mapping_bonus,
                    provenance={
                        "failure_template_retrieval": {
                            "template_rank": int(match.get("rank", 0)),
                            "source_example_index": int(
                                match.get("source_example_index", -1)
                            ),
                            "direct_template": True,
                            "model_used": False,
                            "uses_gold": False,
                        }
                    },
                )
            )
    output.sort(key=lambda item: -item.score)
    return output[:80], {
        "prepared_template_count": len(matches),
        "grounded_candidate_count": min(80, len(output)),
        "endpoint_path_queries": 0,
        "verification": "final_sparql_only",
    }


def _failure_retrieval_plugin_retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    """Retriever plugin consumed by ``replay_failure_path_retrieval.py``."""
    pipeline = context.pipeline
    verbose_diagnostics = bool(getattr(context, "verbose_diagnostics", True))
    decompositions = _failure_retrieval_plugin_decompositions(context)
    anchors = _current_anchors(context)
    diagnostics: dict[str, Any] = {
        "status": "not_compiled",
        "strategy": "failure_only_template_then_endpoint_path",
        "llm_calls": 0,
        "uses_gold_answers": False,
        "anchor_count": len(anchors),
    }
    if not anchors:
        diagnostics["reason"] = "no_linked_anchors"
        return {"answer_ids": [], "diagnostics": diagnostics}

    matches, template_diagnostics = _template_retriever(pipeline).retrieve(
        question=context.question,
        decomposition=decompositions,
        current_anchors=anchors,
        top_k=int(getattr(pipeline, "failure_endpoint_template_top_k", 24)),
        preselect=128,
    )
    diagnostics["template"] = (
        template_diagnostics
        if verbose_diagnostics
        else {
            key: value
            for key, value in template_diagnostics.items()
            if key
            in {
                "strategy",
                "index_size",
                "index_fingerprint",
                "cache_hit",
                "current_anchor_count",
                "considered",
                "rejected_anchor_count",
            }
        }
    )
    grounded, grounding_diagnostics = _direct_template_candidates(
        matches=matches,
        anchors=anchors,
        question=context.question,
    )
    diagnostics["grounding"] = grounding_diagnostics
    diagnostics["grounded_candidate_count"] = len(grounded)
    graph_budget = int(getattr(pipeline, "failure_endpoint_graph_budget", 80))
    graphs, reconstruction_diagnostics = reconstruct_failure_query_graphs(
        grounded,
        ontology=getattr(pipeline.grounder, "ontology", None),
        pipeline_version=str(pipeline.version),
        limit=graph_budget,
        allow_single_path=True,
        allow_hybrid=False,
        max_execution_limit=graph_budget,
        pool_limit=graph_budget,
    )
    diagnostics["reconstruction"] = reconstruction_diagnostics
    seen_signatures: set[str] = set()
    executions, execution_diagnostics, execution_policy = (
        _execute_failure_graphs(
            pipeline=pipeline,
            graphs=graphs,
            question=context.question,
            decompositions=decompositions,
            graph_budget=graph_budget,
            seen_signatures=seen_signatures,
        )
    )
    diagnostics["execution_policy"] = execution_policy
    diagnostics["executions"] = execution_diagnostics

    ordinary_selected = select_failure_execution(
        context.question,
        executions,
        anchors,
        getattr(pipeline.grounder, "ontology", None),
    )
    ordinary_acceptance_reason = ""
    ordinary_acceptance_evidence: dict[str, Any] = {}
    if ordinary_selected is not None:
        projected, projection_diagnostics = _project_cvt_answer_neighbor(
            pipeline,
            ordinary_selected,
            getattr(pipeline.grounder, "ontology", None),
        )
        if projection_diagnostics.get("status") != "not_applicable":
            diagnostics["ordinary_cvt_answer_projection"] = (
                projection_diagnostics
            )
        if projected is not ordinary_selected:
            ordinary_selected = projected
            executions = [projected]
        (
            ordinary_acceptance_reason,
            ordinary_acceptance_evidence,
        ) = _template_acceptance_reason(
            question=context.question,
            selected=ordinary_selected,
            executions=executions,
            lane="ordinary_template",
            anchors=anchors,
            ontology=getattr(pipeline.grounder, "ontology", None),
        )
    ordinary_low_confidence = bool(
        executions and not ordinary_acceptance_reason
    )

    # A score-sorted reconstruction pool can be saturated by several merge
    # variants of the same high-ranked template.  After every ordinary graph
    # is empty *or its selected answer fails the confidence gate*, compile one
    # graph per candidate from the first 24 templates.  Previously executed
    # graph structures are skipped exactly.  Accepted ordinary answers retain
    # the original immediate-return path and pay no additional endpoint cost.
    if not executions or ordinary_low_confidence:
        diagnostics["ordinary_lane"] = {
            "grounding": grounding_diagnostics,
            "reconstruction": reconstruction_diagnostics,
            "execution_policy": execution_policy,
            "executions": execution_diagnostics,
        }
        if ordinary_low_confidence:
            diagnostics["ordinary_lane"]["acceptance"] = {
                "accepted": False,
                "reason": "low_confidence_abstention",
                "uses_gold_answers": False,
                "evidence": ordinary_acceptance_evidence,
            }
        diversity_grounded, diversity_grounding = _direct_template_candidates(
            matches=matches[:_DIVERSITY_TEMPLATE_TOP_K],
            anchors=anchors,
            question=context.question,
        )
        diversity_graph_budget = (
            min(graph_budget, _CONTINUATION_DIVERSITY_GRAPH_BUDGET)
            if ordinary_low_confidence
            else graph_budget
        )
        diversity_graphs, diversity_reconstruction = (
            reconstruct_failure_query_graphs(
                diversity_grounded,
                ontology=getattr(pipeline.grounder, "ontology", None),
                pipeline_version=str(pipeline.version),
                limit=diversity_graph_budget,
                per_candidate_limit=_DIVERSITY_PER_CANDIDATE_LIMIT,
                allow_single_path=True,
                allow_hybrid=False,
                max_execution_limit=diversity_graph_budget,
                pool_limit=diversity_graph_budget,
            )
        )
        (
            executions,
            diversity_execution_diagnostics,
            diversity_execution_policy,
        ) = _execute_failure_graphs(
            pipeline=pipeline,
            graphs=diversity_graphs,
            question=context.question,
            decompositions=decompositions,
            graph_budget=diversity_graph_budget,
            seen_signatures=seen_signatures,
        )
        diagnostics["diversity_fallback"] = {
            "triggered": True,
            "reason": (
                "ordinary_lane_low_confidence"
                if ordinary_low_confidence
                else "ordinary_lane_all_executions_empty"
            ),
            "template_top_k": min(_DIVERSITY_TEMPLATE_TOP_K, len(matches)),
            "per_candidate_limit": _DIVERSITY_PER_CANDIDATE_LIMIT,
            "graph_budget": diversity_graph_budget,
            "grounding": diversity_grounding,
            "reconstruction": diversity_reconstruction,
            "execution_policy": diversity_execution_policy,
            "executions": diversity_execution_diagnostics,
        }
        # Top-level fields always describe the active template lane so trace
        # counters and the eventual selected graph use one consistent scope.
        diagnostics["grounding"] = diversity_grounding
        diagnostics["grounded_candidate_count"] = len(diversity_grounded)
        diagnostics["reconstruction"] = diversity_reconstruction
        diagnostics["execution_policy"] = diversity_execution_policy
        diagnostics["executions"] = diversity_execution_diagnostics
    else:
        diagnostics["diversity_fallback"] = {
            "triggered": False,
            "reason": "ordinary_lane_nonempty",
        }

    active_lane = (
        "diversity_template"
        if diagnostics["diversity_fallback"].get("triggered")
        else "ordinary_template"
    )
    active_selected = select_failure_execution(
        context.question,
        executions,
        anchors,
        getattr(pipeline.grounder, "ontology", None),
    )
    active_acceptance_reason = ""
    active_acceptance_evidence: dict[str, Any] = {}
    if active_selected is not None:
        projected, projection_diagnostics = _project_cvt_answer_neighbor(
            pipeline,
            active_selected,
            getattr(pipeline.grounder, "ontology", None),
        )
        if projection_diagnostics.get("status") != "not_applicable":
            diagnostics["cvt_answer_projection"] = projection_diagnostics
        if projected is not active_selected:
            active_selected = projected
            # The mediator and its entity projection are not competing answer
            # hypotheses.  Keep only the projected execution so the ordinary
            # selector cannot switch back to the anonymous record later.
            executions = [projected]
        (
            active_acceptance_reason,
            active_acceptance_evidence,
        ) = _template_acceptance_reason(
            question=context.question,
            selected=active_selected,
            executions=executions,
            lane=active_lane,
            anchors=anchors,
            ontology=getattr(pipeline.grounder, "ontology", None),
        )
    if (
        ordinary_low_confidence
        and active_acceptance_reason == "diversity_singleton_score"
        and float(active_acceptance_evidence.get("graph_score", 0.0))
        < _CONTINUATION_DIVERSITY_SCORE_FLOOR
    ):
        # A diversity singleton with the ordinary score floor is useful when
        # the ordinary lane was completely empty.  It is weaker evidence when
        # it contradicts an existing non-empty ordinary answer, so the
        # continuation path requires a doubled confidence floor.
        active_acceptance_evidence["continuation_score_floor"] = (
            _CONTINUATION_DIVERSITY_SCORE_FLOOR
        )
        active_acceptance_reason = ""
    if executions and not active_acceptance_reason:
        diagnostics["diversity_fallback"]["acceptance"] = {
            "accepted": False,
            "reason": "low_confidence_abstention",
            "uses_gold_answers": False,
            "evidence": active_acceptance_evidence,
        }
    diagnostics["execution_candidates"] = [
        {
            "graph_id": item.graph.graph_id,
            "score": item.graph.score,
            "answer_count": len(item.answer_ids),
            **(
                {
                    "answer_ids": list(item.answer_ids),
                    "triples": list(item.graph.triples),
                    "answer_var": str(item.graph.answer_var),
                }
                if verbose_diagnostics
                else {}
            ),
        }
        for item in executions
    ]
    if not executions or not active_acceptance_reason:
        # The last lane enumerates only real one-hop endpoint paths.  It runs
        # after both template lanes are empty or rejected by their confidence
        # gate, so it cannot replace an accepted template answer.
        from .failure_retrieval import _failure_endpoint_retrieval_plugin_retrieve as retrieve_paths

        direct = retrieve_paths(context, endpoint)
        diagnostics["direct_endpoint_fallback"] = direct.get(
            "diagnostics",
            {},
        )
        direct_ids = [str(value) for value in direct.get("answer_ids", [])]
        if direct_ids:
            diagnostics["status"] = "selected_direct_endpoint_path"
            diagnostics["selected_lane"] = "direct_endpoint_path"
            diagnostics["selected_graph_id"] = (
                diagnostics["direct_endpoint_fallback"].get(
                    "selected_graph_id",
                    "",
                )
            )
            diagnostics["answer_count"] = len(direct_ids)
            direct_diagnostics = diagnostics["direct_endpoint_fallback"]
            accepted, acceptance_reason, acceptance_evidence = (
                _direct_path_acceptance(
                    context.question,
                    direct_ids,
                    direct_diagnostics,
                )
            )
            challenger_evidence = direct_diagnostics.get(
                "low_confidence_challenger",
                {},
            )
            if (
                isinstance(challenger_evidence, dict)
                and challenger_evidence.get("applied") is True
            ):
                # The endpoint helper is allowed to replace the *rejected*
                # incumbent only for the bounded low-confidence lane.  Never
                # promote that alternative to high confidence after the fact;
                # this also preserves missing-relation first refusal.
                accepted = False
                acceptance_reason = "low_confidence_abstention"
                acceptance_evidence = {
                    **acceptance_evidence,
                    "original_direct_confidence_gate_rejected": True,
                    "low_confidence_challenger_applied": True,
                }
            diagnostics["acceptance"] = {
                "accepted": accepted,
                "reason": acceptance_reason,
                "uses_gold_answers": False,
                "evidence": acceptance_evidence,
            }
            if not accepted:
                best_effort_limit = int(
                    getattr(
                        pipeline,
                        "failure_endpoint_best_effort_answer_limit",
                        64,
                    )
                )
                return_best_effort = bool(
                    getattr(
                        pipeline,
                        "failure_endpoint_return_bounded_direct_best_effort",
                        False,
                    )
                    and 0 < len(direct_ids) <= best_effort_limit
                )
                if return_best_effort:
                    diagnostics["status"] = "selected_bounded_direct_best_effort"
                    diagnostics["acceptance"] = {
                        **diagnostics["acceptance"],
                        "accepted": True,
                        "reason": "bounded_direct_best_effort_after_final_empty",
                        "confidence": "low",
                        "evidence": {
                            **diagnostics["acceptance"]["evidence"],
                            "best_effort_answer_limit": best_effort_limit,
                            "normal_pipeline_was_final_empty": True,
                            "additional_execution_queries": 0,
                            "additional_label_queries": 0,
                        },
                    }
                    return {
                        "answer_ids": direct_ids,
                        "answers": list(direct.get("answers", [])),
                        "selected_graph": direct.get("selected_graph"),
                        "diagnostics": diagnostics,
                    }
                diagnostics["status"] = "abstained_low_confidence"
                diagnostics["reason"] = "direct_path_confidence_gate"
                return {"answer_ids": [], "diagnostics": diagnostics}
            return {
                "answer_ids": direct_ids,
                "answers": list(direct.get("answers", [])),
                "selected_graph": direct.get("selected_graph"),
                "diagnostics": diagnostics,
            }
        if executions:
            diagnostics["status"] = "abstained_low_confidence"
            diagnostics["selected_lane"] = active_lane
            diagnostics["selected_graph_id"] = (
                active_selected.graph.graph_id if active_selected is not None else ""
            )
            diagnostics["answer_count"] = (
                len(active_selected.answer_ids)
                if active_selected is not None
                else 0
            )
            diagnostics["acceptance"] = {
                "accepted": False,
                "reason": "low_confidence_abstention",
                "uses_gold_answers": False,
                "evidence": active_acceptance_evidence,
            }
            diagnostics["reason"] = "template_confidence_gate"
        else:
            diagnostics["reason"] = "all_failure_retrieval_lanes_empty"
        return {"answer_ids": [], "diagnostics": diagnostics}
    selected = select_failure_execution(
        context.question,
        executions,
        anchors,
        getattr(pipeline.grounder, "ontology", None),
    )
    if selected is None:
        diagnostics["reason"] = "selection_failed"
        return {"answer_ids": [], "diagnostics": diagnostics}
    diagnostics["status"] = "selected"
    selected_lane = (
        "diversity_template"
        if diagnostics["diversity_fallback"].get("triggered")
        else "ordinary_template"
    )
    diagnostics["selected_lane"] = selected_lane
    diagnostics["selected_graph_id"] = selected.graph.graph_id
    diagnostics["answer_count"] = len(selected.answer_ids)
    acceptance_reason, acceptance_evidence = _template_acceptance_reason(
        question=context.question,
        selected=selected,
        executions=executions,
        lane=selected_lane,
        anchors=anchors,
        ontology=getattr(pipeline.grounder, "ontology", None),
    )
    diagnostics["acceptance"] = {
        "accepted": bool(acceptance_reason),
        "reason": acceptance_reason or "low_confidence_abstention",
        "uses_gold_answers": False,
        "evidence": acceptance_evidence,
    }
    if not acceptance_reason:
        diagnostics["status"] = "abstained_low_confidence"
        diagnostics["reason"] = "template_confidence_gate"
        return {"answer_ids": [], "diagnostics": diagnostics}
    raw_labels: list[dict[str, str]] = []
    try:
        raw_labels = pipeline.kg.labels(list(selected.answer_ids))
    except Exception as exc:
        diagnostics["selected_label_error"] = f"{type(exc).__name__}: {exc}"
    labels_by_id = {
        str(item.get("id", "")): {
            "id": str(item.get("id", "")),
            "label": str(item.get("label", "")),
        }
        for item in raw_labels
        if isinstance(item, dict) and str(item.get("id", ""))
    }
    selected.answers = [
        labels_by_id.get(answer_id, {"id": answer_id, "label": answer_id})
        for answer_id in selected.answer_ids
    ]
    diagnostics["execution_policy"]["selected_label_queries"] = int(
        bool(selected.answer_ids)
    )
    return {
        "answer_ids": list(selected.answer_ids),
        "answers": list(selected.answers),
        "selected_graph": pipeline._graph_summary(selected.graph),
        "diagnostics": diagnostics,
    }






# ---- failure_gold_sparql_retrieval ----
"""Bounded final-empty retrieval over cached CWQ TRAIN Gold SPARQL.

This module is deliberately isolated from evaluation data.  Runtime inputs are
the current question, reviewed decomposition trace, configured entity links,
and a cache built solely from CWQ TRAIN question/Gold-SPARQL pairs.  It performs
no generative-model call.  The ordinary lane executes at most three final
SELECT queries; only when all three are empty may the optional local-BGE lane
execute at most three different queries (six total for that failure only).
"""



from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Sequence

from .pipeline import _answer_values as _failure_gold_sparql_retrieval_answer_values
from .template_path_retrieval import mask_entities as _failure_gold_sparql_retrieval_mask_entities


_failure_gold_sparql_retrieval_VERSION = "failure-source-train-gold-sparql-production-v4-adaptive-role-quota"
_failure_gold_sparql_retrieval_CACHE_VERSION = 3
_failure_gold_sparql_retrieval_TOKEN_RE = re.compile(r"<entity>|<date>|<year>|<number>|[a-z0-9]+", re.I)
_failure_gold_sparql_retrieval_FULL_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_failure_gold_sparql_retrieval_YEAR_RE = re.compile(r"(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d.])")
_failure_gold_sparql_retrieval_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
_failure_gold_sparql_retrieval_SELECT_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.I)
_failure_gold_sparql_retrieval_ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_failure_gold_sparql_retrieval_RELATION_RE = re.compile(r"\bns:([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)\b")
PLURAL_ANSWER_RE = re.compile(
    r"\b(?:(?:which|what|name|list)\s+(?:are|were)|names|people|actors|"
    r"characters|countries|colleges|movies|films|who\s+(?:are|were))\b",
    re.I,
)
SINGULAR_ANSWER_RE = re.compile(
    r"\b(?:who\s+(?:is|was)|what\s+is\s+the\s+name|"
    r"which\s+(?:person|country|college|movie|film|actor|president|"
    r"prime\s+minister|character)|where\s+did|from\s+which)\b",
    re.I,
)


def _failure_gold_sparql_retrieval_normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _failure_gold_sparql_retrieval_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _failure_gold_sparql_retrieval_tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in _failure_gold_sparql_retrieval_TOKEN_RE.finditer(value)]


def _failure_gold_sparql_retrieval_scalar_slots(text: str) -> list[tuple[str, str]]:
    occupied: list[tuple[int, int]] = []
    output: list[tuple[int, str, str]] = []
    for match in _failure_gold_sparql_retrieval_FULL_DATE_RE.finditer(text):
        occupied.append(match.span())
        output.append((match.start(), "date", match.group(0)))
    for match in _failure_gold_sparql_retrieval_YEAR_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        occupied.append(match.span())
        output.append((match.start(), "year", match.group(0)))
    for match in _failure_gold_sparql_retrieval_NUMBER_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        output.append((match.start(), "number", match.group(0)))
    return [(kind, value) for _, kind, value in sorted(output)]


def _failure_gold_sparql_retrieval_normalize_scalars(text: str) -> str:
    value = _failure_gold_sparql_retrieval_FULL_DATE_RE.sub(" <date> ", str(text))
    value = _failure_gold_sparql_retrieval_YEAR_RE.sub(" <year> ", value)
    value = _failure_gold_sparql_retrieval_NUMBER_RE.sub(" <number> ", value)
    return " ".join(value.split())


def _masked_text(
    question: str,
    decomposition: Sequence[str],
    surfaces: Sequence[str],
) -> str:
    masked_question = _failure_gold_sparql_retrieval_mask_entities(_failure_gold_sparql_retrieval_normalize_scalars(question), surfaces)
    return " ".join(
        [
            masked_question,
            masked_question,
            *(
                _failure_gold_sparql_retrieval_mask_entities(_failure_gold_sparql_retrieval_normalize_scalars(value), surfaces)
                for value in decomposition
                if str(value).strip()
            ),
        ]
    )


def _operator_signature(question: str) -> tuple[str, ...]:
    text = _failure_gold_sparql_retrieval_normalized(question)
    result: list[str] = []
    if re.search(r"\b(?:largest|highest|latest|last|most recent|maximum|greatest)\b", text):
        result.append("argmax")
    if re.search(r"\b(?:smallest|lowest|earliest|first|minimum|least)\b", text):
        result.append("argmin")
    if re.search(r"\b(?:before|earlier than|less than|under)\b", text):
        result.append("lt")
    if re.search(r"\b(?:after|later than|greater than|over|more than)\b", text):
        result.append("gt")
    if re.search(r"\b(?:how many|number of)\b", text):
        result.append("count")
    if (
        _failure_gold_sparql_retrieval_scalar_slots(text)
        and not set(result) & {"argmax", "argmin", "count", "gt", "lt"}
        and re.search(
            r"\b(?:at|during|equal(?:s|\s+to)?|exactly|has|had|having|in|on|with)\b",
            text,
        )
    ):
        result.append("eq")
    return tuple(sorted(set(result)))


def _effective_operator_signature(document: '_failure_gold_sparql_retrieval_Document') -> tuple[str, ...]:
    """Recover scalar equality intent for caches built before ``eq`` existed."""

    return tuple(
        sorted(
            set(map(str, document.operator_signature))
            | set(_operator_signature(document.question))
        )
    )


@dataclass(frozen=True, slots=True)
class _failure_gold_sparql_retrieval_Document:
    source_index: int
    question: str
    decomposition: tuple[str, ...]
    source_surfaces: tuple[str, ...]
    constants: tuple[str, ...]
    constant_roles: tuple[str, ...]
    sparql: str
    answer_var: str
    signature: str
    operator_signature: tuple[str, ...]
    tokens: tuple[str, ...]


def _failure_gold_sparql_retrieval_operator_conflict(source: set[str], target: set[str]) -> bool:
    source_comparison = source & {"eq", "gt", "lt"}
    target_comparison = target & {"eq", "gt", "lt"}
    if (
        source_comparison
        and target_comparison
        and source_comparison != target_comparison
    ):
        return True
    for pair in ({"lt", "gt"}, {"argmin", "argmax"}):
        source_side, target_side = source & pair, target & pair
        if source_side and target_side and source_side != target_side:
            return True
    return False


def _failure_gold_sparql_retrieval_relation_text(sparql: str) -> str:
    relations = _failure_gold_sparql_retrieval_unique(
        relation
        for relation in _failure_gold_sparql_retrieval_RELATION_RE.findall(sparql)
        if not relation.startswith(("m.", "g."))
    )
    return " ; ".join(
        " ".join(relation.replace("_", " ").split("."))
        for relation in relations
    )


def _semantic_document(document: '_failure_gold_sparql_retrieval_Document') -> str:
    """Gold-blind BGE document with entity symbols removed.

    The source question and pseudo decomposition provide paraphrase evidence;
    real relation/constant-role names preserve graph distinctions that vanish
    if every template is represented only by V0/V1-like placeholders.
    """

    return " ".join(
        (
            "passage:",
            _masked_text(
                document.question,
                document.decomposition,
                document.source_surfaces,
            ),
            "relation path:",
            _failure_gold_sparql_retrieval_relation_text(document.sparql),
            "constant roles:",
            " ; ".join(document.constant_roles),
        )
    )


def _cosine_vectors(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / denominator if denominator else 0.0


class SourceGoldSparqlIndex:
    """Read-only cached index with sparse BM25 postings."""

    def __init__(self, cache_path: str | Path) -> None:
        started = time.perf_counter()
        source = Path(cache_path).expanduser()
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _failure_gold_sparql_retrieval_CACHE_VERSION
            or not isinstance(payload.get("documents"), list)
        ):
            raise ValueError(f"incompatible source Gold SPARQL cache: {source}")
        self.documents = [
            _failure_gold_sparql_retrieval_Document(
                source_index=int(item["source_index"]),
                question=str(item["question"]),
                decomposition=tuple(map(str, item.get("decomposition", []))),
                source_surfaces=tuple(map(str, item.get("source_surfaces", []))),
                constants=tuple(map(str, item.get("constants", []))),
                constant_roles=tuple(map(str, item.get("constant_roles", []))),
                sparql=str(item["sparql"]),
                answer_var=str(item.get("answer_var", "x")),
                signature=str(item.get("signature", "")),
                operator_signature=tuple(map(str, item.get("operator_signature", []))),
                tokens=tuple(map(str, item.get("tokens", []))),
            )
            for item in payload["documents"]
            if isinstance(item, dict)
        ]
        self.frequencies: list[Counter[str]] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for index, document in enumerate(self.documents):
            frequency = Counter(document.tokens)
            self.frequencies.append(frequency)
            total_length += len(document.tokens)
            document_frequency.update(set(frequency))
            for token, count in frequency.items():
                self.postings[token].append((index, count))
        count = max(1, len(self.documents))
        self.average_length = total_length / count
        self.idf = {
            token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }
        self.load_seconds = time.perf_counter() - started

    def retrieve(
        self,
        *,
        question: str,
        decomposition: Sequence[str],
        surfaces: Sequence[str],
        entity_count: int,
        top_k: int = 32,
    ) -> list[dict[str, Any]]:
        query = Counter(_failure_gold_sparql_retrieval_tokens(_masked_text(question, decomposition, surfaces)))
        scores: dict[int, float] = defaultdict(float)
        for token, query_frequency in query.items():
            inverse = self.idf.get(token, 0.0)
            for index, frequency in self.postings.get(token, ()):
                document = self.documents[index]
                if len(document.constants) != entity_count:
                    continue
                normalization = 0.25 + 0.75 * len(document.tokens) / self.average_length
                scores[index] += inverse * (
                    frequency * 2.2 / (frequency + 1.2 * normalization)
                ) * min(query_frequency, 2)
        target_operator = set(_operator_signature(question))
        ranked: list[tuple[float, int]] = []
        for index, score in scores.items():
            document = self.documents[index]
            source_operator = set(_effective_operator_signature(document))
            if _failure_gold_sparql_retrieval_operator_conflict(source_operator, target_operator):
                continue
            if target_operator and source_operator == target_operator:
                score += 1.0
            elif target_operator and source_operator and source_operator != target_operator:
                score -= 2.0
            ranked.append((score, index))
        ranked.sort(key=lambda item: (-item[0], self.documents[item[1]].source_index))
        support = Counter(
            self.documents[index].signature for _, index in ranked[:64]
        )
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, (score, index) in enumerate(ranked[:128], start=1):
            document = self.documents[index]
            if document.signature in seen:
                continue
            seen.add(document.signature)
            output.append(
                {
                    "rank": rank,
                    "bm25": score,
                    "support": support[document.signature],
                    "document": document,
                }
            )
            if len(output) >= max(3, min(64, int(top_k))):
                break
        return output


def _failure_gold_sparql_retrieval_token_similarity(left: str, right: str) -> float:
    one = set(_failure_gold_sparql_retrieval_tokens(left)) - {"entity", "date", "year", "number"}
    two = set(_failure_gold_sparql_retrieval_tokens(right)) - {"entity", "date", "year", "number"}
    return len(one & two) / len(one | two) if one and two else 0.0


def _local_context(label: str, text: str, radius: int = 14) -> str:
    words = list(re.finditer(r"[a-z0-9]+", _failure_gold_sparql_retrieval_normalized(text)))
    label_tokens = _failure_gold_sparql_retrieval_tokens(label)
    if not words or not label_tokens:
        return text
    normalized = _failure_gold_sparql_retrieval_normalized(text)
    position = normalized.find(_failure_gold_sparql_retrieval_normalized(label))
    if position < 0:
        return " ".join(label_tokens)
    center = next((i for i, word in enumerate(words) if word.start() >= position), 0)
    return " ".join(
        word.group(0)
        for word in words[
            max(0, center - radius) : min(
                len(words), center + len(label_tokens) + radius
            )
        ]
    )


def _semantic_context(
    label: str,
    question: str,
    decomposition: Sequence[str],
) -> str:
    needle = _failure_gold_sparql_retrieval_normalized(label)
    clauses = [
        str(value)
        for value in decomposition
        if needle and needle in _failure_gold_sparql_retrieval_normalized(value)
    ]
    return " ".join(clauses) if clauses else _local_context(label, question)


def _mapping_candidates(
    document: _failure_gold_sparql_retrieval_Document,
    entities: Sequence[tuple[str, str]],
    question: str,
    decomposition: Sequence[str],
) -> list[tuple[float, tuple[tuple[str, str], ...]]]:
    surfaces = list(document.source_surfaces[: len(document.constants)])
    surfaces.extend([""] * (len(document.constants) - len(surfaces)))
    target = list(entities)
    output: list[tuple[float, tuple[tuple[str, str], ...]]] = []
    for assigned in permutations(target, len(document.constants)):
        score = 0.0
        for position, ((_, label), source_surface) in enumerate(zip(assigned, surfaces)):
            if assigned[position] == target[position]:
                score += 0.10
            score += 0.20 * _failure_gold_sparql_retrieval_token_similarity(source_surface, label)
            role = document.constant_roles[position] if position < len(document.constant_roles) else ""
            score += 2.0 * _failure_gold_sparql_retrieval_token_similarity(
                role, _semantic_context(label, question, decomposition)
            )
        output.append((score, tuple(assigned)))
    output.sort(key=lambda item: (-item[0], tuple(value[0] for value in item[1])))
    return output


def _failure_gold_sparql_retrieval_replace_scalars(
    sparql: str,
    source_question: str,
    target_question: str,
) -> tuple[str, bool]:
    source, target = _failure_gold_sparql_retrieval_scalar_slots(source_question), _failure_gold_sparql_retrieval_scalar_slots(target_question)
    if Counter(kind for kind, _ in source) != Counter(kind for kind, _ in target):
        return sparql, not source
    source_by_kind: dict[str, list[str]] = defaultdict(list)
    target_by_kind: dict[str, list[str]] = defaultdict(list)
    for kind, value in source:
        source_by_kind[kind].append(value)
    for kind, value in target:
        target_by_kind[kind].append(value)
    result = sparql
    for kind, source_values in source_by_kind.items():
        for old, new in zip(source_values, target_by_kind[kind]):
            if kind == "year":
                result = result.replace(f'"{old}-01-01"', f'"{new}-01-01"')
                result = result.replace(f'"{old}-12-31"', f'"{new}-12-31"')
            result = re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d.])", new, result)
    return result, True


def _rewrite_numeric_equality_literal(
    query: str,
    *,
    question: str,
    source_signature: Sequence[str],
) -> tuple[str, int]:
    """Lower one quoted numeric equality like the main SPARQL compiler."""

    target_signature = set(_operator_signature(question))
    if target_signature != {"eq"} or set(source_signature) != {"eq"}:
        return query, 0
    numeric_slots = [
        value for kind, value in _failure_gold_sparql_retrieval_scalar_slots(question) if kind == "number"
    ]
    if len(numeric_slots) != 1:
        return query, 0
    raw_value = numeric_slots[0]
    try:
        numeric_value = float(raw_value)
    except ValueError:
        return query, 0
    variable = "?numeric_equal_0"
    if variable in query:
        return query, 0
    literal_pattern = re.compile(
        rf'(?P<prefix>(?:\?[A-Za-z_][A-Za-z0-9_]*|ns:[A-Za-z0-9_.]+)\s+'
        rf'ns:[A-Za-z][A-Za-z0-9_.]+\s+)"{re.escape(raw_value)}"'
        rf'(?:\^\^[^\s.]+)?\s*\.',
        re.I,
    )
    tolerance = max(0.000001, abs(numeric_value) * 0.000001)
    cast = "bif:atoi" if re.fullmatch(r"[-+]?\d+", raw_value) else "bif:atof"
    replacement = (
        rf"\g<prefix>{variable} .\n"
        f"FILTER (ABS({cast}(STR({variable})) - {numeric_value:.15g}) "
        f"< {tolerance:.15g}) ."
    )
    rewritten, count = literal_pattern.subn(replacement, query)
    return (rewritten, count) if count == 1 else (query, 0)


def _instantiate(
    match: dict[str, Any],
    entities: Sequence[tuple[str, str]],
    question: str,
    decomposition: Sequence[str],
) -> list[dict[str, Any]]:
    document: _failure_gold_sparql_retrieval_Document = match["document"]
    source_signature = _effective_operator_signature(document)
    target_signature = _operator_signature(question)
    if _failure_gold_sparql_retrieval_operator_conflict(set(source_signature), set(target_signature)):
        return []
    output: list[dict[str, Any]] = []
    for mapping_rank, (mapping_score, assigned) in enumerate(
        _mapping_candidates(document, entities, question, decomposition), start=1
    ):
        query, compatible = _failure_gold_sparql_retrieval_replace_scalars(document.sparql, document.question, question)
        if not compatible:
            continue
        query, numeric_rewrite_count = _rewrite_numeric_equality_literal(
            query,
            question=question,
            source_signature=source_signature,
        )
        mapping: dict[str, str] = {}
        for source_id, (target_id, _) in zip(document.constants, assigned):
            mapping[source_id] = target_id
            query = re.sub(
                rf"(?<![A-Za-z0-9_])ns:{re.escape(source_id)}(?![A-Za-z0-9_])",
                f"ns:{target_id}",
                query,
            )
        if not _failure_gold_sparql_retrieval_SELECT_LIMIT_RE.search(query):
            query = f"{query.strip()}\nLIMIT 256"
        output.append(
            {
                "key": hashlib.sha256(query.encode()).hexdigest()[:20],
                "query": query,
                "source_index": document.source_index,
                "source_template_rank": int(match["rank"]),
                "template_signature": document.signature,
                "mapping_rank": mapping_rank,
                "mapping_score": mapping_score,
                "answer_var": document.answer_var,
                "runtime_score": (
                    float(match["bm25"])
                    + 0.30 * math.log1p(int(match["support"]))
                    + 0.35 * mapping_score
                    - 0.04 * (mapping_rank - 1)
                ),
                "entity_mapping": mapping,
                "source_effective_constraint_signature": list(source_signature),
                "target_constraint_signature": list(target_signature),
                "constraint_signature_compatible": True,
                "numeric_equality_cast_rewrite_count": numeric_rewrite_count,
            }
        )
    return output


def _semantic_fallback_candidates(
    *,
    matches: Sequence[dict[str, Any]],
    entities: Sequence[tuple[str, str]],
    question: str,
    decomposition: Sequence[str],
    ranker: Any,
    executed_queries: set[str],
    limit: int,
) -> tuple[list[dict[str, Any]], str, int, int]:
    """Rerank unexecuted TRAIN templates after the ordinary cap three is empty.

    This helper never executes a query.  It uses the already configured local
    embedding ranker, removes entity identifiers/surfaces from similarity
    text, and reuses the same mapping-aware allocator as the ordinary lane.
    Returning an empty list is a safe abstention when embeddings are absent or
    malformed.
    """

    query_text = "query: " + _masked_text(
        question,
        decomposition,
        [label for _, label in entities],
    )
    documents = [_semantic_document(match["document"]) for match in matches]
    scores = list(ranker.score(query_text, documents))
    if len(scores) != len(matches):
        raise ValueError("failure semantic reranker returned the wrong score count")
    semantic_matches: list[dict[str, Any]] = []
    for match, score in zip(matches, scores):
        candidate_match = dict(match)
        candidate_match["embedding_score"] = float(score)
        # Preserve the existing support/mapping tie breakers while making BGE
        # the primary order.  This exact fixed scale is frozen by the full-80
        # production-context audit.
        candidate_match["bm25"] = 100.0 * float(score)
        semantic_matches.append(candidate_match)
    semantic_matches.sort(
        key=lambda item: (
            -float(item["embedding_score"]),
            int(item["rank"]),
            int(item["document"].source_index),
        )
    )
    embedding_calls = 2
    embedded_text_count = len(documents) + 1
    role_vectors: dict[str, list[float]] = {}
    entity_context_vectors: list[list[float]] = []
    embedding_client = getattr(ranker, "client", None)
    if callable(getattr(embedding_client, "embed", None)):
        entity_contexts = [
            "entity semantic context: "
            + _semantic_context(label, question, decomposition)
            for _, label in entities
        ]
        role_texts = _failure_gold_sparql_retrieval_unique(
            "entity relation role: " + role
            for match in semantic_matches
            for role in match["document"].constant_roles
        )
        entity_context_vectors = list(
            embedding_client.embed(entity_contexts, input_type="query")
        )
        role_embeddings = list(
            embedding_client.embed(role_texts, input_type="document")
        )
        if len(entity_context_vectors) != len(entity_contexts) or len(
            role_embeddings
        ) != len(role_texts):
            raise ValueError("failure relation-role embeddings have wrong shape")
        role_vectors = dict(zip(role_texts, role_embeddings))
        embedding_calls += 2
        embedded_text_count += len(entity_contexts) + len(role_texts)
    candidates: list[dict[str, Any]] = []
    for match in semantic_matches:
        instantiated = _instantiate(match, entities, question, decomposition)
        for candidate in instantiated:
            candidate["embedding_score"] = float(match["embedding_score"])
            if role_vectors:
                document = match["document"]
                entity_positions = {
                    entity_id: position
                    for position, (entity_id, _) in enumerate(entities)
                }
                alignments: list[float] = []
                for position, source_id in enumerate(document.constants):
                    target_id = candidate["entity_mapping"].get(source_id, "")
                    target_position = entity_positions.get(target_id)
                    role = (
                        document.constant_roles[position]
                        if position < len(document.constant_roles)
                        else ""
                    )
                    vector = role_vectors.get("entity relation role: " + role)
                    if target_position is not None and vector is not None:
                        alignments.append(
                            _cosine_vectors(
                                entity_context_vectors[target_position], vector
                            )
                        )
                candidate["role_score"] = (
                    sum(alignments) / len(alignments) if alignments else 0.0
                )
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -float(item["runtime_score"]),
            int(item["source_template_rank"]),
            int(item["mapping_rank"]),
            str(item["key"]),
        )
    )
    deduplicated: list[dict[str, Any]] = []
    seen = set(executed_queries)
    for candidate in candidates:
        normalized_query = " ".join(str(candidate["query"]).split())
        if normalized_query in seen:
            continue
        seen.add(normalized_query)
        deduplicated.append(candidate)
    query_limit = min(3, max(1, int(limit)))
    if role_vectors and query_limit >= 3:
        allocated = list(deduplicated[:2])
        allocated_keys = {str(candidate["key"]) for candidate in allocated}
        role_candidates = sorted(
            (
                candidate
                for candidate in deduplicated
                if str(candidate["key"]) not in allocated_keys
            ),
            key=lambda item: (
                -float(item.get("role_score", 0.0)),
                -float(item["runtime_score"]),
                int(item["source_template_rank"]),
                int(item["mapping_rank"]),
                str(item["key"]),
            ),
        )
        if role_candidates:
            allocated.append(role_candidates[0])
        allocation_reason = "bge_top2_plus_relation_role_top1"
    else:
        allocated, allocation_reason = _allocate_cap_three(
            deduplicated,
            entity_count=len(entities),
            limit=query_limit,
        )
    return (
        allocated,
        allocation_reason,
        embedded_text_count,
        embedding_calls,
    )


def _allocate_cap_three(
    candidates: Sequence[dict[str, Any]],
    *,
    entity_count: int,
    limit: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    """Allocate the fixed SELECT budget without evaluation-time evidence.

    A confident slot mapping should not let its alternative permutations fill
    the whole budget.  Conversely, nearly tied mappings are real ambiguity and
    retain the original score order.  The thresholds are absolute because the
    mapping score itself is a bounded sum of fixed lexical/role similarities.
    """

    query_limit = min(3, max(1, int(limit)))
    if not candidates:
        return [], "no_candidates"
    first_signature = str(candidates[0].get("template_signature", ""))
    first_template = [
        candidate
        for candidate in candidates
        if str(candidate.get("template_signature", "")) == first_signature
    ]
    if entity_count >= 3 and len(first_template) >= 3:
        first_to_third_gap = float(first_template[0].get("mapping_score", 0.0)) - float(
            first_template[2].get("mapping_score", 0.0)
        )
        if first_to_third_gap <= 0.20:
            return list(candidates[:query_limit]), "flat_multi_entity_mapping"
    if entity_count > 1 and len(first_template) >= 2:
        mapping_margin = float(first_template[0].get("mapping_score", 0.0)) - float(
            first_template[1].get("mapping_score", 0.0)
        )
        if mapping_margin < 0.50:
            return list(candidates[:query_limit]), "ambiguous_entity_mapping"
    selected: list[dict[str, Any]] = []
    seen_templates: set[str] = set()
    for candidate in candidates:
        signature = str(candidate.get("template_signature", ""))
        if signature in seen_templates:
            continue
        seen_templates.add(signature)
        selected.append(candidate)
        if len(selected) >= query_limit:
            break
    return selected, "confident_mapping_template_diversity"


def _expects_single_answer(question: str) -> bool:
    return not PLURAL_ANSWER_RE.search(question) and bool(
        SINGULAR_ANSWER_RE.search(question)
    )


def _failure_gold_sparql_retrieval_choose_runtime(
    candidates: Sequence[dict[str, Any]], question: str
) -> dict[str, Any] | None:
    nonempty = [candidate for candidate in candidates if candidate.get("answers")]
    if not nonempty:
        return None
    support = Counter(tuple(sorted(candidate["answers"])) for candidate in nonempty)
    singular = _expects_single_answer(question)
    return max(
        nonempty,
        key=lambda item: (
            support[tuple(sorted(item["answers"]))],
            int(singular and len(item["answers"]) == 1),
            -abs(len(item["answers"]) - 1) if singular else 0,
            float(item["runtime_score"]),
            -int(item["source_template_rank"]),
            str(item["key"]),
        ),
    )


def _failure_gold_sparql_retrieval_decompositions(context: Any) -> list[str]:
    traces = (context.trace_bundle.get("decomposition", {}) or {}).get("traces", [])
    for trace in reversed(traces if isinstance(traces, list) else []):
        if isinstance(trace, dict) and trace.get("stage") == "decomposition_review_and_rewrite":
            output = trace.get("output", {})
            values = [
                str(item)
                for candidate in output.get("final_candidates", []) if isinstance(output, dict) and isinstance(candidate, dict)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return _failure_gold_sparql_retrieval_unique(values)
    for trace in reversed(traces if isinstance(traces, list) else []):
        if isinstance(trace, dict) and trace.get("stage") == "semantic_path_generation":
            output = trace.get("output", [])
            values = [
                str(item)
                for candidate in output if isinstance(output, list) and isinstance(candidate, dict)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return _failure_gold_sparql_retrieval_unique(values)
    for trace in reversed(traces if isinstance(traces, list) else []):
        if isinstance(trace, dict) and trace.get("stage") == "final_graph_beam":
            output = trace.get("output", [])
            values = [
                str(item)
                for graph in output if isinstance(output, list) and isinstance(graph, dict)
                for item in (graph.get("provenance") or {}).get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return _failure_gold_sparql_retrieval_unique(values)
    # Strict trace replay can contain the already-reviewed hypotheses under
    # ``decompose_predictions`` without a separate review stage.  Prefer that
    # executed evidence over an absent/empty prediction store.
    for trace in reversed(traces if isinstance(traces, list) else []):
        if isinstance(trace, dict) and trace.get("stage") == "decompose_predictions":
            output = trace.get("output", [])
            values = [
                str(item)
                for candidate in output if isinstance(output, list) and isinstance(candidate, dict)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return _failure_gold_sparql_retrieval_unique(values)
    try:
        candidates = context.pipeline.decompositions.get(context.question)
    except (AttributeError, KeyError):
        candidates = []
    return _failure_gold_sparql_retrieval_unique(
        str(item)
        for candidate in candidates
        for item in getattr(candidate, "decomposition", [])
        if str(item).strip()
    )


def _failure_gold_sparql_retrieval_trace_surfaces(context: Any) -> list[str]:
    traces = (context.trace_bundle.get("decomposition", {}) or {}).get("traces", [])
    surfaces: list[str] = []
    for trace in traces if isinstance(traces, list) else []:
        if not isinstance(trace, dict) or trace.get("stage") != "semantic_path_generation":
            continue
        output = trace.get("output", [])
        current = [
            str(anchor.get("surface", ""))
            for candidate in output if isinstance(output, list) and isinstance(candidate, dict)
            for anchor in (candidate.get("output") or {}).get("anchors", [])
            if isinstance(anchor, dict) and str(anchor.get("surface", "")).strip()
        ]
        if current:
            surfaces = _failure_gold_sparql_retrieval_unique(current)
    return surfaces


def _failure_gold_sparql_retrieval_entities(context: Any) -> list[tuple[str, str]]:
    mapping = getattr(context.pipeline.grounder, "gold_entities", {}).get(
        _failure_gold_sparql_retrieval_normalized(context.question), {}
    )
    items = [(str(entity_id), str(label)) for entity_id, label in mapping.items()]
    normalized = _failure_gold_sparql_retrieval_normalized(context.question)
    positions = {entity_id: index for index, (entity_id, _) in enumerate(items)}
    return sorted(
        items,
        key=lambda item: (
            normalized.find(item[1].casefold())
            if item[1].casefold() in normalized
            else len(normalized) + positions[item[0]],
            positions[item[0]],
        ),
    )


def _failure_gold_sparql_retrieval_index(pipeline: Any) -> SourceGoldSparqlIndex:
    value = getattr(pipeline, "_failure_gold_sparql_index", None)
    if value is not None:
        return value
    lock = getattr(pipeline, "_failure_gold_sparql_index_lock", None)
    if lock is None:
        value = SourceGoldSparqlIndex(pipeline.failure_gold_sparql_cache_path)
        pipeline._failure_gold_sparql_index = value
        return value
    with lock:
        value = getattr(pipeline, "_failure_gold_sparql_index", None)
        if value is None:
            value = SourceGoldSparqlIndex(pipeline.failure_gold_sparql_cache_path)
            pipeline._failure_gold_sparql_index = value
    return value


def _failure_gold_sparql_retrieval_retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    started = time.perf_counter()
    pipeline = context.pipeline
    diagnostics: dict[str, Any] = {
        "version": _failure_gold_sparql_retrieval_VERSION,
        "status": "abstained",
        "trigger": "final_empty_after_existing_failure_lanes",
        "uses_evaluation_gold": False,
        "uses_test_sparql_or_reasoning_information": False,
        "source": "CWQ_TRAIN_gold_sparql_only",
        "llm_calls": 0,
        "embedding_calls": 0,
        "candidate_query_limit": 3,
        "entity_or_relation_whitelist": False,
    }
    entities = _failure_gold_sparql_retrieval_entities(context)
    decomposition = _failure_gold_sparql_retrieval_decompositions(context)
    if not entities:
        diagnostics["reason"] = "no_linked_entities"
        return {"answer_ids": [], "diagnostics": diagnostics}
    index = _failure_gold_sparql_retrieval_index(pipeline)
    matches = index.retrieve(
        question=context.question,
        decomposition=decomposition,
        # Only grounded entities are masks.  Semantic traces can also expose
        # type/category anchors (for example "Country") which are useful
        # retrieval words, not interchangeable entity slots; masking them
        # erases precisely the template distinction this lane needs.
        surfaces=_failure_gold_sparql_retrieval_unique(label for _, label in entities),
        entity_count=len(entities),
        top_k=int(getattr(pipeline, "failure_gold_sparql_top_k", 32)),
    )
    candidates: list[dict[str, Any]] = []
    for match in matches:
        candidates.extend(_instantiate(match, entities, context.question, decomposition))
    candidates.sort(
        key=lambda item: (
            -float(item["runtime_score"]),
            int(item["source_template_rank"]),
            int(item["mapping_rank"]),
            str(item["key"]),
        )
    )
    deduplicated_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized_query = " ".join(candidate["query"].split())
        if normalized_query in seen:
            continue
        seen.add(normalized_query)
        deduplicated_candidates.append(candidate)
    selected_candidates, allocation_reason = _allocate_cap_three(
        deduplicated_candidates,
        entity_count=len(entities),
        limit=int(getattr(pipeline, "failure_gold_sparql_candidate_limit", 3)),
    )
    base_execution_queries = 0
    for candidate in selected_candidates:
        query_started = time.perf_counter()
        try:
            rows = endpoint.execute(candidate["query"])
            answers = _failure_gold_sparql_retrieval_unique(_failure_gold_sparql_retrieval_answer_values(rows, candidate["answer_var"]))
            error = ""
        except Exception as exc:
            answers = []
            error = f"{type(exc).__name__}:{exc}"
        base_execution_queries += 1
        candidate["answers"] = answers
        candidate["error"] = error
        candidate["elapsed_seconds"] = time.perf_counter() - query_started
    selected = _failure_gold_sparql_retrieval_choose_runtime(selected_candidates, context.question)
    adaptive_candidates: list[dict[str, Any]] = []
    adaptive_allocation_reason = "not_triggered"
    adaptive_execution_queries = 0
    adaptive_embedding_calls = 0
    adaptive_embedded_texts = 0
    adaptive_error = ""
    if (
        selected is None
        and bool(
            getattr(
                pipeline,
                "failure_gold_sparql_adaptive_semantic_enabled",
                False,
            )
        )
    ):
        ranker = getattr(getattr(pipeline, "grounder", None), "ranker", None)
        if ranker is None or not callable(getattr(ranker, "score", None)):
            adaptive_error = "embedding_ranker_unavailable"
        else:
            executed_queries = {
                " ".join(str(candidate["query"]).split())
                for candidate in selected_candidates
            }
            try:
                (
                    adaptive_candidates,
                    adaptive_allocation_reason,
                    adaptive_embedded_texts,
                    adaptive_embedding_calls,
                ) = _semantic_fallback_candidates(
                    matches=matches,
                    entities=entities,
                    question=context.question,
                    decomposition=decomposition,
                    ranker=ranker,
                    executed_queries=executed_queries,
                    limit=int(
                        getattr(
                            pipeline,
                            "failure_gold_sparql_adaptive_candidate_limit",
                            3,
                        )
                    ),
                )
                for candidate in adaptive_candidates:
                    query_started = time.perf_counter()
                    try:
                        rows = endpoint.execute(candidate["query"])
                        answers = _failure_gold_sparql_retrieval_unique(
                            _failure_gold_sparql_retrieval_answer_values(rows, candidate["answer_var"])
                        )
                        error = ""
                    except Exception as exc:
                        answers = []
                        error = f"{type(exc).__name__}:{exc}"
                    adaptive_execution_queries += 1
                    candidate["answers"] = answers
                    candidate["error"] = error
                    candidate["elapsed_seconds"] = (
                        time.perf_counter() - query_started
                    )
                selected = _failure_gold_sparql_retrieval_choose_runtime(adaptive_candidates, context.question)
            except Exception as exc:
                adaptive_error = f"{type(exc).__name__}:{exc}"

    execution_queries = base_execution_queries + adaptive_execution_queries
    adaptive_triggered = adaptive_embedding_calls > 0
    selected_from_adaptive = bool(
        selected is not None and selected in adaptive_candidates
    )

    def candidate_trace(candidate: dict[str, Any], lane: str) -> dict[str, Any]:
        trace = {
            "lane": lane,
            "key": candidate["key"],
            "query_sha256": hashlib.sha256(candidate["query"].encode()).hexdigest(),
            "source_index": int(candidate["source_index"]),
            "source_template_rank": int(candidate["source_template_rank"]),
            "mapping_rank": int(candidate["mapping_rank"]),
            "answer_count": len(candidate["answers"]),
            "elapsed_seconds": candidate["elapsed_seconds"],
            "error": candidate["error"],
        }
        if "embedding_score" in candidate:
            trace["embedding_score"] = float(candidate["embedding_score"])
        if "role_score" in candidate:
            trace["role_score"] = float(candidate["role_score"])
        return trace

    diagnostics.update(
        {
            "status": (
                "selected_adaptive_semantic"
                if selected_from_adaptive
                else ("selected" if selected else "all_candidates_empty")
            ),
            "reason": (
                "source_train_gold_sparql_adaptive_semantic_answer_consensus"
                if selected_from_adaptive
                else (
                    "source_train_gold_sparql_answer_consensus"
                    if selected
                    else "all_instantiated_queries_empty"
                )
            ),
            "embedding_calls": adaptive_embedding_calls,
            "embedded_text_count": adaptive_embedded_texts,
            "adaptive_semantic_triggered": adaptive_triggered,
            "adaptive_semantic_error": adaptive_error,
            "adaptive_candidate_query_limit": 3,
            "total_candidate_query_limit": (
                6
                if bool(
                    getattr(
                        pipeline,
                        "failure_gold_sparql_adaptive_semantic_enabled",
                        False,
                    )
                )
                else 3
            ),
            "base_endpoint_execution_queries": base_execution_queries,
            "adaptive_endpoint_execution_queries": adaptive_execution_queries,
            "adaptive_candidate_allocation": adaptive_allocation_reason,
            "adaptive_answer_count": (
                len(selected["answers"]) if selected_from_adaptive else 0
            ),
            "selection_lane": (
                "adaptive_semantic" if selected_from_adaptive else "ordinary"
            ),
            "index_size": len(index.documents),
            "index_load_seconds": index.load_seconds,
            "retrieved_template_count": len(matches),
            "candidate_allocation": allocation_reason,
            "endpoint_execution_queries": execution_queries,
            "answer_count": len(selected["answers"]) if selected else 0,
            "selected_source_index": int(selected["source_index"]) if selected else -1,
            "selected_source_template_rank": int(selected["source_template_rank"]) if selected else -1,
            "selected_mapping_rank": int(selected["mapping_rank"]) if selected else -1,
            "candidate_traces": [
                *(
                    candidate_trace(candidate, "ordinary")
                    for candidate in selected_candidates
                ),
                *(
                    candidate_trace(candidate, "adaptive_semantic")
                    for candidate in adaptive_candidates
                ),
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    answer_ids = list(selected["answers"]) if selected else []
    label_rows: list[dict[str, str]] = []
    label_error = ""
    entity_answer_ids = [
        answer_id for answer_id in answer_ids if _failure_gold_sparql_retrieval_ENTITY_ID_RE.fullmatch(answer_id)
    ]
    if entity_answer_ids:
        try:
            label_rows = list(pipeline.kg.labels(entity_answer_ids))
        except Exception as exc:
            label_error = f"{type(exc).__name__}:{exc}"
    labels_by_id = {
        str(item.get("id", "")): str(item.get("label", ""))
        for item in label_rows
        if isinstance(item, dict) and str(item.get("id", ""))
    }
    diagnostics["selected_label_queries"] = int(bool(entity_answer_ids))
    if label_error:
        diagnostics["selected_label_error"] = label_error
    return {
        "answer_ids": answer_ids,
        "answers": [
            {"id": value, "label": labels_by_id.get(value, value)}
            for value in answer_ids
        ],
        "selected_graph": None,
        "diagnostics": diagnostics,
    }






# ---- missing_relation_retrieval ----
"""Bounded training/ontology retrieval for a missing answer-edge relation.

This lane is deliberately independent from graph selection and every chat
model.  It edits one relation incident to the incumbent answer variable,
executes at most three candidates, and applies runtime-only confidence gates.
It can be invoked after a final ``EMPTY_RESULT`` or as a conservative
challenger to an existing successful answer.
"""



from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import heapq
import json
import math
import re
import time
from typing import Any, Iterable, Sequence

from .contracts import QueryGraphCandidate as _missing_relation_retrieval_QueryGraphCandidate
from .lowering import LoweringError as _missing_relation_retrieval_LoweringError, lower_sparql as _missing_relation_retrieval_lower_sparql
from .ontology import FreebaseOntology as _missing_relation_retrieval_FreebaseOntology, relation_id_from_label as _missing_relation_retrieval_relation_id_from_label
from .template_path_retrieval import mask_entities as _missing_relation_retrieval_mask_entities





_missing_relation_retrieval_TOKEN_RE = re.compile(r"<entity>|[a-z0-9]+", re.IGNORECASE)


def _missing_relation_retrieval_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _missing_relation_retrieval_TOKEN_RE.finditer(str(text))]


def _missing_relation_retrieval_relation_text(relation_id: str) -> str:
    return " ".join(
        part.replace("_", " ") for part in str(relation_id).split(".") if part
    )


def _leaf_tokens(relation_id: str) -> set[str]:
    return set(_missing_relation_retrieval_tokens(str(relation_id).rsplit(".", 1)[-1].replace("_", " ")))


def _missing_relation_retrieval_answer_values(rows: list[dict[str, str]], answer_var: str) -> list[str]:
    key = str(answer_var).lstrip("?")
    output: list[str] = []
    for row in rows:
        value = row.get(key, row.get("answer", ""))
        if not value:
            value = next(
                (
                    candidate
                    for name, candidate in row.items()
                    if str(name).casefold() == key.casefold()
                ),
                "",
            )
        value = str(value).strip()
        prefix = "http://rdf.freebase.com/ns/"
        if value.startswith(prefix):
            value = value[len(prefix) :]
        if value and value not in output:
            output.append(value)
    return output


def _missing_relation_retrieval_trace_outputs(traces: Sequence[dict[str, Any]], stage: str) -> Iterable[Any]:
    for trace in traces:
        if isinstance(trace, dict) and trace.get("stage") == stage:
            yield trace.get("output")


def _missing_relation_retrieval_decompositions(traces: Sequence[dict[str, Any]]) -> list[str]:
    reviewed: list[str] = []
    for output in _missing_relation_retrieval_trace_outputs(traces, "decomposition_review_and_rewrite"):
        if not isinstance(output, dict):
            continue
        values = [
            str(item)
            for candidate in output.get("final_candidates", [])
            if isinstance(candidate, dict)
            for item in candidate.get("decomposition", [])
            if str(item).strip()
        ]
        if values:
            reviewed = values
    if reviewed:
        return list(dict.fromkeys(reviewed))
    values: list[str] = []
    for output in _missing_relation_retrieval_trace_outputs(traces, "decompose_predictions"):
        for candidate in output if isinstance(output, list) else []:
            if isinstance(candidate, dict):
                values.extend(
                    str(item)
                    for item in candidate.get("decomposition", [])
                    if str(item).strip()
                )
    return list(dict.fromkeys(values))


def _missing_relation_retrieval_anchor_surfaces(traces: Sequence[dict[str, Any]]) -> list[str]:
    for output in _missing_relation_retrieval_trace_outputs(traces, "semantic_path_generation"):
        for candidate in output if isinstance(output, list) else []:
            graph = candidate.get("output", {}) if isinstance(candidate, dict) else {}
            values = [
                str(item.get("surface", ""))
                for item in graph.get("anchors", [])
                if isinstance(item, dict) and str(item.get("surface", "")).strip()
            ]
            if values:
                return list(dict.fromkeys(values))
    return []


def _grounded_union(traces: Sequence[dict[str, Any]]) -> Counter[str]:
    output: Counter[str] = Counter()
    for values in _missing_relation_retrieval_trace_outputs(traces, "grounded_semantic_candidates"):
        for candidate in values if isinstance(values, list) else []:
            if not isinstance(candidate, dict):
                continue
            current = Counter(
                str(value)
                for value in (candidate.get("relation_bindings") or {}).values()
                if str(value)
            )
            output |= current
    return output


def _final_incumbent(traces: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for output in _missing_relation_retrieval_trace_outputs(traces, "final_graph_beam"):
        if isinstance(output, list):
            candidates = [value for value in output if isinstance(value, dict)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda value: (
            float(value.get("score", 0.0)),
            -len(value.get("triples", [])),
            str(value.get("graph_id", "")),
        ),
    )


def _missing_relation_retrieval_hard_post_selection_applied(traces: Sequence[dict[str, Any]]) -> bool:
    """Preserve every zero-cost hard-gate decision made by normal selection."""

    latest: dict[str, Any] = {}
    for output in _missing_relation_retrieval_trace_outputs(traces, "glm_final_graph_ranking"):
        if isinstance(output, dict):
            latest = output
    reason_codes = {str(value) for value in latest.get("reason_codes", [])}
    return (
        any(value.startswith("hard_") for value in reason_codes)
        or "morphological_underanswer_expansion_gate" in reason_codes
        or any(str(key).startswith("hard_") for key in latest)
    )


@dataclass(slots=True)
class _missing_relation_retrieval_Document:
    source_index: int
    length: int
    relations: Counter[str]
    answer_incidence: Counter[str]
    triple_count: int
    # Retain the aligned Compose topology in the shared cache.  The narrow
    # extrema-relation lane can then reuse this index instead of building a
    # second training index or invoking any model.
    triples: tuple[tuple[str, str, str], ...]
    answer_var: str
    tokens: list[str]


class _missing_relation_retrieval_AlignedRelationIndex:
    def __init__(
        self,
        semantic_examples: Sequence[dict[str, Any]],
        compose_examples: Sequence[dict[str, Any]],
    ) -> None:
        self.documents: list[_missing_relation_retrieval_Document] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for source_index, (semantic, compose) in enumerate(
            zip(semantic_examples, compose_examples)
        ):
            triples: list[tuple[str, str, str]] = []
            for triple in compose.get("triples", []) if isinstance(compose, dict) else []:
                if not isinstance(triple, dict):
                    continue
                relation_id = _missing_relation_retrieval_relation_id_from_label(triple.get("relation_label", []))
                if relation_id:
                    triples.append(
                        (
                            str(triple.get("subject", "")),
                            relation_id,
                            str(triple.get("object", "")),
                        )
                    )
            answer_var = str(compose.get("answer_var", "")) if isinstance(compose, dict) else ""
            if not triples or not answer_var:
                continue
            graph = semantic.get("semantic_graph", {})
            surfaces = [
                str(item.get("surface", ""))
                for item in graph.get("anchors", [])
                if isinstance(item, dict) and str(item.get("surface", "")).strip()
            ]
            text = " ".join(
                [
                    _missing_relation_retrieval_mask_entities(str(semantic.get("question", "")), surfaces),
                    *(
                        _missing_relation_retrieval_mask_entities(str(value), surfaces)
                        for value in semantic.get("decomposition", [])
                    ),
                ]
            )
            frequencies = Counter(_missing_relation_retrieval_tokens(text))
            relations = Counter(relation for _, relation, _ in triples)
            incidence = Counter(
                f"{relation}|{'out' if subject == answer_var else 'in' if object_ == answer_var else 'internal'}"
                for subject, relation, object_ in triples
            )
            index = len(self.documents)
            document = _missing_relation_retrieval_Document(
                source_index=source_index,
                length=sum(frequencies.values()),
                relations=relations,
                answer_incidence=incidence,
                triple_count=len(triples),
                triples=tuple(triples),
                answer_var=answer_var,
                tokens=list(frequencies.elements()),
            )
            self.documents.append(document)
            document_frequency.update(frequencies)
            for token, frequency in frequencies.items():
                self.postings[token].append((index, frequency))
        count = max(1, len(self.documents))
        self.average_length = sum(value.length for value in self.documents) / count
        self.idf = {
            token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }
        self.max_posting_length = max(64, int(len(self.documents) * 0.18))

    def retrieve(
        self,
        question: str,
        decompositions: Sequence[str],
        surfaces: Sequence[str],
        *,
        top_k: int,
    ) -> list[tuple[float, _missing_relation_retrieval_Document]]:
        text = " ".join(
            [
                _missing_relation_retrieval_mask_entities(question, surfaces),
                *(_missing_relation_retrieval_mask_entities(value, surfaces) for value in decompositions),
            ]
        )
        query = Counter(_missing_relation_retrieval_tokens(text))
        scores: dict[int, float] = defaultdict(float)
        informative = sorted(
            (
                (self.idf.get(token, 0.0), token, frequency)
                for token, frequency in query.items()
                if len(self.postings.get(token, ())) <= self.max_posting_length
            ),
            reverse=True,
        )[:24]
        for _, token, query_frequency in informative:
            inverse_frequency = self.idf.get(token, 0.0)
            for index, frequency in self.postings.get(token, ()):
                document = self.documents[index]
                normalization = 0.25 + 0.75 * document.length / max(
                    self.average_length, 1e-9
                )
                scores[index] += (
                    inverse_frequency
                    * ((frequency * 2.2) / (frequency + 1.2 * normalization))
                    * min(query_frequency, 2)
                )
        return [
            (score, self.documents[index])
            for score, index in heapq.nlargest(
                max(1, top_k), ((score, index) for index, score in scores.items())
            )
        ]


@dataclass(slots=True)
class _missing_relation_retrieval_Evidence:
    relation_id: str
    best_template_rank: int
    normalized_bm25: float
    support_top8: int
    support_top32: int
    weighted_support: float
    best_relation_overlap: float
    preferred_orientation: str
    orientation_confidence: float
    preferred_size_delta: int
    grounded_union: bool
    reverse_of_terminal: bool
    incumbent_weighted_support: float
    template_support_margin: float
    template_support_ratio: float
    lexical_overlap: float
    bge_score: float = 0.0
    incumbent_bge_score: float = 0.0
    candidate_score: float = 0.0
    relation_support: dict[str, float] = field(default_factory=dict)

    def dump(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _missing_relation_retrieval_relation_evidence(
    *,
    question: str,
    decompositions: Sequence[str],
    surfaces: Sequence[str],
    graph: dict[str, Any],
    traces: Sequence[dict[str, Any]],
    index: _missing_relation_retrieval_AlignedRelationIndex,
    ontology: _missing_relation_retrieval_FreebaseOntology,
    top_k: int,
) -> tuple[list[_missing_relation_retrieval_Evidence], list[str]]:
    triples = [
        tuple(map(str, value))
        for value in graph.get("triples", [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    answer_var = str(graph.get("answer_var", ""))
    selected = Counter(relation for _, relation, _ in triples)
    terminal = {
        relation
        for subject, relation, object_ in triples
        if answer_var in {subject, object_}
    }
    grounded = _grounded_union(traces)
    neighbours = index.retrieve(
        question, decompositions, surfaces, top_k=max(3, int(top_k))
    )
    top_score = max((score for score, _ in neighbours), default=1.0)
    aggregate: dict[str, dict[str, Any]] = {}
    all_support: Counter[str] = Counter()
    for rank, (bm25, document) in enumerate(neighbours, 1):
        overlap = sum((document.relations & selected).values()) / max(
            1, sum(selected.values())
        )
        normalized = bm25 / max(top_score, 1e-9)
        document_weight = normalized * (0.35 + 0.65 * overlap)
        for relation_id, count in document.relations.items():
            all_support[relation_id] += document_weight * count
        for relation_id in document.relations - selected:
            value = aggregate.setdefault(
                relation_id,
                {
                    "rank": rank,
                    "bm25": bm25,
                    "support8": 0,
                    "support32": 0,
                    "weight": 0.0,
                    "overlap": 0.0,
                    "orientations": Counter(),
                    "sizes": Counter(),
                },
            )
            value["rank"] = min(value["rank"], rank)
            value["bm25"] = max(value["bm25"], bm25)
            value["support8"] += int(rank <= 8)
            value["support32"] += int(rank <= 32)
            value["weight"] += document_weight
            value["overlap"] = max(value["overlap"], overlap)
            value["sizes"][document.triple_count - len(triples)] += normalized
            for orientation in ("in", "out"):
                count = document.answer_incidence.get(
                    f"{relation_id}|{orientation}", 0
                )
                if count:
                    value["orientations"][orientation] += normalized * count
    incumbent_support = max(
        (float(all_support.get(value, 0.0)) for value in terminal), default=0.0
    )
    question_tokens = set(_missing_relation_retrieval_tokens(question))
    output: list[_missing_relation_retrieval_Evidence] = []
    for relation_id, value in aggregate.items():
        orientations: Counter[str] = value["orientations"]
        orientation = (
            max(orientations, key=lambda key: (orientations[key], key))
            if orientations
            else ""
        )
        orientation_total = sum(orientations.values())
        sizes: Counter[int] = value["sizes"]
        size_delta = max(sizes, key=lambda key: (sizes[key], -abs(key))) if sizes else 0
        relation_tokens = _leaf_tokens(relation_id)
        lexical = (
            len(relation_tokens & question_tokens) / len(relation_tokens)
            if relation_tokens
            else 0.0
        )
        output.append(
            _missing_relation_retrieval_Evidence(
                relation_id=relation_id,
                best_template_rank=int(value["rank"]),
                normalized_bm25=float(value["bm25"] / max(top_score, 1e-9)),
                support_top8=int(value["support8"]),
                support_top32=int(value["support32"]),
                weighted_support=float(value["weight"]),
                best_relation_overlap=float(value["overlap"]),
                preferred_orientation=orientation,
                orientation_confidence=(
                    float(orientations.get(orientation, 0.0) / orientation_total)
                    if orientation_total
                    else 0.0
                ),
                preferred_size_delta=int(size_delta),
                grounded_union=relation_id in grounded,
                reverse_of_terminal=any(
                    relation_id in ontology.reverse_for_relation(old)
                    for old in terminal
                ),
                incumbent_weighted_support=incumbent_support,
                template_support_margin=float(value["weight"] - incumbent_support),
                template_support_ratio=float(
                    value["weight"] / max(incumbent_support, 1e-9)
                ),
                lexical_overlap=lexical,
                relation_support={
                    str(key): float(support)
                    for key, support in all_support.items()
                },
            )
        )
    return output, sorted(terminal)


def _missing_relation_retrieval_type_compatibility(ontology: _missing_relation_retrieval_FreebaseOntology, left: str, right: str) -> float:
    if not left or not right:
        return 0.35
    if left == right:
        return 1.0
    left_supers, right_supers = set(ontology.supertypes(left)), set(ontology.supertypes(right))
    if left in right_supers or right in left_supers:
        return 0.8
    if left_supers & right_supers:
        return 0.55
    return 0.0


def _neighbour_type(
    ontology: _missing_relation_retrieval_FreebaseOntology, relation_id: str, orientation: str
) -> str:
    return (
        ontology.range_for_relation(relation_id)
        if orientation == "out"
        else ontology.domain_for_relation(relation_id)
    )


def _missing_relation_retrieval_score_evidence(
    evidence: list[_missing_relation_retrieval_Evidence],
    *,
    question: str,
    terminal: Sequence[str],
    ranker: Any,
) -> bool:
    relation_ids = list(dict.fromkeys([*terminal, *(item.relation_id for item in evidence[:24])]))
    try:
        scores = ranker.score(question, [_missing_relation_retrieval_relation_text(value) for value in relation_ids])
    except Exception:
        return False
    by_relation = dict(zip(relation_ids, map(float, scores)))
    incumbent_bge = max((by_relation.get(value, 0.0) for value in terminal), default=0.0)
    max_support = max((item.weighted_support for item in evidence), default=1.0)
    for item in evidence:
        item.bge_score = by_relation.get(item.relation_id, 0.0)
        item.incumbent_bge_score = incumbent_bge
        support = item.weighted_support / max(max_support, 1e-9)
        item.candidate_score = (
            0.24 * item.normalized_bm25
            + 0.22 * support
            + 0.19 * item.best_relation_overlap
            + 0.22 * max(0.0, min(1.0, (item.bge_score + 1.0) / 2.0))
            + 0.07 * item.lexical_overlap
            + 0.04 * float(item.grounded_union)
            + 0.02 * item.orientation_confidence
        )
    evidence.sort(
        key=lambda item: (
            -item.candidate_score,
            item.best_template_rank,
            item.relation_id,
        )
    )
    return True


def _preliminary_gate(evidence: Sequence[_missing_relation_retrieval_Evidence]) -> bool:
    if not evidence:
        return False
    best = evidence[0]
    return bool(
        best.best_relation_overlap >= 0.34
        and best.support_top32 >= 2
        and (
            best.bge_score - best.incumbent_bge_score >= -0.04
            or best.lexical_overlap > 0.0
            or best.grounded_union
        )
    )


def _local_evidence_possible(
    evidence: _missing_relation_retrieval_Evidence,
    *,
    graph: dict[str, Any],
    incumbent_count: int,
    final_empty: bool,
) -> bool:
    """BGE-free necessary conditions for any production acceptance branch."""

    if final_empty:
        return bool(
            evidence.relation_id
            in {
                str(triple[1])
                for triple in graph.get("triples", [])
                if isinstance(triple, (list, tuple)) and len(triple) == 3
            }
            or evidence.reverse_of_terminal
            or (
                evidence.best_template_rank == 1
                and evidence.best_relation_overlap >= 2.0 / 3.0
                and evidence.orientation_confidence >= 0.99
            )
        )
    a_possible = bool(
        evidence.incumbent_weighted_support <= 10.0
        and evidence.support_top32 >= 20
        and evidence.orientation_confidence >= 0.99
    )
    b_possible = bool(
        incumbent_count >= 3
        and evidence.support_top32 >= 15
        and evidence.orientation_confidence >= 0.95
        and evidence.best_relation_overlap < 1.0
    )
    answer_var = str(graph.get("answer_var", ""))
    terminal_orientations = {
        "out" if str(triple[0]) == answer_var else "in"
        for triple in graph.get("triples", [])
        if isinstance(triple, (list, tuple))
        and len(triple) == 3
        and answer_var in {str(triple[0]), str(triple[2])}
    }
    d_possible = bool(
        5 <= incumbent_count <= 64
        and evidence.support_top8 >= 5
        and evidence.best_template_rank == 1
        and 0.0 < evidence.best_relation_overlap < 1.0
        and (
            not evidence.preferred_orientation
            or evidence.preferred_orientation in terminal_orientations
        )
    )
    e_possible = bool(
        incumbent_count > 0
        and evidence.best_template_rank == 1
        and 0.5 <= evidence.best_relation_overlap <= 2.0 / 3.0
        and (
            not evidence.preferred_orientation
            or evidence.preferred_orientation in terminal_orientations
        )
    )
    return a_possible or b_possible or d_possible or e_possible


def _pre_execution_candidate(
    candidate: _missing_relation_retrieval_Candidate,
    *,
    incumbent_count: int,
    final_empty: bool,
) -> bool:
    """Apply every gate term known before spending an endpoint query."""

    evidence = candidate.evidence
    if final_empty:
        return bool(
            evidence.relation_id == candidate.replaced_relation
            or evidence.reverse_of_terminal
            or (
                evidence.best_template_rank == 1
                and evidence.best_relation_overlap >= 2.0 / 3.0
                and evidence.orientation_confidence >= 0.99
            )
        )
    a_possible = bool(
        evidence.incumbent_weighted_support <= 10.0
        and evidence.bge_score - evidence.incumbent_bge_score >= 0.0
        and evidence.support_top32 >= 20
        and evidence.orientation_confidence >= 0.99
    )
    b_possible = bool(
        candidate.graph.score >= 0.90
        and incumbent_count >= 3
        and evidence.support_top32 >= 15
        and evidence.orientation_confidence >= 0.95
        and evidence.best_relation_overlap < 1.0
    )
    d_possible = bool(
        5 <= incumbent_count <= 64
        and evidence.support_top8 >= 5
        and evidence.best_template_rank == 1
        and 0.0 < evidence.best_relation_overlap < 1.0
        and candidate.old_orientation == candidate.new_orientation
    )
    e_possible = bool(
        incumbent_count > 0
        and evidence.best_template_rank == 1
        and candidate.old_orientation == candidate.new_orientation
        and evidence.bge_score - evidence.incumbent_bge_score >= 0.0
        and 0.5 <= evidence.best_relation_overlap <= 2.0 / 3.0
    )
    return a_possible or b_possible or d_possible or e_possible


@dataclass(slots=True)
class _missing_relation_retrieval_Candidate:
    graph: _missing_relation_retrieval_QueryGraphCandidate
    evidence: _missing_relation_retrieval_Evidence
    replaced_relation: str
    old_orientation: str
    new_orientation: str
    answers: list[str]
    query_elapsed_seconds: float = 0.0
    error: str = ""
    edge_index: int = -1
    edge_role: str = "answer"
    reversed_edge: bool = False
    schema_compatibility: float = 0.0
    answer_distance: int = -1


def _node_distances_to_answer(
    triples: Sequence[Sequence[str]], answer_var: str
) -> dict[str, int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        if len(triple) != 3:
            continue
        subject, _, object_ = map(str, triple)
        adjacency[subject].add(object_)
        adjacency[object_].add(subject)
    distances = {str(answer_var): 0}
    frontier = [str(answer_var)]
    while frontier:
        node = frontier.pop(0)
        for neighbour in adjacency.get(node, set()):
            if neighbour in distances:
                continue
            distances[neighbour] = distances[node] + 1
            frontier.append(neighbour)
    return distances


def _endpoint_type_hints(
    ontology: _missing_relation_retrieval_FreebaseOntology,
    triples: Sequence[Sequence[str]],
    *,
    edge_index: int,
    node: str,
) -> list[str]:
    hints: list[str] = []
    for index, triple in enumerate(triples):
        if index == edge_index or len(triple) != 3:
            continue
        subject, relation, object_ = map(str, triple)
        value = ""
        if subject == node:
            value = str(ontology.domain_for_relation(relation))
        elif object_ == node:
            value = str(ontology.range_for_relation(relation))
        if value and value not in hints:
            hints.append(value)
    return hints


def _missing_relation_retrieval_best_endpoint_compatibility(
    ontology: _missing_relation_retrieval_FreebaseOntology,
    expected: str,
    hints: Sequence[str],
) -> float:
    if not hints:
        return 0.35 if not expected else 0.50
    return max(_missing_relation_retrieval_type_compatibility(ontology, expected, value) for value in hints)


def _internal_schema_compatibility(
    ontology: _missing_relation_retrieval_FreebaseOntology,
    triples: Sequence[Sequence[str]],
    *,
    edge_index: int,
    new_relation: str,
    reversed_edge: bool,
) -> float:
    subject, old_relation, object_ = map(str, triples[edge_index])
    new_domain = str(ontology.domain_for_relation(new_relation))
    new_range = str(ontology.range_for_relation(new_relation))
    if reversed_edge:
        new_domain, new_range = new_range, new_domain
    old_domain = str(ontology.domain_for_relation(old_relation))
    old_range = str(ontology.range_for_relation(old_relation))
    subject_local = _missing_relation_retrieval_best_endpoint_compatibility(
        ontology,
        new_domain,
        _endpoint_type_hints(
            ontology, triples, edge_index=edge_index, node=subject
        ),
    )
    object_local = _missing_relation_retrieval_best_endpoint_compatibility(
        ontology,
        new_range,
        _endpoint_type_hints(
            ontology, triples, edge_index=edge_index, node=object_
        ),
    )
    subject_score = max(
        subject_local,
        0.75 * _missing_relation_retrieval_type_compatibility(ontology, new_domain, old_domain),
    )
    object_score = max(
        object_local,
        0.75 * _missing_relation_retrieval_type_compatibility(ontology, new_range, old_range),
    )
    return 0.5 * (subject_score + object_score)


def _internal_candidate_possible(candidate: _missing_relation_retrieval_Candidate) -> bool:
    """Generic Gold-blind pre-query gate for the final-empty internal lane."""

    evidence = candidate.evidence
    return bool(
        candidate.edge_role == "internal"
        and evidence.best_relation_overlap >= 0.5
        and evidence.support_top32 >= 2
        and (
            evidence.grounded_union
            or evidence.bge_score - evidence.incumbent_bge_score >= -0.04
            or evidence.lexical_overlap > 0.0
        )
    )


def _compile_internal(
    graph: dict[str, Any],
    evidence: Sequence[_missing_relation_retrieval_Evidence],
    ontology: _missing_relation_retrieval_FreebaseOntology,
    *,
    limit: int,
) -> list[_missing_relation_retrieval_Candidate]:
    """Compile one-edge internal repairs ranked by schema and answer proximity."""

    triples = [
        list(map(str, value))
        for value in graph.get("triples", [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    answer_var = str(graph.get("answer_var", ""))
    internal_edges = [
        (index, value)
        for index, value in enumerate(triples)
        if answer_var not in {value[0], value[2]}
    ]
    if not internal_edges:
        return []
    distances = _node_distances_to_answer(triples, answer_var)
    output: list[_missing_relation_retrieval_Candidate] = []
    seen: set[str] = set()
    for relation in evidence[:12]:
        support = relation.relation_support
        max_support = max(support.values(), default=1.0)
        for edge_index, edge in internal_edges:
            subject, old_relation, object_ = edge
            if relation.relation_id == old_relation:
                continue
            distance = min(
                distances.get(subject, len(triples) + 1),
                distances.get(object_, len(triples) + 1),
            )
            proximity = 1.0 / (1.0 + float(distance))
            support_delta = (
                float(support.get(relation.relation_id, 0.0))
                - float(support.get(old_relation, 0.0))
            ) / max(max_support, 1e-9)
            for reversed_edge in (False, True):
                compatibility = _internal_schema_compatibility(
                    ontology,
                    triples,
                    edge_index=edge_index,
                    new_relation=relation.relation_id,
                    reversed_edge=reversed_edge,
                )
                if compatibility <= 0.0:
                    continue
                revised = deepcopy(triples)
                revised[edge_index] = (
                    [object_, relation.relation_id, subject]
                    if reversed_edge
                    else [subject, relation.relation_id, object_]
                )
                signature = json.dumps(
                    {"triples": revised, "answer_var": answer_var},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                score = (
                    0.48 * relation.candidate_score
                    + 0.25 * compatibility
                    + 0.05 * (1.0 / (1.0 + math.exp(-4.0 * support_delta)))
                    + 0.04 * proximity
                    + 0.02 * float(not reversed_edge)
                )
                candidate_graph = _missing_relation_retrieval_QueryGraphCandidate(
                    graph_id=f"MRI{len(output)}",
                    triples=revised,
                    answer_var=answer_var,
                    operators=deepcopy(list(graph.get("operators", []))),
                    score=score,
                    compose_output={},
                    provenance=deepcopy(dict(graph.get("provenance", {}))),
                )
                candidate_graph.provenance["missing_relation_retrieval"] = {
                    "relation_id": relation.relation_id,
                    "replaced_relation": old_relation,
                    "edge_role": "internal",
                    "edge_index": edge_index,
                    "answer_distance": distance,
                    "uses_gold": False,
                    "model_calls": 0,
                }
                output.append(
                    _missing_relation_retrieval_Candidate(
                        graph=candidate_graph,
                        evidence=relation,
                        replaced_relation=old_relation,
                        old_orientation="internal",
                        new_orientation="internal",
                        answers=[],
                        edge_index=edge_index,
                        edge_role="internal",
                        reversed_edge=reversed_edge,
                        schema_compatibility=compatibility,
                        answer_distance=distance,
                    )
                )
    output.sort(
        key=lambda item: (
            -item.graph.score,
            item.answer_distance,
            item.edge_index,
            item.graph.graph_id,
        )
    )
    return output[: max(1, min(24, int(limit)))]


def _compile(
    graph: dict[str, Any],
    evidence: Sequence[_missing_relation_retrieval_Evidence],
    ontology: _missing_relation_retrieval_FreebaseOntology,
    *,
    limit: int,
) -> list[_missing_relation_retrieval_Candidate]:
    triples = [
        list(map(str, value))
        for value in graph.get("triples", [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    answer_var = str(graph.get("answer_var", ""))
    terminal_edges = [
        (index, value)
        for index, value in enumerate(triples)
        if answer_var in {value[0], value[2]}
    ]
    output: list[_missing_relation_retrieval_Candidate] = []
    seen: set[str] = set()
    for relation in evidence[:12]:
        for edge_index, edge in terminal_edges:
            subject, old_relation, object_ = edge
            old_orientation = "out" if subject == answer_var else "in"
            new_orientation = relation.preferred_orientation or old_orientation
            new_subject, new_object = subject, object_
            if new_orientation != old_orientation:
                new_subject, new_object = object_, subject
            compatibility = _missing_relation_retrieval_type_compatibility(
                ontology,
                _neighbour_type(ontology, old_relation, old_orientation),
                _neighbour_type(ontology, relation.relation_id, new_orientation),
            )
            if compatibility == 0.0:
                continue
            revised = deepcopy(triples)
            revised[edge_index] = [new_subject, relation.relation_id, new_object]
            signature = json.dumps(
                {"triples": revised, "answer_var": answer_var},
                sort_keys=True,
                separators=(",", ":"),
            )
            if signature in seen:
                continue
            seen.add(signature)
            score = (
                relation.candidate_score
                + 0.08 * compatibility
                + 0.04 * float(new_orientation == relation.preferred_orientation)
                + 0.03 * float(relation.preferred_size_delta == 0)
            )
            candidate_graph = _missing_relation_retrieval_QueryGraphCandidate(
                graph_id=f"MR{len(output)}",
                triples=revised,
                answer_var=answer_var,
                operators=deepcopy(list(graph.get("operators", []))),
                score=score,
                compose_output={},
                provenance=deepcopy(dict(graph.get("provenance", {}))),
            )
            candidate_graph.provenance["missing_relation_retrieval"] = {
                "relation_id": relation.relation_id,
                "replaced_relation": old_relation,
                "uses_gold": False,
                "model_calls": 0,
            }
            output.append(
                _missing_relation_retrieval_Candidate(
                    graph=candidate_graph,
                    evidence=relation,
                    replaced_relation=old_relation,
                    old_orientation=old_orientation,
                    new_orientation=new_orientation,
                    answers=[],
                )
            )
    output.sort(key=lambda item: (-item.graph.score, item.graph.graph_id))
    return output[: max(1, min(12, int(limit)))]


def _execute(pipeline: Any, candidates: Sequence[_missing_relation_retrieval_Candidate]) -> int:
    count = 0
    for candidate in candidates[:3]:
        started = time.monotonic()
        try:
            candidate.graph.sparql = _missing_relation_retrieval_lower_sparql(
                candidate.graph, limit=getattr(pipeline, "answer_limit", None)
            )
            count += 1
            rows = pipeline.kg.execute(candidate.graph.sparql)
            candidate.answers = _missing_relation_retrieval_answer_values(rows, candidate.graph.answer_var)
        except Exception as exc:
            candidate.error = f"{type(exc).__name__}:{exc}"
        candidate.query_elapsed_seconds = time.monotonic() - started
    return count


def _answer_relation(candidate: Sequence[str], incumbent: Sequence[str]) -> str:
    left, right = set(map(str, candidate)), set(map(str, incumbent))
    if left == right:
        return "equal"
    if left < right:
        return "subset"
    if left > right:
        return "superset"
    if left & right:
        return "overlap"
    return "disjoint"


def _successful_branch(
    candidate: _missing_relation_retrieval_Candidate, incumbent: Sequence[str]
) -> tuple[str, int]:
    evidence = candidate.evidence
    count, old_count = len(candidate.answers), len(incumbent)
    answer_relation = _answer_relation(candidate.answers, incumbent)
    disjoint = answer_relation == "disjoint"
    a = bool(
        evidence.incumbent_weighted_support <= 10.0
        and count <= 5
        and evidence.bge_score - evidence.incumbent_bge_score >= 0.0
        and evidence.support_top32 >= 20
        and evidence.orientation_confidence >= 0.99
        and disjoint
    )
    b = bool(
        candidate.graph.score >= 0.90
        and count <= 10
        and old_count >= 3
        and evidence.support_top32 >= 15
        and evidence.orientation_confidence >= 0.95
        and disjoint
        and count / max(1, old_count) <= 2.0
    )
    c = bool(
        count <= 3
        and old_count >= 5
        and evidence.support_top8 >= 4
        and disjoint
    )
    d = bool(
        c
        and evidence.best_template_rank == 1
        and candidate.old_orientation == candidate.new_orientation
        and evidence.support_top8 >= 5
        and 0.0 < evidence.best_relation_overlap < 1.0
        and old_count <= 64
    )
    e = bool(
        answer_relation in {"subset", "superset"}
        and evidence.best_template_rank == 1
        and candidate.old_orientation == candidate.new_orientation
        and evidence.bge_score - evidence.incumbent_bge_score >= 0.0
        and 0.5 <= evidence.best_relation_overlap <= 2.0 / 3.0
        and (
            (answer_relation == "superset" and count >= 3)
            or evidence.bge_score - evidence.incumbent_bge_score >= 0.08
            or evidence.template_support_ratio >= 3.0
        )
    )
    if e:
        return "E", 4
    if a:
        return "A", 3
    if d:
        return "D", 2
    if b and not c and evidence.best_relation_overlap < 1.0:
        return "B", 1
    return "", 0


def _empty_branch(candidate: _missing_relation_retrieval_Candidate) -> tuple[str, int]:
    if len(candidate.answers) != 1:
        return "", 0
    evidence = candidate.evidence
    if evidence.relation_id == candidate.replaced_relation:
        return "same_relation_relowering", 3
    if evidence.reverse_of_terminal:
        return "ontology_reverse", 2
    if (
        evidence.best_template_rank == 1
        and evidence.best_relation_overlap >= 2.0 / 3.0
        and evidence.orientation_confidence >= 0.99
    ):
        return "top_template_consensus", 1
    return "", 0


def _missing_relation_retrieval_index(pipeline: Any) -> _missing_relation_retrieval_AlignedRelationIndex | None:
    cached = getattr(pipeline, "_missing_relation_index", None)
    if cached is not None:
        return cached
    semantic = getattr(pipeline.contract, "semantic_examples", [])
    compose = getattr(pipeline.contract, "compose_examples", [])
    if not semantic or len(semantic) != len(compose):
        return None
    lock = getattr(pipeline, "_missing_relation_index_lock", None)
    if lock is None:
        value = _missing_relation_retrieval_AlignedRelationIndex(semantic, compose)
        pipeline._missing_relation_index = value
        return value
    with lock:
        cached = getattr(pipeline, "_missing_relation_index", None)
        if cached is None:
            cached = _missing_relation_retrieval_AlignedRelationIndex(semantic, compose)
            pipeline._missing_relation_index = cached
    return cached


def _graph_dict(graph: _missing_relation_retrieval_QueryGraphCandidate) -> dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "triples": deepcopy(graph.triples),
        "answer_var": graph.answer_var,
        "operators": deepcopy(graph.operators),
        "score": graph.score,
        "provenance": deepcopy(graph.provenance),
    }


def _anchor_ids(graph: dict[str, Any]) -> set[str]:
    bindings = (graph.get("provenance") or {}).get("anchor_bindings", {})
    if not isinstance(bindings, dict):
        return set()
    return {
        str(value.get("id", "") if isinstance(value, dict) else value)
        for value in bindings.values()
        if str(value.get("id", "") if isinstance(value, dict) else value)
    }


def _answer_incidence(graph: dict[str, Any]) -> set[str]:
    answer_var = str(graph.get("answer_var", ""))
    return {
        "out" if str(triple[0]) == answer_var else "in"
        for triple in graph.get("triples", [])
        if isinstance(triple, (list, tuple))
        and len(triple) == 3
        and answer_var in {str(triple[0]), str(triple[2])}
    }


def _broad_answer_kind(graph: dict[str, Any], ontology: Any) -> str:
    answer_var = str(graph.get("answer_var", ""))
    type_ids: set[str] = set()
    for triple in graph.get("triples", []):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation, object_ = map(str, triple)
        if object_ == answer_var:
            type_ids.add(str(ontology.range_for_relation(relation)))
        elif subject == answer_var:
            type_ids.add(str(ontology.domain_for_relation(relation)))
    type_ids.discard("")
    scalar = {
        "type.boolean",
        "type.datetime",
        "type.enumeration",
        "type.float",
        "type.int",
        "type.key",
        "type.rawstring",
        "type.string",
        "type.text",
        "type.uri",
    }
    if type_ids and type_ids <= scalar:
        return "scalar"
    if type_ids:
        return "entity"
    return "unknown"


def _highest_scored_successful_fallback_graph(
    traces: Sequence[dict[str, Any]],
    selected: dict[str, Any],
    ontology: Any,
) -> dict[str, Any] | None:
    """Return a trace-proven compatible source after selected-source refusal."""

    final_graphs: list[dict[str, Any]] = []
    for output in _missing_relation_retrieval_trace_outputs(traces, "final_graph_beam"):
        if isinstance(output, list):
            final_graphs = [item for item in output if isinstance(item, dict)]
    successful_ids: set[str] = set()
    for output in _missing_relation_retrieval_trace_outputs(
        traces, "deterministic_sparql_lowering_and_execution"
    ):
        for item in output if isinstance(output, list) else []:
            if (
                isinstance(item, dict)
                and int(item.get("answer_count", 0)) > 0
                and str(item.get("graph_id", ""))
            ):
                successful_ids.add(str(item["graph_id"]))
    selected_id = str(selected.get("graph_id", ""))
    selected_anchors = _anchor_ids(selected)
    selected_incidence = _answer_incidence(selected)
    selected_kind = _broad_answer_kind(selected, ontology)
    selected_has_operators = bool(selected.get("operators"))
    selected_triple_count = len(selected.get("triples", []))
    candidates = [
        graph
        for graph in final_graphs
        if str(graph.get("graph_id", "")) in successful_ids
        and str(graph.get("graph_id", "")) != selected_id
        and _anchor_ids(graph) == selected_anchors
        and _answer_incidence(graph) == selected_incidence
        and _broad_answer_kind(graph, ontology) == selected_kind
        and bool(graph.get("operators")) == selected_has_operators
        # A source fallback is justified only when it removes structure from a
        # selected graph that could not produce any viable relation repair.
        # Switching laterally to an equally complex graph is an unconstrained
        # re-selection and can disturb an already-correct incumbent.
        and len(graph.get("triples", [])) < selected_triple_count
    ]
    return (
        max(
            candidates,
            key=lambda graph: (
                float(graph.get("score", 0.0)),
                -len(graph.get("triples", [])),
                str(graph.get("graph_id", "")),
            ),
        )
        if candidates
        else None
    )


def _retrieve_missing_relation_once(
    *,
    pipeline: Any,
    question: str,
    result: dict[str, Any],
    final_empty: bool,
    template_top_k: int = 64,
    candidate_limit: int = 3,
    return_bounded_best_effort: bool = False,
    best_effort_answer_limit: int = 64,
    allow_internal_edge_fallback: bool = False,
) -> dict[str, Any]:
    """Return a gated replacement outcome without consulting test Gold."""

    started = time.monotonic()
    diagnostics: dict[str, Any] = {
        "status": "abstained",
        "trigger": "final_empty" if final_empty else "successful_challenger",
        "uses_gold_answers": False,
        "uses_gold_query": False,
        "llm_calls": 0,
        "candidate_query_limit": min(3, max(1, int(candidate_limit))),
        "endpoint_execution_queries": 0,
        "terminal_execution_queries": 0,
        "internal_execution_queries": 0,
        "internal_edge_fallback_allowed": bool(
            final_empty and allow_internal_edge_fallback
        ),
        "bge_ranker_calls": 0,
    }
    ontology = getattr(getattr(pipeline, "grounder", None), "ontology", None)
    ranker = getattr(getattr(pipeline, "grounder", None), "ranker", None)
    traces = result.get("traces", [])
    if ontology is None or ranker is None or not isinstance(traces, list):
        diagnostics["reason"] = "missing_runtime_dependency"
        return {"answer_ids": [], "diagnostics": diagnostics}
    if not final_empty and _missing_relation_retrieval_hard_post_selection_applied(traces):
        diagnostics["reason"] = "preserve_hard_post_selection_gate"
        diagnostics["selection_order"] = "hard_post_selection_before_relation_retrieval"
        return {"answer_ids": [], "diagnostics": diagnostics}
    graph = _final_incumbent(traces) if final_empty else result.get("selected_graph")
    if not isinstance(graph, dict):
        diagnostics["reason"] = "missing_incumbent_graph"
        return {"answer_ids": [], "diagnostics": diagnostics}
    relation_index = _missing_relation_retrieval_index(pipeline)
    if relation_index is None:
        diagnostics["reason"] = "missing_aligned_training_templates"
        return {"answer_ids": [], "diagnostics": diagnostics}
    evidence, terminal = _missing_relation_retrieval_relation_evidence(
        question=question,
        decompositions=_missing_relation_retrieval_decompositions(traces),
        surfaces=_missing_relation_retrieval_anchor_surfaces(traces),
        graph=graph,
        traces=traces,
        index=relation_index,
        ontology=ontology,
        top_k=min(64, max(3, int(template_top_k))),
    )
    if not evidence or not terminal:
        diagnostics["reason"] = "no_relation_proposal"
        return {"answer_ids": [], "diagnostics": diagnostics}
    incumbent = list(map(str, result.get("answer_ids", [])))
    evidence = [
        value
        for value in evidence
        if _local_evidence_possible(
            value,
            graph=graph,
            incumbent_count=len(incumbent),
            final_empty=final_empty,
        )
    ]
    diagnostics["local_prefilter_relation_count"] = len(evidence)
    if not evidence:
        diagnostics["reason"] = "local_pre_embedding_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}
    diagnostics["bge_ranker_calls"] = 1
    if not _missing_relation_retrieval_score_evidence(
        evidence,
        question=question,
        terminal=terminal,
        ranker=ranker,
    ):
        diagnostics["reason"] = "embedding_unavailable"
        return {"answer_ids": [], "diagnostics": diagnostics}
    candidates = _compile(
        graph,
        evidence,
        ontology,
        limit=12,
    )
    diagnostics["compiled_candidate_count"] = len(candidates)
    candidates = [
        candidate
        for candidate in candidates
        if _pre_execution_candidate(
            candidate,
            incumbent_count=len(incumbent),
            final_empty=final_empty,
        )
    ]
    candidates = candidates[: min(3, max(1, int(candidate_limit)))]
    diagnostics["pre_execution_eligible_count"] = len(candidates)
    if not candidates:
        diagnostics["reason"] = "pre_execution_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}
    query_budget = min(3, max(1, int(candidate_limit)))
    terminal_query_count = _execute(pipeline, candidates)
    diagnostics["terminal_execution_queries"] = terminal_query_count
    diagnostics["endpoint_execution_queries"] = terminal_query_count
    accepted: list[tuple[int, _missing_relation_retrieval_Candidate, str]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        branch, priority = (
            _empty_branch(candidate)
            if final_empty
            else _successful_branch(candidate, incumbent)
        )
        candidate_diagnostics.append(
            {
                "relation_id": candidate.evidence.relation_id,
                "replaced_relation": candidate.replaced_relation,
                "score": candidate.graph.score,
                "answer_count": len(candidate.answers),
                "answer_relation_to_incumbent": _answer_relation(
                    candidate.answers, incumbent
                ),
                "accepted_branch": branch,
                "query_elapsed_seconds": candidate.query_elapsed_seconds,
                "error": candidate.error,
            }
        )
        if branch and candidate.answers:
            accepted.append((priority, candidate, branch))
    diagnostics["candidates"] = candidate_diagnostics
    internal_selected: _missing_relation_retrieval_Candidate | None = None
    if (
        final_empty
        and allow_internal_edge_fallback
        and not accepted
        and terminal_query_count > 0
        and terminal_query_count < query_budget
        and all(not candidate.answers for candidate in candidates)
    ):
        remaining = query_budget - terminal_query_count
        internal_candidates = [
            candidate
            for candidate in _compile_internal(
                graph,
                evidence,
                ontology,
                limit=24,
            )
            if _internal_candidate_possible(candidate)
        ][:remaining]
        internal_query_count = _execute(pipeline, internal_candidates)
        diagnostics["internal_execution_queries"] = internal_query_count
        diagnostics["endpoint_execution_queries"] += internal_query_count
        diagnostics["internal_candidates"] = [
            {
                "graph_id": candidate.graph.graph_id,
                "relation_id": candidate.evidence.relation_id,
                "replaced_relation": candidate.replaced_relation,
                "edge_index": candidate.edge_index,
                "edge_role": candidate.edge_role,
                "reversed_edge": candidate.reversed_edge,
                "schema_compatibility": candidate.schema_compatibility,
                "answer_distance": candidate.answer_distance,
                "score": candidate.graph.score,
                "answer_count": len(candidate.answers),
                "answer_ids": list(candidate.answers),
                "query_elapsed_seconds": candidate.query_elapsed_seconds,
                "error": candidate.error,
            }
            for candidate in internal_candidates
        ]
        answer_cap = min(512, max(1, int(best_effort_answer_limit)))
        internal_selected = next(
            (
                candidate
                for candidate in internal_candidates
                if 0 < len(candidate.answers) <= answer_cap
            ),
            None,
        )
    elif final_empty:
        if not allow_internal_edge_fallback:
            diagnostics["internal_edge_fallback_reason"] = "disabled_by_caller"
        elif terminal_query_count <= 0:
            diagnostics["internal_edge_fallback_reason"] = (
                "no_executed_terminal_candidate"
            )
        elif terminal_query_count >= query_budget:
            diagnostics["internal_edge_fallback_reason"] = "shared_query_budget_exhausted"
        elif any(candidate.answers for candidate in candidates):
            diagnostics["internal_edge_fallback_reason"] = (
                "terminal_candidate_was_nonempty"
            )
    bounded_best_effort = False
    internal_best_effort = False
    if accepted:
        # Gate first, then prefer saved-grounding support, branch strength and
        # the pre-endpoint score. This cannot spend more queries than the fixed
        # beam.
        priority, selected, branch = max(
            accepted,
            key=lambda value: (
                bool(value[1].evidence.grounded_union),
                value[0],
                value[1].graph.score,
                value[1].evidence.relation_id,
            ),
        )
    elif internal_selected is not None:
        selected = internal_selected
        priority, branch = 0, "bounded_internal_relation_best_effort"
        bounded_best_effort = True
        internal_best_effort = True
    elif final_empty and return_bounded_best_effort:
        answer_cap = min(512, max(1, int(best_effort_answer_limit)))
        selected = next(
            (
                candidate
                for candidate in candidates
                if 0 < len(candidate.answers) <= answer_cap
            ),
            None,
        )
        if selected is None:
            diagnostics["reason"] = "confidence_gate_rejected"
            diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
            return {"answer_ids": [], "diagnostics": diagnostics}
        priority, branch = 0, "bounded_relation_best_effort"
        bounded_best_effort = True
    else:
        diagnostics["reason"] = "confidence_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}
    raw_labels: list[dict[str, str]] = []
    try:
        raw_labels = pipeline.kg.labels(selected.answers)
    except Exception as exc:
        diagnostics["label_error"] = f"{type(exc).__name__}:{exc}"
    labels = {
        str(value.get("id", "")): str(value.get("label", ""))
        for value in raw_labels
        if isinstance(value, dict) and str(value.get("id", ""))
    }
    diagnostics.update(
        {
            "status": (
                "selected_bounded_internal_relation_best_effort"
                if internal_best_effort
                else (
                    "selected_bounded_relation_best_effort"
                    if bounded_best_effort
                    else "selected"
                )
            ),
            "confidence": "low" if bounded_best_effort else "high",
            "selected_branch": branch,
            "selected_relation_id": selected.evidence.relation_id,
            "selected_replaced_relation": selected.replaced_relation,
            "selected_edge_role": selected.edge_role,
            "selected_edge_index": selected.edge_index,
            "answer_count": len(selected.answers),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    )
    return {
        "answer_ids": list(selected.answers),
        "answers": [
            {"id": value, "label": labels.get(value, value)}
            for value in selected.answers
        ],
        "selected_graph": _graph_dict(selected.graph),
        "diagnostics": diagnostics,
    }


def retrieve_missing_relation(
    *,
    pipeline: Any,
    question: str,
    result: dict[str, Any],
    final_empty: bool,
    template_top_k: int = 64,
    candidate_limit: int = 3,
    return_bounded_best_effort: bool = False,
    best_effort_answer_limit: int = 64,
    allow_internal_edge_fallback: bool = False,
) -> dict[str, Any]:
    """Run relation repair, with one trace-proven source fallback on refusal."""

    kwargs = {
        "pipeline": pipeline,
        "question": question,
        "result": result,
        "final_empty": final_empty,
        "template_top_k": template_top_k,
        "candidate_limit": candidate_limit,
        "return_bounded_best_effort": return_bounded_best_effort,
        "best_effort_answer_limit": best_effort_answer_limit,
        "allow_internal_edge_fallback": allow_internal_edge_fallback,
    }
    primary = _retrieve_missing_relation_once(**kwargs)
    if final_empty or primary.get("answer_ids"):
        return primary
    diagnostics = primary.get("diagnostics", {})
    if (
        not isinstance(diagnostics, dict)
        or int(diagnostics.get("endpoint_execution_queries", 0)) != 0
        or int(diagnostics.get("compiled_candidate_count", -1)) != 0
        or int(diagnostics.get("pre_execution_eligible_count", -1)) != 0
    ):
        return primary
    traces = result.get("traces", [])
    selected = result.get("selected_graph")
    ontology = getattr(getattr(pipeline, "grounder", None), "ontology", None)
    if not isinstance(traces, list) or not isinstance(selected, dict) or ontology is None:
        return primary
    fallback_graph = _highest_scored_successful_fallback_graph(
        traces,
        selected,
        ontology,
    )
    if fallback_graph is None:
        return primary
    fallback_result = {**result, "selected_graph": fallback_graph}
    fallback = _retrieve_missing_relation_once(
        **{**kwargs, "result": fallback_result}
    )
    fallback_diagnostics = fallback.get("diagnostics", {})
    if isinstance(fallback_diagnostics, dict):
        fallback_diagnostics["source_graph_fallback"] = {
            "status": "selected_source_pre_execution_refusal",
            "selected_graph_id": str(selected.get("graph_id", "")),
            "fallback_graph_id": str(fallback_graph.get("graph_id", "")),
            "same_anchor_coverage": True,
            "same_answer_incidence": True,
            "same_broad_answer_kind": True,
            "strictly_simpler_source": True,
            "initial_bge_ranker_calls": int(
                diagnostics.get("bge_ranker_calls", 0)
            ),
            "initial_endpoint_execution_queries": 0,
        }
        fallback_diagnostics["bge_ranker_calls"] = int(
            fallback_diagnostics.get("bge_ranker_calls", 0)
        ) + int(diagnostics.get("bge_ranker_calls", 0))
    return fallback



# ---- failure_endpoint_retrieval_plugin ----
"""Failure-only endpoint path retrieval compiled into executable query graphs.

It only runs as the last production/replay failure lane and consumes linked
anchors, frozen decomposition text, endpoint-verified paths, and ontology schema.
It never consumes an evaluation row,
Gold answer, dataset index, relation allow-list, or chat model.
"""



from dataclasses import dataclass
from itertools import combinations as anchor_subsets
import json
import math
import re
import threading
from typing import Any, Iterable, Mapping, Sequence

from .contracts import EntityCandidate as _failure_endpoint_retrieval_plugin_EntityCandidate, GroundedSemanticCandidate as _failure_endpoint_retrieval_plugin_GroundedSemanticCandidate, QueryGraphCandidate as _failure_endpoint_retrieval_plugin_QueryGraphCandidate
_failure_endpoint_retrieval_plugin_FailurePathCandidate = _failure_path_retrieval_FailurePathCandidate
_failure_endpoint_retrieval_plugin_FailurePathRetriever = _failure_path_retrieval_FailurePathRetriever
from .ontology import relation_label_from_id as _failure_endpoint_retrieval_plugin_relation_label_from_id


_EXECUTION_BUDGET = 30
_INTERSECTION_BEAM = 96
_PATH_QUERY_LIMIT = 1_024
_MAX_SUBSET_ANCHORS = 6
_SUBSET_COMBINATION_BUDGET = 768
_RETRIEVER_CACHE_LOCK = threading.Lock()
_failure_endpoint_retrieval_plugin_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUESTION_BOUNDARIES = {
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "that",
    "was",
    "were",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
}
_QUESTION_PREFIX_SKIP = {"a", "an", "are", "is", "the", "was", "were"}


class _LocalLexicalRanker:
    """EmbeddingRanker-compatible scorer with no service or model calls."""

    def score(self, query: str, candidates: list[str]) -> list[float]:
        query_tokens = _failure_endpoint_retrieval_plugin_TOKEN_RE.findall(str(query).casefold())
        query_set = set(query_tokens)
        output: list[float] = []
        for candidate in candidates:
            candidate_tokens = _failure_endpoint_retrieval_plugin_TOKEN_RE.findall(str(candidate).casefold())
            candidate_set = set(candidate_tokens)
            overlap = len(query_set & candidate_set)
            cosine = overlap / math.sqrt(
                max(1, len(query_set)) * max(1, len(candidate_set))
            )
            coverage = overlap / max(1, len(candidate_set))
            # Consecutive relation words are a useful deterministic signal for
            # phrases such as "place of birth" and "played by".
            query_bigrams = set(zip(query_tokens, query_tokens[1:]))
            candidate_bigrams = set(zip(candidate_tokens, candidate_tokens[1:]))
            bigram = len(query_bigrams & candidate_bigrams) / max(
                1,
                len(candidate_bigrams),
            )
            output.append((0.45 * cosine) + (0.45 * coverage) + (0.10 * bigram))
        return output


_LOCAL_RANKER = _LocalLexicalRanker()
_LOW_CONFIDENCE_SINGLETON_SCORE_WINDOW = 0.30
_LOW_CONFIDENCE_LEXICAL_DELTA = 0.20


@dataclass(frozen=True, slots=True)
class _Anchor:
    anchor_ref: str
    entity_id: str
    surface: str


def _semantic_word_key(value: str) -> str:
    """Small inflection normalizer shared by direct lexical evidence."""

    token = str(value).casefold()
    if token in {"died", "dead"}:
        return "death"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("sed") and len(token) > 4:
        return token[:-1]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _answer_focus_text(question: str) -> str:
    """Extract the same short answer-target phrase used by graph selection."""

    tokens = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", str(question))
    if not tokens:
        return str(question)
    wh_words = {"what", "which", "who", "where", "when", "whom", "whose"}
    starts = [
        index
        for index, token in enumerate(tokens)
        if token.casefold().removesuffix("'s") in wh_words
    ]
    early = [index for index in starts if index <= 1]
    start = early[0] if early else (starts[-1] if starts else 0)
    head = tokens[start].casefold().removesuffix("'s")
    remaining = tokens[start:]
    auxiliaries = {
        "do", "does", "did", "has", "have", "had", "can", "could",
        "would", "will", "shall", "should",
    }
    if head in {"what", "which", "whom", "whose"}:
        if len(remaining) > 1 and remaining[1].casefold() in {
            "is", "are", "was", "were",
        }:
            return " ".join([remaining[0], *remaining[2:10]])
        for index, token in enumerate(remaining[1:], start=1):
            if token.casefold() in auxiliaries:
                return " ".join(remaining[:index])
        return " ".join(remaining[:7])
    if head == "who":
        return " ".join(remaining[:6])
    return " ".join(remaining[:8])


def _direct_terminal_lexical_score(
    question: str,
    candidate: Mapping[str, Any],
) -> float:
    """Token-F1 between answer-focus words and terminal predicate names."""

    ignored = {
        "a", "an", "are", "did", "do", "does", "has", "have", "is",
        "of", "the", "was", "were", "what", "which", "who", "where",
    }
    query_tokens = {
        _semantic_word_key(token)
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9_]*",
            _answer_focus_text(question),
        )
        if token.casefold() not in ignored
    }
    relation_tokens: set[str] = set()
    answer_var = str(candidate.get("answer_var", "V0"))
    for triple in candidate.get("triples", []):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation_id, object_ = map(str, triple)
        if answer_var not in {subject, object_}:
            continue
        relation_tokens.update(
            _semantic_word_key(token)
            for token in re.findall(
                r"[A-Za-z0-9]+",
                relation_id.replace(".", " ").replace("_", " "),
            )
        )
    if not query_tokens or not relation_tokens:
        return 0.0
    overlap = len(query_tokens & relation_tokens)
    precision = overlap / len(relation_tokens)
    recall = overlap / len(query_tokens)
    return (
        0.0
        if precision + recall == 0.0
        else (2.0 * precision * recall) / (precision + recall)
    )


def _select_low_confidence_direct_challenger(
    question: str,
    incumbent: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Choose a narrow challenger from already executed direct candidates.

    This selector is intentionally usable only after the incumbent failed the
    outer confidence gate and bounded best-effort is enabled by its caller.
    It has two schema/shape rules: project away an anonymous CVT singleton to
    a higher-scoring named singleton, or replace a multi-answer incumbent by a
    nearly tied singleton whose terminal predicate has substantially stronger
    lexical evidence.  It never sees Gold answers, IDs from evaluation data,
    or a relation allow-list.
    """

    incumbent_graph_id = str(incumbent.get("graph_id", ""))
    incumbent_score = float(incumbent.get("score", 0.0))
    incumbent_count = int(incumbent.get("answer_count", 0))
    incumbent_lexical = _direct_terminal_lexical_score(question, incumbent)
    evidence: dict[str, Any] = {
        "applied": False,
        "source_graph_id": incumbent_graph_id,
        "source_score": incumbent_score,
        "source_answer_count": incumbent_count,
        "source_terminal_is_cvt": bool(incumbent.get("answer_is_cvt", False)),
        "source_terminal_lexical": incumbent_lexical,
        "uses_existing_executions_only": True,
        "uses_gold_answers": False,
        "entity_or_relation_whitelist": False,
        "additional_execution_queries": 0,
        "additional_llm_calls": 0,
        "additional_embedding_calls": 0,
    }

    if bool(incumbent.get("answer_is_cvt", False)):
        cvt_projection = [
            candidate
            for candidate in candidates
            if str(candidate.get("graph_id", "")) != incumbent_graph_id
            and int(candidate.get("answer_count", 0)) == 1
            and not bool(candidate.get("answer_is_cvt", False))
            and float(candidate.get("score", 0.0)) > incumbent_score + 1e-12
        ]
        if cvt_projection:
            selected = max(
                cvt_projection,
                key=lambda candidate: (
                    float(candidate.get("score", 0.0)),
                    str(candidate.get("graph_id", "")),
                ),
            )
            evidence.update(
                {
                    "applied": True,
                    "reason": "higher_score_non_cvt_singleton_projection",
                    "selected_graph_id": str(selected.get("graph_id", "")),
                    "selected_score": float(selected.get("score", 0.0)),
                    "selected_answer_count": 1,
                    "selected_terminal_is_cvt": False,
                    "selected_terminal_lexical": (
                        _direct_terminal_lexical_score(question, selected)
                    ),
                    "eligible_graph_ids": [
                        str(candidate.get("graph_id", ""))
                        for candidate in sorted(
                            cvt_projection,
                            key=lambda candidate: (
                                -float(candidate.get("score", 0.0)),
                                str(candidate.get("graph_id", "")),
                            ),
                        )
                    ],
                }
            )
            return str(selected.get("graph_id", "")), evidence

    if incumbent_count > 1:
        lexical_singletons: list[tuple[float, Mapping[str, Any]]] = []
        for candidate in candidates:
            if (
                str(candidate.get("graph_id", "")) == incumbent_graph_id
                or int(candidate.get("answer_count", 0)) != 1
                or incumbent_score - float(candidate.get("score", 0.0))
                > _LOW_CONFIDENCE_SINGLETON_SCORE_WINDOW + 1e-12
            ):
                continue
            lexical = _direct_terminal_lexical_score(question, candidate)
            if lexical - incumbent_lexical + 1e-12 < _LOW_CONFIDENCE_LEXICAL_DELTA:
                continue
            lexical_singletons.append((lexical, candidate))
        if lexical_singletons:
            selected_lexical, selected = max(
                lexical_singletons,
                key=lambda item: (
                    item[0],
                    float(item[1].get("score", 0.0)),
                    str(item[1].get("graph_id", "")),
                ),
            )
            evidence.update(
                {
                    "applied": True,
                    "reason": "near_score_lexically_stronger_singleton",
                    "selected_graph_id": str(selected.get("graph_id", "")),
                    "selected_score": float(selected.get("score", 0.0)),
                    "selected_answer_count": 1,
                    "selected_terminal_is_cvt": bool(
                        selected.get("answer_is_cvt", False)
                    ),
                    "selected_terminal_lexical": selected_lexical,
                    "maximum_score_lag": _LOW_CONFIDENCE_SINGLETON_SCORE_WINDOW,
                    "minimum_lexical_delta": _LOW_CONFIDENCE_LEXICAL_DELTA,
                    "eligible_graph_ids": [
                        str(candidate.get("graph_id", ""))
                        for _, candidate in sorted(
                            lexical_singletons,
                            key=lambda item: (
                                -item[0],
                                -float(item[1].get("score", 0.0)),
                                str(item[1].get("graph_id", "")),
                            ),
                        )
                    ],
                }
            )
            return str(selected.get("graph_id", "")), evidence

    evidence["reason"] = "no_safe_low_confidence_challenger"
    return "", evidence


def _failure_endpoint_retrieval_plugin_question_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _failure_endpoint_retrieval_plugin_decompositions(context: Any) -> list[str]:
    """Read the latest reviewed decomposition without invoking a model."""

    values: list[str] = []
    artifact = context.trace_bundle.get("decomposition", {})
    traces = artifact.get("traces", []) if isinstance(artifact, dict) else []
    for trace in reversed(traces if isinstance(traces, list) else []):
        if (
            isinstance(trace, dict)
            and trace.get("stage") == "decomposition_review_and_rewrite"
        ):
            output = trace.get("output")
            candidates = (
                output.get("final_candidates", [])
                if isinstance(output, dict)
                else []
            )
            for candidate in candidates if isinstance(candidates, list) else []:
                if not isinstance(candidate, dict):
                    continue
                values.extend(
                    str(item)
                    for item in candidate.get("decomposition", [])
                    if str(item).strip()
                )
            if values:
                return list(dict.fromkeys(values))
            # Do not resurrect an older review attempt when the latest one is
            # empty or malformed; the immutable input store is the safe
            # fallback.
            break

    try:
        candidates = context.pipeline.decompositions.get(context.question)
    except (AttributeError, KeyError):
        candidates = []
    for candidate in candidates:
        values.extend(
            str(item)
            for item in getattr(candidate, "decomposition", [])
            if str(item).strip()
        )
    if values:
        return list(dict.fromkeys(values))

    # Trace-only replay remains supported when the store is unavailable.
    for trace in reversed(traces if isinstance(traces, list) else []):
        if not isinstance(trace, dict) or trace.get("stage") != "decompose_predictions":
            continue
        candidates = trace.get("output", [])
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            values.extend(
                str(item)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            )
        if values:
            break
    return list(dict.fromkeys(values))


def _anchors(context: Any) -> list[_Anchor]:
    """Normalize the pipeline's already-linked entities in mention order."""

    grounder = context.pipeline.grounder
    entities = getattr(grounder, "gold_entities", {}).get(
        _failure_endpoint_retrieval_plugin_question_key(context.question),
        {},
    )
    items = [
        (str(entity_id), str(label))
        for entity_id, label in entities.items()
        if str(entity_id).strip() and str(label).strip()
    ]
    normalized = context.question.casefold()
    items.sort(
        key=lambda item: (
            normalized.find(item[1].casefold())
            if item[1].casefold() in normalized
            else len(normalized),
            item[1],
            item[0],
        )
    )
    seen_entities: set[str] = set()
    output: list[_Anchor] = []
    for entity_id, label in items:
        if entity_id in seen_entities:
            continue
        seen_entities.add(entity_id)
        output.append(_Anchor(f"A{len(output)}", entity_id, label))
    return output


def _retriever(context: Any, endpoint: Any) -> _failure_endpoint_retrieval_plugin_FailurePathRetriever:
    pipeline = context.pipeline
    with _RETRIEVER_CACHE_LOCK:
        cached = getattr(pipeline, "_failure_endpoint_path_retriever", None)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] is endpoint
        ):
            return cached[1]
        value = _failure_endpoint_retrieval_plugin_FailurePathRetriever(
            ontology=pipeline.grounder.ontology,
            kg=endpoint,
            ranker=_LOCAL_RANKER,
            top_k=24,
            query_limit=_PATH_QUERY_LIMIT,
            max_depth=1,
        )
        # Keep a strong reference to the endpoint next to the retriever.  This
        # avoids object-id reuse and binds cached path rows to exactly one KG.
        pipeline._failure_endpoint_path_retriever = (endpoint, value)
    return value


def _terminal_type(candidate: _failure_endpoint_retrieval_plugin_FailurePathCandidate, ontology: Any) -> str:
    relation_id = candidate.relation_ids[-1]
    if candidate.directions[-1] == "forward":
        return str(ontology.range_for_relation(relation_id) or "")
    return str(ontology.domain_for_relation(relation_id) or "")


def _word_root(value: str) -> str:
    value = str(value).casefold()
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 4 and value.endswith("es"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s"):
        return value[:-1]
    return value


def _question_head_tokens(question: str) -> tuple[str, ...]:
    """Extract a small What/Which answer-head phrase without a type list."""

    tokens = _failure_endpoint_retrieval_plugin_TOKEN_RE.findall(str(question).casefold())
    if not tokens or tokens[0] not in {"what", "which"}:
        return ()
    position = 1
    while position < len(tokens) and tokens[position] in _QUESTION_PREFIX_SKIP:
        position += 1
    head: list[str] = []
    for token in tokens[position:]:
        if token in _QUESTION_BOUNDARIES:
            break
        head.append(_word_root(token))
        if len(head) >= 3:
            break
    return tuple(dict.fromkeys(value for value in head if value))


def _type_leaf_tokens(type_id: str) -> set[str]:
    leaf = str(type_id).rsplit(".", 1)[-1]
    return {
        _word_root(token)
        for token in _failure_endpoint_retrieval_plugin_TOKEN_RE.findall(leaf.replace("_", " "))
        if token
    }


def _answer_type_evidence(
    question: str,
    terminal_types: Sequence[str],
    ontology: Any,
) -> tuple[float, tuple[str, ...]]:
    head = _question_head_tokens(question)
    if not head:
        return 0.0, ()
    head_set = set(head)
    best = 0.0
    for type_id in terminal_types:
        if not type_id:
            continue
        schema_types = [type_id, *ontology.supertypes(type_id)]
        for schema_type in schema_types:
            leaf_tokens = _type_leaf_tokens(schema_type)
            overlap = len(head_set & leaf_tokens)
            if overlap:
                best = max(
                    best,
                    overlap / max(1, min(len(head_set), len(leaf_tokens))),
                )
    return best, head


def _is_cvt_terminal(type_id: str, ontology: Any) -> bool:
    """Infer anonymous-record terminals from schema, never a type list."""

    if not type_id or str(type_id).startswith("type."):
        return False
    explicit = getattr(ontology, "is_cvt_type", None)
    if callable(explicit):
        return bool(explicit(type_id))
    return "common.topic" not in set(ontology.supertypes(type_id))


def _terminal_compatibility(left: str, right: str, ontology: Any) -> float:
    """Require schema evidence before unifying terminals from two anchors."""

    if not left or not right:
        return -1.0
    if left == right:
        return 2.0
    left_supers = set(ontology.supertypes(left))
    right_supers = set(ontology.supertypes(right))
    if left in right_supers or right in left_supers:
        return 1.0
    return -1.0


def _compatible_with(
    candidate: _failure_endpoint_retrieval_plugin_FailurePathCandidate,
    previous: Sequence[_failure_endpoint_retrieval_plugin_FailurePathCandidate],
    ontology: Any,
) -> tuple[bool, float]:
    candidate_type = _terminal_type(candidate, ontology)
    scores = [
        _terminal_compatibility(
            candidate_type,
            _terminal_type(other, ontology),
            ontology,
        )
        for other in previous
    ]
    return bool(scores) and all(score >= 0.0 for score in scores), min(scores, default=0.0)


def _intersection_combinations(
    grouped: Mapping[str, list[_failure_endpoint_retrieval_plugin_FailurePathCandidate]],
    anchor_order: Sequence[str],
    ontology: Any,
    *,
    beam: int = _INTERSECTION_BEAM,
) -> tuple[list[tuple[_failure_endpoint_retrieval_plugin_FailurePathCandidate, ...]], int]:
    """Build a bounded, pairwise terminal-compatible cross-anchor beam."""

    if len(anchor_order) < 2 or any(not grouped.get(anchor) for anchor in anchor_order):
        return [], 0
    states: list[tuple[tuple[_failure_endpoint_retrieval_plugin_FailurePathCandidate, ...], float]] = [
        ((candidate,), float(candidate.score))
        for candidate in grouped[anchor_order[0]]
    ]
    rejected = 0
    for anchor_ref in anchor_order[1:]:
        expanded: list[tuple[tuple[_failure_endpoint_retrieval_plugin_FailurePathCandidate, ...], float]] = []
        for previous, previous_score in states:
            for candidate in grouped[anchor_ref]:
                compatible, type_score = _compatible_with(candidate, previous, ontology)
                if not compatible:
                    rejected += 1
                    continue
                combination = (*previous, candidate)
                # Mean path relevance keeps scores comparable as anchor count grows.
                score = (
                    (previous_score * len(previous))
                    + float(candidate.score)
                    + (0.01 * type_score)
                ) / len(combination)
                expanded.append((combination, score))
        expanded.sort(
            key=lambda item: (
                -item[1],
                tuple(
                    (path.anchor_ref, path.directions, path.relation_ids)
                    for path in item[0]
                ),
            )
        )
        states = expanded[: max(1, int(beam))]
        if not states:
            break
    return [item[0] for item in states], rejected


def _grounded_candidate(
    paths: Sequence[_failure_endpoint_retrieval_plugin_FailurePathCandidate],
    anchors: Mapping[str, _Anchor],
    question: str,
    ontology: Any,
    *,
    total_anchor_count: int,
) -> _failure_endpoint_retrieval_plugin_GroundedSemanticCandidate:
    semantic_paths: list[dict[str, Any]] = []
    relation_bindings: dict[str, str] = {}
    for path_index, path in enumerate(paths):
        path_id = f"P{path_index}"
        source = path.anchor_ref
        steps: list[dict[str, Any]] = []
        for step_index, (relation_id, direction) in enumerate(
            zip(path.relation_ids, path.directions)
        ):
            step_id = f"{path_id}.S{step_index}"
            target = f"{path_id}.V{step_index}"
            steps.append(
                {
                    "id": step_id,
                    "relation_label": _failure_endpoint_retrieval_plugin_relation_label_from_id(relation_id),
                    "direction": direction,
                    "from": source,
                    "to": target,
                }
            )
            relation_bindings[step_id] = relation_id
            source = target
        semantic_paths.append(
            {
                "id": path_id,
                "anchor_ref": path.anchor_ref,
                "goal": question,
                "steps": steps,
                "path_output_var": source,
            }
        )
    used = {path.anchor_ref for path in paths}
    terminal_types = tuple(_terminal_type(path, ontology) for path in paths)
    answer_is_cvt = any(
        _is_cvt_terminal(type_id, ontology) for type_id in terminal_types
    )
    type_evidence, answer_head = _answer_type_evidence(
        question,
        terminal_types,
        ontology,
    )
    return _failure_endpoint_retrieval_plugin_GroundedSemanticCandidate(
        compose_input={
            "question": question,
            "anchors": [
                {"id": anchor_ref, "surface": anchors[anchor_ref].surface}
                for anchor_ref in sorted(used)
            ],
            "semantic_paths": semantic_paths,
        },
        anchor_bindings={
            anchor_ref: _failure_endpoint_retrieval_plugin_EntityCandidate(
                anchors[anchor_ref].entity_id,
                anchors[anchor_ref].surface,
                1.0,
                "failure_endpoint_path",
            )
            for anchor_ref in used
        },
        relation_bindings=relation_bindings,
        score=sum(float(path.score) for path in paths) / max(1, len(paths)),
        provenance={
            "failure_endpoint_path_retrieval": {
                "paths": [path.to_dict() for path in paths],
                "anchor_coverage": len(used),
                "total_anchor_count": int(total_anchor_count),
                "terminal_types": list(terminal_types),
                "answer_head_tokens": list(answer_head),
                "answer_type_evidence": type_evidence,
                "answer_is_cvt": answer_is_cvt,
                "model_used": False,
                "uses_gold": False,
            }
        },
    )


def _compile_query_graph(
    grounded: _failure_endpoint_retrieval_plugin_GroundedSemanticCandidate,
    *,
    graph_id: str,
    pipeline_version: str,
) -> _failure_endpoint_retrieval_plugin_QueryGraphCandidate | None:
    """Lower grounded paths, sharing only their verified terminal variable."""

    semantic_paths = [
        path
        for path in grounded.compose_input.get("semantic_paths", [])
        if isinstance(path, dict)
    ]
    if not semantic_paths:
        return None
    triples: list[list[str]] = []
    next_variable = 1
    for path in semantic_paths:
        anchor_ref = str(path.get("anchor_ref", ""))
        anchor = grounded.anchor_bindings.get(anchor_ref)
        steps = [step for step in path.get("steps", []) if isinstance(step, dict)]
        if anchor is None or not steps:
            return None
        source = str(anchor.entity_id)
        for step_index, step in enumerate(steps):
            relation_id = str(grounded.relation_bindings.get(str(step.get("id", "")), ""))
            direction = str(step.get("direction", ""))
            if not relation_id or direction not in {"forward", "backward"}:
                return None
            if step_index == len(steps) - 1:
                target = "V0"
            else:
                target = f"V{next_variable}"
                next_variable += 1
            triple = (
                [source, relation_id, target]
                if direction == "forward"
                else [target, relation_id, source]
            )
            if triple not in triples:
                triples.append(triple)
            source = target
    if not triples or not any("V0" in (triple[0], triple[2]) for triple in triples):
        return None
    anchor_ids = sorted(
        {candidate.entity_id for candidate in grounded.anchor_bindings.values()}
    )
    path_count = len(semantic_paths)
    grounded_metadata = grounded.provenance.get(
        "failure_endpoint_path_retrieval",
        {},
    )
    score = (10.0 * float(grounded.score)) + (4.0 * path_count)
    if path_count > 1:
        score += 8.0 * (path_count - 1)
    return _failure_endpoint_retrieval_plugin_QueryGraphCandidate(
        graph_id=graph_id,
        triples=triples,
        answer_var="V0",
        operators=[
            {
                "type": "NO_EQUAL",
                "inputs": [],
                "input_var": "V0",
                "attribute_relation_label": [],
                "attribute_relation_labels": [],
                "value": entity_id,
                "value_type": "mid",
                "_value_entity_id": entity_id,
                "_failure_endpoint_anchor_exclusion": True,
            }
            for entity_id in anchor_ids
        ],
        score=score,
        compose_output={
            "source": "failure_endpoint_path_retrieval",
            "strategy": "terminal_intersection" if path_count > 1 else "single_path",
        },
        provenance={
            "version": str(pipeline_version),
            "anchor_bindings": {
                key: {
                    "id": value.entity_id,
                    "label": value.label,
                    "score": value.score,
                }
                for key, value in grounded.anchor_bindings.items()
            },
            "relation_bindings": dict(grounded.relation_bindings),
            "grounded_semantic": grounded.provenance,
            "graph_fallback": {
                "source": "typed_spine_intersection_after_unsuccessful_graphs",
                "model_used": False,
            },
            "failure_endpoint_path_retrieval": {
                "path_count": path_count,
                "anchor_coverage": int(
                    grounded_metadata.get("anchor_coverage", path_count)
                ),
                "total_anchor_count": int(
                    grounded_metadata.get("total_anchor_count", path_count)
                ),
                "terminal_types": list(
                    grounded_metadata.get("terminal_types", [])
                ),
                "answer_head_tokens": list(
                    grounded_metadata.get("answer_head_tokens", [])
                ),
                "answer_type_evidence": float(
                    grounded_metadata.get("answer_type_evidence", 0.0)
                ),
                "answer_is_cvt": bool(
                    grounded_metadata.get("answer_is_cvt", False)
                ),
                "terminal_intersection": path_count > 1,
                "model_used": False,
                "uses_gold": False,
            },
        },
    )


def _query_graphs(
    grounded: Iterable[_failure_endpoint_retrieval_plugin_GroundedSemanticCandidate],
    *,
    pipeline_version: str,
    limit: int = _EXECUTION_BUDGET,
) -> list[_failure_endpoint_retrieval_plugin_QueryGraphCandidate]:
    best: dict[str, _failure_endpoint_retrieval_plugin_QueryGraphCandidate] = {}
    for item in grounded:
        graph = _compile_query_graph(
            item,
            graph_id="FEP",
            pipeline_version=pipeline_version,
        )
        if graph is None:
            continue
        signature = json.dumps(
            {
                "triples": graph.triples,
                "answer_var": graph.answer_var,
                "operators": graph.operators,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        previous = best.get(signature)
        if previous is None or graph.score > previous.score:
            best[signature] = graph
    output = sorted(
        best.values(),
        key=lambda graph: (
            -int(
                float(
                    graph.provenance["failure_endpoint_path_retrieval"].get(
                        "answer_type_evidence",
                        0.0,
                    )
                )
                > 0.0
            ),
            int(
                graph.provenance["failure_endpoint_path_retrieval"].get(
                    "answer_is_cvt",
                    False,
                )
            ),
            -int(
                graph.provenance["failure_endpoint_path_retrieval"].get(
                    "anchor_coverage",
                    0,
                )
            ),
            -graph.score,
            graph.triples,
        ),
    )[: max(1, min(_EXECUTION_BUDGET, int(limit)))]
    for rank, graph in enumerate(output, start=1):
        graph.graph_id = f"FEP{rank}"
        graph.provenance["failure_endpoint_path_retrieval"]["rank"] = rank
        graph.provenance["failure_endpoint_path_retrieval"]["execution_budget"] = len(output)
    return output


def _failure_endpoint_retrieval_plugin_retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    """Harness plugin entry point for direct endpoint path retrieval."""

    pipeline = context.pipeline
    verbose_diagnostics = bool(getattr(context, "verbose_diagnostics", True))
    linked_anchors = _anchors(context)
    anchors = linked_anchors[:_MAX_SUBSET_ANCHORS]
    decompositions = _failure_endpoint_retrieval_plugin_decompositions(context)
    diagnostics: dict[str, Any] = {
        "status": "not_compiled",
        "strategy": "direct_failure_endpoint_paths",
        "llm_calls": 0,
        "external_embedding_calls": 0,
        "local_lexical_ranking": True,
        "uses_gold_answers": False,
        "anchor_count": len(anchors),
        "linked_anchor_count": len(linked_anchors),
        "dropped_anchor_count": len(linked_anchors) - len(anchors),
        "execution_budget": _EXECUTION_BUDGET,
        "path_query_limit": _PATH_QUERY_LIMIT,
        "path_max_depth": 1,
    }
    ontology = getattr(pipeline.grounder, "ontology", None)
    if not anchors or ontology is None:
        diagnostics["reason"] = "missing_anchor_or_ontology"
        return {"answer_ids": [], "diagnostics": diagnostics}

    anchor_by_ref = {anchor.anchor_ref: anchor for anchor in anchors}
    candidates, path_diagnostics = _retriever(context, endpoint).retrieve(
        question=context.question,
        decompositions=decompositions,
        anchor_bindings={
            anchor.anchor_ref: _failure_endpoint_retrieval_plugin_EntityCandidate(
                anchor.entity_id,
                anchor.surface,
                1.0,
                "existing_linked_anchor",
            )
            for anchor in anchors
        },
    )
    diagnostics["path_retrieval"] = (
        path_diagnostics
        if verbose_diagnostics
        else {
            key: value
            for key, value in path_diagnostics.items()
            if key
            in {
                "strategy",
                "top_k",
                "query_limit",
                "max_depth",
                "anchor_count",
                "direction_pattern_count",
                "query_count",
                "cache_hits",
                "query_elapsed_seconds",
                "ranking_elapsed_seconds",
                "raw_candidate_count",
                "deduplicated_candidate_count",
                "schema_rejected_count",
                "retained_candidate_count",
                "retained_by_depth",
                "elapsed_seconds",
                "code",
            }
        }
    )
    grouped: dict[str, list[_failure_endpoint_retrieval_plugin_FailurePathCandidate]] = {
        anchor.anchor_ref: [] for anchor in anchors
    }
    for candidate in candidates:
        grouped.setdefault(candidate.anchor_ref, []).append(candidate)

    path_combinations: list[tuple[_failure_endpoint_retrieval_plugin_FailurePathCandidate, ...]] = []
    rejected_intersections = 0
    anchor_refs = [anchor.anchor_ref for anchor in anchors]
    subset_total = (2 ** len(anchor_refs)) - 1
    per_subset_budget = max(
        1,
        min(_INTERSECTION_BEAM, _SUBSET_COMBINATION_BUDGET // max(1, subset_total)),
    )
    subset_counts: dict[str, int] = {}
    for subset_size in range(len(anchor_refs), 0, -1):
        count_before = len(path_combinations)
        for subset in anchor_subsets(anchor_refs, subset_size):
            if subset_size == 1:
                path_combinations.extend(
                    (candidate,)
                    for candidate in grouped.get(subset[0], [])[:per_subset_budget]
                )
                continue
            values, rejected = _intersection_combinations(
                grouped,
                subset,
                ontology,
                beam=per_subset_budget,
            )
            path_combinations.extend(values)
            rejected_intersections += rejected
        subset_counts[str(subset_size)] = len(path_combinations) - count_before
    grounded = [
        _grounded_candidate(
            combination,
            anchor_by_ref,
            context.question,
            ontology,
            total_anchor_count=len(anchors),
        )
        for combination in path_combinations
    ]
    graphs = _query_graphs(
        grounded,
        pipeline_version=str(pipeline.version),
        limit=_EXECUTION_BUDGET,
    )
    diagnostics["compilation"] = {
        "mode": "all_nonempty_anchor_subsets",
        "combination_count": len(path_combinations),
        "combination_count_by_coverage": subset_counts,
        "subset_count": subset_total,
        "per_subset_combination_budget": per_subset_budget,
        "global_combination_budget": _SUBSET_COMBINATION_BUDGET,
        "terminal_type_rejected_count": rejected_intersections,
        "grounded_candidate_count": len(grounded),
        "query_graph_count": len(graphs),
    }

    executions: list[Any] = []
    execution_traces: list[dict[str, Any]] = []
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    for graph in graphs:
        graph.provenance["original_question"] = context.question
        graph.provenance["decomposition"] = list(decompositions)
        executed, trace = pipeline._execute_graph_candidate(
            graph,
            lookup_labels=False,
        )
        execution_traces.append(
            {
                key: trace[key]
                for key in (
                    "graph_id",
                    "answer_count",
                    "row_count",
                    "query_elapsed_seconds",
                    "elapsed_seconds",
                    "error",
                )
                if key in trace
            }
        )
        if executed is not None and executed.answer_ids:
            executions.append(executed)
    diagnostics["executions"] = execution_traces
    runtime_candidate_records = [
        {
            "graph_id": item.graph.graph_id,
            "score": item.graph.score,
            "answer_count": len(item.answer_ids),
            "answer_ids": list(item.answer_ids),
            "triples": graph_by_id[item.graph.graph_id].triples,
            "answer_var": str(item.graph.answer_var),
            "anchor_coverage": int(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "anchor_coverage",
                    0,
                )
            ),
            "total_anchor_count": int(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "total_anchor_count",
                    0,
                )
            ),
            "terminal_types": list(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "terminal_types",
                    [],
                )
            ),
            "answer_type_evidence": float(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "answer_type_evidence",
                    0.0,
                )
            ),
            "answer_is_cvt": bool(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "answer_is_cvt",
                    False,
                )
            ),
        }
        for item in executions
    ]
    diagnostics["execution_candidates"] = [
        (
            dict(candidate)
            if verbose_diagnostics
            else {
                key: value
                for key, value in candidate.items()
                if key not in {"answer_ids", "triples", "answer_var"}
            }
        )
        for candidate in runtime_candidate_records
    ]
    if not executions:
        diagnostics["reason"] = "all_endpoint_path_graphs_empty"
        return {"answer_ids": [], "diagnostics": diagnostics}

    typed = [
        item
        for item in executions
        if float(
            item.graph.provenance["failure_endpoint_path_retrieval"].get(
                "answer_type_evidence",
                0.0,
            )
        )
        > 0.0
    ]
    selection_pool = typed or executions
    non_cvt = [
        item
        for item in selection_pool
        if not bool(
            item.graph.provenance["failure_endpoint_path_retrieval"].get(
                "answer_is_cvt",
                False,
            )
        )
    ]
    selection_pool = non_cvt or selection_pool
    selected = max(
        selection_pool,
        key=lambda item: (
            int(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "anchor_coverage",
                    0,
                )
            ),
            item.graph.score,
            -len(item.answer_ids),
            item.graph.graph_id,
        ),
    )
    def populate_selected_diagnostics(item: Any) -> None:
        selected_set = set(item.answer_ids)
        selected_answer_set_support = sum(
            set(execution.answer_ids) == selected_set for execution in executions
        )
        selected_metadata = item.graph.provenance[
            "failure_endpoint_path_retrieval"
        ]
        selected_terminal_types = [
            str(value)
            for value in selected_metadata.get("terminal_types", [])
            if str(value)
        ]
        diagnostics["status"] = "selected"
        diagnostics["selected_graph_id"] = item.graph.graph_id
        diagnostics["answer_count"] = len(item.answer_ids)
        diagnostics["selected_anchor_coverage"] = int(
            selected_metadata.get("anchor_coverage", 0)
        )
        diagnostics["selected_answer_type_evidence"] = float(
            selected_metadata.get("answer_type_evidence", 0.0)
        )
        diagnostics["selected_answer_set_support"] = selected_answer_set_support
        diagnostics["nonempty_candidate_count"] = len(executions)
        diagnostics["all_nonempty_candidates_agree"] = bool(
            executions and selected_answer_set_support == len(executions)
        )
        diagnostics["selected_terminal_types"] = selected_terminal_types
        diagnostics["selected_terminal_is_cvt"] = any(
            _is_cvt_terminal(type_id, ontology)
            for type_id in selected_terminal_types
        )
        diagnostics["selected_answer_is_cvt"] = bool(
            selected_metadata.get("answer_is_cvt", False)
        )

    populate_selected_diagnostics(selected)
    if bool(
        getattr(
            pipeline,
            "failure_endpoint_return_bounded_direct_best_effort",
            False,
        )
    ):
        # Import lazily: the failure plugin invokes this module as its final
        # lane, so the shared outer gate is fully initialized at this point.
        from .failure_retrieval import _direct_path_acceptance

        original_accepted, _, _ = _direct_path_acceptance(
            context.question,
            list(selected.answer_ids),
            diagnostics,
        )
        if not original_accepted:
            incumbent_record = next(
                candidate
                for candidate in runtime_candidate_records
                if str(candidate.get("graph_id", "")) == selected.graph.graph_id
            )
            challenger_graph_id, challenger_evidence = (
                _select_low_confidence_direct_challenger(
                    context.question,
                    incumbent_record,
                    runtime_candidate_records,
                )
            )
            diagnostics["low_confidence_challenger"] = {
                **challenger_evidence,
                "best_effort_enabled": True,
                "original_direct_confidence_gate_rejected": True,
            }
            if challenger_graph_id:
                selected = next(
                    execution
                    for execution in executions
                    if execution.graph.graph_id == challenger_graph_id
                )
                populate_selected_diagnostics(selected)

    # Exactly one label lookup is performed, after the final already-executed
    # candidate is fixed.  Challenging never executes another graph query.
    selected_labels: list[dict[str, str]] = []
    try:
        selected_labels = endpoint.labels(list(selected.answer_ids))
    except (RuntimeError, TypeError, ValueError) as exc:
        diagnostics["selected_label_error"] = f"{type(exc).__name__}:{exc}"
    diagnostics["selected_label_lookup_count"] = int(bool(selected.answer_ids))
    labels_by_id = {
        str(item.get("id", "")): {
            "id": str(item.get("id", "")),
            "label": str(item.get("label", "")),
        }
        for item in selected_labels
        if isinstance(item, dict) and str(item.get("id", ""))
    }
    answers = [
        labels_by_id.get(answer_id, {"id": answer_id, "label": answer_id})
        for answer_id in selected.answer_ids
    ]
    if verbose_diagnostics:
        diagnostics["selected_labels"] = answers
    graph_summary = getattr(pipeline, "_graph_summary", None)
    selected_graph = (
        graph_summary(selected.graph)
        if callable(graph_summary)
        else {
            "graph_id": selected.graph.graph_id,
            "triples": list(selected.graph.triples),
            "answer_var": selected.graph.answer_var,
            "operators": list(selected.graph.operators),
            "score": selected.graph.score,
            "provenance": dict(selected.graph.provenance),
        }
    )
    return {
        "answer_ids": list(selected.answer_ids),
        "answers": answers,
        "selected_graph": selected_graph,
        "diagnostics": diagnostics,
    }






# ---- failure_factorized_path_retrieval ----
"""Bounded final-empty fallback over cached CWQ TRAIN relation paths.

This module is intentionally a terminal failure lane.  Its caller must invoke
it only after earlier failure retrieval lanes have all returned no answer.  It
uses the current question, saved decomposition/semantic trace, configured
entity links, local ontology, and a read-only cache built from CWQ TRAIN Gold
SPARQL.  It makes no LLM or embedding call and executes at most three final
SELECT queries.

The frozen policy independently retrieves a constant-to-answer path for every
linked anchor, joins two or more paths at their common answer terminal, and
uses answer-set consensus followed by retrieval score for runtime selection.
Raw queries are never emitted in diagnostics; only SHA-256 hashes are kept.
"""



from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import heapq
from itertools import product
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Sequence

_failure_factorized_path_retrieval_decompositions = _failure_gold_sparql_retrieval_decompositions
_failure_factorized_path_retrieval_entities = _failure_gold_sparql_retrieval_entities
_failure_factorized_path_retrieval_trace_surfaces = _failure_gold_sparql_retrieval_trace_surfaces
from .ontology import FreebaseOntology as _failure_factorized_path_retrieval_FreebaseOntology, relation_id_from_label as _failure_factorized_path_retrieval_relation_id_from_label
from .pipeline import _answer_values as _failure_factorized_path_retrieval_answer_values
from .template_path_retrieval import mask_entities as _failure_factorized_path_retrieval_mask_entities


_failure_factorized_path_retrieval_VERSION = "failure-factorized-source-train-path-production-v1-frozen"
_failure_factorized_path_retrieval_CACHE_VERSION = 2
_failure_factorized_path_retrieval_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_failure_factorized_path_retrieval_FULL_DATE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_failure_factorized_path_retrieval_YEAR_RE = re.compile(r"(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d.])")
_failure_factorized_path_retrieval_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
_failure_factorized_path_retrieval_ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")


def _failure_factorized_path_retrieval_normalized(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _failure_factorized_path_retrieval_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _failure_factorized_path_retrieval_tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _failure_factorized_path_retrieval_TOKEN_RE.finditer(str(value)))


def _failure_factorized_path_retrieval_normalize_scalars(value: Any) -> str:
    text = _failure_factorized_path_retrieval_FULL_DATE_RE.sub(" date ", str(value))
    text = _failure_factorized_path_retrieval_YEAR_RE.sub(" year ", text)
    text = _failure_factorized_path_retrieval_NUMBER_RE.sub(" number ", text)
    return " ".join(text.split())


def _masked(value: Any, surfaces: Sequence[str]) -> str:
    return _failure_factorized_path_retrieval_mask_entities(_failure_factorized_path_retrieval_normalize_scalars(str(value)), surfaces)


def _failure_factorized_path_retrieval_relation_text(relations: Sequence[str]) -> str:
    return " ".join(
        segment.replace("_", " ")
        for relation in relations
        for segment in str(relation).split(".")
    )


def _token_f1(left: Any, right: Any) -> float:
    one, two = set(_failure_factorized_path_retrieval_tokens(left)), set(_failure_factorized_path_retrieval_tokens(right))
    if not one or not two:
        return 0.0
    overlap = len(one & two)
    return 2.0 * overlap / (len(one) + len(two))


def _set_f1(left: Sequence[Any], right: Sequence[Any]) -> float:
    one, two = set(left), set(right)
    if not one or not two:
        return float(one == two)
    overlap = len(one & two)
    return 2.0 * overlap / (len(one) + len(two))


@dataclass(frozen=True, slots=True)
class GoldPath:
    relations: tuple[str, ...]
    directions: tuple[str, ...]

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.relations, self.directions))


@dataclass(frozen=True, slots=True)
class _PathDocument:
    source_index: int
    question: str
    goal: str
    path: GoldPath
    semantic_path: GoldPath
    operator_signature: tuple[str, ...]
    tokens: tuple[str, ...]
    alignment: tuple[float, ...]
    source_kind: str


class TrainGoldPathIndex:
    """Read-only sparse BM25 index over the prebuilt TRAIN path cache."""

    def __init__(self, cache_path: str | Path) -> None:
        started = time.perf_counter()
        source = Path(cache_path).expanduser()
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _failure_factorized_path_retrieval_CACHE_VERSION
            or not isinstance(payload.get("documents"), list)
        ):
            raise ValueError(f"incompatible TRAIN Gold path cache: {source}")
        self.documents = [
            _PathDocument(
                source_index=int(item["source_index"]),
                question=str(item.get("question", "")),
                goal=str(item.get("goal", "")),
                path=GoldPath(
                    tuple(map(str, item.get("relations", []))),
                    tuple(map(str, item.get("directions", []))),
                ),
                semantic_path=GoldPath(
                    tuple(map(str, item.get("semantic_relations", []))),
                    tuple(map(str, item.get("semantic_directions", []))),
                ),
                operator_signature=tuple(
                    map(str, item.get("operator_signature", []))
                ),
                tokens=tuple(map(str, item.get("tokens", []))),
                alignment=tuple(map(float, item.get("alignment", []))),
                source_kind=str(item.get("source_kind", "semantic_aligned")),
            )
            for item in payload["documents"]
            if isinstance(item, dict)
            and item.get("relations")
            and len(item.get("relations", [])) == len(item.get("directions", []))
        ]
        self.frequencies = [Counter(document.tokens) for document in self.documents]
        self.average_length = sum(
            len(document.tokens) for document in self.documents
        ) / max(1, len(self.documents))
        document_frequency: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, frequency in enumerate(self.frequencies):
            document_frequency.update(set(frequency))
            for token, count in frequency.items():
                self.postings[token].append((index, count))
        count = len(self.documents)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.load_seconds = time.perf_counter() - started

    def retrieve(self, query: str, *, top_k: int = 64) -> list[tuple[float, int, _PathDocument]]:
        query_frequency = Counter(_failure_factorized_path_retrieval_tokens(query))
        selected_tokens = sorted(
            query_frequency,
            key=lambda token: (
                self.idf.get(token, 0.0),
                query_frequency[token],
                token,
            ),
            reverse=True,
        )[:28]
        sparse_scores: dict[int, float] = defaultdict(float)
        for token in selected_tokens:
            inverse = self.idf.get(token, 0.0)
            for index, frequency in self.postings.get(token, ()):
                normalization = (
                    0.25
                    + 0.75
                    * len(self.documents[index].tokens)
                    / max(1e-9, self.average_length)
                )
                sparse_scores[index] += inverse * (
                    frequency * 2.2 / (frequency + 1.2 * normalization)
                ) * min(2, query_frequency[token])
        preselected = heapq.nlargest(
            max(256, top_k * 4),
            sparse_scores.items(),
            key=lambda item: (item[1], -item[0]),
        )
        ranked: list[tuple[float, int, _PathDocument]] = []
        for index, lexical_score in preselected:
            document = self.documents[index]
            alignment = document.alignment
            score = float(lexical_score)
            score += 0.35 * float(alignment[1] if len(alignment) > 1 else 0.0)
            score += 0.20 * float(alignment[2] if len(alignment) > 2 else 0.0)
            ranked.append((score, index, document))
        ranked.sort(
            key=lambda value: (-value[0], value[2].source_index, value[1])
        )
        return ranked[: max(1, int(top_k))]


@dataclass(frozen=True, slots=True)
class _SemanticIntent:
    surface: str
    goal: str
    path: GoldPath


def _semantic_intents(context: Any) -> list[_SemanticIntent]:
    traces = (context.trace_bundle.get("decomposition", {}) or {}).get("traces", [])
    output: list[_SemanticIntent] = []
    seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for event in traces if isinstance(traces, list) else []:
        if not isinstance(event, dict) or event.get("stage") != "semantic_path_generation":
            continue
        candidates = event.get("output", [])
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            payload = candidate.get("output") or {}
            anchors = {
                str(anchor.get("id", "")): str(anchor.get("surface", ""))
                for anchor in payload.get("anchors", [])
                if isinstance(anchor, dict)
            }
            for path in payload.get("semantic_paths", []):
                if not isinstance(path, dict):
                    continue
                relations: list[str] = []
                directions: list[str] = []
                for step in path.get("steps", []):
                    if not isinstance(step, dict):
                        continue
                    relation = _failure_factorized_path_retrieval_relation_id_from_label(step.get("relation_label", []))
                    if not relation:
                        continue
                    direction = str(step.get("direction", "forward")).casefold()
                    relations.append(relation)
                    directions.append(
                        direction if direction in {"forward", "backward"} else "forward"
                    )
                if not relations:
                    continue
                surface = anchors.get(str(path.get("anchor_ref", "")), "")
                goal = str(path.get("goal", ""))
                key = (surface, goal, tuple(relations), tuple(directions))
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    _SemanticIntent(
                        surface,
                        goal,
                        GoldPath(tuple(relations), tuple(directions)),
                    )
                )
    return output


def _intent_entity_score(
    intent: _SemanticIntent, label: str, question: str
) -> float:
    one, two = _failure_factorized_path_retrieval_normalized(intent.surface), _failure_factorized_path_retrieval_normalized(label)
    exact = float(bool(one and two and one == two))
    contained = float(bool(one and two and (one in two or two in one)))
    mention = float(bool(two and two in _failure_factorized_path_retrieval_normalized(intent.goal)))
    question_mention = float(bool(two and two in _failure_factorized_path_retrieval_normalized(question)))
    return (
        5.0 * exact
        + 1.5 * contained
        + mention
        + 0.1 * question_mention
        + _token_f1(one, two)
    )


def _assign_intents(
    intents: Sequence[_SemanticIntent],
    entities: Sequence[tuple[str, str]],
    question: str,
) -> dict[str, list[_SemanticIntent]]:
    output: dict[str, list[_SemanticIntent]] = {
        entity_id: [] for entity_id, _ in entities
    }
    for intent in intents:
        ranked = sorted(
            (
                (_intent_entity_score(intent, label, question), entity_id)
                for entity_id, label in entities
            ),
            key=lambda value: (-value[0], value[1]),
        )
        if ranked and ranked[0][0] >= 1.0:
            output[ranked[0][1]].append(intent)
    return output


def _local_clauses(
    label: str,
    decomposition: Sequence[str],
    intents: Sequence[_SemanticIntent],
) -> list[str]:
    needle = _failure_factorized_path_retrieval_normalized(label)
    explicit = [
        str(value)
        for value in decomposition
        if needle and needle in _failure_factorized_path_retrieval_normalized(value)
    ]
    goals = [intent.goal for intent in intents if intent.goal]
    return _failure_factorized_path_retrieval_unique([*explicit, *goals]) or list(map(str, decomposition))


def _failure_factorized_path_retrieval_path_similarity(
    left: GoldPath, right: GoldPath
) -> tuple[float, float, float, float]:
    return (
        float(left.pairs == right.pairs),
        _set_f1(left.pairs, right.pairs),
        _set_f1(left.relations, right.relations),
        min(len(left.relations), len(right.relations))
        / max(1, max(len(left.relations), len(right.relations))),
    )


def _path_start_type(path: GoldPath, ontology: _failure_factorized_path_retrieval_FreebaseOntology) -> str:
    if not path.relations:
        return ""
    relation, direction = path.relations[0], path.directions[0]
    return (
        ontology.domain_for_relation(relation)
        if direction == "forward"
        else ontology.range_for_relation(relation)
    )


def _path_terminal_type(path: GoldPath, ontology: _failure_factorized_path_retrieval_FreebaseOntology) -> str:
    if not path.relations:
        return ""
    relation, direction = path.relations[-1], path.directions[-1]
    return (
        ontology.range_for_relation(relation)
        if direction == "forward"
        else ontology.domain_for_relation(relation)
    )


def _failure_factorized_path_retrieval_type_compatibility(
    left: str, right: str, ontology: _failure_factorized_path_retrieval_FreebaseOntology
) -> float:
    if not left or not right:
        return 0.25
    if left == right:
        return 1.0
    left_supers = set(ontology.supertypes(left))
    right_supers = set(ontology.supertypes(right))
    if left in right_supers or right in left_supers:
        return 0.9
    overlap = (left_supers & right_supers) - {"common.topic", "type.object"}
    return 0.65 if overlap else 0.0


@dataclass(slots=True)
class _RetrievedPath:
    entity_id: str
    entity_label: str
    path: GoldPath
    score: float
    retrieval_rank: int
    source_index: int
    source_kind: str
    support: int

    @property
    def signature(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.path.relations, self.path.directions


def _retrieve_for_entity(
    index: TrainGoldPathIndex,
    ontology: _failure_factorized_path_retrieval_FreebaseOntology,
    *,
    question: str,
    decomposition: Sequence[str],
    surfaces: Sequence[str],
    entity_id: str,
    entity_label: str,
    intents: Sequence[_SemanticIntent],
    top_k: int,
) -> list[_RetrievedPath]:
    clauses = _local_clauses(entity_label, decomposition, intents)
    semantic_relations = [
        relation for intent in intents for relation in intent.path.relations
    ]
    query = " ".join(
        [
            _masked(question, surfaces),
            _masked(question, surfaces),
            *(_masked(clause, surfaces) for clause in clauses for _ in range(3)),
            _failure_factorized_path_retrieval_relation_text(semantic_relations),
        ]
    )
    retrieved = index.retrieve(query, top_k=max(64, top_k * 8))
    support: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter(
        (document.path.relations, document.path.directions)
        for _, _, document in retrieved[:64]
    )
    ranked: list[_RetrievedPath] = []
    for retrieval_rank, (bm25, _, document) in enumerate(retrieved, start=1):
        similarities = [
            _failure_factorized_path_retrieval_path_similarity(intent.path, document.path) for intent in intents
        ]
        best = max(similarities, default=(0.0, 0.0, 0.0, 0.0))
        start_compatibility = max(
            (
                _failure_factorized_path_retrieval_type_compatibility(
                    _path_start_type(intent.path, ontology),
                    _path_start_type(document.path, ontology),
                    ontology,
                )
                for intent in intents
            ),
            default=0.25,
        )
        signature = (document.path.relations, document.path.directions)
        score = (
            float(bm25)
            + 2.4 * best[0]
            + 1.4 * best[1]
            + 0.7 * best[2]
            + 0.6 * start_compatibility
            + 0.25 * math.log1p(support[signature])
            + 0.25 * float(document.alignment[1] if len(document.alignment) > 1 else 0.0)
        )
        ranked.append(
            _RetrievedPath(
                entity_id=entity_id,
                entity_label=entity_label,
                path=document.path,
                score=score,
                retrieval_rank=retrieval_rank,
                source_index=document.source_index,
                source_kind=document.source_kind,
                support=support[signature],
            )
        )
    ranked.sort(
        key=lambda value: (
            -value.score,
            value.retrieval_rank,
            value.source_index,
            value.signature,
        )
    )
    output: list[_RetrievedPath] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for value in ranked:
        if value.signature in seen:
            continue
        seen.add(value.signature)
        output.append(value)
        if len(output) >= max(1, int(top_k)):
            break
    return output


def _render_path(path: _RetrievedPath, path_index: int) -> list[str]:
    current = f"ns:{path.entity_id}"
    triples: list[str] = []
    for step_index, (relation, direction) in enumerate(path.path.pairs):
        terminal = step_index == len(path.path.relations) - 1
        next_node = "?x" if terminal else f"?p{path_index}_{step_index}"
        if direction == "forward":
            triples.append(f"{current} ns:{relation} {next_node} .")
        else:
            triples.append(f"{next_node} ns:{relation} {current} .")
        current = next_node
    return triples


def _candidate_query(paths: Sequence[_RetrievedPath]) -> str:
    body = "\n".join(
        triple
        for path_index, path in enumerate(paths)
        for triple in _render_path(path, path_index)
    )
    return (
        "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
        "SELECT DISTINCT ?x\nWHERE {\n"
        f"{body}\n"
        "FILTER (!isLiteral(?x) || lang(?x) = '' || langMatches(lang(?x), 'en'))\n"
        "}\nLIMIT 256"
    )


def _build_candidates(
    entity_paths: dict[str, list[_RetrievedPath]],
    ontology: _failure_factorized_path_retrieval_FreebaseOntology,
    *,
    combination_beam: int,
) -> list[dict[str, Any]]:
    active = [(entity_id, paths) for entity_id, paths in entity_paths.items() if paths]
    if len(active) < 2:
        return []
    combinations: list[tuple[float, tuple[_RetrievedPath, ...]]] = []
    for values in product(
        *(paths[: max(1, int(combination_beam))] for _, paths in active)
    ):
        terminal_types = [
            _path_terminal_type(value.path, ontology) for value in values
        ]
        compatibilities = [
            _failure_factorized_path_retrieval_type_compatibility(
                terminal_types[left], terminal_types[right], ontology
            )
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ]
        if compatibilities and min(compatibilities) <= 0.0:
            continue
        score = sum(value.score for value in values)
        score += 1.2 * sum(compatibilities) / max(1, len(compatibilities))
        score += 0.15 * len({value.source_index for value in values})
        combinations.append((score, tuple(values)))
    combinations.sort(
        key=lambda value: (
            -value[0],
            tuple(path.retrieval_rank for path in value[1]),
            tuple(path.signature for path in value[1]),
        )
    )
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, paths in combinations:
        query = _candidate_query(paths)
        compact = " ".join(query.split())
        if compact in seen:
            continue
        seen.add(compact)
        output.append(
            {
                "key": hashlib.sha256(query.encode()).hexdigest()[:20],
                "query": query,
                "runtime_score": score,
                "answer_var": "x",
                "paths": [
                    {
                        "entity_id": path.entity_id,
                        "entity_label": path.entity_label,
                        "relations": list(path.path.relations),
                        "directions": list(path.path.directions),
                        "retrieval_rank": path.retrieval_rank,
                        "source_index": path.source_index,
                        "source_kind": path.source_kind,
                        "support_top64": path.support,
                    }
                    for path in paths
                ],
            }
        )
        if len(output) >= 32:
            break
    return output


def _failure_factorized_path_retrieval_choose_runtime(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    nonempty = [candidate for candidate in candidates if candidate.get("answers")]
    if not nonempty:
        return None
    support = Counter(
        tuple(sorted(map(str, candidate["answers"]))) for candidate in nonempty
    )
    return max(
        nonempty,
        key=lambda candidate: (
            support[tuple(sorted(map(str, candidate["answers"])))],
            float(candidate.get("runtime_score", 0.0)),
            str(candidate.get("key", "")),
        ),
    )


def _failure_factorized_path_retrieval_index(pipeline: Any) -> TrainGoldPathIndex:
    cached = getattr(pipeline, "_failure_factorized_path_index", None)
    if cached is not None:
        return cached
    lock = getattr(pipeline, "_failure_factorized_path_index_lock", None)
    if lock is None:
        cached = TrainGoldPathIndex(pipeline.failure_factorized_path_cache_path)
        pipeline._failure_factorized_path_index = cached
        return cached
    with lock:
        cached = getattr(pipeline, "_failure_factorized_path_index", None)
        if cached is None:
            cached = TrainGoldPathIndex(pipeline.failure_factorized_path_cache_path)
            pipeline._failure_factorized_path_index = cached
    return cached


def _failure_factorized_path_retrieval_retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    """Run the frozen terminal fallback and return a pipeline-style outcome."""
    started = time.perf_counter()
    pipeline = context.pipeline
    diagnostics: dict[str, Any] = {
        "version": _failure_factorized_path_retrieval_VERSION,
        "status": "abstained",
        "trigger": "final_empty_after_temporal_direct_and_adaptive_lanes",
        "source": "CWQ_TRAIN_gold_sparql_path_cache_only",
        "uses_evaluation_gold": False,
        "uses_test_sparql_or_reasoning_information": False,
        "llm_calls": 0,
        "embedding_calls": 0,
        "entity_or_relation_whitelist": False,
        "candidate_query_limit": 3,
        "raw_queries_exposed": False,
    }
    entities = _failure_factorized_path_retrieval_entities(context)
    if len(entities) < 2:
        diagnostics["reason"] = "requires_at_least_two_linked_entities"
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return {"answer_ids": [], "answers": [], "selected_graph": None, "diagnostics": diagnostics}
    decomposition = _failure_factorized_path_retrieval_decompositions(context)
    intents = _semantic_intents(context)
    assigned = _assign_intents(intents, entities, context.question)
    ontology = getattr(pipeline, "ontology", None)
    if ontology is None:
        diagnostics["reason"] = "ontology_unavailable"
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return {"answer_ids": [], "answers": [], "selected_graph": None, "diagnostics": diagnostics}
    index = _failure_factorized_path_retrieval_index(pipeline)
    surfaces = _failure_factorized_path_retrieval_unique(
        [label for _, label in entities] + _failure_factorized_path_retrieval_trace_surfaces(context)
    )
    entity_paths: dict[str, list[_RetrievedPath]] = {}
    top_k = max(1, int(getattr(pipeline, "failure_factorized_path_top_k", 6)))
    for entity_id, entity_label in entities:
        entity_paths[entity_id] = _retrieve_for_entity(
            index,
            ontology,
            question=context.question,
            decomposition=decomposition,
            surfaces=surfaces,
            entity_id=entity_id,
            entity_label=entity_label,
            intents=assigned.get(entity_id, []),
            top_k=top_k,
        )
    candidates = _build_candidates(
        entity_paths,
        ontology,
        combination_beam=max(
            1, int(getattr(pipeline, "failure_factorized_path_combination_beam", 5))
        ),
    )
    limit = min(
        3,
        max(1, int(getattr(pipeline, "failure_factorized_path_candidate_limit", 3))),
    )
    selected_candidates = candidates[:limit]
    execution_queries = 0
    for candidate in selected_candidates:
        query_started = time.perf_counter()
        try:
            rows = endpoint.execute(candidate["query"])
            answers = _failure_factorized_path_retrieval_unique(_failure_factorized_path_retrieval_answer_values(rows, candidate["answer_var"]))
            error = ""
        except Exception as exc:
            answers = []
            error = f"{type(exc).__name__}:{exc}"
        execution_queries += 1
        candidate["answers"] = answers
        candidate["error"] = error
        candidate["elapsed_seconds"] = time.perf_counter() - query_started
    selected = _failure_factorized_path_retrieval_choose_runtime(selected_candidates)
    answer_ids = list(selected.get("answers", [])) if selected else []

    label_rows: list[dict[str, str]] = []
    label_error = ""
    entity_answer_ids = [
        answer_id for answer_id in answer_ids if _failure_factorized_path_retrieval_ENTITY_ID_RE.fullmatch(answer_id)
    ]
    if entity_answer_ids:
        try:
            label_rows = list(pipeline.kg.labels(entity_answer_ids))
        except Exception as exc:
            label_error = f"{type(exc).__name__}:{exc}"
    labels_by_id = {
        str(item.get("id", "")): str(item.get("label", ""))
        for item in label_rows
        if isinstance(item, dict) and str(item.get("id", ""))
    }
    candidate_traces: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        query = str(candidate.pop("query"))
        candidate_traces.append(
            {
                "rank": rank,
                "key": str(candidate["key"]),
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "runtime_score": float(candidate["runtime_score"]),
                "answer_count": len(candidate["answers"]),
                "elapsed_seconds": float(candidate["elapsed_seconds"]),
                "error": str(candidate["error"]),
                "paths": candidate["paths"],
            }
        )
    diagnostics.update(
        {
            "status": "selected" if selected else "all_candidates_empty",
            "reason": (
                "factorized_train_path_answer_consensus"
                if selected
                else "all_factorized_queries_empty"
            ),
            "index_size": len(index.documents),
            "index_load_seconds": index.load_seconds,
            "linked_entity_count": len(entities),
            "retrieved_path_counts": {
                entity_id: len(paths) for entity_id, paths in entity_paths.items()
            },
            "generated_candidate_count": len(candidates),
            "endpoint_execution_queries": execution_queries,
            "answer_count": len(answer_ids),
            "selected_candidate_key": str(selected.get("key", "")) if selected else "",
            "candidate_traces": candidate_traces,
            "selected_label_queries": int(bool(entity_answer_ids)),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    if label_error:
        diagnostics["selected_label_error"] = label_error
    return {
        "answer_ids": answer_ids,
        "answers": [
            {"id": value, "label": labels_by_id.get(value, value)}
            for value in answer_ids
        ],
        "selected_graph": None,
        "diagnostics": diagnostics,
    }






# ---- extrema_relation_retrieval ----
"""Narrow, model-free relation repair for explicit entity extrema.

The normal pipeline occasionally keeps the right ``ARGMIN``/``ARGMAX``
operator but attaches the answer variable through the wrong relation.  This
lane repairs exactly one answer edge on at most three graphs from the saved
``final_graph_beam``.  It reuses the aligned training/ontology/BGE index owned
by :mod:`missing_relation_retrieval`, executes at most three candidate
queries, and never consults test answers or an LLM.

The acceptance gate is intentionally much narrower than the research
generator.  A candidate must preserve the unique surface extrema, retain the
answer-edge direction, be schema compatible, and return one answer.  It also
needs either an ontology reverse proof on an object-side answer edge (R), or a
strong aligned-template/topology proof (X).
"""



from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
from typing import Any, Iterable, Sequence

from .contracts import QueryGraphCandidate as _extrema_relation_retrieval_QueryGraphCandidate
from .lowering import LoweringError as _extrema_relation_retrieval_LoweringError, lower_sparql as _extrema_relation_retrieval_lower_sparql
_extrema_relation_retrieval_AlignedRelationIndex = _missing_relation_retrieval_AlignedRelationIndex
_extrema_relation_retrieval_Evidence = _missing_relation_retrieval_Evidence
_extrema_relation_retrieval_anchor_surfaces = _missing_relation_retrieval_anchor_surfaces
_extrema_relation_retrieval_answer_values = _missing_relation_retrieval_answer_values
_extrema_relation_retrieval_best_endpoint_compatibility = _missing_relation_retrieval_best_endpoint_compatibility
_extrema_relation_retrieval_decompositions = _missing_relation_retrieval_decompositions
_extrema_relation_retrieval_hard_post_selection_applied = _missing_relation_retrieval_hard_post_selection_applied
_extrema_relation_retrieval_index = _missing_relation_retrieval_index
_extrema_relation_retrieval_relation_evidence = _missing_relation_retrieval_relation_evidence
_extrema_relation_retrieval_score_evidence = _missing_relation_retrieval_score_evidence
_extrema_relation_retrieval_type_compatibility = _missing_relation_retrieval_type_compatibility





_VARIABLE_RE = re.compile(r"^(?:V\d+|P\d+\.V\d+)$", re.IGNORECASE)
_MAXIMUM_RE = re.compile(
    r"\b(?:latest|last|most recent|biggest|largest|highest|maximum|newest|longest)\b",
    re.IGNORECASE,
)
_MINIMUM_RE = re.compile(
    r"\b(?:earliest|least|smallest|lowest|minimum|first|oldest|youngest|shortest)\b",
    re.IGNORECASE,
)
_NON_EXTREMA_NAME_RE = re.compile(r"\b(?:first|last)\s+name\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r'["“][^"”]*["”]')


def _unique_extrema_operator(question: str) -> str:
    """Return the one explicit extrema polarity, or ``""`` if ambiguous."""

    text = _QUOTED_RE.sub(" ", str(question))
    text = _NON_EXTREMA_NAME_RE.sub(" ", text)
    maximum = bool(_MAXIMUM_RE.search(text))
    minimum = bool(_MINIMUM_RE.search(text))
    if maximum == minimum:
        return ""
    return "ARGMAX" if maximum else "ARGMIN"


def _extrema_relation_retrieval_trace_outputs(traces: Sequence[dict[str, Any]], stage: str) -> Iterable[Any]:
    for trace in traces:
        if isinstance(trace, dict) and trace.get("stage") == stage:
            yield trace.get("output")


def _final_graph_beam(traces: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    latest: list[dict[str, Any]] = []
    for output in _extrema_relation_retrieval_trace_outputs(traces, "final_graph_beam"):
        if isinstance(output, list):
            values = [value for value in output if isinstance(value, dict)]
            if values:
                latest = values
    return latest[: max(1, min(3, int(limit)))]


def _selection_reason_codes(traces: Sequence[dict[str, Any]]) -> set[str]:
    latest: dict[str, Any] = {}
    for output in _extrema_relation_retrieval_trace_outputs(traces, "glm_final_graph_ranking"):
        if isinstance(output, dict):
            latest = output
    return {
        str(value)
        for value in latest.get("reason_codes", [])
        if str(value).strip()
    }


def _interlock_reason(result: dict[str, Any], traces: Sequence[dict[str, Any]]) -> str:
    """Protect prior hard/path/template decisions before doing any BGE work."""

    if _extrema_relation_retrieval_hard_post_selection_applied(traces):
        return "preserve_hard_post_selection_gate"
    reason_codes = _selection_reason_codes(traces)
    if "path_alignment_gate" in reason_codes:
        return "preserve_path_alignment_gate"
    if "source_verified_template_consensus_gate" in reason_codes:
        return "preserve_template_consensus_gate"
    blocked_stages = {
        "failure_endpoint_retrieval",
        "failure_endpoint_template_retrieval",
        "missing_relation_retrieval",
        "extrema_relation_retrieval",
    }
    if any(
        isinstance(value, dict) and value.get("stage") in blocked_stages
        for value in traces
    ):
        return "preserve_existing_retrieval_lane"
    recovery = result.get("failure_recovery")
    if isinstance(recovery, dict):
        strategy = str(recovery.get("strategy", ""))
        if "path" in strategy:
            return "preserve_failure_path_retrieval"
        if "template" in strategy:
            return "preserve_failure_template_retrieval"
    graph = result.get("selected_graph")
    provenance = graph.get("provenance", {}) if isinstance(graph, dict) else {}
    if isinstance(provenance, dict):
        if "failure_endpoint_path_retrieval" in provenance:
            return "preserve_failure_path_retrieval"
        if "failure_template_retrieval" in provenance:
            return "preserve_failure_template_retrieval"
    return ""


def _is_variable(value: str) -> bool:
    return bool(_VARIABLE_RE.fullmatch(str(value)))


def _multiset_f1(left: Counter[str], right: Counter[str]) -> float:
    left_size, right_size = sum(left.values()), sum(right.values())
    if not left_size or not right_size:
        return float(left_size == right_size)
    return 2.0 * sum((left & right).values()) / (left_size + right_size)


@dataclass(frozen=True, slots=True)
class _GraphFeatures:
    triples: tuple[tuple[str, str, str], ...]
    answer_var: str
    relations: Counter[str]
    answer_incidence: Counter[str]
    paths: tuple[tuple[tuple[str, str], ...], ...]
    abstract_paths: tuple[tuple[str, ...], ...]
    constant_count: int
    answer_degree: int

    @property
    def triple_count(self) -> int:
        return len(self.triples)


def _graph_features(
    triples: Iterable[Sequence[Any]], answer_var: Any
) -> _GraphFeatures:
    normalized = tuple(
        (str(value[0]), str(value[1]), str(value[2]))
        for value in triples
        if isinstance(value, (list, tuple)) and len(value) == 3 and str(value[1])
    )
    answer = str(answer_var)
    relations = Counter(relation for _, relation, _ in normalized)
    incidence = Counter(
        f"{relation}|{'out' if subject == answer else 'in' if object_ == answer else 'internal'}"
        for subject, relation, object_ in normalized
    )
    nodes = {
        node for subject, _, object_ in normalized for node in (subject, object_)
    }
    adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for subject, relation, object_ in normalized:
        adjacency[subject].append((object_, relation, "out"))
        adjacency[object_].append((subject, relation, "in"))
    constants = sorted(node for node in nodes if not _is_variable(node))
    paths: list[tuple[tuple[str, str], ...]] = []
    for constant in constants:
        queue: deque[tuple[str, tuple[tuple[str, str], ...]]] = deque(
            [(constant, ())]
        )
        visited = {constant}
        while queue:
            node, path = queue.popleft()
            if node == answer:
                paths.append(path)
                break
            for neighbour, relation, direction in sorted(adjacency.get(node, [])):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                queue.append((neighbour, (*path, (relation, direction))))
    ordered = tuple(sorted(paths))
    return _GraphFeatures(
        triples=normalized,
        answer_var=answer,
        relations=relations,
        answer_incidence=incidence,
        paths=ordered,
        abstract_paths=tuple(
            sorted(tuple(direction for _, direction in path) for path in paths)
        ),
        constant_count=len(constants),
        answer_degree=len(adjacency.get(answer, [])),
    )


def _extrema_relation_retrieval_path_similarity(
    left: tuple[tuple[str, str], ...], right: tuple[tuple[str, str], ...]
) -> float:
    if not left or not right:
        return float(left == right)
    denominator = max(len(left), len(right))
    aligned = sum(a == b for a, b in zip(left, right)) / denominator
    left_relations = Counter(value[0] for value in left)
    right_relations = Counter(value[0] for value in right)
    relation = _multiset_f1(left_relations, right_relations)
    direction = sum(a[1] == b[1] for a, b in zip(left, right)) / denominator
    return 0.55 * aligned + 0.25 * relation + 0.20 * direction


def _path_set_similarity(
    left: tuple[tuple[tuple[str, str], ...], ...],
    right: tuple[tuple[tuple[str, str], ...], ...],
) -> float:
    if not left or not right:
        return float(left == right)

    def directed(source: Any, target: Any) -> float:
        return sum(
            max(_extrema_relation_retrieval_path_similarity(path, other) for other in target)
            for path in source
        ) / len(source)

    return 0.5 * (directed(left, right) + directed(right, left))


def _topology_similarity(left: _GraphFeatures, right: _GraphFeatures) -> float:
    relation = _multiset_f1(left.relations, right.relations)
    incidence = _multiset_f1(left.answer_incidence, right.answer_incidence)
    paths = _path_set_similarity(left.paths, right.paths)
    abstract_exact = float(left.abstract_paths == right.abstract_paths)
    metadata = sum(
        (
            left.constant_count == right.constant_count,
            left.triple_count == right.triple_count,
            left.answer_degree == right.answer_degree,
        )
    ) / 3.0
    return (
        0.25 * relation
        + 0.20 * incidence
        + 0.35 * paths
        + 0.12 * abstract_exact
        + 0.08 * metadata
    )


def _weighted_train_features(
    neighbours: Sequence[tuple[float, Any]], selected: _GraphFeatures
) -> tuple[list[tuple[float, _GraphFeatures]], dict[str, float], dict[str, float]]:
    top = max((float(score) for score, _ in neighbours), default=1.0)
    weighted: list[tuple[float, _GraphFeatures]] = []
    relation_support: dict[str, float] = defaultdict(float)
    role_support: dict[str, float] = defaultdict(float)
    for bm25, document in neighbours:
        graph = _graph_features(document.triples, document.answer_var)
        normalized = float(bm25) / max(top, 1e-9)
        overlap = sum((graph.relations & selected.relations).values()) / max(
            1, sum(selected.relations.values())
        )
        weight = normalized * (0.35 + 0.65 * overlap)
        weighted.append((weight, graph))
        for relation, count in graph.relations.items():
            relation_support[relation] += weight * count
        for key, count in graph.answer_incidence.items():
            role_support[key] += weight * count
    return weighted, dict(relation_support), dict(role_support)


def _topology_fit(
    graph: _GraphFeatures,
    neighbours: Sequence[tuple[float, _GraphFeatures]],
) -> tuple[float, float]:
    values = sorted(
        (
            float(weight) * _topology_similarity(graph, document)
            for weight, document in neighbours
        ),
        reverse=True,
    )
    if not values:
        return 0.0, 0.0
    return values[0], sum(values[:3]) / min(3, len(values))


def _endpoint_hints(
    ontology: Any,
    triples: Sequence[Sequence[str]],
    edge_index: int,
    node: str,
) -> list[str]:
    output: list[str] = []
    for index, triple in enumerate(triples):
        if index == edge_index or len(triple) != 3:
            continue
        subject, relation, object_ = map(str, triple)
        value = ""
        if subject == node:
            value = str(ontology.domain_for_relation(relation))
        elif object_ == node:
            value = str(ontology.range_for_relation(relation))
        if value and value not in output:
            output.append(value)
    return output


def _edge_schema_compatibility(
    ontology: Any,
    triples: Sequence[Sequence[str]],
    edge_index: int,
    new_relation: str,
    reversed_edge: bool,
) -> tuple[float, dict[str, Any]]:
    subject, old_relation, object_ = map(str, triples[edge_index])
    new_domain = str(ontology.domain_for_relation(new_relation))
    new_range = str(ontology.range_for_relation(new_relation))
    if reversed_edge:
        new_domain, new_range = new_range, new_domain
    old_domain = str(ontology.domain_for_relation(old_relation))
    old_range = str(ontology.range_for_relation(old_relation))
    subject_hints = _endpoint_hints(ontology, triples, edge_index, subject)
    object_hints = _endpoint_hints(ontology, triples, edge_index, object_)
    subject_score = max(
        _extrema_relation_retrieval_best_endpoint_compatibility(ontology, new_domain, subject_hints),
        0.75 * _extrema_relation_retrieval_type_compatibility(ontology, new_domain, old_domain),
    )
    object_score = max(
        _extrema_relation_retrieval_best_endpoint_compatibility(ontology, new_range, object_hints),
        0.75 * _extrema_relation_retrieval_type_compatibility(ontology, new_range, old_range),
    )
    return 0.5 * (subject_score + object_score), {
        "new_domain": new_domain,
        "new_range": new_range,
        "old_domain": old_domain,
        "old_range": old_range,
        "subject_hints": subject_hints,
        "object_hints": object_hints,
        "subject_compatibility": subject_score,
        "object_compatibility": object_score,
    }


def _sigmoid(value: float) -> float:
    if value >= 30.0:
        return 1.0
    if value <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(slots=True)
class _extrema_relation_retrieval_Candidate:
    key: str
    source_graph_rank: int
    edge_index: int
    edge_role: str
    relation_id: str
    replaced_relation: str
    reversed_edge: bool
    graph: _extrema_relation_retrieval_QueryGraphCandidate
    evidence: dict[str, Any]
    sparql: str
    answers: list[str]
    error: str = ""
    elapsed_seconds: float = 0.0


def _edge_role(triple: Sequence[str], answer_var: str) -> str:
    subject, _, object_ = map(str, triple)
    if subject == answer_var:
        return "out"
    if object_ == answer_var:
        return "in"
    return "internal"


def _node_distances(
    triples: Sequence[Sequence[str]], answer_var: str
) -> dict[str, int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        if len(triple) != 3:
            continue
        subject, _, object_ = map(str, triple)
        adjacency[subject].add(object_)
        adjacency[object_].add(subject)
    distances = {answer_var: 0}
    frontier = [answer_var]
    while frontier:
        node = frontier.pop(0)
        for neighbour in adjacency.get(node, set()):
            if neighbour in distances:
                continue
            distances[neighbour] = distances[node] + 1
            frontier.append(neighbour)
    return distances


def _compile_candidates(
    *,
    graph: dict[str, Any],
    evidence: Sequence[_extrema_relation_retrieval_Evidence],
    ontology: Any,
    neighbours: Sequence[tuple[float, Any]],
    source_graph_rank: int,
) -> list[_extrema_relation_retrieval_Candidate]:
    triples = [
        list(map(str, value))
        for value in graph.get("triples", [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    answer_var = str(graph.get("answer_var", ""))
    if not triples or not answer_var:
        return []
    incumbent = _graph_features(triples, answer_var)
    weighted, relation_support, role_support = _weighted_train_features(
        neighbours, incumbent
    )
    incumbent_best, incumbent_mean = _topology_fit(incumbent, weighted)
    max_relation_support = max(relation_support.values(), default=1.0)
    max_role_support = max(role_support.values(), default=1.0)
    operators = deepcopy(list(graph.get("operators", [])))
    distances = _node_distances(triples, answer_var)
    output: list[_extrema_relation_retrieval_Candidate] = []
    seen: set[str] = set()

    for relation in evidence[:16]:
        for edge_index, triple in enumerate(triples):
            subject, old_relation, object_ = triple
            if relation.relation_id == old_relation:
                continue
            role = _edge_role(triple, answer_var)
            distance = min(
                distances.get(subject, len(triples) + 1),
                distances.get(object_, len(triples) + 1),
            )
            proximity = 1.0 / (1.0 + float(distance))
            orientations = [False]
            if (
                role != "internal"
                and relation.preferred_orientation
                and relation.preferred_orientation != role
            ):
                orientations.insert(0, True)
            elif role == "internal":
                orientations.append(True)
            for reversed_edge in orientations:
                compatibility, schema = _edge_schema_compatibility(
                    ontology,
                    triples,
                    edge_index,
                    relation.relation_id,
                    reversed_edge,
                )
                if compatibility <= 0.0:
                    continue
                revised = deepcopy(triples)
                revised[edge_index] = (
                    [object_, relation.relation_id, subject]
                    if reversed_edge
                    else [subject, relation.relation_id, object_]
                )
                signature = json.dumps(
                    {
                        "triples": revised,
                        "answer_var": answer_var,
                        "operators": operators,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                revised_features = _graph_features(revised, answer_var)
                best_fit, mean_fit = _topology_fit(revised_features, weighted)
                topology_delta = 0.6 * (best_fit - incumbent_best) + 0.4 * (
                    mean_fit - incumbent_mean
                )
                support_delta = (
                    relation_support.get(relation.relation_id, 0.0)
                    - relation_support.get(old_relation, 0.0)
                ) / max(max_relation_support, 1e-9)
                new_role = (
                    "in" if role == "out" else "out"
                ) if reversed_edge and role != "internal" else role
                role_support_value = role_support.get(
                    f"{relation.relation_id}|{new_role}", 0.0
                ) / max(max_role_support, 1e-9)
                score = (
                    0.48 * relation.candidate_score
                    + 0.18 * best_fit
                    + 0.10 * mean_fit
                    + 0.25 * compatibility
                    + 0.07 * role_support_value
                    + 0.05 * _sigmoid(4.0 * support_delta)
                    + 0.06 * max(-0.5, min(0.5, topology_delta))
                    + 0.04 * proximity
                    + 0.02 * float(not reversed_edge)
                    - 0.012 * (source_graph_rank - 1)
                )
                candidate_graph = _extrema_relation_retrieval_QueryGraphCandidate(
                    graph_id=f"ER{hashlib.sha256(signature.encode()).hexdigest()[:16]}",
                    triples=revised,
                    answer_var=answer_var,
                    operators=operators,
                    score=score,
                    compose_output={},
                    provenance=deepcopy(dict(graph.get("provenance", {}))),
                )
                candidate_graph.provenance["extrema_relation_retrieval"] = {
                    "source_graph_rank": source_graph_rank,
                    "edge_index": edge_index,
                    "edge_role": role,
                    "relation_id": relation.relation_id,
                    "replaced_relation": old_relation,
                    "uses_gold": False,
                    "model_calls": 0,
                }
                try:
                    sparql = _extrema_relation_retrieval_lower_sparql(candidate_graph)
                except (_extrema_relation_retrieval_LoweringError, ValueError):
                    continue
                output.append(
                    _extrema_relation_retrieval_Candidate(
                        key=candidate_graph.graph_id[2:],
                        source_graph_rank=source_graph_rank,
                        edge_index=edge_index,
                        edge_role=role,
                        relation_id=relation.relation_id,
                        replaced_relation=old_relation,
                        reversed_edge=reversed_edge,
                        graph=candidate_graph,
                        evidence={
                            **relation.dump(),
                            **schema,
                            "incumbent_topology_best": incumbent_best,
                            "candidate_topology_best": best_fit,
                            "incumbent_topology_mean3": incumbent_mean,
                            "candidate_topology_mean3": mean_fit,
                            "topology_delta": topology_delta,
                            "old_relation_support": relation_support.get(
                                old_relation, 0.0
                            ),
                            "new_relation_support": relation_support.get(
                                relation.relation_id, 0.0
                            ),
                            "normalized_support_delta": support_delta,
                            "normalized_role_support": role_support_value,
                            "schema_compatibility": compatibility,
                            "answer_distance": distance,
                            "answer_proximity": proximity,
                        },
                        sparql=sparql,
                        answers=[],
                    )
                )
    output.sort(
        key=lambda value: (
            -value.graph.score,
            value.source_graph_rank,
            value.edge_index,
            value.key,
        )
    )
    return output


def _proof_from_values(
    *,
    edge_role: str,
    reversed_edge: bool,
    operators: Sequence[dict[str, Any]],
    evidence: dict[str, Any],
    wanted_operator: str,
) -> str:
    """Shared production proof gate used by runtime and offline audit."""

    operator_types = {
        str(value.get("type", "")).upper()
        for value in operators
        if isinstance(value, dict)
    }
    opposite = "ARGMIN" if wanted_operator == "ARGMAX" else "ARGMAX"
    base = bool(
        edge_role in {"in", "out"}
        and not reversed_edge
        and str(evidence.get("preferred_orientation", "")) == edge_role
        and wanted_operator in operator_types
        and opposite not in operator_types
        and float(evidence.get("schema_compatibility", 0.0)) >= 0.75
    )
    if not base:
        return ""
    reverse = bool(
        edge_role == "in" and evidence.get("reverse_of_terminal")
    )
    template = bool(
        int(evidence.get("best_template_rank", 10**9)) == 1
        and int(evidence.get("support_top8", 0)) >= 8
        and evidence.get("grounded_union")
        and float(evidence.get("topology_delta", 0.0)) >= 0.20
        and float(evidence.get("normalized_support_delta", 0.0)) >= 0.90
    )
    return "R" if reverse else "X" if template else ""


def _proof_branch(candidate: _extrema_relation_retrieval_Candidate, wanted_operator: str) -> str:
    return _proof_from_values(
        edge_role=candidate.edge_role,
        reversed_edge=candidate.reversed_edge,
        operators=candidate.graph.operators,
        evidence=candidate.evidence,
        wanted_operator=wanted_operator,
    )


def _candidate_audit(candidate: _extrema_relation_retrieval_Candidate, branch: str) -> dict[str, Any]:
    return {
        "graph_id": candidate.graph.graph_id,
        "source_graph_rank": candidate.source_graph_rank,
        "edge_index": candidate.edge_index,
        "edge_role": candidate.edge_role,
        "relation_id": candidate.relation_id,
        "replaced_relation": candidate.replaced_relation,
        "reversed_edge": candidate.reversed_edge,
        "score": candidate.graph.score,
        "schema_compatibility": candidate.evidence.get(
            "schema_compatibility", 0.0
        ),
        "topology_delta": candidate.evidence.get("topology_delta", 0.0),
        "normalized_support_delta": candidate.evidence.get(
            "normalized_support_delta", 0.0
        ),
        "proof_branch": branch,
        "answer_count": len(candidate.answers),
        "answer_ids": list(candidate.answers),
        "error": candidate.error,
        "query_elapsed_seconds": candidate.elapsed_seconds,
    }


def retrieve_extrema_relation(
    *,
    pipeline: Any,
    question: str,
    result: dict[str, Any],
    template_top_k: int = 64,
    source_graph_limit: int = 3,
    candidate_limit: int = 3,
) -> dict[str, Any]:
    """Return a Gold-blind answer-edge extrema repair, or abstain."""

    started = time.monotonic()
    query_limit = min(3, max(1, int(candidate_limit)))
    diagnostics: dict[str, Any] = {
        "status": "abstained",
        "uses_gold_answers": False,
        "uses_gold_query": False,
        "uses_relation_or_sample_whitelist": False,
        "llm_calls": 0,
        "trigger": "successful_explicit_unique_extrema",
        "candidate_query_limit": query_limit,
        "endpoint_execution_queries": 0,
        "bge_ranker_calls": 0,
        "source_graph_limit": min(3, max(1, int(source_graph_limit))),
    }
    incumbent = list(map(str, result.get("answer_ids", [])))
    if not incumbent or result.get("failure") is not None:
        diagnostics["reason"] = "requires_successful_nonempty_incumbent"
        return {"answer_ids": [], "diagnostics": diagnostics}
    wanted = _unique_extrema_operator(question)
    diagnostics["wanted_operator"] = wanted
    if not wanted:
        diagnostics["reason"] = "surface_extrema_not_unique"
        return {"answer_ids": [], "diagnostics": diagnostics}
    traces = result.get("traces", [])
    if not isinstance(traces, list):
        diagnostics["reason"] = "missing_trace_bundle"
        return {"answer_ids": [], "diagnostics": diagnostics}
    interlock = _interlock_reason(result, traces)
    if interlock:
        diagnostics["reason"] = interlock
        diagnostics["interlock_applied"] = True
        return {"answer_ids": [], "diagnostics": diagnostics}
    ontology = getattr(getattr(pipeline, "grounder", None), "ontology", None)
    ranker = getattr(getattr(pipeline, "grounder", None), "ranker", None)
    if ontology is None or ranker is None:
        diagnostics["reason"] = "missing_runtime_dependency"
        return {"answer_ids": [], "diagnostics": diagnostics}
    relation_index: _extrema_relation_retrieval_AlignedRelationIndex | None = _extrema_relation_retrieval_index(pipeline)
    if relation_index is None:
        diagnostics["reason"] = "missing_aligned_training_templates"
        return {"answer_ids": [], "diagnostics": diagnostics}
    beam = _final_graph_beam(traces, source_graph_limit)
    diagnostics["source_graph_count"] = len(beam)
    if not beam:
        diagnostics["reason"] = "missing_final_graph_beam"
        return {"answer_ids": [], "diagnostics": diagnostics}

    decompositions = _extrema_relation_retrieval_decompositions(traces)
    surfaces = _extrema_relation_retrieval_anchor_surfaces(traces)
    all_candidates: list[_extrema_relation_retrieval_Candidate] = []
    graph_audit: list[dict[str, Any]] = []
    for graph_rank, graph in enumerate(beam, 1):
        evidence, terminal = _extrema_relation_retrieval_relation_evidence(
            question=question,
            decompositions=decompositions,
            surfaces=surfaces,
            graph=graph,
            traces=traces,
            index=relation_index,
            ontology=ontology,
            top_k=min(64, max(3, int(template_top_k))),
        )
        if not evidence or not terminal:
            continue
        diagnostics["bge_ranker_calls"] += 1
        if not _extrema_relation_retrieval_score_evidence(
            evidence,
            question=question,
            terminal=terminal,
            ranker=ranker,
        ):
            continue
        neighbours = relation_index.retrieve(
            question,
            decompositions,
            surfaces,
            top_k=min(64, max(3, int(template_top_k))),
        )
        compiled = _compile_candidates(
            graph=graph,
            evidence=evidence,
            ontology=ontology,
            neighbours=neighbours,
            source_graph_rank=graph_rank,
        )
        all_candidates.extend(compiled)
        graph_audit.append(
            {
                "source_graph_rank": graph_rank,
                "terminal_relations": list(terminal),
                "relation_evidence_count": len(evidence),
                "compiled_candidate_count": len(compiled),
            }
        )
    diagnostics["graph_audit"] = graph_audit
    all_candidates.sort(
        key=lambda value: (
            -value.graph.score,
            value.source_graph_rank,
            value.edge_index,
            value.key,
        )
    )
    eligible: list[tuple[_extrema_relation_retrieval_Candidate, str]] = []
    seen_sparql: set[str] = set()
    for candidate in all_candidates:
        if candidate.sparql in seen_sparql:
            continue
        seen_sparql.add(candidate.sparql)
        branch = _proof_branch(candidate, wanted)
        if branch:
            eligible.append((candidate, branch))
        if len(eligible) >= query_limit:
            break
    diagnostics["compiled_candidate_count"] = len(all_candidates)
    diagnostics["pre_execution_eligible_count"] = len(eligible)
    if not eligible:
        diagnostics["reason"] = "extrema_proof_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}

    for candidate, _ in eligible:
        query_started = time.monotonic()
        try:
            diagnostics["endpoint_execution_queries"] += 1
            rows = pipeline.kg.execute(candidate.sparql)
            candidate.answers = _extrema_relation_retrieval_answer_values(rows, candidate.graph.answer_var)
        except Exception as exc:
            candidate.error = f"{type(exc).__name__}:{exc}"
        candidate.elapsed_seconds = time.monotonic() - query_started
    accepted = [
        (candidate, branch)
        for candidate, branch in eligible
        if not candidate.error and len(candidate.answers) == 1
    ]
    diagnostics["candidates"] = [
        _candidate_audit(candidate, branch) for candidate, branch in eligible
    ]
    if not accepted:
        diagnostics["reason"] = "singleton_answer_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}
    selected, branch = accepted[0]
    diagnostics.update(
        {
            "selected_graph_id": selected.graph.graph_id,
            "selected_proof_branch": branch,
            "selected_relation_id": selected.relation_id,
            "selected_replaced_relation": selected.replaced_relation,
            "selected_edge_role": selected.edge_role,
            "selected_source_graph_rank": selected.source_graph_rank,
            "answer_count": 1,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    )
    if set(selected.answers) == set(incumbent):
        diagnostics["status"] = "unchanged_equal_answer"
        diagnostics["reason"] = "candidate_answer_equals_incumbent"
        return {"answer_ids": [], "diagnostics": diagnostics}
    labels: dict[str, str] = {}
    try:
        labels = {
            str(value.get("id", "")): str(value.get("label", ""))
            for value in pipeline.kg.labels(selected.answers)
            if isinstance(value, dict) and str(value.get("id", ""))
        }
    except Exception as exc:
        diagnostics["label_error"] = f"{type(exc).__name__}:{exc}"
    diagnostics["status"] = "selected"
    diagnostics["reason"] = f"extrema_relation_proof_{branch}"
    return {
        "answer_ids": list(selected.answers),
        "answers": [
            {"id": value, "label": labels.get(value, value)}
            for value in selected.answers
        ],
        "selected_graph": {
            "graph_id": selected.graph.graph_id,
            "triples": deepcopy(selected.graph.triples),
            "answer_var": selected.graph.answer_var,
            "operators": deepcopy(selected.graph.operators),
            "score": selected.graph.score,
            "provenance": deepcopy(selected.graph.provenance),
        },
        "diagnostics": diagnostics,
    }



# ---- temporal_interval_retrieval ----
"""Bounded failure retrieval for event/office interval-overlap questions.

The lane is an exclusive router in front of broad source-SPARQL retrieval.  It
indexes only the interval-overlap templates already stored in the CWQ TRAIN
cache, keeps one structurally best entity assignment per source template, and
executes at most three SELECT queries.  It never reads evaluation answers,
sample indexes, or an entity/relation allow-list, and performs no model call.
"""



from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import permutations
import math
import re
import time
from typing import Any, Iterable, Sequence

_temporal_interval_retrieval_decompositions = _failure_gold_sparql_retrieval_decompositions
_temporal_interval_retrieval_entities = _failure_gold_sparql_retrieval_entities
_temporal_interval_retrieval_index = _failure_gold_sparql_retrieval_index
_temporal_interval_retrieval_operator_conflict = _failure_gold_sparql_retrieval_operator_conflict
_temporal_interval_retrieval_token_similarity = _failure_gold_sparql_retrieval_token_similarity
_temporal_interval_retrieval_tokens = _failure_gold_sparql_retrieval_tokens
_temporal_interval_retrieval_unique = _failure_gold_sparql_retrieval_unique
from .pipeline import _answer_values as _temporal_interval_retrieval_answer_values
from .template_path_retrieval import mask_entities as _temporal_interval_retrieval_mask_entities


_temporal_interval_retrieval_VERSION = "failure-temporal-interval-retrieval-production-v1"
_temporal_interval_retrieval_FULL_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
US_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4})(?!\d)")
MONTH_DATE_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    r"(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<month_day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<month_year>\d{4})"
    r"|(?P<day_day>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<day_month>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"[,]?\s+(?P<day_year>\d{4}))",
    re.I,
)
_temporal_interval_retrieval_YEAR_RE = re.compile(r"(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d.])")
_temporal_interval_retrieval_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
_temporal_interval_retrieval_ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_temporal_interval_retrieval_SELECT_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.I)
_temporal_interval_retrieval_RELATION_RE = re.compile(r"\bns:([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)\b")
MONTHS = {
    name.casefold(): position
    for position, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}

TERM_PATTERN = r'(?:\?[A-Za-z_][A-Za-z0-9_]*|ns:[A-Za-z0-9_.]+|"(?:[^"\\]|\\.)*"(?:\^\^[A-Za-z0-9_:.-]+)?|[-+]?\d+(?:\.\d+)?)'
REL_PATTERN = r"ns:([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)"
TRIPLE_RE = re.compile(
    rf"(?P<s>{TERM_PATTERN})\s+{REL_PATTERN}\s+(?P<o>{TERM_PATTERN})",
    re.I,
)
LESS_FILTER_RE = re.compile(
    r"FILTER\s*\(\s*(?P<left>\?[A-Za-z_][A-Za-z0-9_]*)\s*<\s*(?P<right>\?[A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.I,
)
GREATER_FILTER_RE = re.compile(
    r"FILTER\s*\(\s*(?P<left>\?[A-Za-z_][A-Za-z0-9_]*)\s*>\s*(?P<right>\?[A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class _IntervalSchema:
    event_subject: str
    event_start_relation: str
    event_end_relation: str
    tenure_subject: str
    answer_to_tenure_relation: str
    office_title_relation: str
    office_constant: str
    tenure_from_relation: str
    tenure_to_relation: str


def _temporal_interval_retrieval_normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _traces(context: Any) -> list[dict[str, Any]]:
    bundle = getattr(context, "trace_bundle", {}) or {}
    values = (bundle.get("decomposition", {}) or {}).get("traces", [])
    return [value for value in values if isinstance(value, dict)]


def _type_hints(context: Any) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for trace in _traces(context):
        if trace.get("stage") != "decomposition_review_and_rewrite":
            continue
        value = trace.get("output", {})
        for candidate in value.get("candidates", []) if isinstance(value, dict) else []:
            for event in candidate.get("events", []) if isinstance(candidate, dict) else []:
                item = event.get("input", {}) if isinstance(event, dict) else {}
                for entity in item.get("entity_context", []) if isinstance(item, dict) else []:
                    if not isinstance(entity, dict):
                        continue
                    label = str(entity.get("name", ""))
                    if label:
                        output[label] = [str(value) for value in entity.get("types", [])]
    return output


def _eligibility_evidence(context: Any) -> tuple[bool, dict[str, bool]]:
    question = str(context.question)
    decomposition = _temporal_interval_retrieval_decompositions(context)
    type_hints = _type_hints(context)
    text = _temporal_interval_retrieval_normalized(" ".join([question, *decomposition]))
    question_text = _temporal_interval_retrieval_normalized(question)
    office_language = bool(
        re.search(
            r"\b(?:president|governor|prime minister|government(?:al)? position|office holder|tenure|leader)\b",
            text,
        )
    )
    temporal_join_language = bool(
        re.search(r"\b(?:during|while|at the same time|time period|tenure)\b", text)
    )
    # A legislative-session year window uses a different schema and remains
    # in the broad direct lane.
    event_language = bool(
        re.search(r"\b(?:war|battle|conflict|event|time period)\b", text)
    )
    explicit_bounds = bool(
        ("start and end date" in text or "start and end dates" in text)
        or re.search(r"\bbegan before\b[^.?!]*\bended after\b", text)
        or re.search(r"\bstarted before\b[^.?!]*\bended after\b", text)
    )
    typed_event = any(
        any(token in _temporal_interval_retrieval_normalized(" ".join(types)) for token in ("event", "war", "conflict", "battle"))
        for types in type_hints.values()
    )
    typed_office = any(
        any(token in _temporal_interval_retrieval_normalized(" ".join(types)) for token in ("government office", "appointed role"))
        for types in type_hints.values()
    )
    surface_join = bool(
        re.search(r"\b(?:during|while|at the same time|time period)\b", question_text)
        and event_language
    )
    evidence = {
        "office_language": office_language,
        "temporal_join_language": temporal_join_language,
        "event_language": event_language,
        "explicit_bounds": explicit_bounds,
        "typed_event": typed_event,
        "typed_office": typed_office,
        "surface_join": surface_join,
    }
    return (
        office_language
        and temporal_join_language
        and (explicit_bounds or (typed_event and typed_office) or surface_join),
        evidence,
    )


def eligible(context: Any) -> bool:
    """Return whether this row exclusively belongs to the interval lane."""

    return _eligibility_evidence(context)[0]


def _parse_triples(sparql: str) -> list[tuple[str, str, str]]:
    text = "\n".join(line.split("#", 1)[0] for line in str(sparql).splitlines())
    triples: list[tuple[str, str, str]] = []
    for match in TRIPLE_RE.finditer(text):
        subject, relation, object_ = match.group("s"), match.group(2), match.group("o")
        triples.append((subject, relation, object_))
        cursor = match.end()
        current_relation = relation
        while True:
            separator = re.match(r"\s*([;,])", text[cursor:])
            if not separator:
                break
            cursor += separator.end()
            if separator.group(1) == ";":
                continuation = re.match(
                    rf"\s*{REL_PATTERN}\s+(?P<o>{TERM_PATTERN})",
                    text[cursor:],
                    re.I,
                )
                if not continuation:
                    break
                current_relation = continuation.group(1)
                object_ = continuation.group("o")
            else:
                continuation = re.match(
                    rf"\s*(?P<o>{TERM_PATTERN})", text[cursor:], re.I
                )
                if not continuation:
                    break
                object_ = continuation.group("o")
            cursor += continuation.end()
            triples.append((subject, current_relation, object_))
    return list(dict.fromkeys(triples))


def _answer_term(sparql: str) -> str:
    match = re.search(
        r"\bSELECT\s+(?:DISTINCT\s+)?(?P<answer>\?[A-Za-z_][A-Za-z0-9_]*)",
        sparql,
        re.I,
    )
    return match.group("answer") if match else "?x"


def _derive_interval_schema(sparql: str) -> _IntervalSchema | None:
    """Discover the overlap topology without comparing any relation ID."""

    triples = _parse_triples(sparql)
    answer = _answer_term(sparql)
    by_object: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    by_subject: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for triple in triples:
        by_subject[triple[0]].append(triple)
        by_object[triple[2]].append(triple)
    for less in LESS_FILTER_RE.finditer(sparql):
        from_var, end_var = less.group("left"), less.group("right")
        for greater in GREATER_FILTER_RE.finditer(sparql):
            to_var, start_var = greater.group("left"), greater.group("right")
            for from_triple in by_object.get(from_var, []):
                tenure = from_triple[0]
                to_triple = next(
                    (value for value in by_object.get(to_var, []) if value[0] == tenure),
                    None,
                )
                if to_triple is None:
                    continue
                for start_triple in by_object.get(start_var, []):
                    event = start_triple[0]
                    end_triple = next(
                        (value for value in by_object.get(end_var, []) if value[0] == event),
                        None,
                    )
                    if end_triple is None:
                        continue
                    answer_edge = next(
                        (value for value in by_subject.get(answer, []) if value[2] == tenure),
                        None,
                    )
                    if answer_edge is None:
                        continue
                    title_edge = next(
                        (
                            value
                            for value in by_subject.get(tenure, [])
                            if value[2].startswith(("ns:m.", "ns:g."))
                            and value[1] not in {from_triple[1], to_triple[1]}
                        ),
                        None,
                    )
                    if title_edge is None:
                        continue
                    return _IntervalSchema(
                        event_subject=event,
                        event_start_relation=start_triple[1],
                        event_end_relation=end_triple[1],
                        tenure_subject=tenure,
                        answer_to_tenure_relation=answer_edge[1],
                        office_title_relation=title_edge[1],
                        office_constant=title_edge[2].removeprefix("ns:"),
                        tenure_from_relation=from_triple[1],
                        tenure_to_relation=to_triple[1],
                    )
    return None


def _is_interval_template(sparql: str) -> bool:
    return _derive_interval_schema(sparql) is not None


def _source_role(schema: _IntervalSchema, constant: str) -> str:
    if schema.event_subject == f"ns:{constant}":
        return "event"
    if schema.office_constant == constant:
        return "office"
    return "constraint"


def _temporal_interval_retrieval_relation_text(sparql: str) -> str:
    output: list[str] = []
    for relation in _temporal_interval_retrieval_RELATION_RE.findall(sparql):
        if relation.startswith(("m.", "g.")):
            continue
        output.extend(part.replace("_", " ") for part in relation.split("."))
    return " ".join(output)


def _interval_operator_signature(question: str, sparql: str = "") -> set[str]:
    text = _temporal_interval_retrieval_normalized(question)
    output: set[str] = set()
    if re.search(r"\b(?:last|latest|most recent|highest|largest|greatest)\b", text):
        output.add("argmax")
    if re.search(r"\b(?:first|earliest|lowest|smallest|least)\b", text):
        output.add("argmin")
    if re.search(r"\b(?:after|later than|greater than|more than|over)\b", text):
        output.add("gt")
    if re.search(r"\b(?:before|earlier than|less than|under|prior to)\b", text):
        output.add("lt")
    upper = sparql.upper()
    if "ORDER BY DESC" in upper:
        output.add("argmax")
    elif "ORDER BY" in upper:
        output.add("argmin")
    filtered = re.sub(r"\?from\s*<\s*\?end|\?to\s*>\s*\?start", "", sparql)
    if re.search(r"FILTER\s*\([^)]*\?num\s*>", filtered, re.I):
        output.add("gt")
    if re.search(r"FILTER\s*\([^)]*\?num\s*<", filtered, re.I):
        output.add("lt")
    return output


def _temporal_interval_retrieval_scalar_slots(text: str) -> list[tuple[str, str]]:
    occupied: list[tuple[int, int]] = []
    output: list[tuple[int, str, str]] = []
    for pattern in (_temporal_interval_retrieval_FULL_DATE_RE, US_DATE_RE):
        for match in pattern.finditer(text):
            if any(left <= match.start() < right for left, right in occupied):
                continue
            occupied.append(match.span())
            value = match.group(0)
            if pattern is US_DATE_RE:
                value = f"{int(match.group(3)):04d}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"
            output.append((match.start(), "date", value))
    for match in MONTH_DATE_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        occupied.append(match.span())
        month = match.group("month_name") or match.group("day_month")
        day = match.group("month_day") or match.group("day_day")
        year = match.group("month_year") or match.group("day_year")
        output.append(
            (
                match.start(),
                "date",
                f"{int(year):04d}-{MONTHS[month.casefold()]:02d}-{int(day):02d}",
            )
        )
    for match in _temporal_interval_retrieval_YEAR_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        occupied.append(match.span())
        output.append((match.start(), "year", match.group(0)))
    for match in _temporal_interval_retrieval_NUMBER_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        if re.search(r"world\s+war\s*$", text[max(0, match.start() - 16):match.start()], re.I):
            continue
        output.append((match.start(), "number", match.group(0)))
    return [(kind, value) for _, kind, value in sorted(output)]


def _temporal_interval_retrieval_normalize_scalars(text: str) -> str:
    value = _temporal_interval_retrieval_FULL_DATE_RE.sub(" <date> ", str(text))
    value = US_DATE_RE.sub(" <date> ", value)
    value = MONTH_DATE_RE.sub(" <date> ", value)
    value = _temporal_interval_retrieval_YEAR_RE.sub(" <year> ", value)
    output: list[str] = []
    cursor = 0
    for match in _temporal_interval_retrieval_NUMBER_RE.finditer(value):
        output.append(value[cursor:match.start()])
        before = value[max(0, match.start() - 16):match.start()]
        output.append(match.group(0) if re.search(r"world\s+war\s*$", before, re.I) else " <number> ")
        cursor = match.end()
    output.append(value[cursor:])
    return " ".join("".join(output).split())


def _temporal_interval_retrieval_replace_scalars(query: str, source_question: str, target_question: str) -> tuple[str, bool]:
    source, target = _temporal_interval_retrieval_scalar_slots(source_question), _temporal_interval_retrieval_scalar_slots(target_question)
    if Counter(kind for kind, _ in source) != Counter(kind for kind, _ in target):
        return query, not source and not target
    source_by_kind: dict[str, list[str]] = defaultdict(list)
    target_by_kind: dict[str, list[str]] = defaultdict(list)
    for kind, value in source:
        source_by_kind[kind].append(value)
    for kind, value in target:
        target_by_kind[kind].append(value)
    for kind, values in source_by_kind.items():
        for old, new in zip(values, target_by_kind[kind]):
            if kind == "year":
                query = query.replace(f'"{old}-01-01"', f'"{new}-01-01"')
                query = query.replace(f'"{old}-12-31"', f'"{new}-12-31"')
            query = re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d.])", new, query)
    return query, True


def _target_role_score(role: str, label: str, types: Sequence[str], context: str) -> float:
    type_text, label_text, context_text = (
        _temporal_interval_retrieval_normalized(" ".join(types)),
        _temporal_interval_retrieval_normalized(label),
        _temporal_interval_retrieval_normalized(context),
    )
    score = 0.0
    if role == "event":
        if any(value in type_text for value in ("event", "war", "conflict", "battle")):
            score += 3.0
        if re.search(rf"\b(?:during|while|period|war|battle)\b[^.?!]*\b{re.escape(label_text)}\b", context_text):
            score += 0.8
    elif role == "office":
        if any(value in type_text for value in ("government office", "appointed role")):
            score += 3.0
        if any(value in label_text for value in ("president", "governor", "minister", "secretary", "office", "title")):
            score += 0.8
    return score


def _mapping(
    document: Any,
    schema: _IntervalSchema,
    entities: Sequence[tuple[str, str]],
    type_hints: dict[str, list[str]],
    context: str,
) -> tuple[float, tuple[tuple[str, str], ...]] | None:
    output: list[tuple[float, tuple[tuple[str, str], ...]]] = []
    roles = [_source_role(schema, value) for value in document.constants]
    for assigned in permutations(entities, len(document.constants)):
        score = 0.0
        for position, ((target_id, label), source_id) in enumerate(zip(assigned, document.constants)):
            if target_id == source_id:
                score += 4.0
            score += _target_role_score(roles[position], label, type_hints.get(label, []), context)
            role_text = document.constant_roles[position] if position < len(document.constant_roles) else ""
            score += 2.0 * _temporal_interval_retrieval_token_similarity(role_text, context)
        output.append((score, tuple(assigned)))
    return max(output, default=None, key=lambda item: (item[0], tuple(value[0] for value in item[1])))


def _interval_documents(pipeline: Any, index: Any) -> tuple[tuple[Any, _IntervalSchema], ...]:
    cached = getattr(pipeline, "_temporal_interval_documents", None)
    if cached is not None:
        return cached
    documents = tuple(
        (document, schema)
        for document in index.documents
        if (schema := _derive_interval_schema(document.sparql)) is not None
    )
    pipeline._temporal_interval_documents = documents
    return documents


def _retrieve_documents(
    context: Any,
    interval_documents: Sequence[tuple[Any, _IntervalSchema]],
    entities: Sequence[tuple[str, str]],
) -> list[tuple[float, Any, _IntervalSchema]]:
    decomposition = _temporal_interval_retrieval_decompositions(context)
    surfaces = [label for _, label in entities]
    masked_question = _temporal_interval_retrieval_mask_entities(_temporal_interval_retrieval_normalize_scalars(context.question), surfaces)
    query_tokens = Counter(
        _temporal_interval_retrieval_tokens(
            " ".join(
                [masked_question, masked_question]
                + [_temporal_interval_retrieval_mask_entities(_temporal_interval_retrieval_normalize_scalars(value), surfaces) for value in decomposition]
            )
        )
    )
    compatible = [
        (document, schema)
        for document, schema in interval_documents
        if len(document.constants) == len(entities)
    ]
    document_tokens = [
        _temporal_interval_retrieval_tokens(" ".join([_temporal_interval_retrieval_normalize_scalars(document.question), _temporal_interval_retrieval_relation_text(document.sparql)]))
        for document, _ in compatible
    ]
    frequencies = [Counter(values) for values in document_tokens]
    average_length = (
        sum(len(values) for values in document_tokens) / len(document_tokens)
        if document_tokens else 0.0
    )
    document_frequency: Counter[str] = Counter()
    for values in document_tokens:
        document_frequency.update(set(values))
    count = len(document_tokens)
    idf = {
        token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
        for token, frequency in document_frequency.items()
    }
    target_operator = _interval_operator_signature(context.question)
    ranked: list[tuple[float, Any, _IntervalSchema]] = []
    for position, (document, schema) in enumerate(compatible):
        source_operator = _interval_operator_signature(
            document.question, document.sparql
        )
        if _temporal_interval_retrieval_operator_conflict(source_operator, target_operator):
            continue
        frequency = frequencies[position]
        normalization = 0.25 + (
            0.75 * len(document_tokens[position]) / average_length
            if average_length else 0.0
        )
        score = sum(
            idf.get(token, 0.0)
            * (count * 2.2 / (count + 1.2 * normalization))
            * min(query_count, 2)
            for token, query_count in query_tokens.items()
            if (count := frequency.get(token, 0))
        )
        score += 12.0 * _temporal_interval_retrieval_token_similarity(str(context.question), document.question)
        if target_operator == source_operator:
            score += 1.0
        ranked.append((score, document, schema))
    ranked.sort(key=lambda item: (-item[0], item[1].source_index))
    return ranked[:32]


def _instantiate_templates(
    context: Any,
    entities: Sequence[tuple[str, str]],
    interval_documents: Sequence[tuple[Any, _IntervalSchema]],
) -> list[dict[str, Any]]:
    decomposition = _temporal_interval_retrieval_decompositions(context)
    type_hints = _type_hints(context)
    text = " ".join([str(context.question), *decomposition])
    output: list[dict[str, Any]] = []
    for rank, (score, document, schema) in enumerate(
        _retrieve_documents(context, interval_documents, entities), start=1
    ):
        assigned = _mapping(document, schema, entities, type_hints, text)
        if assigned is None:
            continue
        mapping_score, values = assigned
        query, compatible = _temporal_interval_retrieval_replace_scalars(document.sparql, document.question, context.question)
        if not compatible:
            continue
        mapping: dict[str, str] = {}
        for source_id, (target_id, _) in zip(document.constants, values):
            mapping[source_id] = target_id
            query = re.sub(
                rf"(?<![A-Za-z0-9_])ns:{re.escape(source_id)}(?![A-Za-z0-9_])",
                f"ns:{target_id}",
                query,
            )
        if not _temporal_interval_retrieval_SELECT_LIMIT_RE.search(query):
            query = f"{query.strip()}\nLIMIT 256"
        output.append(
            {
                "key": hashlib.sha256(query.encode()).hexdigest()[:20],
                "query": query,
                "source_index": document.source_index,
                "source_rank": rank,
                "mapping_score": mapping_score,
                "answer_var": document.answer_var,
                "runtime_score": score + 0.8 * mapping_score,
                "mode": "train_interval_template",
            }
        )
    return output


def _relation_id(value: Any) -> str:
    if isinstance(value, list):
        return ".".join(str(part).strip().replace(" ", "_") for part in value)
    return str(value).strip()


def _event_clause(
    context: Any,
    entities: Sequence[tuple[str, str]],
    type_hints: dict[str, list[str]],
    office_id: str,
) -> tuple[list[str], str, dict[str, Any]] | None:
    for entity_id, label in entities:
        if entity_id == office_id:
            continue
        type_text = _temporal_interval_retrieval_normalized(" ".join(type_hints.get(label, [])))
        if any(value in type_text for value in ("event", "war", "conflict", "battle")):
            return [], f"ns:{entity_id}", {"mode": "direct_typed_event"}
    # Type hints can be absent after a guarded decomposition rollback even
    # though entity linking still gives an unambiguous event/office pair.  A
    # single remaining entity whose surface itself denotes the event is safer
    # than splicing an unrelated semantic path.  This is surface grammar, not
    # an entity or relation allow-list.
    remaining = [
        (entity_id, label)
        for entity_id, label in entities
        if entity_id != office_id
    ]
    if len(remaining) == 1:
        entity_id, label = remaining[0]
        label_text = _temporal_interval_retrieval_normalized(label)
        question_text = _temporal_interval_retrieval_normalized(str(context.question))
        if (
            label_text
            and label_text in question_text
            and re.search(r"\b(?:event|war|battle|conflict)\b", label_text)
            and re.search(r"\b(?:during|while|at the same time)\b", question_text)
        ):
            return [], f"ns:{entity_id}", {"mode": "direct_surface_event"}
    entity_by_label = {
        _temporal_interval_retrieval_normalized(label): entity_id for entity_id, label in entities if entity_id != office_id
    }
    for trace in _traces(context):
        if trace.get("stage") != "semantic_path_generation":
            continue
        for candidate in trace.get("output", []) if isinstance(trace.get("output"), list) else []:
            value = candidate.get("output", {}) if isinstance(candidate, dict) else {}
            anchors = {
                str(item.get("id", "")): str(item.get("surface", ""))
                for item in value.get("anchors", []) if isinstance(item, dict)
            }
            for path in value.get("semantic_paths", []):
                if not isinstance(path, dict):
                    continue
                anchor_ref = str(path.get("anchor_ref", ""))
                entity_id = entity_by_label.get(_temporal_interval_retrieval_normalized(anchors.get(anchor_ref, "")))
                steps = [item for item in path.get("steps", []) if isinstance(item, dict)]
                if not entity_id or not steps:
                    continue
                # The reviewed decomposition names the event-producing path;
                # its first projected node is the event.  This avoids a
                # relation-name allow-list when locating the splice point.
                event_position = 1
                triples: list[str] = []
                current, event_term = f"ns:{entity_id}", ""
                for position, step in enumerate(steps[:event_position], start=1):
                    relation = _relation_id(step.get("relation_label", ""))
                    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", relation):
                        triples = []
                        break
                    target = f"?event_path_{position}"
                    triples.append(
                        f"{target} ns:{relation} {current} ."
                        if str(step.get("direction", "")).casefold() == "backward"
                        else f"{current} ns:{relation} {target} ."
                    )
                    current, event_term = target, target
                if triples:
                    return triples, event_term, {
                        "mode": "frozen_semantic_event_path",
                        "hop_count": len(triples),
                    }
    return None


def _canonical_candidate(
    context: Any,
    entities: Sequence[tuple[str, str]],
    schema: _IntervalSchema | None,
) -> dict[str, Any] | None:
    if schema is None:
        return None
    decomposition, type_hints = _temporal_interval_retrieval_decompositions(context), _type_hints(context)
    text = " ".join([str(context.question), *decomposition])
    office_id, office_label = max(
        entities,
        key=lambda item: _target_role_score("office", item[1], type_hints.get(item[1], []), text),
    )
    if _target_role_score("office", office_label, type_hints.get(office_label, []), text) <= 0.0:
        return None
    event = _event_clause(context, entities, type_hints, office_id)
    if event is None:
        return None
    event_triples, event_term, event_evidence = event
    question_text = _temporal_interval_retrieval_normalized(context.question)
    slots = _temporal_interval_retrieval_scalar_slots(context.question)
    argmax = bool(re.search(r"\b(?:last|latest|most recent)\b", question_text))
    argmin = bool(re.search(r"\b(?:first|earliest)\b", question_text))
    if not slots and not argmax and not argmin and not event_triples:
        return None
    extra: list[str] = []
    ordering = "LIMIT 256"
    scalar_evidence: dict[str, Any] = {}
    if slots:
        dates = [value for kind, value in slots if kind in {"date", "year"}]
        if len(dates) != 1 or not re.search(r"\b(?:left|ended|end|terminated|until)\b", question_text):
            return None
        value = dates[0] if len(dates[0]) > 4 else f"{dates[0]}-01-01"
        if re.search(r"\b(?:after|later than)\b", question_text):
            comparison = ">"
        elif re.search(r"\b(?:before|prior to|earlier than)\b", question_text):
            comparison = "<"
        else:
            return None
        extra.extend(
            [
                f"?tenure ns:{schema.tenure_to_relation} ?constraint_date .",
                f'FILTER (xsd:dateTime(?constraint_date) {comparison} "{value}T00:00:00"^^xsd:dateTime)',
            ]
        )
        scalar_evidence = {
            "predicate_role": "tenure_end_from_train_schema",
            "comparison": comparison,
            "normalized_value": value,
        }
    elif argmax or argmin:
        ordering = f"ORDER BY {'DESC' if argmax else 'ASC'}(?to) LIMIT 1"
    lines = [
        "PREFIX ns: <http://rdf.freebase.com/ns/>",
        "SELECT DISTINCT ?x",
        "WHERE {",
        *(f"  {value}" for value in event_triples),
        f"  {event_term} ns:{schema.event_start_relation} ?start ;",
        f"             ns:{schema.event_end_relation} ?end .",
        f"  ?x ns:{schema.answer_to_tenure_relation} ?tenure .",
        f"  ?tenure ns:{schema.office_title_relation} ns:{office_id} ;",
        f"           ns:{schema.tenure_from_relation} ?from ;",
        f"           ns:{schema.tenure_to_relation} ?to .",
        "  FILTER (?from < ?end)",
        "  FILTER (?to > ?start)",
        *(f"  {value}" for value in extra),
        "}",
        ordering,
    ]
    query = "\n".join(lines)
    return {
        "key": hashlib.sha256(query.encode()).hexdigest()[:20],
        "query": query,
        "source_index": -1,
        "source_rank": 0,
        "mapping_score": 0.0,
        "answer_var": "x",
        "runtime_score": 1_000.0,
        "mode": "canonical_interval_splice",
        "canonical_evidence": {
            "event": event_evidence,
            "scalar": scalar_evidence,
            "operator": "argmax" if argmax else ("argmin" if argmin else "none"),
        },
    }


def _temporal_interval_retrieval_retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    started = time.perf_counter()
    is_eligible, eligibility = _eligibility_evidence(context)
    diagnostics: dict[str, Any] = {
        "version": _temporal_interval_retrieval_VERSION,
        "status": "abstained",
        "reason": "not_explicit_event_office_interval",
        "trigger": "final_empty_exclusive_interval_route",
        "eligible": is_eligible,
        "eligibility_evidence": eligibility,
        "claimed_query_budget": is_eligible,
        "uses_evaluation_gold": False,
        "uses_test_sparql_or_reasoning_information": False,
        "source": "CWQ_TRAIN_interval_gold_sparql_only",
        "llm_calls": 0,
        "embedding_calls": 0,
        "candidate_query_limit": 3,
        "entity_or_relation_whitelist": False,
    }
    if not is_eligible:
        return {"answer_ids": [], "diagnostics": diagnostics}
    entities = _temporal_interval_retrieval_entities(context)
    if len(entities) < 2:
        diagnostics["reason"] = "fewer_than_two_linked_entities"
        return {"answer_ids": [], "diagnostics": diagnostics}
    index = _temporal_interval_retrieval_index(context.pipeline)
    interval_documents = _interval_documents(context.pipeline, index)
    candidates = _instantiate_templates(context, entities, interval_documents)
    schema = interval_documents[0][1] if interval_documents else None
    canonical = _canonical_candidate(context, entities, schema)
    if canonical is not None:
        candidates.append(canonical)
    candidates.sort(
        key=lambda item: (
            -float(item["runtime_score"]),
            int(item["source_rank"]),
            str(item["key"]),
        )
    )
    selected_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = " ".join(candidate["query"].split())
        if compact in seen:
            continue
        seen.add(compact)
        selected_candidates.append(candidate)
        if len(selected_candidates) >= min(
            3,
            max(1, int(getattr(context.pipeline, "temporal_interval_candidate_limit", 3))),
        ):
            break
    for candidate in selected_candidates:
        query_started = time.perf_counter()
        try:
            rows = endpoint.execute(candidate["query"])
            candidate["answers"] = _temporal_interval_retrieval_unique(_temporal_interval_retrieval_answer_values(rows, candidate["answer_var"]))
            candidate["error"] = ""
        except Exception as exc:
            candidate["answers"] = []
            candidate["error"] = f"{type(exc).__name__}:{exc}"
        candidate["elapsed_seconds"] = time.perf_counter() - query_started
    nonempty = [value for value in selected_candidates if value["answers"]]
    support = Counter(tuple(sorted(value["answers"])) for value in nonempty)
    selected = max(
        nonempty,
        key=lambda item: (
            support[tuple(sorted(item["answers"]))],
            float(item["runtime_score"]),
            -int(item["source_rank"]),
            str(item["key"]),
        ),
        default=None,
    )
    diagnostics.update(
        {
            "status": "selected" if selected else "all_candidates_empty",
            "reason": "temporal_interval_answer_consensus" if selected else "all_interval_queries_empty",
            "interval_template_count": len(interval_documents),
            "endpoint_execution_queries": len(selected_candidates),
            "answer_count": len(selected["answers"]) if selected else 0,
            "selected_source_index": int(selected["source_index"]) if selected else -1,
            "selected_source_rank": int(selected["source_rank"]) if selected else -1,
            "selected_mode": str(selected["mode"]) if selected else "",
            "candidate_traces": [
                {
                    "key": candidate["key"],
                    "query_sha256": hashlib.sha256(candidate.pop("query").encode()).hexdigest(),
                    "source_index": int(candidate["source_index"]),
                    "source_rank": int(candidate["source_rank"]),
                    "mode": str(candidate["mode"]),
                    "answer_count": len(candidate["answers"]),
                    "elapsed_seconds": candidate["elapsed_seconds"],
                    "error": candidate["error"],
                    **(
                        {"canonical_evidence": candidate["canonical_evidence"]}
                        if "canonical_evidence" in candidate else {}
                    ),
                }
                for candidate in selected_candidates
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    answer_ids = list(selected["answers"]) if selected else []
    entity_ids = [value for value in answer_ids if _temporal_interval_retrieval_ENTITY_ID_RE.fullmatch(value)]
    labels: list[dict[str, str]] = []
    label_error = ""
    if entity_ids:
        try:
            labels = list(context.pipeline.kg.labels(entity_ids))
        except Exception as exc:
            label_error = f"{type(exc).__name__}:{exc}"
    label_by_id = {
        str(item.get("id", "")): str(item.get("label", ""))
        for item in labels if isinstance(item, dict) and str(item.get("id", ""))
    }
    diagnostics["selected_label_queries"] = int(bool(entity_ids))
    if label_error:
        diagnostics["selected_label_error"] = label_error
    return {
        "answer_ids": answer_ids,
        "answers": [{"id": value, "label": label_by_id.get(value, value)} for value in answer_ids],
        "selected_graph": None,
        "diagnostics": diagnostics,
    }






__all__ = ['GoldPath', 'SourceGoldSparqlIndex', 'TrainGoldPathIndex', '_failure_endpoint_retrieval_plugin_retrieve', '_failure_factorized_path_retrieval_retrieve', '_failure_gold_sparql_retrieval_retrieve', '_failure_path_retrieval_FailurePathCandidate', '_failure_path_retrieval_FailurePathRetriever', '_failure_retrieval_plugin_retrieve', '_temporal_interval_retrieval_retrieve', 'eligible', 'retrieve_extrema_relation', 'retrieve_missing_relation', 'select_failure_execution']
