"""Consolidated time constraints; original rules and budgets retained."""
from __future__ import annotations


# ---- numeric_temporal_slot_normalization ----
"""Fixed-budget ontology normalization for explicit scalar constraints.

The caller supplies an already parsed date/number/extrema specification.  The
module ranks schema-backed attribute paths using anchor-masked lexical
evidence and returns deterministic graph replacements.  It never accesses
Gold data, calls a model, or names an entity/relation/sample.  Callers replace
existing execution slots; they must not append beyond their fixed beam.
"""



from collections import Counter
from copy import deepcopy
import json
import re
from typing import Any, Callable, Iterable, Mapping as _numeric_temporal_slot_normalization_Mapping, Sequence as _numeric_temporal_slot_normalization_Sequence

from .constraint_repairs import answer_set_safety_reason, constraint_focus_phrase, property_rank_evidence
from .contracts import ExecutedGraph, QueryGraphCandidate as _numeric_temporal_slot_normalization_QueryGraphCandidate
from .ontology import relation_label_from_id as _numeric_temporal_slot_normalization_relation_label_from_id


EPS = 1e-12
SCALAR_RANGES = frozenset(
    {"type.datetime", "type.enumeration", "type.float", "type.int"}
)
INTERVAL_ROLE_RE = re.compile(
    r"\b(?:during|served?|president|governor|office|position|tenure|held|current)\b",
    re.I,
)
GENERIC_WORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "by", "did", "do", "does",
        "for", "has", "have", "in", "is", "of", "on", "or", "the",
        "that", "to", "was", "were", "what", "when", "where", "which",
        "who", "with", "first", "last", "latest", "earliest", "largest",
        "smallest", "highest", "lowest", "most", "recent", "number",
        "value", "date", "year", "time", "event", "film", "movie",
        "country", "location", "organization", "organisation", "person",
        "school", "structure", "team", "war",
    }
)
TEMPORAL_MORPHOLOGY = frozenset(
    {"birth", "death", "end", "found", "open", "publication", "release", "runtime", "start"}
)
RANGE_TYPES = frozenset(
    {"GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL"}
)
ENTITY_ID_RE = re.compile(r"^(?:m|g)\.[A-Za-z0-9_]+$")
SCALAR_LITERAL_RE = re.compile(
    r"^(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"[+-]?\d{4,}(?:-\d{2}(?:-\d{2})?)?(?:Z|[+-]\d{2}:\d{2})?)$"
)
MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_numeric_temporal_slot_normalization_DATE_RE = re.compile(
    rf"(?:\b\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}\b|"
    rf"\b\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{4}}\b|"
    rf"\b(?:{MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}\b|"
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTH_NAMES})[,]?\s+\d{{4}}\b)",
    re.I,
)
SCALAR_CUE_RE = re.compile(
    r"\b(?:age|amount|area|army|attendance|capacity|casualt(?:y|ies)|code|"
    r"count|duration|episode|episodes|force|gdp|height|id|identifier|length|"
    r"minutes?|hours?|miles?|kilomet(?:er|re)s?|number|population|"
    r"postgraduates?|rate|runtime|size|soldiers?|tons?|undergraduates?|"
    r"visitors?|weight|width|percent(?:age)?)\b",
    re.I,
)
TEMPORAL_CUE_RE = re.compile(
    r"\b(?:date|year|during|served?|president|governor|office|position|tenure|"
    r"held|founded|established|opened|released?|ended?|began|started?|born|"
    r"died|death|current|until|since)\b",
    re.I,
)
COMPARISON_RE = re.compile(
    r"\b(?:no more than|no less than|at most|at least|prior to|less than|"
    r"fewer than|lower than|smaller than|under|below|before|more than|"
    r"greater than|higher than|larger than|over|above|after|later than)\b",
    re.I,
)


def _numeric_temporal_slot_normalization_lexical_tokens(value: str) -> set[str]:
    """Small generic morphology used only for numeric/temporal attributes."""

    aliases = {
        "died": "death", "dead": "death", "released": "release",
        "releasing": "release", "founded": "found", "founding": "found",
        "began": "start", "begun": "start", "ended": "end",
        "ending": "end", "visitors": "visitor", "minutes": "runtime",
        "minute": "runtime", "long": "runtime",
        "ids": "id",
    }
    output: set[str] = set()
    for raw in re.findall(r"[A-Za-z]+", str(value).casefold()):
        token = aliases.get(raw, raw)
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        output.add(token)
    return output


def _schema_leaf_exact_match(relation_id: str, focus: str) -> bool:
    """Match a complete schema leaf against the anchor-masked question.

    This is deliberately relation-agnostic.  Every token in the final schema
    component must occur in the question after the same small morphology used
    by scalar retrieval, and at least one token must carry non-generic lexical
    content.  It is used only to break otherwise equal property ranks.
    """

    leaf_tokens = _numeric_temporal_slot_normalization_lexical_tokens(str(relation_id).rsplit(".", 1)[-1])
    focus_tokens = _numeric_temporal_slot_normalization_lexical_tokens(focus)
    return bool(
        leaf_tokens
        and leaf_tokens <= focus_tokens
        and (leaf_tokens - GENERIC_WORDS - {"id"})
    )


def mask_anchor_labels(question: str, graph: _numeric_temporal_slot_normalization_QueryGraphCandidate) -> str:
    """Remove linked entity labels before recognizing a scalar literal."""

    text = str(question)
    bindings = graph.provenance.get("anchor_bindings", {})
    if isinstance(bindings, dict):
        for binding in bindings.values():
            label = str(binding.get("label", "")) if isinstance(binding, dict) else ""
            if label:
                text = re.sub(re.escape(label), " ", text, flags=re.I)
    text = re.sub(r"\bearlies\b", "earliest", text, flags=re.I)
    text = re.sub(r"\bmost\s+recently\b", "latest", text, flags=re.I)
    return " ".join(text.split())


def _equality_focus(text: str, value: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+(?:[.,/-][A-Za-z0-9]+)*", str(text))
    digits = re.sub(r"[^0-9]", "", str(value))
    indexes = [
        index
        for index, token in enumerate(tokens)
        if digits and re.sub(r"[^0-9]", "", token) == digits
    ]
    if not indexes:
        return constraint_focus_phrase(text)
    index = indexes[-1]
    preceding = tokens[index - 1].casefold() if index else ""
    end = index + 1 if preceding in {
        "at", "during", "from", "in", "on", "since", "throughout", "until",
    } else min(len(tokens), index + 4)
    return " ".join(tokens[max(0, index - 6):end])


def normalize_explicit_spec(
    question: str,
    graph: _numeric_temporal_slot_normalization_QueryGraphCandidate,
    resolver: Callable[[str], _numeric_temporal_slot_normalization_Mapping[str, Any] | None],
) -> dict[str, Any] | None:
    """Resolve one explicit constraint after entity-label/value masking."""

    text = mask_anchor_labels(question, graph)
    raw = resolver(text)
    if not isinstance(raw, _numeric_temporal_slot_normalization_Mapping):
        return None
    spec = dict(raw)
    operator_type = str(spec.get("type", "")).upper()
    if operator_type in {"ARGMIN", "ARGMAX"}:
        spec["focus"] = text
        return spec
    if operator_type in RANGE_TYPES:
        comparisons = list(COMPARISON_RE.finditer(text))
        if comparisons:
            cue = comparisons[-1]
            tail = text[cue.end():]
            date = _numeric_temporal_slot_normalization_DATE_RE.search(tail)
            number = re.search(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", tail)
            choices = [match for match in (date, number) if match is not None]
            if choices:
                literal = min(choices, key=lambda match: match.start())
                nearest = resolver(text[: cue.end()] + tail[: literal.end()])
                if (
                    isinstance(nearest, _numeric_temporal_slot_normalization_Mapping)
                    and str(nearest.get("type", "")).upper() in RANGE_TYPES
                ):
                    spec = dict(nearest)
        spec["focus"] = text
        return spec
    if operator_type != "EQUAL":
        return None

    values = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text)
    dates = list(_numeric_temporal_slot_normalization_DATE_RE.finditer(text))
    if len(values) != 1 and not (len(dates) == 1 and len(values) <= 3):
        return None
    value = str(spec.get("value", ""))
    spec["focus"] = _equality_focus(text, value)
    if dates:
        spec["value_type"] = "dateTime"
        return spec
    if re.fullmatch(r"[12]\d{3}", value) and TEMPORAL_CUE_RE.search(text):
        if INTERVAL_ROLE_RE.search(text) and re.search(
            rf"\b(?:in|during|throughout|at)\s+(?:the\s+year\s+)?{re.escape(value)}\b",
            text,
            re.I,
        ):
            spec["type"] = "TC"
            spec["value_type"] = "year"
        else:
            spec["value_type"] = "dateTime"
        return spec
    if SCALAR_CUE_RE.search(text):
        spec["value_type"] = "number"
        return spec
    return None


def _substantive_types(graph: _numeric_temporal_slot_normalization_QueryGraphCandidate) -> set[str]:
    return {
        str(operator.get("type", "")).upper()
        for operator in graph.operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() not in {"", "NO_EQUAL"}
    }


def _equivalent_types(operator_type: str) -> set[str]:
    return {
        "LESS_THAN": {"LESS_THAN", "LESS_OR_EQUAL"},
        "GREATER_THAN": {"GREATER_THAN", "GREATER_OR_EQUAL"},
    }.get(str(operator_type).upper(), {str(operator_type).upper()})


def build_fixed_slot_variants(
    graph: _numeric_temporal_slot_normalization_QueryGraphCandidate,
    spec: _numeric_temporal_slot_normalization_Mapping[str, Any],
    ontology: Any,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return at most ``limit`` deterministic replacements for ``graph``.

    The result is ordered.  A caller may try the variants in order, but once
    a variant returns a nonempty answer set it must not fall through to a
    lower-ranked property merely to obtain a smaller set.
    """

    if ontology is None or not graph.triples or int(limit) < 1:
        return []
    variable_types: dict[str, set[str]] = {}
    predecessors: dict[str, list[str]] = {}
    for subject, relation, object_ in graph.triples:
        domain = str(ontology.domain_for_relation(relation))
        range_id = str(ontology.range_for_relation(relation))
        if str(subject).startswith("V") and domain:
            variable_types.setdefault(str(subject), set()).add(domain)
        if str(object_).startswith("V") and range_id:
            variable_types.setdefault(str(object_), set()).add(range_id)
        if str(subject).startswith("V") and str(object_).startswith("V"):
            predecessors.setdefault(str(object_), []).append(str(subject))

    owners = list(dict.fromkeys([
        str(graph.answer_var),
        *predecessors.get(str(graph.answer_var), []),
    ]))
    local_focus = constraint_focus_phrase(str(spec.get("focus", "")), window=9)
    question_tokens = _numeric_temporal_slot_normalization_lexical_tokens(local_focus)
    operator_type = str(spec.get("type", "")).upper()
    candidates: list[tuple[float, str, str, str, Any, bool, bool]] = []
    seen_paths: set[tuple[str, str, str]] = set()
    for owner in owners:
        declared = variable_types.get(owner, set())
        declared_namespaces = {
            value.split(".", 1)[0] for value in declared if "." in value
        }
        expanded: set[str] = set()
        for type_id in declared:
            expanded.update(ontology.supertypes(type_id))
            expanded.update(ontology.subtypes(type_id, max_depth=2))
        for owner_type in expanded:
            for first in ontology.relations_for_domain(owner_type):
                middle_type = str(ontology.range_for_relation(first))
                paths = [(first, "")] if middle_type in SCALAR_RANGES else []
                if middle_type and middle_type not in SCALAR_RANGES:
                    paths.extend(
                        (first, second)
                        for second in ontology.relations_for_domain(middle_type)
                        if str(ontology.range_for_relation(second)) in SCALAR_RANGES
                    )
                for first_relation, second_relation in paths:
                    path_key = (owner, first_relation, second_relation)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)
                    terminal = second_relation or first_relation
                    relation_text = " ".join(
                        _numeric_temporal_slot_normalization_relation_label_from_id(first_relation)
                        + (
                            _numeric_temporal_slot_normalization_relation_label_from_id(second_relation)
                            if second_relation else []
                        )
                    )
                    relation_tokens = _numeric_temporal_slot_normalization_lexical_tokens(relation_text)
                    overlap = (
                        (question_tokens - GENERIC_WORDS)
                        & (relation_tokens - GENERIC_WORDS)
                    )
                    tail = terminal.rsplit(".", 1)[-1].casefold()
                    tc_start = bool(
                        operator_type == "TC"
                        and tail in {"from", "from_date", "start", "start_date"}
                        and INTERVAL_ROLE_RE.search(str(spec.get("focus", "")))
                    )
                    temporal_overlap = bool(
                        question_tokens & relation_tokens & TEMPORAL_MORPHOLOGY
                    )
                    evidence = property_rank_evidence(
                        first_relation=first_relation,
                        second_relation=second_relation,
                        spec=spec,
                        terminal_range=str(ontology.range_for_relation(terminal)),
                        semantic_similarity=0.0,
                        graph_score=float(graph.score),
                    )
                    if not evidence.matched_tokens and not tc_start and not temporal_overlap:
                        continue
                    score = (
                        4.0 * len(overlap)
                        + 2.0 * int(temporal_overlap)
                        + 1.5 * int(tc_start)
                        + 2.0 * int(
                            str(ontology.domain_for_relation(first_relation)).split(".", 1)[0]
                            in declared_namespaces
                        )
                        + evidence.family_score
                        + (0.20 if not second_relation else 0.0)
                        + evidence.score
                    )
                    candidates.append(
                        (
                            score,
                            owner,
                            first_relation,
                            second_relation,
                            evidence,
                            _schema_leaf_exact_match(
                                terminal,
                                str(spec.get("focus", "")),
                            ),
                            bool(
                                owner == str(graph.answer_var)
                                or any(
                                    str(subject) == owner
                                    and str(relation) == first_relation
                                    and str(object_) == str(graph.answer_var)
                                    for subject, relation, object_ in graph.triples
                                )
                            ),
                        )
                    )

    output: list[dict[str, Any]] = []
    seen_graphs: set[str] = set()
    exact_terminals = {
        second_relation or first_relation
        for _, _, first_relation, second_relation, _, exact, _ in candidates
        if exact
    }
    unique_exact_terminal = (
        next(iter(exact_terminals)) if len(exact_terminals) == 1 else ""
    )
    for (
        score,
        owner,
        first_relation,
        second_relation,
        evidence,
        exact,
        answer_path_bound,
    ) in sorted(
        candidates,
        key=lambda item: (
            -item[0],
            -int(bool(unique_exact_terminal) and (item[3] or item[2]) == unique_exact_terminal),
            item[1],
            item[2],
            item[3],
        ),
    ):
        sibling = deepcopy(graph)
        sibling.graph_id = f"{graph.graph_id}N{len(output)}"
        sibling.sparql = ""
        operator: dict[str, Any] = {
            "type": operator_type,
            "inputs": [],
            "input_var": owner,
            "attribute_relation_label": _numeric_temporal_slot_normalization_relation_label_from_id(
                second_relation or first_relation
            ),
            "attribute_relation_labels": [],
            "value": str(spec.get("value", "")),
            "value_type": str(spec.get("value_type", "")),
            "attribute_relation_id": second_relation or first_relation,
            "_fixed_slot_numeric_temporal_repair": True,
        }
        if second_relation:
            variables = {
                str(value)
                for triple in sibling.triples
                for value in (triple[0], triple[2])
                if re.fullmatch(r"V\d+", str(value))
            }
            number = max((int(value[1:]) for value in variables), default=-1) + 1
            middle = f"V{number}"
            sibling.triples.append([owner, first_relation, middle])
            operator["input_var"] = middle
        sibling.operators.append(operator)
        signature = json.dumps(
            {
                "triples": sibling.triples,
                "answer_var": sibling.answer_var,
                "operators": sibling.operators,
            },
            sort_keys=True,
        )
        if signature in seen_graphs:
            continue
        seen_graphs.add(signature)
        sibling.provenance = deepcopy(graph.provenance)
        sibling.provenance["fixed_slot_numeric_temporal_normalization"] = {
            "source_graph_id": graph.graph_id,
            "source_answer_var": graph.answer_var,
            "source_triples": deepcopy(graph.triples),
            "first_relation": first_relation,
            "second_relation": second_relation,
            "lexical_rank_score": score,
            "matched_tokens": list(evidence.matched_tokens),
            "unique_schema_leaf_exact_match": bool(
                unique_exact_terminal
                and (second_relation or first_relation) == unique_exact_terminal
            ),
            "answer_path_unique_schema_leaf_exact_match": bool(
                answer_path_bound
                and unique_exact_terminal
                and (second_relation or first_relation) == unique_exact_terminal
            ),
            "additional_model_calls": 0,
            "additional_endpoint_queries": 0,
        }
        output.append(
            {
                "graph": sibling,
                "first_relation": first_relation,
                "second_relation": second_relation,
                "evidence": evidence,
                "lexical_rank_score": score,
                "unique_schema_leaf_exact_match": bool(
                    unique_exact_terminal
                    and (second_relation or first_relation)
                    == unique_exact_terminal
                ),
                "answer_path_unique_schema_leaf_exact_match": bool(
                    answer_path_bound
                    and unique_exact_terminal
                    and (second_relation or first_relation)
                    == unique_exact_terminal
                ),
            }
        )
        if len(output) >= int(limit):
            break
    return output


def choose_fixed_slot_prediction(
    source_ids: _numeric_temporal_slot_normalization_Sequence[str],
    attempts: _numeric_temporal_slot_normalization_Sequence[_numeric_temporal_slot_normalization_Mapping[str, Any]],
    *,
    operator_type: str,
    source_answers: _numeric_temporal_slot_normalization_Sequence[_numeric_temporal_slot_normalization_Mapping[str, Any]] = (),
) -> tuple[list[str], int | None, str]:
    """Choose the first safe variant, with nonempty top-property interlock."""

    for index, attempt in enumerate(attempts):
        answers = list(map(str, attempt.get("answer_ids", [])))
        reason = answer_set_safety_reason(
            source_ids,
            answers,
            operator_type=operator_type,
            source_answers=source_answers,
            repaired_answers=attempt.get("answers", []),
        )
        if not reason:
            return answers, index, "strict_subset"
        if answers:
            return [], None, f"top_nonempty_{reason}"
    return [], None, "all_variants_empty"


def prepare_fixed_execution_slots(
    question: str,
    graphs: _numeric_temporal_slot_normalization_Sequence[_numeric_temporal_slot_normalization_QueryGraphCandidate],
    ontology: Any,
    *,
    spec_resolver: Callable[[str], _numeric_temporal_slot_normalization_Mapping[str, Any] | None],
    execution_budget: int,
    max_replacements: int = 3,
    expects_entity: Callable[[str], bool] | None = None,
    protected_graph_ids: Iterable[str] = (),
) -> tuple[list[_numeric_temporal_slot_normalization_QueryGraphCandidate], dict[str, Any]]:
    """Insert normalized variants while keeping graph/query counts constant.

    The highest-scoring eligible source remains in the beam.  Its variants
    replace the same number of lowest-priority execution slots.  Variants are
    marked so the normal selector can ignore them; they are considered only by
    :func:`postselect_fixed_slot_execution` after an incumbent is chosen.
    """

    original = list(graphs)
    protected_ids = {str(value) for value in protected_graph_ids if str(value)}
    budget = min(len(original), max(0, int(execution_budget)))
    replacement_cap = min(max(0, int(max_replacements)), max(0, budget - 1))
    diagnostics: dict[str, Any] = {
        "status": "not_applied",
        "execution_budget": budget,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "uses_whitelist": False,
        "protected_graph_ids": sorted(protected_ids),
    }
    if ontology is None or budget < 2 or replacement_cap < 1:
        diagnostics["reason"] = "missing_ontology_or_execution_slots"
        return original, diagnostics

    require_entity = bool(expects_entity and expects_entity(question))
    eligible: list[tuple[float, int, _numeric_temporal_slot_normalization_QueryGraphCandidate, dict[str, Any]]] = []
    # Keep enough tail positions available for equal-count replacement.
    for index, graph in enumerate(original[:budget]):
        spec = normalize_explicit_spec(question, graph, spec_resolver)
        if spec is None:
            continue
        operator_type = str(spec.get("type", "")).upper()
        if _substantive_types(graph) & _equivalent_types(operator_type):
            continue
        if (
            operator_type in {"ARGMIN", "ARGMAX"}
            and (
                graph.provenance.get("grounded_semantic", {}) or {}
            ).get("dropped_paths")
        ):
            continue
        if require_entity:
            answer_types: set[str] = set()
            for subject, relation, object_ in graph.triples:
                if str(object_) == str(graph.answer_var):
                    answer_types.add(str(ontology.range_for_relation(relation)))
                if str(subject) == str(graph.answer_var):
                    answer_types.add(str(ontology.domain_for_relation(relation)))
            known = {value for value in answer_types if value}
            if known and all(value.startswith("type.") for value in known):
                continue
        eligible.append((float(graph.score), -index, graph, spec))
    if not eligible:
        diagnostics["reason"] = "no_eligible_source_graph"
        return original, diagnostics

    ranked_eligible = sorted(
        eligible,
        key=lambda item: (-item[0], -item[1], item[2].graph_id),
    )
    prepared_sources: list[
        tuple[float, int, _numeric_temporal_slot_normalization_QueryGraphCandidate, dict[str, Any], list[dict[str, Any]]]
    ] = []
    top_schema_score: float | None = None
    for score, negative_index, candidate, candidate_spec in ranked_eligible:
        # Once a schema-backed source exists, only near-score challengers can
        # displace it.  This bounds ontology work and protects normal ranking.
        if top_schema_score is not None and top_schema_score - score > 0.030000001:
            break
        candidate_variants = build_fixed_slot_variants(
            candidate,
            candidate_spec,
            ontology,
            limit=replacement_cap,
        )
        if not candidate_variants:
            continue
        if top_schema_score is None:
            top_schema_score = score
        prepared_sources.append(
            (
                score,
                negative_index,
                candidate,
                candidate_spec,
                candidate_variants,
            )
        )
    if not prepared_sources:
        diagnostics["reason"] = "no_schema_backed_variant"
        return original, diagnostics

    chosen = prepared_sources[0]
    source_selection_reason = "highest_score_schema_backed_source"
    for challenger in prepared_sources[1:]:
        incumbent_top = chosen[4][0]
        challenger_top = challenger[4][0]
        score_gap = chosen[0] - challenger[0]
        lexical_delta = float(challenger_top["lexical_rank_score"]) - float(
            incumbent_top["lexical_rank_score"]
        )
        direct_schema_upgrade = bool(
            score_gap <= 0.020000001
            and incumbent_top.get("second_relation")
            and not challenger_top.get("second_relation")
            and lexical_delta >= 0.4 - EPS
        )
        unique_leaf_upgrade = bool(
            score_gap <= 0.020000001
            and challenger_top.get(
                "answer_path_unique_schema_leaf_exact_match"
            )
            and not incumbent_top.get(
                "answer_path_unique_schema_leaf_exact_match"
            )
        )
        if direct_schema_upgrade or unique_leaf_upgrade:
            chosen = challenger
            source_selection_reason = (
                "near_score_direct_schema_leaf_upgrade"
                if direct_schema_upgrade
                else "near_score_unique_schema_leaf_exact_match"
            )

    _, negative_index, source, spec, variants = chosen
    source_index = -negative_index
    source_order = [
        chosen,
        *(item for item in prepared_sources if item is not chosen),
    ]
    diverse_source_count = max(
        1,
        min(
            len(source_order),
            replacement_cap,
            max(1, budget // 2),
        ),
    )
    active_source_order = source_order[:diverse_source_count]
    active_replacement_cap = min(
        replacement_cap,
        max(0, budget - diverse_source_count),
    )
    replacement_items: list[dict[str, Any]] = []
    replacement_keys: set[str] = set()

    def retain(item: dict[str, Any]) -> None:
        graph = item["graph"]
        key = json.dumps(
            {
                "triples": graph.triples,
                "answer_var": graph.answer_var,
                "operators": graph.operators,
            },
            sort_keys=True,
        )
        if key not in replacement_keys:
            replacement_keys.add(key)
            replacement_items.append(item)

    # Give each near-score source one slot before spending the remaining slots
    # on lower-ranked properties of a single source. Query count is unchanged.
    for prepared_source in active_source_order:
        retain(prepared_source[4][0])
        if len(replacement_items) >= active_replacement_cap:
            break
    if len(replacement_items) < active_replacement_cap:
        for prepared_source in active_source_order:
            for item in prepared_source[4][1:]:
                retain(item)
                if len(replacement_items) >= active_replacement_cap:
                    break
            if len(replacement_items) >= active_replacement_cap:
                break

    count = min(len(replacement_items), active_replacement_cap)
    evicted: list[int] = []
    diverse_source_ids = {
        str(item[2].graph_id) for item in active_source_order
    }
    # Never evict a graph ranked above the source merely to manufacture more
    # repair slots.  A low-priority source may consume only still-lower tail
    # slots; if it is already last, the fixed-budget lane abstains.
    for index in range(budget - 1, source_index, -1):
        if (
            str(original[index].graph_id) in protected_ids
            or str(original[index].graph_id) in diverse_source_ids
        ):
            continue
        evicted.append(index)
        if len(evicted) == count:
            break
    count = min(count, len(evicted))
    if not count:
        diagnostics["reason"] = "no_replaceable_tail_slot"
        return original, diagnostics
    evicted_set = set(evicted[:count])
    prefix = [
        graph for index, graph in enumerate(original[:budget])
        if index not in evicted_set
    ]
    replacement_graphs = [item["graph"] for item in replacement_items[:count]]
    prepared = [*prefix, *replacement_graphs, *original[budget:]]
    diagnostics.update(
        {
            "status": "prepared",
            "constraint": dict(spec),
            "source_graph_id": source.graph_id,
            "source_selection_reason": source_selection_reason,
            "slot_allocation": "near_score_source_diverse_round_robin",
            "schema_backed_sources_considered": [
                item[2].graph_id for item in prepared_sources
            ],
            "variant_graph_ids": [graph.graph_id for graph in replacement_graphs],
            "replaced_graph_ids": [original[index].graph_id for index in sorted(evicted_set)],
            "replacement_query_count": count,
            "query_count_before": budget,
            "query_count_after": min(len(prepared), budget),
        }
    )
    return prepared, diagnostics


def is_fixed_slot_variant(graph: _numeric_temporal_slot_normalization_QueryGraphCandidate) -> bool:
    return isinstance(
        graph.provenance.get("fixed_slot_numeric_temporal_normalization"),
        dict,
    )


def partition_fixed_slot_executions(
    executions: _numeric_temporal_slot_normalization_Sequence[ExecutedGraph],
) -> tuple[list[ExecutedGraph], list[ExecutedGraph]]:
    normal, variants = [], []
    for execution in executions:
        (variants if is_fixed_slot_variant(execution.graph) else normal).append(execution)
    return normal, variants


def _provenance_source_graph_ids(graph: _numeric_temporal_slot_normalization_QueryGraphCandidate) -> list[str]:
    """Return explicit source/base graph ids in stable provenance order."""

    output: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, _numeric_temporal_slot_normalization_Mapping):
            for key, item in value.items():
                if key in {"source_graph_id", "base_graph_id"}:
                    graph_id = str(item).strip()
                    if graph_id and graph_id not in output:
                        output.append(graph_id)
                elif isinstance(item, (_numeric_temporal_slot_normalization_Mapping, list, tuple)):
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(graph.provenance)
    return output


def _triple_counter(graph: _numeric_temporal_slot_normalization_QueryGraphCandidate) -> Counter[tuple[str, str, str]]:
    return Counter(
        (str(subject), str(relation), str(object_))
        for subject, relation, object_ in graph.triples
    )


def _is_safe_source_derivation(
    selected: ExecutedGraph,
    source: ExecutedGraph,
) -> bool:
    """Verify that a derived execution is the same answered source core.

    Attribute-repair executions use ids such as ``G1A5`` even though their
    unchanged answer-producing spine comes from ``G1``.  A fixed-slot variant
    may challenge that derived execution only when the provenance/base id,
    answer variable, complete current answer set, and source core triples all
    agree.  This prevents a shared graph-id prefix from linking unrelated
    candidates.
    """

    if str(selected.graph.answer_var) != str(source.graph.answer_var):
        return False
    if set(map(str, selected.answer_ids)) != set(map(str, source.answer_ids)):
        return False
    source_core = _triple_counter(source.graph)
    selected_core = _triple_counter(selected.graph)
    return all(selected_core[triple] >= count for triple, count in source_core.items())


def _selected_source_graph_id(
    selected: ExecutedGraph,
    source_executions: _numeric_temporal_slot_normalization_Sequence[ExecutedGraph],
    available_source_ids: set[str],
) -> tuple[str, str]:
    """Resolve the selected execution to one safely equivalent beam source."""

    selected_id = str(selected.graph.graph_id)
    if selected_id in available_source_ids:
        return selected_id, "selected_is_source"

    explicit = _provenance_source_graph_ids(selected.graph)
    candidates = explicit
    evidence = "provenance_source_graph_id"
    if not candidates:
        match = re.match(r"^(G\d+)", selected_id)
        candidates = [match.group(1)] if match else []
        evidence = "canonical_graph_base"

    executions_by_id = {
        str(execution.graph.graph_id): execution
        for execution in source_executions
    }
    for source_id in candidates:
        if source_id not in available_source_ids:
            continue
        source = executions_by_id.get(source_id)
        if source is None:
            continue
        if _is_safe_source_derivation(selected, source):
            return source_id, evidence
    return "", "no_safe_selected_source"


def _unique_extrema_operator(question: str) -> str:
    text = re.sub(r'["“][^"”]*["”]', " ", str(question))
    maximum = bool(
        re.search(
            r"\b(?:latest|last|most recent|biggest|largest|highest|maximum)\b",
            text,
            re.I,
        )
        and not re.search(r"\blast name\b", text, re.I)
    )
    minimum = bool(
        re.search(
            r"\b(?:earliest|least|smallest|lowest|minimum|first)\b",
            text,
            re.I,
        )
        and not re.search(r"\bfirst name\b", text, re.I)
    )
    if maximum == minimum:
        return ""
    return "ARGMAX" if maximum else "ARGMIN"


def _answer_terminal_types(
    graph: _numeric_temporal_slot_normalization_QueryGraphCandidate,
    ontology: Any,
) -> set[str]:
    if ontology is None:
        return set()
    output: set[str] = set()
    answer_var = str(graph.answer_var)
    for subject, relation, object_ in graph.triples:
        if str(object_) == answer_var:
            output.add(str(ontology.range_for_relation(relation)))
        if str(subject) == answer_var:
            output.add(str(ontology.domain_for_relation(relation)))
    return {value for value in output if value}


def _ontology_type_compatible(
    answer_types: set[str],
    attribute_domain: str,
    ontology: Any,
) -> bool:
    if not answer_types or not attribute_domain or ontology is None:
        return False
    attribute_closure = {attribute_domain}
    attribute_closure.update(ontology.supertypes(attribute_domain))
    for answer_type in answer_types:
        closure = {answer_type}
        closure.update(ontology.supertypes(answer_type))
        closure.update(ontology.subtypes(answer_type, max_depth=2))
        if attribute_domain in closure or answer_type in attribute_closure:
            return True
    return False


def _entity_projection_source(
    selected: ExecutedGraph,
    variants: _numeric_temporal_slot_normalization_Sequence[ExecutedGraph],
    source_executions: _numeric_temporal_slot_normalization_Sequence[ExecutedGraph],
    *,
    question: str,
    ontology: Any,
    expects_entity: Callable[[str], bool] | None,
) -> tuple[ExecutedGraph | None, str]:
    """Resolve a scalar extrema projection to one complete entity source."""

    entity_intent_texts = [str(question)]
    for source in source_executions:
        decomposition = source.graph.provenance.get("decomposition", [])
        if isinstance(decomposition, (list, tuple)):
            entity_intent_texts.extend(map(str, decomposition))
    if not expects_entity or not any(
        expects_entity(text) for text in entity_intent_texts
    ):
        return None, "question_does_not_expect_entity"
    if not selected.answer_ids or not all(
        SCALAR_LITERAL_RE.fullmatch(str(value).strip())
        for value in selected.answer_ids
    ):
        return None, "selected_answers_not_all_scalar_literals"
    extrema = _unique_extrema_operator(question)
    if not extrema:
        return None, "question_not_unique_extrema"

    variants_by_source: dict[str, list[ExecutedGraph]] = {}
    for execution in variants:
        repair = execution.graph.provenance.get(
            "fixed_slot_numeric_temporal_normalization",
            {},
        )
        source_id = str((repair or {}).get("source_graph_id", ""))
        if source_id:
            variants_by_source.setdefault(source_id, []).append(execution)

    eligible: list[ExecutedGraph] = []
    for source in source_executions:
        source_id = str(source.graph.graph_id)
        source_variants = variants_by_source.get(source_id, [])
        if not source_variants or not 2 <= len(set(source.answer_ids)) <= 256:
            continue
        if not all(ENTITY_ID_RE.fullmatch(str(value)) for value in source.answer_ids):
            continue
        answer_types = _answer_terminal_types(source.graph, ontology)
        if not answer_types or any(value in SCALAR_RANGES for value in answer_types):
            continue
        top_variant = max(
            source_variants,
            key=lambda execution: float(
                (
                    execution.graph.provenance.get(
                        "fixed_slot_numeric_temporal_normalization",
                        {},
                    )
                    or {}
                ).get("lexical_rank_score", 0.0)
            ),
        )
        repair_operator = next(
            (
                operator
                for operator in top_variant.graph.operators
                if isinstance(operator, dict)
                and operator.get("_fixed_slot_numeric_temporal_repair")
            ),
            None,
        )
        repair = top_variant.graph.provenance.get(
            "fixed_slot_numeric_temporal_normalization",
            {},
        )
        first_relation = str((repair or {}).get("first_relation", ""))
        if (
            not isinstance(repair_operator, _numeric_temporal_slot_normalization_Mapping)
            or str(repair_operator.get("type", "")).upper() != extrema
            or str(repair_operator.get("input_var", ""))
            != str(source.graph.answer_var)
            or not _ontology_type_compatible(
                answer_types,
                str(ontology.domain_for_relation(first_relation)),
                ontology,
            )
        ):
            continue
        eligible.append(source)

    if not eligible:
        return None, "no_complete_entity_extrema_source"
    eligible.sort(
        key=lambda execution: (
            -float(execution.graph.score),
            execution.graph.graph_id,
        )
    )
    source = eligible[0]
    if (
        len(eligible) > 1
        and abs(
            float(source.graph.score) - float(eligible[1].graph.score)
        ) <= EPS
    ):
        return None, "entity_source_top_score_not_unique"
    if float(source.graph.score) + EPS < float(selected.graph.score):
        return None, "entity_source_scores_below_scalar_selected"
    return source, "entity_projection_source"


def postselect_fixed_slot_execution(
    selected: ExecutedGraph,
    variants: _numeric_temporal_slot_normalization_Sequence[ExecutedGraph],
    *,
    source_executions: _numeric_temporal_slot_normalization_Sequence[ExecutedGraph] = (),
    question: str = "",
    ontology: Any = None,
    expects_entity: Callable[[str], bool] | None = None,
) -> tuple[ExecutedGraph, dict[str, Any]]:
    """Apply a fixed-slot variant only to its selected source graph."""

    diagnostics: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
    }
    variant_sources = {
        str(
            (
                execution.graph.provenance.get(
                    "fixed_slot_numeric_temporal_normalization", {}
                )
                or {}
            ).get("source_graph_id", "")
        )
        for execution in variants
    }
    variant_sources.discard("")
    source_graph_id, source_evidence = _selected_source_graph_id(
        selected,
        source_executions,
        variant_sources,
    )
    challenge_source = selected
    if not source_graph_id:
        sources_by_id = {
            str(execution.graph.graph_id): execution
            for execution in source_executions
        }
        cross_source: list[tuple[ExecutedGraph, ExecutedGraph]] = []
        for execution in variants:
            repair = execution.graph.provenance.get(
                "fixed_slot_numeric_temporal_normalization", {}
            ) or {}
            candidate_source_id = str(repair.get("source_graph_id", ""))
            source = sources_by_id.get(candidate_source_id)
            if source is None or not execution.answer_ids:
                continue
            if not repair.get(
                "answer_path_unique_schema_leaf_exact_match"
            ):
                continue
            if (
                float(selected.graph.score) - float(source.graph.score)
                > 0.030000001
            ):
                continue
            candidate_answers = set(map(str, execution.answer_ids))
            source_answers = set(map(str, source.answer_ids))
            if not candidate_answers or not candidate_answers < source_answers:
                continue
            cross_source.append((execution, source))
        cross_answer_sets = {
            tuple(sorted(set(map(str, execution.answer_ids))))
            for execution, _ in cross_source
        }
        if cross_source and len(cross_answer_sets) == 1:
            challenger, source = max(
                cross_source,
                key=lambda item: (
                    float(item[0].graph.score),
                    str(item[0].graph.graph_id),
                ),
            )
            diagnostics.update(
                {
                    "status": "applied",
                    "reason": "cross_source_unique_exact_constraint",
                    "source_resolution": "cross_source_exact_schema_leaf",
                    "source_graph_id": source.graph.graph_id,
                    "selected_graph_id": challenger.graph.graph_id,
                    "answer_count": len(challenger.answer_ids),
                    "variant_graph_ids": [
                        execution.graph.graph_id
                        for execution, _ in cross_source
                    ],
                }
            )
            return challenger, diagnostics
        entity_source, entity_reason = _entity_projection_source(
            selected,
            variants,
            source_executions,
            question=question,
            ontology=ontology,
            expects_entity=expects_entity,
        )
        if entity_source is not None:
            challenge_source = entity_source
            source_graph_id = str(entity_source.graph.graph_id)
            source_evidence = entity_reason
        elif not 2 <= len(set(selected.answer_ids)) <= 256:
            diagnostics["source_resolution"] = entity_reason
            diagnostics["reason"] = "source_cardinality_outside_gate"
            return selected, diagnostics
    diagnostics["source_resolution"] = source_evidence
    if not source_graph_id:
        diagnostics["reason"] = "no_variant_for_selected_source"
        return selected, diagnostics
    diagnostics["source_graph_id"] = source_graph_id
    matching = [
        execution
        for execution in variants
        if str(
            (
                execution.graph.provenance.get(
                    "fixed_slot_numeric_temporal_normalization", {}
                )
                or {}
            ).get("source_graph_id", "")
        ) == source_graph_id
    ]
    matching.sort(
        key=lambda execution: (
            -float(
                (
                    execution.graph.provenance.get(
                        "fixed_slot_numeric_temporal_normalization", {}
                    )
                    or {}
                ).get("lexical_rank_score", 0.0)
            ),
            execution.graph.graph_id,
        )
    )
    if not matching:
        diagnostics["reason"] = "no_variant_for_selected_source"
        return selected, diagnostics
    operator_type = next(
        (
            str(operator.get("type", "")).upper()
            for operator in matching[0].graph.operators
            if isinstance(operator, dict)
            and operator.get("_fixed_slot_numeric_temporal_repair")
        ),
        "",
    )
    prediction, selected_index, reason = choose_fixed_slot_prediction(
        challenge_source.answer_ids,
        [
            {"answer_ids": execution.answer_ids, "answers": execution.answers}
            for execution in matching
        ],
        operator_type=operator_type,
        source_answers=challenge_source.answers,
    )
    diagnostics.update(
        {
            "reason": reason,
            "operator_type": operator_type,
            "variant_graph_ids": [execution.graph.graph_id for execution in matching],
        }
    )
    if selected_index is None:
        return selected, diagnostics
    challenger = matching[selected_index]
    diagnostics.update(
        {
            "status": "applied",
            "selected_graph_id": challenger.graph.graph_id,
            "answer_count": len(prediction),
        }
    )
    return challenger, diagnostics



# ---- relative_temporal ----
"""Bind relative episode dates from KG evidence, without Gold answers or LLM calls."""
from copy import deepcopy
import json
import re

from .evaluation import canonicalize_answer_id
from .ontology import relation_label_from_id as _relative_temporal_relation_label_from_id


def relative_reference(question):
    q = question.strip().rstrip(" ?.")
    match = re.search(r"\b(before|after)\s+(?:(?:he|she|they)\s+(was|became|joined)\s+)?(.+)$", q, re.I)
    if not match:
        return None
    direction, verb, name = match.groups()
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I).strip()
    if not name or re.search(r"\d|\b(?:how|many|died|death|born|named)\b", name, re.I):
        return None
    return {"direction": direction.lower(), "verb": (verb or "episode").lower(), "name": name}


def semantic_projection_items(question, decomposition):
    """Defer time-only declarations; retain relation questions and bound entity conditions."""
    relative = relative_reference(question)
    original_years = set(re.findall(r"\b[12]\d{3}\b", question))
    kept, deferred = [], []
    for item in decomposition:
        is_question = re.match(r"\s*(?:what|who|when|where|which|how)\b", item, re.I)
        comparison = relative and re.search(r"\b(before|after)\b", item, re.I) and re.search(r"\b(from|to|start|end)(?:\s+date)?\s+of\b", item, re.I)
        item_years = set(re.findall(r"\b[12]\d{3}\b", item))
        explicit_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", question))
        item_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", item))
        literal_comparison = (item_numbers and item_numbers <= explicit_numbers
                              and re.search(r"\b(before|after|less than|more than|greater than|at least|at most)\b", item, re.I))
        year_condition = (item_years and item_years <= original_years
                          and re.search(r"\b(covers|during|in year)\b", item, re.I))
        current_condition = (
            re.search(r"\b(?:covers|during|current)\b", item, re.I)
            and re.search(r"\b(?:now|current)\b", item, re.I)
        )
        ordinal_condition = (re.search(r"\b(first|earliest)\b", question, re.I)
                             and re.search(r"\bearliest\b", item, re.I) and re.search(r"\b(from|start)\b", item, re.I))
        if not is_question and (
            comparison
            or year_condition
            or current_condition
            or literal_comparison
            or ordinal_condition
        ):
            deferred.append(item)
        else:
            kept.append(item)
    return (kept, deferred) if kept else (list(decomposition), [])


class RelativeTemporalBinder:
    def __init__(self, kg, ontology):
        self.kg = kg
        self.ontology = ontology
        self.cache = {}

    def bind(self, question, graph):
        type_binding = self.bind_declared_types(graph)
        ordinal = self.bind_first(question, graph)
        ordinal["declared_type_constraints"] = type_binding
        literal = self.bind_literal(question, graph)
        if literal["status"] != "not_applicable":
            return literal
        intent = relative_reference(question)
        if not intent or self.ontology is None:
            return ordinal
        for owner, role_relation, target in graph.triples:
            if target != graph.answer_var or not owner.startswith("V"):
                continue
            role_range = self.ontology.range_for_relation(role_relation)
            if role_range.startswith("type."):
                continue
            domain = self.ontology.domain_for_relation(role_relation)
            dates = {r.rsplit(".", 1)[-1]: r for r in self.ontology.relations_for_domain(domain)
                     if self.ontology.range_for_relation(r) == "type.datetime"}
            start = dates.get("from") or dates.get("start_date")
            end = dates.get("to") or dates.get("end_date")
            if not start:
                continue
            incoming = next(((s, r) for s, r, o in graph.triples
                             if o == owner and re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", s)), None)
            if not incoming:
                continue
            subject, incoming_relation = incoming
            key = (subject, incoming_relation, role_relation, start, end, intent["name"].casefold())
            if key not in self.cache:
                optional_end = f"OPTIONAL {{ ?record <http://rdf.freebase.com/ns/{end}> ?end . }}" if end else ""
                prefix = json.dumps(intent["name"].casefold())
                query = f'''SELECT DISTINCT ?record ?reference ?start ?end WHERE {{
 <http://rdf.freebase.com/ns/{subject}> <http://rdf.freebase.com/ns/{incoming_relation}> ?record .
 ?record <http://rdf.freebase.com/ns/{role_relation}> ?reference .
 ?reference <http://rdf.freebase.com/ns/type.object.name> ?name .
 FILTER(LANGMATCHES(LANG(?name), "EN"))
 FILTER(STRSTARTS(LCASE(STR(?name)), {prefix}))
 FILTER(STRLEN(STR(?name)) = {len(intent['name'])} || SUBSTR(STR(?name), {len(intent['name']) + 1}, 1) IN (" ", ".", "-"))
 ?record <http://rdf.freebase.com/ns/{start}> ?start .
 {optional_end}
}}'''
                self.cache[key] = (query, self.kg.execute(query))
            query, evidence = self.cache[key]
            cutoff_field = "start" if intent["direction"] == "before" or intent["verb"] in {"became", "joined"} else "end"
            values = [canonicalize_answer_id(r[cutoff_field]) for r in evidence if r.get(cutoff_field)]
            values = [v for v in values if re.fullmatch(r"[12]\d{3}(?:-\d{2}-\d{2})?", v)]
            if not values:
                return {"status": "unresolved", "reason": "reference_episode_date_missing", "query": query}
            cutoff = min(values) if intent["direction"] == "before" else max(values)
            successor = role_relation.rsplit(".", 1)[-1] in {"office_holder", "coach"} and re.match(r"who\b", question.strip(), re.I)
            after_end = intent["direction"] == "after" and cutoff_field == "end"
            operator = {"type": "LESS_THAN" if intent["direction"] == "before" else "GREATER_OR_EQUAL" if after_end else "GREATER_THAN",
                        "inputs": [], "input_var": owner, "attribute_relation_label": _relative_temporal_relation_label_from_id(start),
                        "attribute_relation_labels": [], "value": cutoff,
                        "value_type": "year" if len(cutoff) == 4 else "date"}
            if len(cutoff) == 4:
                operator["_comparison_date_precision"] = "year"
            else:
                operator["_calendar_day_boundary"] = True
            # Only supersede a comparative on this same episode date. Preserve
            # every unrelated type/entity/time/numeric restriction.
            graph.operators = [op for op in graph.operators if not (
                (op.get("type") in {"LESS_THAN", "GREATER_THAN", "LESS_OR_EQUAL", "GREATER_OR_EQUAL"}
                and op.get("input_var") == owner
                and op.get("attribute_relation_label") == operator["attribute_relation_label"])
                or (op.get("type") == "TC" and op.get("input_var") == owner
                    and not re.search(r"\b[12]\d{3}\b", question)))]
            graph.operators.append(operator)
            if successor:
                graph.operators.append({"type": "ARGMAX" if intent["direction"] == "before" else "ARGMIN",
                    "inputs": [], "input_var": owner, "attribute_relation_label": _relative_temporal_relation_label_from_id(start),
                    "attribute_relation_labels": [], "value": "", "value_type": ""})
            return {"status": "bound", "intent": intent, "owner": owner, "reference_date": cutoff,
                    "reference_date_field": cutoff_field, "query": query, "evidence": deepcopy(evidence),
                    "operator": deepcopy(operator), "uses_gold_answers": False, "llm_calls": 0}
        return ordinal

    def bind_declared_types(self, graph):
        if self.ontology is None:
            return []
        added = []
        for item in graph.provenance.get("decomposition", []):
            match = re.fullmatch(r"\s*(.+?) is the type of the (.+?)\.\s*", item, re.I)
            if not match:
                continue
            type_name, role_name = [s.casefold().strip() for s in match.groups()]
            types = [d for d in self.ontology.domain_to_relation_ids
                     if d.rsplit(".", 1)[-1].replace("_", " ").casefold() == type_name]
            if len(types) != 1:
                continue
            target = next((o for _, r, o in graph.triples
                           if r.rsplit(".", 1)[-1].replace("_", " ").casefold() == role_name), None)
            if target:
                triple = [target, "type.object.type", types[0]]
                if triple not in graph.triples:
                    graph.triples.append(triple)
                    added.append(triple)
        return added

    def bind_first(self, question, graph):
        if self.ontology is None or re.search(r"first name|first language", question, re.I):
            return {"status": "not_applicable"}
        masked = question
        for entity in graph.provenance.get("anchor_bindings", {}).values():
            if entity.get("label"):
                masked = re.sub(re.escape(entity["label"]), "", masked, flags=re.I)
        if not re.search(r"\b(first|earliest)\b", masked, re.I):
            return {"status": "not_applicable"}
        for owner, role, target in graph.triples:
            if target != graph.answer_var or not owner.startswith("V"):
                continue
            domain = self.ontology.domain_for_relation(role)
            relation = next((r for r in self.ontology.relations_for_domain(domain)
                             if r.rsplit(".", 1)[-1] in {"from", "start_date"}
                             and self.ontology.range_for_relation(r) == "type.datetime"), None)
            if relation:
                graph.operators = [op for op in graph.operators if not (
                    op.get("type") in {"ARGMIN", "ARGMAX"} and op.get("input_var") == owner)]
                op = {"type": "ARGMIN", "inputs": [], "input_var": owner,
                      "attribute_relation_label": _relative_temporal_relation_label_from_id(relation),
                      "attribute_relation_labels": [], "value": "", "value_type": "",
                      "_order_temporal_lexical": True}
                graph.operators.append(op)
                return {"status": "bound", "intent": "first_episode", "operator": deepcopy(op),
                        "uses_gold_answers": False, "llm_calls": 0}
        return {"status": "not_applicable"}

    @staticmethod
    def words(text):
        aliases = {"died": "death", "dead": "death", "born": "birth", "released": "release"}
        result = set()
        for word in re.findall(r"[a-z]+", text.casefold()):
            word = aliases.get(word, word)
            if word.endswith("ing") and len(word) > 5:
                word = word[:-3]
                if len(word) > 2 and word[-1] == word[-2]:
                    word = word[:-1]
            elif word.endswith("s") and len(word) > 3:
                word = word[:-1]
            result.add(word)
        return result

    def bind_literal(self, question, graph):
        if self.ontology is None:
            return {"status": "not_applicable"}
        decomposition_text = " ".join(
            str(item) for item in graph.provenance.get("decomposition", [])
        )
        decomposition_years = [
            match.group(1)
            for match in re.finditer(
                r"\b(?:covers|during|in year)\s+([12]\d{3})\b",
                decomposition_text,
                re.I,
            )
        ]
        date_pattern = r"\b(before|after|in)\s+(?:the year\s+)?([12]\d{3}-\d{2}-\d{2}|[12]\d{3})(?![-\d])"
        dates = list(re.finditer(date_pattern, question, re.I))
        seen_dates = {
            (match.group(1).casefold(), match.group(2)) for match in dates
        }
        for match in re.finditer(date_pattern, decomposition_text, re.I):
            key = (match.group(1).casefold(), match.group(2))
            if key not in seen_dates:
                dates.append(match)
                seen_dates.add(key)
        numeric = re.search(r"\b(less than|more than|greater than|at least|at most)\s+(-?\d+(?:\.\d+)?)", question, re.I)
        numeric_equality = self.explicit_numeric_equality(question, graph)
        current = bool(
            re.search(r"\b(?:current|now)\b", question, re.I)
            or re.search(r"\b(?:covers|during)\s+(?:the\s+)?(?:current|now)\b", decomposition_text, re.I)
        )
        masked_question = question
        for entity in graph.provenance.get("anchor_bindings", {}).values():
            label = entity.get("label", "")
            if label and re.search(r"\d", label):
                masked_question = re.sub(re.escape(label), "", masked_question, flags=re.I)
        standalone_years = re.findall(r"\b[12]\d{3}\b", masked_question)
        standalone_year = standalone_years[0] if not dates and not numeric and len(standalone_years) == 1 else None
        if (
            not dates
            and not numeric
            and numeric_equality is None
            and not standalone_year
            and not decomposition_years
            and not current
        ):
            return {"status": "not_applicable"}
        owners = []
        for subject, relation, target in graph.triples:
            if target != graph.answer_var:
                continue
            target_type = self.ontology.range_for_relation(relation)
            if target_type.startswith("type."):
                continue
            domain = self.ontology.domain_for_relation(relation)
            if subject.startswith("V"):
                owners.append((subject, domain))
            owners.append((target, target_type))
        matches = []
        qwords = self.words(question)
        for owner, domain in owners:
            for relation in self.ontology.relations_for_domain(domain):
                value_type = self.ontology.range_for_relation(relation)
                name = relation.rsplit(".", 1)[-1].replace("_", " ")
                overlap = self.words(name) & qwords
                if (
                    dates
                    or (standalone_year and owner.startswith("V") and name in {"from", "start date"})
                    or (decomposition_years and owner.startswith("V") and name in {"from", "start date"})
                    or (current and owner.startswith("V") and name in {"from", "start date"})
                ) and value_type == "type.datetime":
                    priority = 2 if name in {"from", "start date"} else 1 if "release" in name else 0
                    matches.append((len(overlap) * 3 + priority, owner, relation, value_type))
                elif numeric and value_type in {"type.int", "type.float", "type.rawstring", "type.text", "type.enumeration"} and overlap - {"id", "number"}:
                    matches.append((len(overlap), owner, relation, value_type))
        if not matches:
            measurement_year = (
                standalone_year
                if standalone_year
                and re.search(
                    r"\b(?:rate|percentage|percent|value|amount|population|gdp|cpi|"
                    r"inflation|emission|emissions|labor|labour|metric ton|metric tons)\b",
                    question,
                    re.I,
                )
                else None
            )
            if (numeric_equality is None and measurement_year is None) or not owners:
                return {"status": "not_applicable"}
            # Semantic can omit a scalar attribute even though the original
            # question contains an explicit value constraint.  Preserve that
            # constraint as a deliberately unresolved operator.  The bounded
            # ontology/embedding repair in the pipeline resolves its direct or
            # CVT attribute path before an answer is accepted.  This uses no
            # entity/value whitelist and does not add a model call.
            owner = next(
                (candidate for candidate, _ in owners if candidate == graph.answer_var),
                owners[0][0],
            )
            if numeric_equality is not None:
                value, attribute_words = numeric_equality
                value_type = "number"
                intent = "explicit_numeric_equality_coverage"
            else:
                value = str(measurement_year)
                attribute_words = " ".join(
                    word
                    for word in re.findall(r"[A-Za-z]+", question.casefold())
                    if word not in {
                        "a", "an", "the", "that", "which", "what", "where",
                        "who", "whose", "of", "in", "on", "once", "with",
                        "and", "or", "had", "has", "have", "was", "were",
                        "is", "are", "country", "countries", "location",
                    }
                )[-160:]
                value_type = "year"
                intent = "explicit_measurement_year_coverage"
            graph.operators = [
                old
                for old in graph.operators
                if not (
                    str(old.get("type", "")).upper() in {"ARGMIN", "ARGMAX"}
                    and not re.search(
                        r"\b(?:largest|smallest|highest|lowest|biggest|latest|earliest|most recent|first|last)\b",
                        question,
                        re.I,
                    )
                )
            ]
            if measurement_year is not None:
                existing = next(
                    (
                        old
                        for old in graph.operators
                        if str(old.get("type", "")).upper()
                        in {
                            "EQUAL",
                            "GREATER_THAN",
                            "GREATER_OR_EQUAL",
                            "LESS_THAN",
                            "LESS_OR_EQUAL",
                        }
                        and str(old.get("value", "")).strip() == str(measurement_year)
                    ),
                    None,
                )
                if existing is not None:
                    existing["type"] = "EQUAL"
                    existing["value_type"] = "year"
                    existing["_constraint_coverage_repair"] = True
                    existing.pop("_numeric_comparison_cast", None)
                    existing.pop("_numeric_cast_compare", None)
                    graph.provenance["explicit_numeric_constraint_repair"] = {
                        "value": str(measurement_year),
                        "attribute_words": attribute_words,
                        "owner": str(existing.get("input_var", owner)),
                        "reused_existing_operator": True,
                        "uses_gold_answers": False,
                        "llm_calls": 0,
                    }
                    return {
                        "status": "bound",
                        "intent": intent,
                        "operator": deepcopy(existing),
                        "uses_gold_answers": False,
                        "llm_calls": 0,
                    }
            operator = {
                "type": "EQUAL",
                "inputs": [],
                "input_var": owner,
                "attribute_relation_label": ["explicit numeric constraint", attribute_words],
                "attribute_relation_labels": [],
                "value": value,
                "value_type": value_type,
                "_numeric_cast_compare": True,
                "_constraint_coverage_repair": True,
            }
            graph.operators.append(operator)
            graph.provenance["explicit_numeric_constraint_repair"] = {
                "value": value,
                "attribute_words": attribute_words,
                "owner": owner,
                "uses_gold_answers": False,
                "llm_calls": 0,
            }
            return {
                "status": "bound",
                "intent": intent,
                "operator": deepcopy(operator),
                "uses_gold_answers": False,
                "llm_calls": 0,
            }
        _, owner, relation, value_type = max(matches)
        bound = []
        if value_type == "type.datetime":
            constraints = (
                [(match.group(1).lower(), match.group(2)) for match in dates]
                or ([("in", standalone_year)] if standalone_year else [])
            )
            for year in decomposition_years:
                if ("in", year) not in constraints:
                    constraints.append(("in", year))
            if current:
                constraints.append(("in", "NOW"))
        else:
            constraints = [(numeric.group(1).lower(), numeric.group(2))]
        types = {"before": "LESS_THAN", "after": "GREATER_THAN", "in": "TC", "less than": "LESS_THAN",
                 "more than": "GREATER_THAN", "greater than": "GREATER_THAN", "at least": "GREATER_OR_EQUAL", "at most": "LESS_OR_EQUAL"}
        for direction, value in constraints:
            op = {"type": types[direction], "inputs": [], "input_var": owner,
                  "attribute_relation_label": _relative_temporal_relation_label_from_id(relation), "attribute_relation_labels": [],
                  "value": value, "value_type": "year" if value_type == "type.datetime" and len(value) == 4 else "date" if value_type == "type.datetime" else "number"}
            if value_type == "type.datetime" and op["type"] != "TC":
                op["_comparison_date_precision"] = "year" if len(value) == 4 else "day"
                op["_calendar_day_boundary"] = len(value) == 10
            elif value_type in {"type.rawstring", "type.text", "type.enumeration"}:
                op["_numeric_comparison_cast"] = True
            graph.operators = [old for old in graph.operators if not (
                old.get("input_var") == owner
                and old.get("attribute_relation_label") == op["attribute_relation_label"]
                and old.get("type") == op["type"]
                and (
                    old.get("type") != "TC"
                    or str(old.get("value")) == value
                )
            )]
            graph.operators.append(op)
            bound.append(deepcopy(op))
        return {"status": "bound", "intent": "explicit_date_or_number", "operators": bound, "uses_gold_answers": False, "llm_calls": 0}

    @staticmethod
    def explicit_numeric_equality(question, graph):
        """Return an uncovered scalar equality expressed by the question.

        Four-digit values are normally years and are left to temporal binding.
        They count as scalar values only when an adjacent measurement unit makes
        that reading explicit.  Numbers inside already-linked entity labels are
        masked before matching, preventing event titles such as ``2010 FIFA
        World Cup`` from becoming accidental numeric filters.
        """
        masked = str(question)
        for entity in graph.provenance.get("anchor_bindings", {}).values():
            label = str(entity.get("label", "")).strip()
            if label and re.search(r"\d", label):
                masked = re.sub(re.escape(label), " ", masked, flags=re.I)

        # Comparison operators and dates already have dedicated semantics.
        if re.search(
            r"\b(?:less than|more than|greater than|at least|at most|before|after)\s+-?\d",
            masked,
            re.I,
        ):
            return None
        masked = re.sub(r"\b[12]\d{3}[-/]\d{1,2}[-/]\d{1,2}\b", " ", masked)
        matches = list(re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", masked))
        if not matches:
            return None

        scalar_cues = re.compile(
            r"\b(?:rate|percentage|percent|value|amount|number|count|population|"
            r"episode|episodes|gdp|cpi|inflation|emission|emissions|labor|labour|"
            r"metric ton|metric tons|ton|tons|kilomet(?:er|re)|mile|dollar)\b",
            re.I,
        )
        candidates = []
        for match in matches:
            value = match.group(0)
            start, end = match.span()
            context = masked[max(0, start - 72):min(len(masked), end + 32)]
            if not scalar_cues.search(context):
                continue
            # A four-digit scalar in CWQ's dated measurement templates is a
            # year even when the unit follows it (for example, ``2009 metric
            # ton``).  It is handled by measurement-year coverage above.
            if re.fullmatch(r"[12]\d{3}", value):
                continue
            candidates.append((match, value, context))
        if len(candidates) != 1:
            return None

        _, value, context = candidates[0]
        normalized_value = value.lstrip("+")
        for operator in graph.operators:
            old_value = str(operator.get("value", "")).strip().replace(",", "")
            if old_value == normalized_value and str(operator.get("type", "")).upper() in {
                "EQUAL",
                "GREATER_THAN",
                "GREATER_OR_EQUAL",
                "LESS_THAN",
                "LESS_OR_EQUAL",
            }:
                return None

        words = re.findall(r"[A-Za-z]+", context.casefold())
        stop = {
            "a", "an", "the", "that", "which", "what", "where", "who",
            "whose", "do", "does", "did", "had", "has", "have", "was",
            "were", "is", "are", "at", "of", "in", "on", "once", "with",
            "and", "or", "people", "country", "countries", "location",
        }
        attribute_words = " ".join(word for word in words if word not in stop)[-160:]
        if not attribute_words:
            return None
        return normalized_value, attribute_words



# ---- bound_year_equality_safety ----
"""Pure compiler for an explicit year equality on an existing bound field.

No relation/projection/anchor/other operator changes; no execution, selection,
model, embedding, file or Gold access. A runtime caller owns saved-request/time and nonregression
gates. Source singleton correction is a diagnostic opportunity, not a claim
that an already-perfect singleton can be changed safely.
"""


from collections.abc import Mapping as _bound_year_equality_safety_Mapping
from copy import deepcopy
import re
from typing import Any


_bound_year_equality_safety_COMPARISONS = frozenset({"EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"})
_VARIABLE = re.compile(r"V\d+\Z")
_bound_year_equality_safety_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")


def _bound_year_equality_safety_words(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", str(value).replace("_", " ").casefold()))


def _attribute_words(predicate: str) -> tuple[str, ...]:
    return tuple(w for w in _bound_year_equality_safety_words(str(predicate).rsplit(".", 1)[-1]) if w not in {"date", "time", "of"})


def _bound_year_equality_safety_contains(text: str, phrase: tuple[str, ...]) -> bool:
    words = _bound_year_equality_safety_words(text)
    return bool(phrase) and any(words[i:i + len(phrase)] == phrase for i in range(len(words) - len(phrase) + 1))


def compile_bound_year_equality(
    question: str, source_graph: Any, ontology: Any, *, graph_id: str = "bound_year_equality_diagnostic",
) -> tuple[Any | None, dict[str, Any]]:
    """Compile one uniquely attested event-year comparison; never select IDs."""
    evidence = {"status": "not_compiled", "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "uses_entity_or_question_whitelist": False, "compiler_selects_answers": False,
                "requires_actual_saved_budget_before_runtime_execution": True,
                "pure_compiler_does_not_execute_or_select_answers": True}
    def abstain(reason):
        evidence["reason"] = reason
        return None, evidence

    question = str(question)
    if (re.search(r'["“”]|\b(?:not|never|before|after|prior|later|earlier|less|more|greater|lower|higher|during|through|until|since|between|or|either)\b', question, re.I)
            or re.findall(r"(?<!\w)\d+(?!\w)", question) == []):
        return abstain("question_is_not_one_unambiguous_event_year_equality")
    years = re.findall(r"(?<!\w)([12]\d{3})(?!\w)", question)
    numbers = re.findall(r"(?<!\w)\d+(?!\w)", question)
    if len(years) != 1 or numbers != years:
        return abstain("question_has_other_numbers_or_multiple_years")
    year = years[0]
    triples = getattr(source_graph, "triples", None)
    operators = getattr(source_graph, "operators", None)
    provenance = getattr(source_graph, "provenance", None)
    if (ontology is None or not isinstance(triples, list) or len(triples) != 2
            or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)
            or not isinstance(operators, list) or any(not isinstance(o, _bound_year_equality_safety_Mapping) for o in operators)
            or not isinstance(provenance, _bound_year_equality_safety_Mapping)):
        return abstain("not_one_existing_entity_path_and_bound_datetime_field")
    if any(provenance.get(key) for key in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("postprocessed_source_not_one_query")
    answer = str(source_graph.answer_var)
    fields = [(str(a), str(p), str(b)) for a, p, b in triples
              if str(a) == answer and str(ontology.range_for_relation(str(p))) == "type.datetime"]
    if len(fields) != 1 or not _VARIABLE.fullmatch(fields[0][2]):
        return abstain("year_field_is_not_unique_or_not_owned_by_answer")
    owner, attribute, value_var = fields[0]
    context = [tuple(map(str, t)) for t in triples if tuple(map(str, t)) != fields[0]]
    if len(context) != 1 or context[0][2] != owner or not _bound_year_equality_safety_ENTITY.fullmatch(context[0][0]):
        return abstain("context_is_not_one_anchor_to_answer_entity_role")
    attribute_words = _attribute_words(attribute)
    attribute_pattern = r"\s+".join(re.escape(word) for word in attribute_words)
    if not attribute_words or not re.search(r"\b" + attribute_pattern + r"\s+in\s+" + year + r"\b", question, re.I):
        return abstain("question_does_not_explicitly_name_bound_field_in_year")
    entity_type = str(ontology.range_for_relation(context[0][1]) or "")
    domain = str(ontology.domain_for_relation(attribute) or "")
    if entity_type != domain or not entity_type or entity_type.startswith("type."):
        return abstain("entity_role_and_datetime_owner_not_exactly_same_declared_type")
    bindings = provenance.get("anchor_bindings")
    if not isinstance(bindings, _bound_year_equality_safety_Mapping):
        return abstain("actual_anchor_label_binding_missing")
    anchors = [value for value in bindings.values() if isinstance(value, _bound_year_equality_safety_Mapping) and str(value.get("id")) == context[0][0]]
    if len(anchors) != 1 or not str(anchors[0].get("label", "")) or not _bound_year_equality_safety_contains(question, _bound_year_equality_safety_words(anchors[0]["label"])):
        return abstain("actual_anchor_name_is_not_unique_in_question")
    decomposition = provenance.get("decomposition")
    if not isinstance(decomposition, list):
        return abstain("independent_decomposition_year_field_attachment_missing")
    entity_words = _bound_year_equality_safety_words(entity_type.rsplit(".", 1)[-1])
    attached = [text for text in decomposition if isinstance(text, str)
                and _bound_year_equality_safety_contains(text, attribute_words) and _bound_year_equality_safety_contains(text, entity_words)
                and re.search(r"\b(?:is|equals?|equal\s+to)\s+" + year + r"\b", text, re.I)]
    if not attached:
        return abstain("decomposition_does_not_confirm_same_entity_field_year_equality")
    comparison = [(i, op) for i, op in enumerate(operators) if str(op.get("type", "")).upper() in _bound_year_equality_safety_COMPARISONS]
    if len(comparison) != 1:
        return abstain("not_one_existing_comparison_operator")
    index, original = comparison[0]
    if (str(original.get("input_var")) != value_var or original.get("attribute_relation_id") not in (None, "")
            or original.get("attribute_relation_label") not in (None, [])
            or str(original.get("value")) != year or original.get("inputs") not in (None, [])):
        return abstain("source_comparison_is_not_on_that_already_bound_year_field")
    for i, op in enumerate(operators):
        if i == index or op.get("type") == "NO_EQUAL":
            continue
        if (op.get("type") != "TC" or str(op.get("input_var")) != owner
                or str(op.get("attribute_relation_id") or "") not in {"", attribute}
                or str(op.get("value")) != year
                or op.get("attribute_relation_label") not in (None, [], [x.replace("_", " ") for x in attribute.split(".")])):
            return abstain("another_substantive_operator_has_different_owner_field_or_value")
    if original.get("type") == "EQUAL" and original.get("_comparison_date_precision") == "year":
        return abstain("source_already_has_exact_bound_year_equality")
    result = deepcopy(source_graph)
    result.graph_id, result.sparql = graph_id, ""
    replacement = deepcopy(original)
    replacement.update(type="EQUAL", input_var=value_var, value=year, value_type="dateTime",
                       attribute_relation_id="", attribute_relation_label=[], attribute_relation_labels=[],
                       _attribute_value_already_bound=True, _comparison_date_precision="year", _calendar_day_boundary=False)
    for key in ("_numeric_comparison_cast", "_numeric_cast_compare", "_numeric_cast_kind", "_temporal_exists_semantics"):
        replacement.pop(key, None)
    result.operators[index] = replacement
    evidence.update(status="compiled_not_executed", reason="question_and_decomposition_attest_same_bound_event_year",
                    year=year, owner_var=owner, value_var=value_var, scalar_relation_id=attribute,
                    anchor_id=context[0][0], anchor_name=anchors[0]["label"],
                    source_graph_id=source_graph.graph_id, compiled_graph_id=graph_id,
                    preserves_all_source_triples=True, answer_var_unchanged=True, all_other_operators_unchanged=True,
                    independent_decomposition_matches=attached, original_operator=dict(original), compiled_operator=replacement)
    result.provenance["bound_year_equality_diagnostic"] = deepcopy(evidence)
    return result, evidence



# ---- budgeted_literal_constraint_repair ----
"""Compile a proven literal/operator correction after the model stages.

No relation, node, answer projection or other operator is added or removed.
This module never executes a query or selects answers. The caller must spend
an exact-query saved request AND remaining measured time credit, enforce the
bounded endpoint timeout, and validate the full returned denotation. Singleton
sources are protected by default: a semantic correction is not measured F1
improvement, and can change an already-perfect answer.
"""


from collections.abc import Mapping as _budgeted_literal_constraint_repair_Mapping
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any

from .normalization import normalize_comparison_constraint_spec, parse_english_calendar_date as _budgeted_literal_constraint_repair_parse_english_calendar_date
from .constraint_repairs import _decomposition_items, _semantic_views
from .contracts import QueryGraphCandidate as _budgeted_literal_constraint_repair_QueryGraphCandidate
from .safety import _nested_name_schema_guard_complete as _complete
from .ontology import relation_id_from_label as _budgeted_literal_constraint_repair_relation_id_from_label


_budgeted_literal_constraint_repair_COMPARISONS = frozenset({"EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"})
_budgeted_literal_constraint_repair_VAR = re.compile(r"V\d+\Z")
_budgeted_literal_constraint_repair_ENTITY = re.compile(r"(?:[mg]\.[A-Za-z0-9_]+|http://rdf\.freebase\.com/ns/[mg]\.[A-Za-z0-9_]+)\Z")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_ON_ISO = re.compile(r"\bon\s+(\d{4}-\d{2}-\d{2})(?!\w|[Tt:]|\.\d)", re.I)
_NUMBERS = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")
_MONTH = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_ON_NAMED_DATE = re.compile(
    rf"\bon\s+((?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH})\s*,?\s+\d{{4}}|"
    rf"(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*\d{{4}}))(?!\w)", re.I)


def _evidence(reason: str) -> dict[str, Any]:
    return {"status": "not_compiled", "reason": reason, "additional_model_calls": 0,
            "additional_endpoint_queries": 0, "additional_embedding_calls": 0,
            "uses_gold_answers": False, "uses_entity_or_question_whitelist": False,
            "requires_saved_request_and_remaining_measured_time": True,
            "requires_post_model_execution": True, "compiler_selects_answers": False}


def _unquoted(text: str) -> str | None:
    if text.count('"') % 2 or text.count("“") != text.count("”"):
        return None
    # Dates/number thresholds appearing inside quotes do not establish a
    # filtering obligation, even if another valid-looking cue also exists.
    quoted = re.findall(r'"[^\"]*"|“[^”]*”', text)
    if any(re.search(r"\d|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", x, re.I) for x in quoted):
        return None
    return re.sub(r'"[^\"]*"|“[^”]*”', lambda m: " " * len(m[0]), text)


def _budgeted_literal_constraint_repair_spec(question: str) -> dict[str, Any] | None:
    text = _unquoted(str(question))
    if text is None or re.search(r"\b(?:not|never|or|either|except|around|approximately|about|nearest|between)\b", text, re.I):
        return None
    normalized = normalize_comparison_constraint_spec(text)
    if normalized is not None:
        if normalized["value_type"] == "dateTime":
            if (normalized["type"] != "EQUAL" or normalized["_source_comparison_cue"].casefold() != "on"
                    or normalized.get("_comparison_date_precision") != "day"):
                return None
            return {"type": "EQUAL", "value": normalized["value"], "value_type": "date",
                    "datatype": "type.datetime", "normalization": normalized}
        if normalized.get("_normalization_kind") == "english_cardinal_words":
            return {"type": normalized["type"], "value": normalized["value"], "value_type": "number",
                    "datatype": "number", "normalization": normalized}
    named = list(_ON_NAMED_DATE.finditer(text))
    if len(named) == 1:
        match = named[0]
        remaining = text[:match.start(1)] + text[match.end(1):]
        if (_NUMBERS.search(remaining)
                or re.search(r"\b(?:before|after|prior|later|earlier|less|more|greater|lower|higher)\b", remaining, re.I)):
            return None
        value = _budgeted_literal_constraint_repair_parse_english_calendar_date(match[1])
        if value is not None:
            return {"type": "EQUAL", "value": value, "value_type": "date", "datatype": "type.datetime",
                    "normalization": {"_source_literal": match[1], "_source_comparison_cue": "on",
                                      "_source_literal_span": [match.start(1), match.end(1)],
                                      "_comparison_date_precision": "day"}}
    matches = list(_ON_ISO.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    # Nothing else numeric/temporal is interpreted or silently discarded.
    remaining = text[:match.start(1)] + text[match.end(1):]
    if (_NUMBERS.search(remaining) or re.search(r"\b(?:before|after|prior|later|earlier|less|more|greater|lower|higher)\b", remaining, re.I)
            or re.search(r"\d{4}-\d{2}-\d{2}[Tt]\d", text)):
        return None
    try:
        date.fromisoformat(match[1])
    except ValueError:
        return None
    return {"type": "EQUAL", "value": match[1], "value_type": "date", "datatype": "type.datetime",
            "normalization": {"_source_literal": match[1], "_source_comparison_cue": "on",
                              "_comparison_date_precision": "day"}}


def _number(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value).replace(",", "").strip())
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def _value_matches(operator: _budgeted_literal_constraint_repair_Mapping, spec: _budgeted_literal_constraint_repair_Mapping) -> bool:
    if spec["value_type"] == "date":
        return str(operator.get("value", "")) == spec["value"]
    value = _number(operator.get("value"))
    return value is not None and value == _number(spec["value"])


def _semantic_field_certificate(question: str, source: Any, field: tuple[str, str, str],
                                spec: _budgeted_literal_constraint_repair_Mapping, semantics: Any, decompositions: Any) -> dict[str, Any] | None:
    """Embed the original full path uniquely into actual BGP and Compose.

    All source BGP edges, not merely the date edge, must be accounted for by
    the one certificate. This rejects unrequested containment/country/area
    joins instead of dropping them to create a convenient owner.
    """
    graph = source.graph
    provenance = getattr(graph, "provenance", None)
    compose = getattr(graph, "compose_output", None)
    if not isinstance(provenance, _budgeted_literal_constraint_repair_Mapping) or not isinstance(compose, _budgeted_literal_constraint_repair_Mapping):
        return None
    if str(compose.get("answer_var")) != str(graph.answer_var):
        return None
    anchors = provenance.get("anchor_bindings")
    if not isinstance(anchors, _budgeted_literal_constraint_repair_Mapping):
        return None
    triples = {tuple(map(str, t)) for t in graph.triples}
    items = _decomposition_items(decompositions)
    certificates = set()
    for semantic in _semantic_views(semantics):
        for path in semantic["semantic_paths"]:
            if not isinstance(path, _budgeted_literal_constraint_repair_Mapping) or not isinstance(path.get("steps"), list):
                continue
            goal = " ".join(str(path.get("goal", "")).split()).casefold()
            if goal not in items:
                continue
            condition = _budgeted_literal_constraint_repair_spec(goal)
            if condition is None or any(condition[k] != spec[k] for k in ("type", "value", "value_type")):
                continue
            anchor_id = str(path.get("anchor_ref", ""))
            linked = anchors.get(anchor_id)
            original = [a for a in semantic.get("anchors", []) if isinstance(a, _budgeted_literal_constraint_repair_Mapping) and str(a.get("id")) == anchor_id]
            if not isinstance(linked, _budgeted_literal_constraint_repair_Mapping) or len(original) != 1:
                continue
            surface = str(original[0].get("surface", ""))
            if not surface or surface.casefold() != str(linked.get("label", "")).casefold() or surface.casefold() not in question.casefold():
                continue
            entities = [e for e in compose.get("entities", []) if isinstance(e, _budgeted_literal_constraint_repair_Mapping)
                        and str(e.get("surface", "")).casefold() == surface.casefold()]
            if len(entities) != 1:
                continue
            substitutions = {str(entities[0].get("id")): str(linked.get("id"))}
            composed = {(substitutions.get(str(t.get("subject")), str(t.get("subject"))),
                         _budgeted_literal_constraint_repair_relation_id_from_label(t.get("relation_label")),
                         substitutions.get(str(t.get("object")), str(t.get("object"))))
                        for t in compose.get("triples", []) if isinstance(t, _budgeted_literal_constraint_repair_Mapping)}
            current = str(linked.get("id"))
            consumed = set()
            mapped = {}
            valid = True
            for step in path["steps"]:
                if not isinstance(step, _budgeted_literal_constraint_repair_Mapping):
                    valid = False
                    break
                predicate = _budgeted_literal_constraint_repair_relation_id_from_label(step.get("relation_label"))
                if step.get("direction") == "forward":
                    edges = {(s, p, o) for s, p, o in triples & composed if s == current and p == predicate}
                    outputs = {o for _, _, o in edges}
                elif step.get("direction") == "backward":
                    edges = {(s, p, o) for s, p, o in triples & composed if o == current and p == predicate}
                    outputs = {s for s, _, _ in edges}
                else:
                    valid = False
                    break
                if len(edges) != 1 or len(outputs) != 1:
                    valid = False
                    break
                consumed.update(edges)
                current = next(iter(outputs))
                mapped[str(step.get("to"))] = current
            if valid and consumed == triples and field in consumed and current == field[2] and graph.answer_var in mapped.values():
                certificates.add((tuple(sorted(consumed)), field, str(graph.answer_var)))
    if len(certificates) != 1:
        return None
    return {"unique_full_semantic_compose_bgp_mapping": True, "owner_var": field[0], "value_var": field[2],
            "attribute_relation": field[1], "answer_var_unchanged": graph.answer_var,
            "all_original_bgp_joins_preserved": True, "question_decomposition_date_or_number_equal": True}


def compile_budgeted_literal_constraint_repair(
    question: str, source: Any, ontology: Any, *, semantic_graphs: Any,
    decompositions: Any, graph_id: str = "saved_budget_literal_repair",
    model_calls_complete: bool = False, saved_request_credits: int = 0,
    saved_time_seconds: float = 0.0, allow_singleton_compile: bool = False,
) -> tuple[_budgeted_literal_constraint_repair_QueryGraphCandidate | None, dict[str, Any]]:
    """Compile only; default singleton protection is independent of Gold.

    ``allow_singleton_compile`` is for explicit shadow/risk compilation. It
    does not authorize execution or choosing the returned graph's answers.
    The root must still validate request/time credits at the exact execution.
    """
    if model_calls_complete is not True:
        return None, _evidence("model_stages_not_proven_complete")
    if (not isinstance(saved_request_credits, int) or isinstance(saved_request_credits, bool)
            or saved_request_credits <= 0):
        return None, _evidence("no_paid_saved_request_credit")
    if (not isinstance(saved_time_seconds, (int, float)) or not math.isfinite(saved_time_seconds)
            or saved_time_seconds < 0.02):
        return None, _evidence("no_remaining_measured_saved_time_credit")
    if not _complete(source) or not all(_budgeted_literal_constraint_repair_ENTITY.fullmatch(str(x)) for x in source.answer_ids):
        return None, _evidence("source_full_entity_execution_not_proven")
    if len(source.answer_ids) <= 1 and not allow_singleton_compile:
        return None, _evidence("singleton_incumbent_protected_by_default")
    if len(source.answer_ids) >= 256:
        return None, _evidence("source_result_near_or_above_repair_row_budget")
    graph = source.graph
    if (not isinstance(graph.triples, list) or not graph.triples
            or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in graph.triples)
            or not isinstance(graph.operators, list) or any(not isinstance(x, _budgeted_literal_constraint_repair_Mapping) for x in graph.operators)):
        return None, _evidence("source_graph_contract_missing")
    provenance = getattr(graph, "provenance", None)
    if not isinstance(provenance, _budgeted_literal_constraint_repair_Mapping) or any(provenance.get(k) for k in (
            "selection_answer_union", "selection_answer_intersection")):
        return None, _evidence("source_is_postprocessed_or_provenance_unknown")
    scalar = [(index, op) for index, op in enumerate(graph.operators) if str(op.get("type", "")).upper() != "NO_EQUAL"]
    if len(scalar) != 1 or str(scalar[0][1].get("type", "")).upper() not in _budgeted_literal_constraint_repair_COMPARISONS:
        return None, _evidence("not_one_explicit_comparison_no_contains_or_extrema")
    spec = _budgeted_literal_constraint_repair_spec(question)
    if spec is None:
        return None, _evidence("not_one_unquoted_full_on_date_or_exact_word_number")
    index, original_operator = scalar[0]
    if not _value_matches(original_operator, spec) or original_operator.get("inputs") not in (None, []):
        return None, _evidence("source_threshold_not_identical_or_has_other_inputs")
    attribute = str(original_operator.get("attribute_relation_id") or _budgeted_literal_constraint_repair_relation_id_from_label(original_operator.get("attribute_relation_label")))
    input_var = str(original_operator.get("input_var", ""))
    if not _budgeted_literal_constraint_repair_VAR.fullmatch(input_var):
        return None, _evidence("comparison_operand_not_one_actual_variable")
    possible = []
    for subject, predicate, object_ in graph.triples:
        datatype = str(ontology.range_for_relation(str(predicate))) if ontology is not None else ""
        allowed = datatype == "type.datetime" if spec["value_type"] == "date" else datatype in {"type.int", "type.float"}
        if not allowed or (attribute and attribute != str(predicate)):
            continue
        if str(object_) == input_var or (attribute and str(subject) == input_var):
            possible.append((str(subject), str(predicate), str(object_)))
    if len(possible) != 1:
        return None, _evidence("primitive_attribute_or_owner_not_uniquely_already_bound")
    field = possible[0]
    certificate = _semantic_field_certificate(question, source, field, spec, semantic_graphs, decompositions)
    if certificate is None:
        return None, _evidence("question_original_semantic_compose_full_bgp_not_one_owner")
    old_kind = str(original_operator.get("type", "")).upper()
    if spec["value_type"] == "date":
        if (old_kind == "EQUAL" and original_operator.get("_comparison_date_precision") == "day"
                and not original_operator.get("_numeric_comparison_cast") and not original_operator.get("_numeric_cast_compare")):
            return None, _evidence("already_same_complete_day_equality")
    elif old_kind == spec["type"] and str(original_operator.get("value_type", "")).casefold() == "number":
        numeric_type = ontology.range_for_relation(field[1])
        cast_is_exact = (not original_operator.get("_numeric_comparison_cast")
                         or original_operator.get("_numeric_cast_kind") == ("float" if numeric_type == "type.float" else "integer"))
        if cast_is_exact and not (old_kind == "EQUAL" and original_operator.get("_numeric_cast_compare")):
            return None, _evidence("already_same_complete_word_number_comparison")
    repaired = deepcopy(graph)
    repaired.graph_id = graph_id
    repaired.sparql = ""
    operator = deepcopy(original_operator)
    operator.update(type=spec["type"], input_var=field[2], value=spec["value"], value_type=spec["value_type"],
                    attribute_relation_id="", attribute_relation_label=[], attribute_relation_labels=[],
                    _attribute_value_already_bound=True)
    for key in ("_numeric_comparison_cast", "_numeric_cast_compare", "_numeric_cast_kind",
                "_calendar_day_boundary", "_comparison_date_precision", "_temporal_exists_semantics"):
        operator.pop(key, None)
    if spec["value_type"] == "date":
        operator.update(_comparison_date_precision="day", _calendar_day_boundary=True)
    elif spec["type"] != "EQUAL":
        operator.update(_numeric_comparison_cast=True,
                        _numeric_cast_kind="float" if ontology.range_for_relation(field[1]) == "type.float" else "integer")
    repaired.operators[index] = operator
    evidence = _evidence("single_literal_comparison_corrected_on_same_bound_attribute")
    evidence.update(status="compiled_not_executed", source_graph_id=graph.graph_id, compiled_graph_id=graph_id,
                    original_operator=deepcopy(original_operator), compiled_operator=deepcopy(operator),
                    obligation=spec, owner_certificate=certificate, singleton_shadow_only=len(source.answer_ids) == 1,
                    preserves_all_other_operators=True, preserves_exact_source_triples=True,
                    measured_available_credit_seconds=saved_time_seconds)
    repaired.provenance["budgeted_literal_constraint_repair"] = deepcopy(evidence)
    return repaired, evidence



# ---- bound_calendar_date_append ----
"""Pure append of an explicit day filter to an already bound field.

The compiler adds exactly one EQUAL operator and preserves every source BGP
edge, answer projection and operator. It does not fix unrelated bad joins or
operators, execute queries, choose answers, access files or inspect Gold.
An actual caller still owns full-execution, saved-budget and nonregression
checks. A semantic obligation does not prove that changing an answer is safe.
"""


from collections.abc import Mapping as _bound_calendar_date_append_Mapping
from copy import deepcopy
import re
from typing import Any

_bound_calendar_date_append_contains = _bound_year_equality_safety_contains
_bound_calendar_date_append_words = _bound_year_equality_safety_words
_bound_calendar_date_append_spec = _budgeted_literal_constraint_repair_spec
from .normalization import parse_english_calendar_date as _bound_calendar_date_append_parse_english_calendar_date
_bound_calendar_date_append_DATE_RE = _numeric_temporal_slot_normalization_DATE_RE
_bound_calendar_date_append_lexical_tokens = _numeric_temporal_slot_normalization_lexical_tokens
from .ontology import relation_id_from_label as _bound_calendar_date_append_relation_id_from_label


_bound_calendar_date_append_VAR = re.compile(r"V\d+\Z")
_bound_calendar_date_append_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_FILTER_TYPES = frozenset({"NO_EQUAL", "EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL", "TC"})
_AUXILIARY = frozenset({"is", "are", "was", "were", "has", "have", "had", "been", "being", "date", "time", "of", "the", "a", "an"})


def _bound_calendar_date_append_tokens(value: Any) -> set[str]:
    # Reuse existing scalar morphology. The extra irregular form is lexical,
    # not a relation/entity/question allowlist.
    return {"birth" if token == "born" else token for token in _bound_calendar_date_append_lexical_tokens(str(value))}


def _leaf_tokens(predicate: str) -> set[str]:
    return _bound_calendar_date_append_tokens(str(predicate).rsplit(".", 1)[-1]) - _AUXILIARY


def _normalized_dates(text: str, *, known_literal: str = "", known_value: str = "") -> list[str | None]:
    # The shared parser may normalize fully spelled-out day/year literals.
    # Recognize that exact attested span, not an arbitrary month substring.
    if known_literal and known_value and known_literal != known_value:
        text = re.sub(re.escape(known_literal), lambda match: known_value, text, flags=re.I)
    values = []
    for match in _bound_calendar_date_append_DATE_RE.finditer(text):
        value = match[0]
        values.append(value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
                      else _bound_calendar_date_append_parse_english_calendar_date(value))
    return values


def compile_bound_calendar_date_equality_append(
    question: str, source_graph: Any, ontology: Any, *, graph_id: str = "bound_day_append_shadow",
) -> tuple[Any | None, dict[str, Any]]:
    """Compile only a positively attached, uniquely represented day equality.

    There is deliberately no source-answer or index argument. Runtime use is
    not authorized merely by successful compilation. A runtime caller must
    require saved budget, a complete source and a nonempty strict subset.
    """
    evidence = {"status": "not_compiled", "pure_compiler_only": True,
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "uses_entity_or_question_whitelist": False, "compiler_selects_answers": False,
                "requires_actual_full_source_execution": True,
                "requires_saved_request_and_remaining_measured_time": True,
                "requires_post_model_execution": True, "compiler_does_not_authorize_answer_replacement": True}

    def abstain(reason: str):
        evidence["reason"] = reason
        return None, evidence

    question = str(question)
    spec = _bound_calendar_date_append_spec(question)
    if spec is None or spec.get("datatype") != "type.datetime" or spec.get("type") != "EQUAL":
        return abstain("question_not_one_unquoted_explicit_on_full_calendar_day")
    value = str(spec["value"])
    normalization = spec.get("normalization", {})
    span = normalization.get("_source_literal_span")
    literal = str(normalization.get("_source_literal", ""))
    # _spec's ISO fallback does not retain spans; recover the exact literal,
    # never infer an unexpressed date from an operator or decomposition.
    if not isinstance(span, list) or len(span) != 2:
        literal = value
        matches = list(re.finditer(r"\bon\s+(" + re.escape(literal) + r")(?!\w)", question, re.I))
        if len(matches) != 1:
            return abstain("normalized_question_literal_span_not_unique")
        span = [matches[0].start(1), matches[0].end(1)]
    if _normalized_dates(question, known_literal=literal, known_value=value) != [value]:
        return abstain("question_has_another_invalid_or_conflicting_calendar_date")

    triples = getattr(source_graph, "triples", None)
    operators = getattr(source_graph, "operators", None)
    provenance = getattr(source_graph, "provenance", None)
    answer = str(getattr(source_graph, "answer_var", ""))
    if (ontology is None or not _bound_calendar_date_append_VAR.fullmatch(answer) or not isinstance(triples, list)
            or not triples or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)
            or not isinstance(operators, list) or any(not isinstance(op, _bound_calendar_date_append_Mapping) for op in operators)
            or not isinstance(provenance, _bound_calendar_date_append_Mapping)):
        return abstain("source_graph_contract_or_ontology_missing")
    if any(provenance.get(k) for k in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("postprocessed_incumbent_is_not_one_preserved_query")
    if any(str(op.get("type", "")).upper() not in _FILTER_TYPES for op in operators):
        return abstain("non_filter_operator_cannot_prove_denotation_refinement")

    fields = [tuple(map(str, t)) for t in triples if str(t[0]) == answer
              and str(ontology.range_for_relation(str(t[1]))) == "type.datetime"]
    if len(fields) != 1 or not _bound_calendar_date_append_VAR.fullmatch(fields[0][2]):
        return abstain("datetime_value_not_uniquely_bound_directly_on_entity_answer")
    owner, attribute, value_var = fields[0]
    if any(str(s) == value_var or (str(o) == value_var and tuple(map(str, (s, p, o))) != fields[0]) for s, p, o in triples):
        return abstain("bound_datetime_is_reused_as_entity_or_another_field")
    primary = [tuple(map(str, t)) for t in triples if str(t[2]) == owner and _bound_calendar_date_append_MID.fullmatch(str(t[0]))]
    if len(primary) != 1:
        return abstain("entity_answer_primary_anchor_role_not_unique")
    anchor_id, primary_relation, _ = primary[0]
    role_type = str(ontology.range_for_relation(primary_relation) or "")
    domain = str(ontology.domain_for_relation(attribute) or "")
    role_closure = set(ontology.supertypes(role_type))
    domain_closure = set(ontology.supertypes(domain))
    # A specialized owner (e.g. an event-participating person) is allowed only
    # by explicit ontology inheritance, never by ordinary-role similarity.
    if (not role_type or not domain or role_type.startswith("type.") or domain.startswith("type.")
            or "common.topic" not in role_closure or "common.topic" not in domain_closure
            or not (domain in role_closure or role_type in domain_closure)):
        return abstain("primary_entity_role_and_datetime_owner_not_proven_compatible")
    role_words, field_words = _leaf_tokens(primary_relation), _leaf_tokens(attribute)
    if not role_words or not field_words:
        return abstain("primary_role_or_datetime_field_has_no_explicit_lexical_content")

    prefix = question[:span[0]]
    clause = re.search(r"\b(?P<role>[A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:who|which|that)\s+"
                       r"(?P<event>[A-Za-z]+(?:\s+[A-Za-z]+){0,4})\s+on\s*$", prefix, re.I)
    if clause is None or not role_words <= _bound_calendar_date_append_tokens(clause["role"]):
        return abstain("question_does_not_attach_date_event_to_the_primary_answer_role")
    event_words = _bound_calendar_date_append_tokens(clause["event"]) - _AUXILIARY
    if event_words != field_words:
        return abstain("question_event_verb_does_not_uniquely_name_existing_datetime_field")
    bindings = provenance.get("anchor_bindings")
    if not isinstance(bindings, _bound_calendar_date_append_Mapping):
        return abstain("actual_topic_bindings_missing")
    anchors = [v for v in bindings.values() if isinstance(v, _bound_calendar_date_append_Mapping) and str(v.get("id")) == anchor_id]
    if len(anchors) != 1 or not str(anchors[0].get("label", "")) or not _bound_calendar_date_append_contains(question, _bound_calendar_date_append_words(anchors[0]["label"])):
        return abstain("primary_actual_topic_name_not_unique_in_question")

    decomposition = provenance.get("decomposition")
    attached = []
    if isinstance(decomposition, list):
        for text in decomposition:
            if not isinstance(text, str) or _normalized_dates(text, known_literal=literal, known_value=value) != [value]:
                continue
            words = _bound_calendar_date_append_tokens(text)
            if (role_words <= words and field_words <= words
                    and re.search(r"\b(?:is|equals?|equal\s+to)\s+(?:" + re.escape(literal) + "|" + re.escape(value) + r")(?!\w)", text, re.I)):
                attached.append(text)
    if not attached:
        return abstain("decomposition_does_not_independently_attest_same_role_field_day_equality")

    for op in operators:
        kind = str(op.get("type", "")).upper()
        if kind == "NO_EQUAL":
            continue
        input_var = str(op.get("input_var", ""))
        op_attribute = str(op.get("attribute_relation_id") or _bound_calendar_date_append_relation_id_from_label(op.get("attribute_relation_label")))
        if input_var == value_var or (input_var == owner and (op_attribute == attribute or (kind == "TC" and not op_attribute))):
            return abstain("same_bound_datetime_already_has_equality_interval_or_comparison")
        op_value = str(op.get("value", ""))
        op_dates = _normalized_dates(op_value)
        date_typed = (str(op.get("value_type", "")).casefold() in {"date", "datetime", "year"}
                      or str(ontology.range_for_relation(op_attribute)) == "type.datetime")
        if (op_dates and op_dates != [value]) or (date_typed and op_value != value):
            return abstain("another_operator_has_conflicting_or_unproved_date_literal")
    result = deepcopy(source_graph)
    result.graph_id, result.sparql = graph_id, ""
    new_operator = {"type": "EQUAL", "inputs": [], "input_var": value_var,
                    "attribute_relation_id": "", "attribute_relation_label": [], "attribute_relation_labels": [],
                    "value": value, "value_type": "dateTime", "_attribute_value_already_bound": True,
                    "_comparison_date_precision": "day", "_calendar_day_boundary": True}
    result.operators.append(new_operator)
    evidence.update(status="compiled_not_executed", reason="explicit_question_and_decomposition_day_missing_on_existing_bound_field",
                    source_graph_id=source_graph.graph_id, compiled_graph_id=graph_id, owner_var=owner,
                    value_var=value_var, datetime_relation=attribute, primary_relation=primary_relation,
                    primary_entity_type=role_type, datetime_owner_type=domain,
                    primary_anchor_id=anchor_id, primary_anchor_name=anchors[0]["label"], normalized_day=value,
                    question_literal_normalization=deepcopy(normalization), question_attachment=clause[0],
                    independent_decomposition_matches=attached, appended_operator=deepcopy(new_operator),
                    preserves_exact_source_triples=True, preserves_all_source_operators=True,
                    answer_var_unchanged=True, added_relations=0, appended_operator_count=1,
                    does_not_correct_unrelated_joins_or_operators=True,
                    subset_is_not_nonregression_or_f1_improvement_proof=True)
    result.provenance["bound_calendar_date_append_shadow"] = deepcopy(evidence)
    return result, evidence



# ---- original_datetime_owner_restore ----
"""Isolated, fail-closed restoration of one originally grounded date owner.

Only a two-edge incumbent that repeats the original primary predicate and
projects its second entity can qualify. The complete original two-step path,
actual complete grounding, topic identity, question grammar and decomposition
must independently attest the primary entity and its existing datetime field.
No models, queries, file/Gold access, answer selection or production hook.
"""
from collections.abc import Mapping as _original_datetime_owner_restore_Mapping, Sequence as _original_datetime_owner_restore_Sequence
from copy import deepcopy
import re
from typing import Any

_original_datetime_owner_restore_tokens = _bound_calendar_date_append_tokens
_original_datetime_owner_restore_spec = _budgeted_literal_constraint_repair_spec
from .contracts import QueryGraphCandidate as _original_datetime_owner_restore_QueryGraphCandidate
from .lowering import lower_sparql
from .ontology import relation_id_from_label as _original_datetime_owner_restore_relation_id_from_label


_original_datetime_owner_restore_VAR = re.compile(r"V\d+\Z")
_original_datetime_owner_restore_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_XSD = "http://www.w3.org/2001/XMLSchema#"
_TEMPORAL_DATATYPES = frozenset({_XSD + "gYear", _XSD + "date", _XSD + "dateTime"})
_YEAR_LITERAL = re.compile(r"(?P<year>[1-9]\d{0,3})\s*(?P<era>BCE|BC|CE|AD)?\Z", re.I)
_YEAR_SHAPE = r"^[+-]?[0-9]+(?:Z|[+-][0-9]{2}:[0-9]{2}|-[0-9]{2}-[0-9]{2}(?:[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:[.][0-9]+)?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?)?$"


def _original_datetime_owner_restore_words(value: Any) -> set[str]:
    return {"child" if word == "children" else word for word in _original_datetime_owner_restore_tokens(value)}


def _leaf(predicate: str) -> set[str]:
    return _original_datetime_owner_restore_words(str(predicate).rsplit(".", 1)[-1]) - {"date", "time", "of"}


def _proof(reason: str) -> dict[str, Any]:
    return {"status": "not_compiled", "reason": reason, "pure_compiler_only": True,
            "model_calls": 0, "endpoint_queries": 0, "embedding_calls": 0,
            "uses_gold_answers": False, "uses_entity_question_or_relation_whitelist": False,
            "compiler_selects_answers": False, "compiler_does_not_authorize_answer_replacement": True,
            "requires_post_model_execution_and_saved_request_time_budget": True,
            "new_denotation_verified": False}


def _literal(value: str) -> dict[str, Any] | None:
    match = _YEAR_LITERAL.fullmatch(value.strip())
    if match and (match["era"] or len(match["year"]) == 4):
        magnitude = int(match["year"])
        era = str(match["era"] or "CE").upper()
        return {"precision": "year", "year": -magnitude if era in {"BC", "BCE"} else magnitude,
                "source_literal": value.strip(), "explicit_era": era,
                "year_zero_adjustment": 0, "normalization_contract": "historical_era_signed_same_magnitude_no_offset"}
    parsed = _original_datetime_owner_restore_spec("An event happened on " + value.strip() + "?")
    if parsed and parsed.get("datatype") == "type.datetime":
        return {"precision": "day", "value": str(parsed["value"]), "source_literal": value.strip(),
                "normalization": deepcopy(parsed["normalization"])}
    return None


def datetime_question_obligation(question: str) -> dict[str, Any] | None:
    """One equality date after 'on'; no hidden extra literal or comparator."""
    text = str(question)
    if re.search(r'["“”]|\b(?:not|never|or|either|except|before|after|prior|later|earlier|less|more|greater|lower|higher|during|until|since|between|about|approximately|most|least|first|last|latest|earliest)\b', text, re.I):
        return None
    cues = list(re.finditer(r"\bon\s+", text, re.I))
    if len(cues) != 1:
        return None
    cue = cues[0]
    tail = text[cue.end():]
    end = re.search(r"\s+and\b|[?;!]", tail, re.I)
    literal = tail[:end.start()] if end else tail
    spec = _literal(literal.strip().rstrip("."))
    if spec is None:
        return None
    spec["literal_span"] = [cue.end(), cue.end() + len(literal.rstrip())]
    outside = text[:spec["literal_span"][0]] + text[spec["literal_span"][1]:]
    if re.search(r"\d", outside):
        return None
    return spec


def datetime_owner_restore_necessary_gate(question: str, current: Any) -> dict[str, Any]:
    """Current-graph-only necessary gates; no original-context substitution."""
    spec = datetime_question_obligation(question)
    if spec is None:
        return _proof("question_not_one_explicit_event_day_or_era_year_equality")
    triples = getattr(current, "triples", None)
    if (not isinstance(triples, list) or len(triples) != 2 or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)
            or getattr(current, "operators", None) != []):
        return _proof("current_not_one_unconstrained_two_edge_path")
    first, last = [tuple(map(str, t)) for t in triples]
    if (_original_datetime_owner_restore_MID.fullmatch(first[0]) and _original_datetime_owner_restore_VAR.fullmatch(first[2]) and last == (first[2], first[1], str(current.answer_var))
            and _original_datetime_owner_restore_VAR.fullmatch(last[2]) and first[2] != last[2]):
        direction, anchor, owner = "forward", first[0], first[2]
    elif (_original_datetime_owner_restore_MID.fullmatch(first[2]) and _original_datetime_owner_restore_VAR.fullmatch(first[0]) and last == (str(current.answer_var), first[1], first[0])
            and _original_datetime_owner_restore_VAR.fullmatch(last[0]) and first[0] != last[0]):
        direction, anchor, owner = "backward", first[2], first[0]
    else:
        return _proof("current_second_edge_not_exact_repeated_primary_projection_drift")
    return {**_proof("necessary_current_graph_shape_only_original_context_still_required"),
            "necessary_gate": True, "obligation": spec, "primary_relation": first[1],
            "primary_direction": direction, "anchor_id": anchor, "original_owner_var": owner,
            "wrong_entity_projection": str(current.answer_var)}


def _grounding_tuple(grounding: Any):
    if isinstance(grounding, _original_datetime_owner_restore_Mapping):
        return grounding.get("compose_input"), grounding.get("relation_bindings"), grounding.get("anchor_bindings"), grounding.get("provenance", {})
    return (getattr(grounding, "compose_input", None), getattr(grounding, "relation_bindings", None),
            getattr(grounding, "anchor_bindings", None), getattr(grounding, "provenance", {}))


def _entity(value: Any) -> tuple[str, str]:
    if isinstance(value, _original_datetime_owner_restore_Mapping):
        return str(value.get("id", value.get("entity_id", ""))), str(value.get("label", value.get("name", "")))
    if isinstance(value, str):
        return value, ""
    return str(getattr(value, "entity_id", "")), str(getattr(value, "label", ""))


def compile_original_datetime_owner_restore(
    question: str, current: Any, original_semantic: _original_datetime_owner_restore_Mapping[str, Any], actual_groundings: _original_datetime_owner_restore_Sequence[Any],
    topic_bindings: _original_datetime_owner_restore_Mapping[str, Any], ontology: Any, *, graph_id: str = "original_datetime_owner_shadow",
    linked_entity_context: _original_datetime_owner_restore_Sequence[_original_datetime_owner_restore_Mapping[str, Any]] | None = None,
    datetime_schema: _original_datetime_owner_restore_Mapping[str, Any] | None = None,
    verify_temporal_datatype_in_query: bool = False,
) -> tuple[_original_datetime_owner_restore_QueryGraphCandidate | None, dict[str, Any]]:
    """Restore only a proved owner drift; signed years require real datatype."""
    proof = datetime_owner_restore_necessary_gate(question, current)
    if not proof.get("necessary_gate"):
        return None, proof

    def abstain(reason: str):
        proof.update(status="not_compiled", reason=reason)
        return None, proof

    provenance = getattr(current, "provenance", None)
    if not isinstance(provenance, _original_datetime_owner_restore_Mapping) or any(provenance.get(k) for k in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("current_not_one_question_owned_query")
    if provenance.get("original_question", question) != question:
        return abstain("current_question_contract_conflict")
    paths = original_semantic.get("semantic_paths", [])
    if ontology is None or len(paths) != 1 or not isinstance(paths[0], _original_datetime_owner_restore_Mapping):
        return abstain("original_not_one_complete_path")
    path = paths[0]
    steps = path.get("steps", [])
    if len(steps) != 2 or any(not isinstance(s, _original_datetime_owner_restore_Mapping) for s in steps):
        return abstain("original_not_exact_primary_plus_datetime_path")
    first, last = steps
    root = str(path.get("anchor_ref", ""))
    if (first.get("from") != root or first.get("to") != last.get("from") or last.get("to") != path.get("path_output_var")
            or last.get("direction") != "forward" or first.get("direction") != proof["primary_direction"]
            or _original_datetime_owner_restore_relation_id_from_label(first.get("relation_label")) != proof["primary_relation"]
            or first.get("id") == last.get("id") or root not in topic_bindings):
        return abstain("original_primary_variable_id_or_direction_does_not_match_current")
    anchor, label = _entity(topic_bindings[root])
    originals = [a for a in original_semantic.get("anchors", []) if isinstance(a, _original_datetime_owner_restore_Mapping) and str(a.get("id")) == root]
    actuals = provenance.get("anchor_bindings", {})
    actual_id, actual_label = _entity(actuals.get(root)) if isinstance(actuals, _original_datetime_owner_restore_Mapping) else ("", "")
    if (anchor != proof["anchor_id"] or actual_id != anchor or not label or label.casefold() != actual_label.casefold()
            or len(originals) != 1 or str(originals[0].get("surface", "")).casefold() != label.casefold()
            or not re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", question, re.I)):
        return abstain("original_and_actual_topic_id_label_question_identity_not_proven")
    literal = proof["obligation"]["source_literal"]
    for item in original_semantic.get("anchors", []):
        if not isinstance(item, _original_datetime_owner_restore_Mapping) or item.get("id") == root:
            continue
        if str(item.get("surface", "")).strip().casefold() != literal.casefold() or item.get("id") in topic_bindings:
            return abstain("extra_original_topic_or_unknown_kind_is_not_a_proven_scalar_literal")
    for item in linked_entity_context or []:
        name = str(item.get("name", item.get("label", "")))
        if (name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, re.I)
                and str(item.get("id", "")) != anchor):
            return abstain("another_explicit_linked_question_topic_not_in_original_path")

    original_field = _original_datetime_owner_restore_relation_id_from_label(last.get("relation_label"))
    fields = set()
    witnesses = []
    for grounding in actual_groundings:
        composed, relations, bindings, meta = _grounding_tuple(grounding)
        if not isinstance(composed, _original_datetime_owner_restore_Mapping) or not isinstance(relations, _original_datetime_owner_restore_Mapping) or not isinstance(bindings, _original_datetime_owner_restore_Mapping) or not isinstance(meta, _original_datetime_owner_restore_Mapping) or meta.get("dropped_paths"):
            continue
        actual_paths = composed.get("semantic_paths", [])
        if len(actual_paths) != 1 or not isinstance(actual_paths[0], _original_datetime_owner_restore_Mapping):
            continue
        actual_path = actual_paths[0]
        actual_steps = actual_path.get("steps", [])
        if (len(actual_steps) != 2 or any(not isinstance(s, _original_datetime_owner_restore_Mapping) for s in actual_steps)
                or any(actual_path.get(k) != path.get(k) for k in ("id", "anchor_ref", "path_output_var", "goal"))
                or any(any(a.get(k) != b.get(k) for k in ("id", "from", "to", "direction")) for a, b in zip(actual_steps, steps))):
            continue
        bound_anchor, _ = _entity(bindings.get(root))
        primary = str(relations.get(str(first["id"]), ""))
        field = str(relations.get(str(last["id"]), ""))
        if (bound_anchor != anchor or primary != proof["primary_relation"]
                or _original_datetime_owner_restore_relation_id_from_label(actual_steps[0].get("relation_label")) != primary
                or _original_datetime_owner_restore_relation_id_from_label(actual_steps[1].get("relation_label")) != field
                or field.split(".", 1)[0] != original_field.split(".", 1)[0]
                or field.rsplit(".", 1)[-1] != original_field.rsplit(".", 1)[-1]
                or str(ontology.range_for_relation(field)) != "type.datetime"):
            continue
        if ontology.domain_for_relation(original_field) and field != original_field:
            continue
        fields.add(field)
        witnesses.append({"actual_relation_bindings": dict(relations), "complete_path_id": path["id"],
                          "actual_grounding_provenance": deepcopy(dict(meta)),
                          "original_and_grounded_step_ids_variables_directions_identical": True})
    if len(fields) != 1:
        return abstain("actual_complete_grounding_does_not_uniquely_prove_original_datetime_rid")
    field = next(iter(fields))
    primary = proof["primary_relation"]
    role_type = str(ontology.range_for_relation(primary) if first["direction"] == "forward" else ontology.domain_for_relation(primary))
    domain = str(ontology.domain_for_relation(field))
    if (not role_type or not domain or "common.topic" not in set(ontology.supertypes(role_type))
            or "common.topic" not in set(ontology.supertypes(domain))
            or not (role_type in set(ontology.supertypes(domain)) or domain in set(ontology.supertypes(role_type)))):
        return abstain("actual_primary_entity_and_datetime_owner_not_ontology_compatible")
    role, event = _leaf(primary), _leaf(field)
    before = question[:proof["obligation"]["literal_span"][0]]
    event_match = re.search(r"\b([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+on\s*$", before, re.I)
    if event_match is None or not event <= _original_datetime_owner_restore_words(event_match[1]):
        return abstain("question_event_does_not_match_the_actual_datetime_field")
    event_prefix = re.sub(r"\s+on\s*$", "", before, flags=re.I)
    event_tokens = list(re.finditer(r"[A-Za-z]+", event_prefix))
    event_start = None
    for size in range(1, min(4, len(event_tokens)) + 1):
        suffix = event_prefix[event_tokens[-size].start():]
        if _original_datetime_owner_restore_words(suffix) - {"is", "was", "were", "are", "has", "have", "had", "been"} == event:
            event_start = event_tokens[-size].start()
            break
    head = event_prefix[:event_start].strip() if event_start is not None else ""
    head = re.sub(r"\s+(?:is|was|were|are|has|have|had|been)\s*$", "", head, flags=re.I)
    if not re.fullmatch(r"(?:who|(?:what|which)\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,3})", head, re.I):
        return abstain("question_does_not_request_an_entity_event_owner")
    after = question[proof["obligation"]["literal_span"][1]:].strip().rstrip("? .!")
    tail = re.fullmatch(r"and\s+(?:was|is)\s+(?:a|an|the)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+of\s+(.+)", after, re.I)
    if tail is None or _original_datetime_owner_restore_words(tail[1]) != role:
        return abstain("question_has_extra_scope_or_primary_role_attachment_not_exact")
    topic_text = tail[2].strip()
    # Titles are not alternate topic identities; require the linked name as
    # the complete trailing identity and allow at most one preceding word.
    if not re.fullmatch(r"(?:[A-Za-z]+\s+)?" + re.escape(label), topic_text, re.I):
        return abstain("question_topic_tail_has_another_unaccounted_entity_or_constraint")
    goal = str(path.get("goal", ""))
    decompositions = provenance.get("decomposition", [])
    independent = [text for text in decompositions if isinstance(text, str) and text.strip() == goal.strip()]
    goal_values = re.findall(r"\b(?:is|equals?|equal\s+to)\s+(.+?)(?=[?;!.]|$)", goal, re.I)
    goal_specs = [_literal(text.strip()) for text in goal_values]
    matching = [s for s in goal_specs if s and s.get("precision") == proof["obligation"]["precision"]
                and s.get("year", s.get("value")) == proof["obligation"].get("year", proof["obligation"].get("value"))]
    if not independent or len(matching) != 1 or not role <= _original_datetime_owner_restore_words(goal) or not event <= _original_datetime_owner_restore_words(goal):
        return abstain("original_goal_and_actual_decomposition_do_not_attest_same_role_field_equal_literal")
    proof.update(original_path_id=path["id"], actual_grounding_witnesses=witnesses,
                 datetime_relation_id=field, primary_entity_type=role_type, datetime_owner_type=domain,
                 original_datetime_label=deepcopy(last["relation_label"]), actual_topic_id=anchor, actual_topic_label=label,
                 original_owner_var=proof["original_owner_var"], original_question_and_decomposition_match=True,
                 wh_entity_subject_supported_by_explicit_primary_relationship_conjunct=True,
                 no_wh_kind_synonym_or_entity_concept_routing=True,
                 contradiction="repeated_primary_entity_hop_replaced_original_bound_datetime_and_owner_projection")

    spec = proof["obligation"]
    if spec["precision"] == "year" and spec["year"] < 0:
        if datetime_schema is None and verify_temporal_datatype_in_query is True:
            proof["datatype_validation_deferred_to_same_query"] = True
            proof["requires_nonempty_typed_literal_validated_query_result"] = True
        elif (not isinstance(datetime_schema, _original_datetime_owner_restore_Mapping) or datetime_schema.get("error")
              or datetime_schema.get("scalar_relation_id") != field or datetime_schema.get("primary_relation_id") != primary
              or datetime_schema.get("root_anchor_id") != anchor):
            return abstain("bce_year_requires_actual_same_path_datetime_binding_schema")
        else:
            raw = datetime_schema.get("raw_sparql_json", {})
            bindings = raw.get("results", {}).get("bindings", []) if isinstance(raw, _original_datetime_owner_restore_Mapping) else []
            if len(bindings) != 1 or not isinstance(bindings[0], _original_datetime_owner_restore_Mapping):
                return abstain("bce_datetime_binding_not_unique_complete_observed_literal")
            term = bindings[0].get("V1", {})
            value = str(term.get("value", "")) if isinstance(term, _original_datetime_owner_restore_Mapping) else ""
            datatype = str(term.get("datatype", "")) if isinstance(term, _original_datetime_owner_restore_Mapping) else ""
            shape = r"^[+-]?[0-9]+(?:Z|[+-][0-9]{2}:[0-9]{2})?$" if datatype == _XSD + "gYear" else _YEAR_SHAPE
            if (term.get("type") not in {"literal", "typed-literal"} or datatype not in _TEMPORAL_DATATYPES
                    or term.get("xml:lang") or not re.fullmatch(shape, value)):
                return abstain("bce_observed_term_not_proven_temporal_typed_signed_year_shape")
            observed_year = int(re.match(r"^[+-]?\d+", value)[0])
            if observed_year != spec["year"]:
                return abstain("bce_observed_signed_year_does_not_match_question_without_offset")
            proof["actual_datetime_binding_schema"] = {"datatype": datatype, "value": value,
                                                        "observed_signed_year": observed_year,
                                                        "no_padding_or_year_zero_offset_assumed": True}
    owner, scalar = proof["original_owner_var"], str(current.answer_var)
    first_triple = list(map(str, current.triples[0]))
    op = {"type": "EQUAL", "inputs": [], "input_var": scalar, "attribute_relation_id": "",
          "attribute_relation_label": [], "attribute_relation_labels": [], "_attribute_value_already_bound": True,
          "value_type": "dateTime", "_comparison_date_precision": spec["precision"],
          "_verify_temporal_datatype_in_query": True}
    if spec["precision"] == "year":
        op.update(value=str(spec["year"]), _original_datetime_signed_year_equality=True)
    else:
        op.update(value=spec["value"], _calendar_day_boundary=True)
    candidate = _original_datetime_owner_restore_QueryGraphCandidate(graph_id, [first_triple, [owner, field, scalar]], owner, [op],
                                    float(getattr(current, "score", 0.0)), None, deepcopy(dict(provenance)))
    # The restored query must not claim the wrong repeated-primary grounding
    # as its own. Preserve that original state in the repair proof and attach
    # the actual complete datetime grounding that authorized the new query.
    proof["original_source_relation_bindings"] = deepcopy(provenance.get("relation_bindings", {}))
    proof["original_source_grounded_semantic"] = deepcopy(provenance.get("grounded_semantic", {}))
    candidate.provenance["relation_bindings"] = deepcopy(witnesses[0]["actual_relation_bindings"])
    candidate.provenance["grounded_semantic"] = deepcopy(witnesses[0]["actual_grounding_provenance"])
    proof.update(status="compiled_not_executed", reason="complete_original_grounded_datetime_owner_restored",
                 source_graph_id=current.graph_id, compiled_graph_id=graph_id, restored_answer_var=owner,
                 removes_only_proven_wrong_repeated_primary_edge=True, preserves_original_complete_path=True,
                 preserved_existing_operators=0, graph_role_change_is_not_subset_evidence=True)
    candidate.provenance["original_datetime_owner_restore"] = deepcopy(proof)
    return candidate, proof


def lower_original_datetime_owner_restore(candidate: _original_datetime_owner_restore_QueryGraphCandidate, *, limit: int = 256) -> str:
    """Lower just the isolated signed-year equality, never alter global code."""
    signed = [op for op in candidate.operators if op.get("_original_datetime_signed_year_equality")]
    if not signed:
        query = lower_sparql(candidate, limit=limit)
        exact = [op for op in candidate.operators if op.get("_verify_temporal_datatype_in_query")]
        if len(exact) != 1:
            raise ValueError("Isolated calendar day requires one explicit datatype-verified operator")
        variable = str(exact[0].get("input_var", ""))
        if not _original_datetime_owner_restore_VAR.fullmatch(variable):
            raise ValueError("Calendar day operand invalid")
        guard = f'  FILTER(isLiteral(?{variable}) && LANG(?{variable}) = "")\n'
        guard += f'  FILTER(DATATYPE(?{variable}) IN (<{_XSD}date>, <{_XSD}dateTime>))\n'
        return query.replace("\n}\n", "\n" + guard + "}\n", 1)
    if len(candidate.operators) != 1 or len(signed) != 1:
        raise ValueError("Isolated signed-year lowering requires one and only one operator")
    op = signed[0]
    value = str(op.get("value", ""))
    variable = str(op.get("input_var", ""))
    if not re.fullmatch(r"-?[1-9]\d{0,3}", value) or not _original_datetime_owner_restore_VAR.fullmatch(variable) or op.get("type") != "EQUAL":
        raise ValueError("Signed-year equality contract invalid")
    base = deepcopy(candidate)
    base.operators = []
    query = lower_sparql(base, limit=limit)
    filter_text = (f'  FILTER(isLiteral(?{variable}) && LANG(?{variable}) = "")\n'
                   f'  FILTER(DATATYPE(?{variable}) IN (<{_XSD}gYear>, <{_XSD}date>, <{_XSD}dateTime>))\n'
                   f'  FILTER(REGEX(STR(?{variable}), "{_YEAR_SHAPE}"))\n'
                   f'  FILTER(<{_XSD}integer>(REPLACE(STR(?{variable}), "^([+-]?[0-9]+).*$", "$1")) = {value})\n')
    marker = "\n}\n"
    if marker not in query:
        raise ValueError("Simple source lowering does not contain one outer WHERE terminator")
    return query.replace(marker, "\n" + filter_text + "}\n", 1)


def compile_datetime_owner_restore_from_current_stages(
    question: str, current: Any, stages: _original_datetime_owner_restore_Sequence[_original_datetime_owner_restore_Mapping[str, Any]], ontology: Any, *,
    graph_id: str = "original_datetime_owner_shadow", datetime_schema: _original_datetime_owner_restore_Mapping[str, Any] | None = None,
    verify_temporal_datatype_in_query: bool = False,
) -> tuple[_original_datetime_owner_restore_QueryGraphCandidate | None, dict[str, Any]]:
    """Resolve only the actual originating ordered Semantic/Grounding batch.

    ``stages`` is in-memory current-flow stage data supplied by the caller,
    not file/trace access in this module. A source grounding with the current
    semantic index/rank, relation IDs and topic IDs must actually exist in
    that ordered batch before another grounding in the batch can be used.
    """
    meta = getattr(current, "provenance", {}).get("grounded_semantic", {})
    if (not isinstance(meta, _original_datetime_owner_restore_Mapping) or not isinstance(meta.get("semantic_graph_index"), int)
            or isinstance(meta.get("semantic_graph_index"), bool)):
        return None, _proof("current_originating_semantic_index_missing")
    index = meta["semantic_graph_index"]
    current_relations = getattr(current, "provenance", {}).get("relation_bindings", {})
    topics = getattr(current, "provenance", {}).get("anchor_bindings", {})
    if (not isinstance(topics, _original_datetime_owner_restore_Mapping) or not topics
            or not isinstance(current_relations, _original_datetime_owner_restore_Mapping) or not current_relations):
        return None, _proof("current_actual_topic_or_relation_bindings_missing")
    pending, context = None, []
    candidates = []
    for stage in stages:
        if not isinstance(stage, _original_datetime_owner_restore_Mapping):
            continue
        if stage.get("stage") == "linked_entity_context" and isinstance(stage.get("output"), list):
            context = stage["output"]
        elif stage.get("stage") == "semantic_path_generation":
            output = stage.get("output", [])
            pending = [s["output"] for s in output if isinstance(s, _original_datetime_owner_restore_Mapping) and s.get("valid") is True and isinstance(s.get("output"), _original_datetime_owner_restore_Mapping)] if isinstance(output, list) else None
        elif stage.get("stage") == "grounded_semantic_candidates" and pending is not None:
            output = stage.get("output", [])
            if not isinstance(output, list) or not 0 <= index < len(pending):
                continue
            actual = [g for g in output if isinstance(g, _original_datetime_owner_restore_Mapping) and g.get("provenance", {}).get("semantic_graph_index") == index]
            origins = [g for g in actual if g.get("relation_bindings") == current_relations
                       and g.get("provenance", {}).get("rank") == meta.get("rank")
                       and isinstance(g.get("anchor_bindings"), _original_datetime_owner_restore_Mapping)
                       and set(g["anchor_bindings"]) == set(topics)
                       and all(_entity(v)[0] and _entity(topics[k])[0] == _entity(v)[0]
                               for k, v in g["anchor_bindings"].items())]
            if len(origins) != 1:
                continue
            candidate, proof = compile_original_datetime_owner_restore(question, current, pending[index], actual, topics,
                                                                        ontology, graph_id=graph_id,
                                                                        linked_entity_context=context,
                                                                        datetime_schema=datetime_schema,
                                                                        verify_temporal_datatype_in_query=verify_temporal_datatype_in_query)
            candidates.append((candidate, proof))
    positives = [(c, p) for c, p in candidates if c is not None]
    if not positives:
        return candidates[0] if len(candidates) == 1 else (None, _proof("actual_originating_semantic_grounding_batch_not_uniquely_proven"))
    keys = {(tuple(tuple(t) for t in c.triples), c.answer_var, str(c.operators)) for c, _ in positives}
    if len(keys) != 1 or len(candidates) != 1:
        return None, _proof("multiple_current_originating_batches_or_restoration_programs_are_ambiguous")
    candidate, proof = positives[0]
    proof["actual_ordered_originating_batch_and_source_grounding_verified"] = True
    candidate.provenance["original_datetime_owner_restore"] = deepcopy(proof)
    return candidate, proof



__all__ = ['RelativeTemporalBinder', '_numeric_temporal_slot_normalization_lexical_tokens', 'build_fixed_slot_variants', 'choose_fixed_slot_prediction', 'compile_bound_calendar_date_equality_append', 'compile_bound_year_equality', 'compile_budgeted_literal_constraint_repair', 'compile_datetime_owner_restore_from_current_stages', 'compile_original_datetime_owner_restore', 'datetime_owner_restore_necessary_gate', 'datetime_question_obligation', 'is_fixed_slot_variant', 'lower_original_datetime_owner_restore', 'mask_anchor_labels', 'normalize_explicit_spec', 'partition_fixed_slot_executions', 'postselect_fixed_slot_execution', 'prepare_fixed_execution_slots', 'relative_reference', 'semantic_projection_items']
