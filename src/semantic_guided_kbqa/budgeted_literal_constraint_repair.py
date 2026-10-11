"""Compile a proven literal/operator correction after the model stages.

No relation, node, answer projection or other operator is added or removed.
This module never executes a query or selects answers. The caller must spend
an exact-query saved request AND remaining measured time credit, enforce the
bounded endpoint timeout, and validate the full returned denotation. Singleton
sources are protected by default: a semantic correction is not measured F1
improvement, and can change an already-perfect answer.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any

from .comparison_number_normalization import normalize_comparison_constraint_spec, parse_english_calendar_date
from .constraint_execution_safety import _decomposition_items, _semantic_views
from .contracts import QueryGraphCandidate
from .nested_name_schema_guard import _complete
from .ontology import relation_id_from_label


_COMPARISONS = frozenset({"EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"})
_VAR = re.compile(r"V\d+\Z")
_ENTITY = re.compile(r"(?:[mg]\.[A-Za-z0-9_]+|http://rdf\.freebase\.com/ns/[mg]\.[A-Za-z0-9_]+)\Z")
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


def _spec(question: str) -> dict[str, Any] | None:
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
        value = parse_english_calendar_date(match[1])
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


def _value_matches(operator: Mapping, spec: Mapping) -> bool:
    if spec["value_type"] == "date":
        return str(operator.get("value", "")) == spec["value"]
    value = _number(operator.get("value"))
    return value is not None and value == _number(spec["value"])


def _semantic_field_certificate(question: str, source: Any, field: tuple[str, str, str],
                                spec: Mapping, semantics: Any, decompositions: Any) -> dict[str, Any] | None:
    """Embed the original full path uniquely into actual BGP and Compose.

    All source BGP edges, not merely the date edge, must be accounted for by
    the one certificate. This rejects unrequested containment/country/area
    joins instead of dropping them to create a convenient owner.
    """
    graph = source.graph
    provenance = getattr(graph, "provenance", None)
    compose = getattr(graph, "compose_output", None)
    if not isinstance(provenance, Mapping) or not isinstance(compose, Mapping):
        return None
    if str(compose.get("answer_var")) != str(graph.answer_var):
        return None
    anchors = provenance.get("anchor_bindings")
    if not isinstance(anchors, Mapping):
        return None
    triples = {tuple(map(str, t)) for t in graph.triples}
    items = _decomposition_items(decompositions)
    certificates = set()
    for semantic in _semantic_views(semantics):
        for path in semantic["semantic_paths"]:
            if not isinstance(path, Mapping) or not isinstance(path.get("steps"), list):
                continue
            goal = " ".join(str(path.get("goal", "")).split()).casefold()
            if goal not in items:
                continue
            condition = _spec(goal)
            if condition is None or any(condition[k] != spec[k] for k in ("type", "value", "value_type")):
                continue
            anchor_id = str(path.get("anchor_ref", ""))
            linked = anchors.get(anchor_id)
            original = [a for a in semantic.get("anchors", []) if isinstance(a, Mapping) and str(a.get("id")) == anchor_id]
            if not isinstance(linked, Mapping) or len(original) != 1:
                continue
            surface = str(original[0].get("surface", ""))
            if not surface or surface.casefold() != str(linked.get("label", "")).casefold() or surface.casefold() not in question.casefold():
                continue
            entities = [e for e in compose.get("entities", []) if isinstance(e, Mapping)
                        and str(e.get("surface", "")).casefold() == surface.casefold()]
            if len(entities) != 1:
                continue
            substitutions = {str(entities[0].get("id")): str(linked.get("id"))}
            composed = {(substitutions.get(str(t.get("subject")), str(t.get("subject"))),
                         relation_id_from_label(t.get("relation_label")),
                         substitutions.get(str(t.get("object")), str(t.get("object"))))
                        for t in compose.get("triples", []) if isinstance(t, Mapping)}
            current = str(linked.get("id"))
            consumed = set()
            mapped = {}
            valid = True
            for step in path["steps"]:
                if not isinstance(step, Mapping):
                    valid = False
                    break
                predicate = relation_id_from_label(step.get("relation_label"))
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
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
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
    if not _complete(source) or not all(_ENTITY.fullmatch(str(x)) for x in source.answer_ids):
        return None, _evidence("source_full_entity_execution_not_proven")
    if len(source.answer_ids) <= 1 and not allow_singleton_compile:
        return None, _evidence("singleton_incumbent_protected_by_default")
    if len(source.answer_ids) >= 256:
        return None, _evidence("source_result_near_or_above_repair_row_budget")
    graph = source.graph
    if (not isinstance(graph.triples, list) or not graph.triples
            or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in graph.triples)
            or not isinstance(graph.operators, list) or any(not isinstance(x, Mapping) for x in graph.operators)):
        return None, _evidence("source_graph_contract_missing")
    provenance = getattr(graph, "provenance", None)
    if not isinstance(provenance, Mapping) or any(provenance.get(k) for k in (
            "selection_answer_union", "selection_answer_intersection")):
        return None, _evidence("source_is_postprocessed_or_provenance_unknown")
    scalar = [(index, op) for index, op in enumerate(graph.operators) if str(op.get("type", "")).upper() != "NO_EQUAL"]
    if len(scalar) != 1 or str(scalar[0][1].get("type", "")).upper() not in _COMPARISONS:
        return None, _evidence("not_one_explicit_comparison_no_contains_or_extrema")
    spec = _spec(question)
    if spec is None:
        return None, _evidence("not_one_unquoted_full_on_date_or_exact_word_number")
    index, original_operator = scalar[0]
    if not _value_matches(original_operator, spec) or original_operator.get("inputs") not in (None, []):
        return None, _evidence("source_threshold_not_identical_or_has_other_inputs")
    attribute = str(original_operator.get("attribute_relation_id") or relation_id_from_label(original_operator.get("attribute_relation_label")))
    input_var = str(original_operator.get("input_var", ""))
    if not _VAR.fullmatch(input_var):
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
