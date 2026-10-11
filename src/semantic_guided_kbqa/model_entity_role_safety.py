"""Zero-call final protection of a captured model-selected entity answer.

This only reverses a proven local rule-score entity-to-primitive override. It
does not choose a new model/candidate, rebind constraints, or assert temporal
fidelity. Positive head vocabulary is generated from the loaded ontology.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

from .evaluation import canonicalize_answer_id
from .nested_name_schema_guard import (
    _anchors, _complete, _entity_answers, _schema_status, _singular,
    _substantive_operators, _type_index,
)


_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_HEAD = re.compile(r"^\s*(?:what|which)\s+([A-Za-z]+)(.*)\Z", re.I | re.S)
_BOUNDARY = re.compile(
    r"^(?:that|which|who|whose|with|without|where|from|in|of|by|for|at|on|having|"
    r"is|was|are|were|did|does|do|has|have|had|can|could|will|would|should)\b", re.I)
# These are generic scalar-question grammar exclusions, not positive entity
# head vocabulary; every accepted entity head must come from the ontology.
_SCALAR_HEAD = re.compile(r"(?:date|time|year|month|day|hour|minute|second|number)\Z", re.I)
_PRIMITIVE_TYPES = frozenset({"type.int", "type.float", "type.datetime", "type.text", "type.boolean", "type.rawstring", "type.uri"})


def model_entity_head(question: str) -> str:
    """Parse only one immediate noun, without reducing compound/quoted heads."""
    match = _HEAD.match(str(question))
    if not match:
        return ""
    head, tail = _singular(match.group(1)), match.group(2)
    if _SCALAR_HEAD.fullmatch(head):
        return ""
    # No apostrophe/hyphen/digit or adjacent word can be part of a simple head.
    if tail and not tail[0].isspace() and tail[0] not in "?.!":
        return ""
    rest = tail.strip()
    if rest and not re.fullmatch(r"[?.!]+", rest) and not _BOUNDARY.match(rest):
        return ""
    return head


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _graph_state(graph: Any) -> str:
    triples = getattr(graph, "triples", None)
    operators = getattr(graph, "operators", None)
    if not isinstance(triples, list) or not triples or not isinstance(operators, list):
        raise ValueError("Missing graph state")
    if any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples):
        raise ValueError("Malformed graph triple")
    if any(not isinstance(o, dict) for o in operators):
        raise ValueError("Malformed graph operator")
    return _stable({"triples": sorted(tuple(map(str, t)) for t in triples),
                    "answer_var": str(graph.answer_var), "operators": operators})


def _captured_complete(candidate: Mapping[str, Any]) -> bool:
    values = candidate.get("answer_ids")
    count = candidate.get("answer_count")
    return (isinstance(values, list) and bool(values)
            and isinstance(count, int) and not isinstance(count, bool)
            and count == len(values) and not candidate.get("answers_truncated", True)
            and len(set(map(str, values))) == len(values))


def _candidate_graph(candidate: Mapping[str, Any]) -> Any:
    # No contracts/pipeline imports, and no mutation of the captured mapping.
    from types import SimpleNamespace
    return SimpleNamespace(triples=candidate.get("triples"),
                           answer_var=candidate.get("answer_var", ""),
                           operators=candidate.get("operators"))


def _primitive_schema(graph: Any, ontology: Any) -> tuple[bool, dict[str, Any]]:
    declared: set[str] = set()
    unknown = False
    incident_count = 0
    for subject, relation, object_ in graph.triples:
        for term, method in ((subject, "domain_for_relation"), (object_, "range_for_relation")):
            if str(term) != str(graph.answer_var):
                continue
            incident_count += 1
            function = getattr(ontology, method, None)
            value = str(function(str(relation)) or "") if callable(function) else ""
            declared.add(value)
            if value not in _PRIMITIVE_TYPES:
                unknown = True
    evidence = {"declared_types": sorted(declared), "incident_schema_count": incident_count,
                "has_unknown_generic_or_nonprimitive_type": unknown}
    return bool(incident_count and declared and not unknown), evidence


def model_entity_role_challenger(
    question: str, selected: Any, executions: list[Any], ontology: Any,
    prior_selection: Mapping[str, Any] | None,
) -> tuple[Any | None, dict[str, Any]]:
    """Return the one existing captured model execution, or abstain.

    The captured selection must explicitly record model_selected_graph_id and
    local_rule_score_guard. Final rule graph, complete answers, anchors, context
    and substantive operators are checked against captured/current state. No
    score, label, Gold, model, endpoint, or embedding access is used.
    """
    head = model_entity_head(question)
    evidence: dict[str, Any] = {
        "status": "not_applied", "source_graph_id": str(selected.graph.graph_id),
        "head": head, "uses_existing_executions_only": True,
        "vocabulary_source": "loaded_ontology_generated_type_index",
        "additional_model_calls": 0, "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0, "uses_gold_answers": False,
        "protects_answer_role_only": True, "proves_comparison_semantics": False,
    }

    def abstain(reason: str, unqualified: bool = False):
        evidence["reason"] = reason
        if unqualified:
            evidence["status"] = "unqualified"
        return None, evidence

    if not head:
        return abstain("not_strict_single_word_entity_head")
    if not isinstance(prior_selection, Mapping):
        return abstain("captured_prior_selection_missing")
    reasons = prior_selection.get("reason_codes")
    if not isinstance(reasons, (list, tuple)) or "local_rule_score_guard" not in reasons:
        return abstain("not_proven_local_rule_score_override")
    model_id = str(prior_selection.get("model_selected_graph_id") or "")
    rule_id = str(prior_selection.get("selected_graph_id") or "")
    evidence.update(model_selected_graph_id=model_id, captured_rule_graph_id=rule_id)
    if not model_id or not rule_id or model_id == rule_id:
        return abstain("explicit_distinct_model_choice_missing")
    if evidence["source_graph_id"] != rule_id:
        return abstain("late_selected_graph_differs_from_captured_rule_graph")
    if prior_selection.get("answer_union_graph_ids"):
        return abstain("captured_union_not_single_rule_execution")
    try:
        expected = set(_type_index(ontology).get(head, ()))
        evidence["expected_types"] = sorted(expected)
        if not expected:
            return abstain("head_not_in_concrete_ontology_vocabulary")
        if not _complete(selected):
            return abstain("source_denotation_incomplete_or_unproven", True)
        source_ids = set(canonicalize_answer_id(v) for v in selected.answer_ids)
        if not source_ids or not all(source_ids) or any(_ENTITY.fullmatch(v) for v in source_ids):
            return abstain("source_not_pure_primitive_answers")
        source_state = _graph_state(selected.graph)
        primitive, source_schema = _primitive_schema(selected.graph, ontology)
        evidence["source_schema"] = source_schema
        if not primitive:
            return abstain("source_incident_schema_not_known_primitive")
        candidates = prior_selection.get("candidates")
        if not isinstance(candidates, list):
            return abstain("captured_candidate_state_missing", True)
        captured_rule = [c for c in candidates if isinstance(c, Mapping) and str(c.get("graph_id")) == rule_id]
        captured_model = [c for c in candidates if isinstance(c, Mapping) and str(c.get("graph_id")) == model_id]
        if len(captured_rule) != 1 or len(captured_model) != 1:
            return abstain("captured_choice_identity_ambiguous_or_missing", True)
        rule, model = captured_rule[0], captured_model[0]
        if not _captured_complete(rule) or not _captured_complete(model):
            return abstain("captured_choice_denotation_incomplete_or_unproven", True)
        if (_graph_state(_candidate_graph(rule)) != source_state
                or {canonicalize_answer_id(v) for v in rule["answer_ids"]} != source_ids):
            return abstain("late_source_state_differs_from_captured_rule_execution")
        model_ids = {canonicalize_answer_id(v) for v in model["answer_ids"]}
        if not all(_ENTITY.fullmatch(v) for v in model_ids):
            return abstain("captured_model_not_pure_entity_answers")
        matching = [e for e in executions if str(e.graph.graph_id) == model_id]
        if not matching:
            return abstain("captured_model_execution_missing", True)
        model_state = _graph_state(_candidate_graph(model))
        states = set()
        for execution in matching:
            ids = _entity_answers(execution)
            if ids is None or not _complete(execution):
                return abstain("model_execution_incomplete_or_wrong_kind", True)
            states.add((_graph_state(execution.graph), tuple(sorted(ids))))
        if len(states) != 1 or states != {(model_state, tuple(sorted(model_ids)))}:
            return abstain("model_execution_denotation_or_graph_ambiguous", True)
        replacement = matching[0]
        schema_status, schema = _schema_status(replacement.graph, expected, ontology)
        evidence["model_schema"] = schema
        if schema_status != "compatible":
            return abstain("model_schema_unknown_generic_or_wrong_head")
        source_anchors = _anchors(selected.graph)
        if not source_anchors or not source_anchors <= _anchors(replacement.graph):
            return abstain("source_anchor_not_preserved")
        source_operators = _substantive_operators(selected.graph)
        model_operators = _substantive_operators(replacement.graph)
        if source_operators is None or model_operators is None or not source_operators <= model_operators:
            return abstain("substantive_source_constraint_not_preserved")
        # The removed primitive projection cannot itself carry an operator:
        # equal variable spelling in the entity graph is not role-mapping proof.
        for operator in selected.graph.operators:
            refs = [str(operator.get("input_var", "")), *(str(v) for v in operator.get("inputs", []))]
            if operator.get("value_type") == "variable":
                refs.append(str(operator.get("value", "")))
            if str(selected.graph.answer_var) in refs:
                return abstain("source_constraint_touches_removed_scalar_projection")
        incident = [tuple(map(str, t)) for t in selected.graph.triples
                    if str(t[0]) == str(selected.graph.answer_var) or str(t[2]) == str(selected.graph.answer_var)]
        if len(incident) != 1:
            return abstain("source_scalar_projection_edge_not_unique")
        context = {tuple(map(str, t)) for t in selected.graph.triples} - set(incident)
        if not context <= {tuple(map(str, t)) for t in replacement.graph.triples}:
            return abstain("substantive_source_context_not_preserved")
        evidence.update(status="applied", reason="preserve_captured_model_entity_answer_role",
                        selected_graph_id=model_id, selected_answer_count=len(model_ids),
                        compatible_model_denotation_count=1,
                        preserved_source_anchor_count=len(source_anchors),
                        preserved_source_context_edge_count=len(context),
                        preserved_substantive_operator_count=len(source_operators))
        return replacement, evidence
    except Exception as exc:
        evidence["error"] = str(exc)
        return abstain("schema_or_captured_state_error", True)
