"""Undo a proven single-measurement scope conflict in an existing union.

Gold-blind, final-only, and restricted to an explicitly supported, more
specific owner property. All model requests and endpoint executions have
already finished; this module never generates or queries candidates.
"""
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .contracts import ExecutedGraph, QueryGraphCandidate


def _number(value: Any) -> Decimal | None:
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _single_numeric_operator(graph: QueryGraphCandidate) -> dict | None:
    substantive = [operator for operator in graph.operators
                   if isinstance(operator, dict)
                   and str(operator.get("type", "")).upper() != "NO_EQUAL"]
    if len(substantive) != 1:
        return None
    operator = substantive[0]
    if (str(operator.get("type", "")).upper() != "EQUAL"
            or str(operator.get("value_type", "")).casefold() != "number"
            or not operator.get("attribute_relation_id")
            or _number(operator.get("value")) is None):
        return None
    return operator


def _owner_property(graph: QueryGraphCandidate, operator: dict, ontology: Any) -> str:
    leaf = str(operator["attribute_relation_id"])
    if ontology.range_for_relation(leaf) not in {"type.int", "type.float"}:
        return ""
    leaf_domain = str(ontology.domain_for_relation(leaf))
    if not leaf_domain:
        return ""
    properties = {
        str(relation) for subject, relation, object_ in graph.triples
        if str(subject) == str(graph.answer_var)
        and str(object_) == str(operator.get("input_var", ""))
        and str(ontology.range_for_relation(str(relation))) == leaf_domain
    }
    return next(iter(properties)) if len(properties) == 1 else ""


def scalar_owner_union_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
    ontology: Any,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Use full existing executions; serialized previews are not executions."""
    evidence = {"status": "not_applied", "source_graph_id": selected.graph.graph_id,
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_evaluation_gold": False}
    union = selected.graph.provenance.get("selection_answer_union")
    if not isinstance(union, dict) or ontology is None:
        evidence["reason"] = "no_existing_union_or_ontology"
        return None, evidence
    by_id = {execution.graph.graph_id: execution for execution in executions}
    primary = by_id.get(str(union.get("primary_graph_id", "")))
    secondary = by_id.get(str(union.get("secondary_graph_id", "")))
    if primary is None or secondary is None or primary is secondary:
        evidence["reason"] = "union_sources_not_materialized"
        return None, evidence
    if (selected.graph.triples != primary.graph.triples
            or selected.graph.operators != primary.graph.operators
            or selected.graph.answer_var != primary.graph.answer_var):
        evidence["reason"] = "union_graph_changed_by_late_repair"
        return None, evidence
    primary_ids, secondary_ids = set(primary.answer_ids), set(secondary.answer_ids)
    if (not primary_ids or not secondary_ids or primary_ids & secondary_ids
            or set(selected.answer_ids) != primary_ids | secondary_ids):
        evidence["reason"] = "not_exact_disjoint_union"
        return None, evidence
    p_operator, s_operator = (_single_numeric_operator(item.graph)
                              for item in (primary, secondary))
    if p_operator is None or s_operator is None:
        evidence["reason"] = "not_single_numeric_equality"
        return None, evidence
    if (p_operator["attribute_relation_id"] != s_operator["attribute_relation_id"]
            or _number(p_operator["value"]) != _number(s_operator["value"])):
        evidence["reason"] = "different_scalar_semantics"
        return None, evidence
    text = re.sub(r'["“][^"”]*["”]', " ", str(question))
    literals = re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w.])", text)
    if (len(literals) != 1 or _number(literals[0]) != _number(p_operator["value"])
            or re.search(r"\b(?:and|or|both|either)\b", text, re.I)):
        evidence["reason"] = "not_unambiguous_single_measurement_request"
        return None, evidence
    p_property = _owner_property(primary.graph, p_operator, ontology)
    s_property = _owner_property(secondary.graph, s_operator, ontology)
    if not p_property or not s_property or p_property == s_property:
        evidence["reason"] = "owner_property_unknown_or_same"
        return None, evidence
    p_domain = ontology.domain_for_relation(p_property)
    if not p_domain or p_domain != ontology.domain_for_relation(s_property):
        evidence["reason"] = "different_owner_types"
        return None, evidence
    p_words = set(re.findall(r"[A-Za-z]+", p_property.rsplit(".", 1)[-1].casefold()))
    s_words = set(re.findall(r"[A-Za-z]+", s_property.rsplit(".", 1)[-1].casefold()))
    question_words = set(re.findall(r"[A-Za-z]+", text.casefold()))
    if not s_words or not s_words < p_words or not p_words <= question_words:
        evidence["reason"] = "primary_specificity_not_explicitly_supported"
        return None, evidence
    # Both interpretations must use the same supplied entity anchors; this
    # guard does not decide between competing entity-linking hypotheses.
    def anchors(graph):
        return {str(term) for triple in graph.triples for term in (triple[0], triple[2])
                if str(term).startswith(("m.", "g.", "http://rdf.freebase.com/ns/m.",
                                         "http://rdf.freebase.com/ns/g."))}
    p_anchors = anchors(primary.graph)
    if not p_anchors or p_anchors != anchors(secondary.graph):
        evidence["reason"] = "different_entity_anchors"
        return None, evidence
    evidence.update({"status": "applied", "reason": "explicit_more_specific_measurement_scope",
                     "selected_graph_id": primary.graph.graph_id,
                     "discarded_union_graph_id": secondary.graph.graph_id,
                     "primary_owner_property": p_property, "secondary_owner_property": s_property,
                     "scalar_leaf_property": p_operator["attribute_relation_id"],
                     "explicit_numeric_value": str(p_operator["value"]),
                     "source_answer_count": len(selected.answer_ids),
                     "selected_answer_count": len(primary.answer_ids)})
    return primary, evidence
