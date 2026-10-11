"""Resolve numerical comparisons onto scalar leaves using local schema only.

This guard is used by model-free challenger generation, not by the ordinary
query lowering path.  It does not inspect answers, questions, or dataset IDs,
and does not issue endpoint/model requests.  Missing schema is left unchanged;
a proven entity-valued numerical target is never cast as though it were a
number.  Anonymous value records can be repaired only when their schema has
exactly one numerical leaf.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .ontology import relation_id_from_label


_NUMERIC_TYPES = frozenset({"type.int", "type.float"})


def _attribute_relation(operator: dict[str, Any]) -> str:
    direct = str(operator.get("attribute_relation_id", ""))
    if direct:
        return direct
    label = operator.get("attribute_relation_label", [])
    if not label:
        label = operator.get("attribute_relation_labels", [])
    return relation_id_from_label(label) if label else ""


def _numeric_cast(operator: dict[str, Any]) -> bool:
    kind = str(operator.get("type", "")).upper()
    if kind in {"LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"}:
        return bool(operator.get("_numeric_comparison_cast"))
    if kind == "EQUAL":
        return bool(operator.get("_numeric_cast_compare"))
    return False


def _node_types(node: str, triples: list[list[str]], ontology: Any) -> set[str]:
    types: set[str] = set()
    for subject, relation, object_ in triples:
        if str(object_) == node:
            value = str(ontology.range_for_relation(str(relation)))
            if value:
                types.add(value)
        if str(subject) == node:
            value = str(ontology.domain_for_relation(str(relation)))
            if value:
                types.add(value)
    return types


def _unique_numeric_leaf(ontology: Any, type_id: str) -> str:
    # Freebase named entities inherit common.topic.  They are not value
    # records, and their numeric attributes must not be chosen implicitly.
    explicit = getattr(ontology, "is_cvt_type", None)
    if callable(explicit):
        if not explicit(type_id):
            return ""
    elif "common.topic" in set(ontology.supertypes(type_id)):
        return ""
    properties = {
        str(relation)
        for relation in ontology.relations_for_domain(type_id)
        if str(ontology.range_for_relation(str(relation))) in _NUMERIC_TYPES
    }
    return next(iter(properties)) if len(properties) == 1 else ""


def schema_safe_numeric_operators(
    *,
    operators: list[dict[str, Any]],
    triples: list[list[str]],
    answer_var: str,
    ontology: Any,
) -> tuple[list[dict[str, Any]], list[list[str]], list[dict[str, Any]]] | None:
    """Return scalar-safe copies, or abstain from a proven unsafe candidate.

    Only numeric-cast operators are inspected.  Existing scalar/text casts and
    targets without local type evidence remain byte-for-byte unchanged.  A
    known nonprimitive target requires a unique numeric leaf of an anonymous
    value-record type; incompatible types or ambiguous leaves cause abstention.
    """

    repaired = deepcopy(operators)
    graph_triples = deepcopy(triples)
    audit: list[dict[str, Any]] = []
    if ontology is None:
        return repaired, graph_triples, audit
    # Multiple constraints can target records of the same type.  Keep this
    # invocation-local cache so a schema closure is computed only once, without
    # retaining stale decisions if the ontology is replaced or updated.
    leaf_by_type: dict[str, str] = {}
    for operator in repaired:
        if not _numeric_cast(operator):
            continue
        input_var = str(operator.get("input_var") or answer_var)
        attribute = _attribute_relation(operator)
        if attribute:
            range_id = str(ontology.range_for_relation(attribute))
            types = {range_id} if range_id else set()
        else:
            types = _node_types(input_var, graph_triples, ontology)
        # Preserve unknown schemas and primitive values (including numeric
        # strings). The targeted bug is casting a proven entity/record URI.
        entity_types = {value for value in types if not value.startswith("type.")}
        if not entity_types:
            continue
        if len(types) != 1:
            return None
        record_type = next(iter(entity_types))
        if record_type not in leaf_by_type:
            leaf_by_type[record_type] = _unique_numeric_leaf(ontology, record_type)
        leaf = leaf_by_type[record_type]
        if not leaf:
            return None
        record_var = input_var
        if attribute:
            existing = sorted({
                str(object_)
                for subject, relation, object_ in graph_triples
                if str(subject) == input_var and str(relation) == attribute
                and str(object_).startswith("V")
            })
            if len(existing) > 1:
                return None
            if existing:
                record_var = existing[0]
            else:
                used = {str(value) for triple in graph_triples for value in (triple[0], triple[2])}
                counter = 0
                while f"VNumericRecord{counter}" in used:
                    counter += 1
                record_var = f"VNumericRecord{counter}"
                graph_triples.append([input_var, attribute, record_var])
        operator["input_var"] = record_var
        operator["attribute_relation_id"] = leaf
        numeric_type = str(ontology.range_for_relation(leaf))
        # The threshold's lexical form does not determine the property's
        # datatype: a float record compared with integer 15 must preserve 15.7,
        # rather than silently truncating it to 15.  Set this only after a
        # record-to-scalar repair; existing scalar paths are left untouched.
        operator["_numeric_cast_kind"] = (
            "float" if numeric_type == "type.float" else "integer"
        )
        audit.append({
            "record_type": record_type,
            "record_var": record_var,
            "numeric_leaf": leaf,
            "numeric_type": numeric_type,
            "numeric_cast_kind": operator["_numeric_cast_kind"],
            "additional_model_calls": 0,
            "additional_endpoint_requests": 0,
        })
    return repaired, graph_triples, audit
