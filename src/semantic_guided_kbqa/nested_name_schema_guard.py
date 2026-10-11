"""Conservative, zero-call final selection protection for nested name heads.

Only existing executed denotations are considered. Type vocabulary comes from
the loaded schema, not entity/predicate/question lists. Generic schema roots,
literal name answers, ambiguous denotations and incomplete executions abstain.
This module does not import pipeline.py and must run after late challengers.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import re
from threading import Lock
from typing import Any

from .evaluation import canonicalize_answer_id


_NESTED_NAME = re.compile(
    r"^\s*(?:what|which)\s+(?:(?:is|was|are|were)\s+)?(?:the\s+)?"
    r"names?\s+of\s+(?:the|a|an)\s+(.+?)\s*$", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z-]*")
_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_VARIABLE = re.compile(r"(?:[AVE]\d+|P\d+\.V\d+|\?[A-Za-z_]\w*)\Z")
_BOUNDARIES = frozenset({"that", "which", "who", "whose", "with", "without", "where",
                         "from", "in", "of", "by", "for", "at", "on", "having"})
_INDEX_CACHE: dict[int, tuple[Any, dict[str, frozenset[str]]]] = {}
_INDEX_LOCK = Lock()


def _singular(word: str) -> str:
    token = word.casefold()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def nested_name_head(question: str) -> str:
    """Return a strict nested target noun phrase, or empty on ambiguity.

    Multiword heads require a schema phrase match downstream; the parser does
    not reinterpret 'team captain' as 'team'. Only a grammatical relative or
    prepositional boundary, punctuation, or end can terminate the head phrase.
    """
    match = _NESTED_NAME.match(str(question))
    if not match:
        return ""
    rest = match.group(1)
    tokens: list[str] = []
    position = 0
    while position < len(rest):
        while position < len(rest) and rest[position].isspace():
            position += 1
        if position == len(rest) or rest[position] in "?,.;:!":
            break
        word = _WORD.match(rest, position)
        if word is None:
            return ""
        token = _singular(word.group())
        if token in _BOUNDARIES and tokens:
            break
        tokens.append(token)
        if len(tokens) > 4:
            return ""
        position = word.end()
    return " ".join(tokens)


def _concrete_type(type_id: str) -> bool:
    value = str(type_id).strip()
    if "." not in value or value.startswith(("base.", "freebase.", "type.")):
        return False
    # Universal/topic roots are not evidence of a concrete answer role.
    return value.rsplit(".", 1)[-1] not in {"topic", "object"}


def _type_index(ontology: Any) -> dict[str, frozenset[str]]:
    if ontology is None:
        return {}
    key = id(ontology)
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None and cached[0] is ontology:
            return cached[1]
    ids: set[str] = set()
    for field in ("relation_domains", "relation_ranges"):
        values = getattr(ontology, field, {})
        if isinstance(values, Mapping):
            ids.update(str(value) for value in values.values() if value)
    parents = getattr(ontology, "type_parents", {})
    if isinstance(parents, Mapping):
        ids.update(str(value) for value in parents)
        ids.update(str(parent) for values in parents.values() for parent in values)
    index: dict[str, set[str]] = {}
    for type_id in ids:
        if not _concrete_type(type_id):
            continue
        leaf = type_id.rsplit(".", 1)[-1]
        if not re.fullmatch(r"[A-Za-z]+(?:_[A-Za-z]+)*", leaf):
            continue
        words = [_singular(word) for word in leaf.split("_")]
        index.setdefault(words[-1], set()).add(type_id)
        index.setdefault(" ".join(words), set()).add(type_id)
    frozen = {head: frozenset(values) for head, values in index.items()}
    with _INDEX_LOCK:
        # Bound strong references while avoiding reuse of an ontology object's ID.
        if len(_INDEX_CACHE) >= 8:
            _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
        _INDEX_CACHE[key] = (ontology, frozen)
    return frozen


def _schema_status(graph: Any, expected: set[str], ontology: Any) -> tuple[str, dict[str, Any]]:
    answer_var = str(graph.answer_var)
    declared: list[str] = []
    unknown = False
    for triple in graph.triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return "unknown", {"reason": "malformed_graph_triple"}
        subject, predicate, object_ = map(str, triple)
        sides = []
        if object_ == answer_var:
            sides.append("range_for_relation")
        if subject == answer_var:
            sides.append("domain_for_relation")
        for side in sides:
            function = getattr(ontology, side, None)
            if not callable(function):
                unknown = True
                continue
            type_id = str(function(predicate) or "").strip()
            if not _concrete_type(type_id):
                unknown = True
            elif type_id not in declared:
                declared.append(type_id)
    closures = {}
    for type_id in declared:
        function = getattr(ontology, "supertypes", None)
        values = function(type_id) if callable(function) else ()
        closures[type_id] = sorted({type_id, *(str(value) for value in values if _concrete_type(str(value)))})
    union = {value for values in closures.values() for value in values}
    evidence = {"declared_types": sorted(declared), "concrete_type_closures": closures,
                "has_unknown_or_generic_incident_type": unknown}
    if not declared or unknown:
        return "unknown", evidence
    return ("compatible" if union & expected else "conflict"), evidence


def _entity_answers(execution: Any) -> set[str] | None:
    ids = getattr(execution, "answer_ids", None)
    if not isinstance(ids, list) or not ids:
        return None
    canonical = [canonicalize_answer_id(value) for value in ids]
    if not all(_ENTITY.fullmatch(value) for value in canonical):
        return None
    return set(canonical)


def _complete(execution: Any) -> bool:
    ids = getattr(execution, "answer_ids", None)
    count = getattr(execution, "row_count", None)
    # DISTINCT one-column execution has one row per answer. If this invariant
    # cannot be established, do not treat a preview as a full denotation.
    return (isinstance(ids, list) and bool(ids) and isinstance(count, int)
            and not isinstance(count, bool) and count == len(ids)
            and len(set(map(str, ids))) == len(ids))


def _anchors(graph: Any) -> set[str]:
    return {canonicalize_answer_id(term) for triple in graph.triples
            for term in (str(triple[0]), str(triple[2])) if term and not _VARIABLE.fullmatch(term)}


def _substantive_operators(graph: Any) -> set[str] | None:
    signatures = set()
    for operator in graph.operators:
        if not isinstance(operator, dict):
            return None
        kind = str(operator.get("type", "")).upper()
        # Even variable NO_EQUAL may express a genuine exclusion. Preserve
        # every operator and field, including variables and casting metadata.
        signatures.add(json.dumps(operator, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return signatures


def _source_spine(graph: Any) -> tuple[set[tuple[str, str, str]], str]:
    incoming = [tuple(map(str, triple)) for triple in graph.triples
                if str(triple[2]) == str(graph.answer_var)]
    if len(incoming) != 1:
        return set(), "source_projection_edge_not_unique"
    projection = incoming[0]
    spine = {tuple(map(str, triple)) for triple in graph.triples if tuple(map(str, triple)) != projection}
    return (spine, "") if spine else (set(), "source_context_spine_missing")


def nested_name_schema_challenger(
    question: str, selected: Any, executions: list[Any], ontology: Any,
) -> tuple[Any | None, dict[str, Any]]:
    """Return a conservative existing replacement, otherwise None + reason.

    This protects answer role, not KG factual correctness. A positive result
    requires unique compatible denotation, complete answers, preserved anchor
    constants, source context spine and all substantive operators. Callers must
    run it after every other challenger; it neither modifies nor executes graphs.
    """
    head = nested_name_head(question)
    evidence: dict[str, Any] = {
        "status": "not_applied", "source_graph_id": selected.graph.graph_id,
        "head": head, "uses_existing_executions_only": True,
        "vocabulary_source": "ontology_type_leaf_phrase_and_final_word",
        "additional_model_calls": 0, "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0, "uses_gold_answers": False,
    }
    if not head:
        evidence["reason"] = "not_strict_nested_name_head"
        return None, evidence
    try:
        expected = set(_type_index(ontology).get(head, ()))
        evidence["expected_types"] = sorted(expected)
        if not expected:
            evidence["reason"] = "head_not_unambiguously_in_schema_vocabulary"
            return None, evidence
        if _entity_answers(selected) is None:
            evidence["reason"] = "scalar_text_or_mixed_answer_preserved"
            return None, evidence
        if not _complete(selected):
            evidence["status"], evidence["reason"] = "unqualified", "source_denotation_incomplete_or_unproven"
            return None, evidence
        source_status, source_types = _schema_status(selected.graph, expected, ontology)
        evidence["source_schema"] = source_types
        if source_status != "conflict":
            evidence["reason"] = "source_already_compatible" if source_status == "compatible" else "source_schema_unknown_or_generic"
            return None, evidence
        anchors = _anchors(selected.graph)
        spine, spine_error = _source_spine(selected.graph)
        operators = _substantive_operators(selected.graph)
        if not anchors or spine_error or operators is None:
            evidence["reason"] = spine_error or "source_anchor_or_operator_evidence_missing"
            return None, evidence
        eligible, unknown, incomplete, excluded = [], [], [], []
        for execution in executions:
            if execution is selected:
                continue
            graph = execution.graph
            if _entity_answers(execution) is None:
                continue
            status, types = _schema_status(graph, expected, ontology)
            if status == "conflict":
                continue
            candidate_operators = _substantive_operators(graph)
            candidate_triples = {tuple(map(str, triple)) for triple in graph.triples}
            if (not anchors <= _anchors(graph) or not spine <= candidate_triples
                    or candidate_operators is None or not operators <= candidate_operators):
                excluded.append(str(graph.graph_id))
                continue
            if status == "unknown":
                unknown.append(str(graph.graph_id))
                continue
            if not _complete(execution):
                incomplete.append(str(graph.graph_id))
                continue
            eligible.append((execution, types))
        evidence.update({"eligible_graph_ids": [str(execution.graph.graph_id) for execution, _ in eligible],
                         "unknown_schema_candidate_ids": unknown, "incomplete_compatible_candidate_ids": incomplete,
                         "excluded_constraint_or_anchor_candidate_ids": excluded})
        if unknown or incomplete:
            evidence["status"], evidence["reason"] = "unqualified", "candidate_type_or_completeness_uncertain"
            return None, evidence
        denotations = {_canonical_denotation(execution) for execution, _ in eligible}
        evidence["compatible_denotation_count"] = len(denotations)
        if len(denotations) != 1:
            evidence["reason"] = "no_compatible_candidate" if not denotations else "multiple_compatible_denotations"
            return None, evidence
        challenger, challenger_types = max(eligible, key=lambda item: (
            float(item[0].graph.score), str(item[0].graph.graph_id)))
        # Correcting an answer role is not evidence for broadening its set.
        # In particular, a singleton already-perfect answer may itself carry
        # several Freebase types; an alternative's larger set is not safer.
        if len(_entity_answers(challenger)) > len(_entity_answers(selected)):
            evidence["reason"] = "role_replacement_would_expand_answer_cardinality"
            return None, evidence
        if _canonical_denotation(challenger) == _canonical_denotation(selected):
            evidence["reason"] = "compatible_denotation_already_selected"
            return None, evidence
        evidence.update({"status": "applied", "reason": "unique_existing_nested_name_schema_denotation",
                         "selected_graph_id": challenger.graph.graph_id,
                         "selected_schema": challenger_types, "source_anchor_count": len(anchors),
                         "preserved_context_spine_edge_count": len(spine),
                         "preserved_substantive_operator_count": len(operators),
                         "selected_answer_count": len(challenger.answer_ids)})
        return challenger, evidence
    except Exception as exc:
        evidence.update({"status": "unqualified", "reason": "schema_or_execution_evidence_error",
                         "error": f"{type(exc).__name__}: {exc}"})
        return None, evidence


def _canonical_denotation(execution: Any) -> tuple[str, ...]:
    return tuple(sorted(canonicalize_answer_id(value) for value in execution.answer_ids))
