"""Consolidated safety; original rules and budgets retained."""
from __future__ import annotations


# ---- nested_name_schema_guard ----
"""Conservative, zero-call final selection protection for nested name heads.

Only existing executed denotations are considered. Type vocabulary comes from
the loaded schema, not entity/predicate/question lists. Generic schema roots,
literal name answers, ambiguous denotations and incomplete executions abstain.
This module does not import pipeline.py and must run after late challengers.
"""


from collections.abc import Mapping
import json
import re
from threading import Lock
from typing import Any

from .evaluation import canonicalize_answer_id as _nested_name_schema_guard_canonicalize_answer_id


_NESTED_NAME = re.compile(
    r"^\s*(?:what|which)\s+(?:(?:is|was|are|were)\s+)?(?:the\s+)?"
    r"names?\s+of\s+(?:the|a|an)\s+(.+?)\s*$", re.I)
_nested_name_schema_guard_WORD = re.compile(r"[A-Za-z][A-Za-z-]*")
_nested_name_schema_guard_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_VARIABLE = re.compile(r"(?:[AVE]\d+|P\d+\.V\d+|\?[A-Za-z_]\w*)\Z")
_BOUNDARIES = frozenset({"that", "which", "who", "whose", "with", "without", "where",
                         "from", "in", "of", "by", "for", "at", "on", "having"})
_INDEX_CACHE: dict[int, tuple[Any, dict[str, frozenset[str]]]] = {}
_INDEX_LOCK = Lock()


def _nested_name_schema_guard_singular(word: str) -> str:
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
        word = _nested_name_schema_guard_WORD.match(rest, position)
        if word is None:
            return ""
        token = _nested_name_schema_guard_singular(word.group())
        if token in _BOUNDARIES and tokens:
            break
        tokens.append(token)
        if len(tokens) > 4:
            return ""
        position = word.end()
    return " ".join(tokens)


def _nested_name_schema_guard_concrete_type(type_id: str) -> bool:
    value = str(type_id).strip()
    if "." not in value or value.startswith(("base.", "freebase.", "type.")):
        return False
    # Universal/topic roots are not evidence of a concrete answer role.
    return value.rsplit(".", 1)[-1] not in {"topic", "object"}


def _nested_name_schema_guard_type_index(ontology: Any) -> dict[str, frozenset[str]]:
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
        if not _nested_name_schema_guard_concrete_type(type_id):
            continue
        leaf = type_id.rsplit(".", 1)[-1]
        if not re.fullmatch(r"[A-Za-z]+(?:_[A-Za-z]+)*", leaf):
            continue
        words = [_nested_name_schema_guard_singular(word) for word in leaf.split("_")]
        index.setdefault(words[-1], set()).add(type_id)
        index.setdefault(" ".join(words), set()).add(type_id)
    frozen = {head: frozenset(values) for head, values in index.items()}
    with _INDEX_LOCK:
        # Bound strong references while avoiding reuse of an ontology object's ID.
        if len(_INDEX_CACHE) >= 8:
            _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
        _INDEX_CACHE[key] = (ontology, frozen)
    return frozen


def _nested_name_schema_guard_schema_status(graph: Any, expected: set[str], ontology: Any) -> tuple[str, dict[str, Any]]:
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
            if not _nested_name_schema_guard_concrete_type(type_id):
                unknown = True
            elif type_id not in declared:
                declared.append(type_id)
    closures = {}
    for type_id in declared:
        function = getattr(ontology, "supertypes", None)
        values = function(type_id) if callable(function) else ()
        closures[type_id] = sorted({type_id, *(str(value) for value in values if _nested_name_schema_guard_concrete_type(str(value)))})
    union = {value for values in closures.values() for value in values}
    evidence = {"declared_types": sorted(declared), "concrete_type_closures": closures,
                "has_unknown_or_generic_incident_type": unknown}
    if not declared or unknown:
        return "unknown", evidence
    return ("compatible" if union & expected else "conflict"), evidence


def _nested_name_schema_guard_entity_answers(execution: Any) -> set[str] | None:
    ids = getattr(execution, "answer_ids", None)
    if not isinstance(ids, list) or not ids:
        return None
    canonical = [_nested_name_schema_guard_canonicalize_answer_id(value) for value in ids]
    if not all(_nested_name_schema_guard_ENTITY.fullmatch(value) for value in canonical):
        return None
    return set(canonical)


def _nested_name_schema_guard_complete(execution: Any) -> bool:
    ids = getattr(execution, "answer_ids", None)
    count = getattr(execution, "row_count", None)
    # DISTINCT one-column execution has one row per answer. If this invariant
    # cannot be established, do not treat a preview as a full denotation.
    return (isinstance(ids, list) and bool(ids) and isinstance(count, int)
            and not isinstance(count, bool) and count == len(ids)
            and len(set(map(str, ids))) == len(ids))


def _nested_name_schema_guard_anchors(graph: Any) -> set[str]:
    return {_nested_name_schema_guard_canonicalize_answer_id(term) for triple in graph.triples
            for term in (str(triple[0]), str(triple[2])) if term and not _VARIABLE.fullmatch(term)}


def _nested_name_schema_guard_substantive_operators(graph: Any) -> set[str] | None:
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
        expected = set(_nested_name_schema_guard_type_index(ontology).get(head, ()))
        evidence["expected_types"] = sorted(expected)
        if not expected:
            evidence["reason"] = "head_not_unambiguously_in_schema_vocabulary"
            return None, evidence
        if _nested_name_schema_guard_entity_answers(selected) is None:
            evidence["reason"] = "scalar_text_or_mixed_answer_preserved"
            return None, evidence
        if not _nested_name_schema_guard_complete(selected):
            evidence["status"], evidence["reason"] = "unqualified", "source_denotation_incomplete_or_unproven"
            return None, evidence
        source_status, source_types = _nested_name_schema_guard_schema_status(selected.graph, expected, ontology)
        evidence["source_schema"] = source_types
        if source_status != "conflict":
            evidence["reason"] = "source_already_compatible" if source_status == "compatible" else "source_schema_unknown_or_generic"
            return None, evidence
        anchors = _nested_name_schema_guard_anchors(selected.graph)
        spine, spine_error = _source_spine(selected.graph)
        operators = _nested_name_schema_guard_substantive_operators(selected.graph)
        if not anchors or spine_error or operators is None:
            evidence["reason"] = spine_error or "source_anchor_or_operator_evidence_missing"
            return None, evidence
        eligible, unknown, incomplete, excluded = [], [], [], []
        for execution in executions:
            if execution is selected:
                continue
            graph = execution.graph
            if _nested_name_schema_guard_entity_answers(execution) is None:
                continue
            status, types = _nested_name_schema_guard_schema_status(graph, expected, ontology)
            if status == "conflict":
                continue
            candidate_operators = _nested_name_schema_guard_substantive_operators(graph)
            candidate_triples = {tuple(map(str, triple)) for triple in graph.triples}
            if (not anchors <= _nested_name_schema_guard_anchors(graph) or not spine <= candidate_triples
                    or candidate_operators is None or not operators <= candidate_operators):
                excluded.append(str(graph.graph_id))
                continue
            if status == "unknown":
                unknown.append(str(graph.graph_id))
                continue
            if not _nested_name_schema_guard_complete(execution):
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
        if len(_nested_name_schema_guard_entity_answers(challenger)) > len(_nested_name_schema_guard_entity_answers(selected)):
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
    return tuple(sorted(_nested_name_schema_guard_canonicalize_answer_id(value) for value in execution.answer_ids))



# ---- query_contradiction_safety ----
"""Recognize a bound RDF-subject self-inequality, without weakening filters.

A literal NaN need not equal itself, so mere FILTER(?x != ?x) is insufficient.
Only a variable bound as a mandatory RDF triple subject is proved nonliteral.
COUNT over an empty graph can return zero; aggregate queries are not skipped.
"""
import re
from typing import Any


def bound_subject_self_inequality(graph: Any, sparql: str) -> dict | None:
    if not re.match(r"\A\s*SELECT\s+DISTINCT\s+\?[A-Za-z_]\w*\s+WHERE\b", sparql, re.I):
        return None
    for operator in graph.operators:
        if not isinstance(operator, dict):
            continue
        if (str(operator.get("type", "")).upper() != "NO_EQUAL"
                or str(operator.get("value_type", "")).casefold() != "variable"
                or any(operator.get(field) for field in
                       ("attribute_relation_id", "attribute_relation_label", "attribute_relation_labels"))):
            continue
        left = str(operator.get("input_var", "")).removeprefix("?")
        right = str(operator.get("value", "")).removeprefix("?")
        if left != right or not re.fullmatch(r"[A-Za-z_]\w*", left):
            continue
        subjects = [triple for triple in graph.triples
                    if isinstance(triple, (list, tuple)) and len(triple) == 3
                    and str(triple[0]).removeprefix("?") == left]
        if not subjects:
            continue
        pattern = r"\bFILTER\s*\(\s*\?" + re.escape(left) + r"\s*!=\s*\?" + re.escape(left) + r"\s*\)"
        if not re.search(pattern, sparql):
            continue
        return {"status": "proven_empty", "reason": "bound_nonliteral_subject_self_inequality",
                "variable": left, "subject_binding_edge_count": len(subjects),
                "filter_retained": True, "NaN_literal_case_excluded": True,
                "aggregate_empty_value_case_excluded": True,
                "additional_model_calls": 0, "additional_endpoint_queries": 0}
    return None



# ---- scalar_owner_union_safety ----
"""Undo a proven single-measurement scope conflict in an existing union.

Gold-blind, final-only, and restricted to an explicitly supported, more
specific owner property. All model requests and endpoint executions have
already finished; this module never generates or queries candidates.
"""
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .contracts import ExecutedGraph as _scalar_owner_union_safety_ExecutedGraph, QueryGraphCandidate as _scalar_owner_union_safety_QueryGraphCandidate


def _scalar_owner_union_safety_number(value: Any) -> Decimal | None:
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _single_numeric_operator(graph: _scalar_owner_union_safety_QueryGraphCandidate) -> dict | None:
    substantive = [operator for operator in graph.operators
                   if isinstance(operator, dict)
                   and str(operator.get("type", "")).upper() != "NO_EQUAL"]
    if len(substantive) != 1:
        return None
    operator = substantive[0]
    if (str(operator.get("type", "")).upper() != "EQUAL"
            or str(operator.get("value_type", "")).casefold() != "number"
            or not operator.get("attribute_relation_id")
            or _scalar_owner_union_safety_number(operator.get("value")) is None):
        return None
    return operator


def _owner_property(graph: _scalar_owner_union_safety_QueryGraphCandidate, operator: dict, ontology: Any) -> str:
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
    selected: _scalar_owner_union_safety_ExecutedGraph,
    executions: list[_scalar_owner_union_safety_ExecutedGraph],
    ontology: Any,
) -> tuple[_scalar_owner_union_safety_ExecutedGraph | None, dict[str, Any]]:
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
            or _scalar_owner_union_safety_number(p_operator["value"]) != _scalar_owner_union_safety_number(s_operator["value"])):
        evidence["reason"] = "different_scalar_semantics"
        return None, evidence
    text = re.sub(r'["“][^"”]*["”]', " ", str(question))
    literals = re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w.])", text)
    if (len(literals) != 1 or _scalar_owner_union_safety_number(literals[0]) != _scalar_owner_union_safety_number(p_operator["value"])
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



# ---- terminal_union_scope_safety ----
"""Final-only removal of an unsupported terminal-role interpretation.

This is intentionally not a selector or an answer generator. It can return only
the complete, already executed primary of an actual two-source answer union.
Question text, graph identity, full denotations and concrete ontology types must
independently establish the narrow scope. No labels, gold, learned scores,
endpoint requests or model/embedding calls are used.
"""


from collections import Counter
import re
from typing import Any

from .contracts import ExecutedGraph as _terminal_union_scope_safety_ExecutedGraph, QueryGraphCandidate as _terminal_union_scope_safety_QueryGraphCandidate


_ENTITY_ID = re.compile(r"(?:https?://rdf\.freebase\.com/ns/)?[mg]\.[A-Za-z0-9_]+")
_GRAMMAR = frozenset({
    "a", "an", "are", "as", "at", "be", "been", "being", "by", "do", "does",
    "did", "for", "from", "had", "has", "have", "how", "in", "into", "is",
    "it", "of", "on", "that", "the", "there", "these", "this", "those", "to",
    "was", "were", "what", "when", "where", "which", "who", "whom", "with",
})
_AMBIGUOUS_ROLE = re.compile(
    r"\b(?:and|or|nor|both|also|either|plus|not|never|without|except|excluding)\b"
    r"|\bas\s+well\s+as\b|[&/]", re.I,
)


def _token_forms(token: str) -> frozenset[str]:
    """Conservative spelling morphology, not a task/relation vocabulary."""
    token = token.casefold()
    forms = {token}
    if len(token) > 4 and token.endswith("ies"):
        forms.add(token[:-3] + "y")
    elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        forms.add(token[:-1])
        if token.endswith(("ches", "shes", "xes", "zes", "oes")):
            forms.add(token[:-2])
    if len(token) > 5 and token.endswith("ed"):
        base = token[:-2]
        forms.update((base, token[:-1]))
        if len(base) > 3 and base[-1] == base[-2]:
            forms.add(base[:-1])
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        forms.update((base, base + "e"))
        if len(base) > 3 and base[-1] == base[-2]:
            forms.add(base[:-1])
    if len(token) > 4 and token.endswith("ly"):
        forms.add(token[:-2])
    return frozenset(forms)


def _terminal_union_scope_safety_tokens(value: str, *, informative: bool = False) -> list[str]:
    tokens = re.findall(r"[a-z]+", str(value).casefold())
    result: list[str] = []
    for token in tokens:
        if informative and (len(token) < 3 or token in _GRAMMAR):
            continue
        # Repeated inflections of one word are not two informative role words.
        if not any(_token_forms(token) & _token_forms(old) for old in result):
            result.append(token)
    return result


def _unquoted_question(question: str) -> str | None:
    text = str(question)
    for pattern in (r'"[^"\n]*"', r"“[^”\n]*”", r"(?<!\w)'[^'\n]*'(?!\w)",
                    r"(?<!\w)‘[^’\n]*’(?!\w)"):
        text = re.sub(pattern, " ", text)
    # Unbalanced quoting has no reliable distinction between an entity title
    # and a requested role. In-word apostrophes (there's) are not quotation.
    if re.search(r'["“”‘]|(?<!\w)[\'’]|[\'’](?!\w)', text):
        return None
    return text


def _matches(token: str, other: str) -> bool:
    return bool(_token_forms(token) & _token_forms(other))


def _possibly_mentioned(token: str, question_tokens: list[str]) -> bool:
    # A derivation such as "officially" must never count as absent evidence
    # for "official". Prefixes are used only to abstain, never to establish
    # primary support; this deliberately favors preserving a valid union.
    def possible(other: str) -> bool:
        if _matches(token, other):
            return True
        if min(len(token), len(other)) >= 4 and (token in other or other in token):
            return True
        prefix_length = 0
        for left, right in zip(token, other):
            if left != right:
                break
            prefix_length += 1
        # Derivations such as administrative/administration are ambiguous
        # role evidence. Shared long stems only veto; they never add support.
        return prefix_length >= max(6, min(len(token), len(other)) - 3)

    return any(possible(other) for other in question_tokens)


def _complete_entities(execution: _terminal_union_scope_safety_ExecutedGraph) -> bool:
    ids = execution.answer_ids
    return (isinstance(ids, list) and bool(ids)
            and isinstance(execution.row_count, int)
            and not isinstance(execution.row_count, bool)
            and execution.row_count == len(ids)
            and all(isinstance(value, str) and _ENTITY_ID.fullmatch(value) for value in ids)
            and len(set(ids)) == len(ids))


def _terminal_edge(graph: _terminal_union_scope_safety_QueryGraphCandidate) -> tuple[str, str, str] | None:
    triples = graph.triples
    if (not isinstance(triples, list)
            or any(not isinstance(row, (list, tuple)) or len(row) != 3
                   or not all(isinstance(term, str) for term in row) for row in triples)):
        return None
    incident = [tuple(row) for row in triples
                if graph.answer_var in (row[0], row[2])]
    if len(incident) != 1 or incident[0][0] == incident[0][2]:
        return None
    return incident[0]


def _terminal_union_scope_safety_concrete_type(value: str) -> bool:
    # "type" contains primitive scalars and the generic object root;
    # "common" contains cross-domain topic/taggable scaffolding. Neither is
    # sufficient evidence for changing an entity-denoting terminal role.
    return bool(value and "." in value and value.split(".", 1)[0] not in {"type", "common"})


def _compatible_concrete(left: str, right: str, ontology: Any) -> bool:
    if not _terminal_union_scope_safety_concrete_type(left) or not _terminal_union_scope_safety_concrete_type(right):
        return False
    if left == right:
        return True
    supertypes = getattr(ontology, "supertypes", None)
    if not callable(supertypes):
        return False
    # Merely sharing an ancestor does not make sibling types interchangeable.
    return left in supertypes(right) or right in supertypes(left)


def terminal_union_scope_challenger(
    question: str,
    selected: _terminal_union_scope_safety_ExecutedGraph,
    executions: list[_terminal_union_scope_safety_ExecutedGraph],
    ontology: Any,
) -> tuple[_terminal_union_scope_safety_ExecutedGraph | None, dict[str, Any]]:
    """Return an existing complete primary only when union scope is proven.

    ``executions`` must contain the original full execution objects, not saved
    candidate previews. Ambiguous identities or incomplete row counts abstain.
    Inputs (including union provenance, graph fields and answer lists) are never
    mutated. The caller applies the returned object only at finalization.
    """
    evidence: dict[str, Any] = {
        "status": "not_applied", "source_graph_id": selected.graph.graph_id,
        "additional_model_calls": 0, "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0, "uses_evaluation_gold": False,
    }

    def abstain(reason: str):
        evidence["reason"] = reason
        return None, evidence

    union = selected.graph.provenance.get("selection_answer_union")
    if not isinstance(union, dict) or ontology is None:
        return abstain("no_existing_union_or_ontology")
    primary_id, secondary_id = (union.get(key) for key in ("primary_graph_id", "secondary_graph_id"))
    if (not isinstance(primary_id, str) or not isinstance(secondary_id, str)
            or not primary_id or not secondary_id or primary_id == secondary_id):
        return abstain("ambiguous_union_provenance")
    primary_matches = [item for item in executions if item.graph.graph_id == primary_id]
    secondary_matches = [item for item in executions if item.graph.graph_id == secondary_id]
    if len(primary_matches) != 1 or len(secondary_matches) != 1:
        return abstain("union_sources_not_uniquely_materialized")
    primary, secondary = primary_matches[0], secondary_matches[0]
    if (primary is secondary or primary is selected or secondary is selected
            or primary.graph.provenance.get("selection_answer_union")
            or secondary.graph.provenance.get("selection_answer_union")):
        return abstain("union_sources_not_original_executions")
    p_graph, s_graph = primary.graph, secondary.graph
    if (selected.graph.graph_id != primary_id
            or any(getattr(selected.graph, field) != getattr(p_graph, field)
                   for field in ("triples", "answer_var", "operators", "compose_output", "sparql"))):
        return abstain("union_graph_changed_by_late_repair")
    if not all(_complete_entities(item) for item in (selected, primary, secondary)):
        return abstain("denotations_not_complete_entity_sets")
    p_ids, s_ids = set(primary.answer_ids), set(secondary.answer_ids)
    if set(selected.answer_ids) != p_ids | s_ids or not s_ids - p_ids:
        return abstain("not_exact_union_with_secondary_extras")
    if p_graph.answer_var != s_graph.answer_var or p_graph.operators != s_graph.operators:
        # Exact operator equality retains every NO_EQUAL field and all scalar,
        # temporal, aggregate and comparison constraints without normalization.
        return abstain("answer_variable_or_operators_differ")
    p_edge, s_edge = _terminal_edge(p_graph), _terminal_edge(s_graph)
    if (p_edge is None or s_edge is None or p_edge[1] == s_edge[1]
            or p_edge[0] != s_edge[0] or p_edge[2] != s_edge[2]):
        return abstain("not_single_same_orientation_terminal_predicate_change")
    p_context = Counter(tuple(row) for row in p_graph.triples if tuple(row) != p_edge)
    s_context = Counter(tuple(row) for row in s_graph.triples if tuple(row) != s_edge)
    if p_context != s_context:
        return abstain("different_graph_context")
    # Identical context must anchor the same entity-linking interpretation; a
    # completely unanchored free query cannot establish the requested scope.
    if not any(_ENTITY_ID.fullmatch(term) for row in p_graph.triples for term in (row[0], row[2])):
        return abstain("no_shared_entity_anchor")
    p_domain = str(ontology.domain_for_relation(p_edge[1]))
    s_domain = str(ontology.domain_for_relation(s_edge[1]))
    p_range = str(ontology.range_for_relation(p_edge[1]))
    s_range = str(ontology.range_for_relation(s_edge[1]))
    if (not _compatible_concrete(p_domain, s_domain, ontology)
            or not _compatible_concrete(p_range, s_range, ontology)):
        return abstain("terminal_schema_not_concretely_compatible")
    text = _unquoted_question(question)
    if text is None or _AMBIGUOUS_ROLE.search(text):
        return abstain("ambiguous_or_multiple_role_request")
    p_tokens = _terminal_union_scope_safety_tokens(p_edge[1].rsplit(".", 1)[-1], informative=True)
    s_tokens = _terminal_union_scope_safety_tokens(s_edge[1].rsplit(".", 1)[-1], informative=True)
    q_tokens = _terminal_union_scope_safety_tokens(text)
    if len(p_tokens) < 2 or not all(any(_matches(token, word) for word in q_tokens) for token in p_tokens):
        return abstain("primary_terminal_role_not_explicitly_supported")
    shared = [token for token in s_tokens if any(_matches(token, word) for word in p_tokens)]
    modifiers = [token for token in s_tokens if not any(_matches(token, word) for word in p_tokens)]
    if not shared or not modifiers or any(_possibly_mentioned(token, q_tokens) for token in modifiers):
        return abstain("secondary_extra_modifier_not_proven_absent")
    evidence.update({
        "status": "applied", "reason": "explicit_primary_terminal_scope_excludes_unmentioned_modifier",
        "selected_graph_id": primary_id, "discarded_union_graph_id": secondary_id,
        "primary_terminal_relation": p_edge[1], "secondary_terminal_relation": s_edge[1],
        "supported_primary_leaf_tokens": p_tokens, "unsupported_secondary_modifiers": modifiers,
        "primary_domain": p_domain, "secondary_domain": s_domain,
        "primary_range": p_range, "secondary_range": s_range,
        "same_context_and_operators": True,
        "source_answer_count": len(selected.answer_ids), "selected_answer_count": len(primary.answer_ids),
        "removed_answer_count": len(s_ids - p_ids),
        "primary_full_row_count": primary.row_count, "secondary_full_row_count": secondary.row_count,
    })
    return primary, evidence



# ---- unsupported_intersection_execution_guard ----
"""Pure, final restoration of a proved unsupported answer intersection.

This is deliberately not a plural-parser change and not a scalar repair.
It may return only the actual complete pre-intersection execution. A missing
explicit scalar condition is still missing after restoration. No query, model,
file, Gold, relinking, or answer-ID synthesis is performed here.
"""


from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .budgeted_schema_repair import (
    NUMERIC_TYPES, STRING_TYPES, _validated_provided_scalar_spec, schema_type_overlap,
)
from .normalization import parse_english_calendar_date, parse_english_number
from .contracts import ExecutedGraph as _unsupported_intersection_execution_guard_ExecutedGraph
from .ontology import relation_id_from_label as _unsupported_intersection_execution_guard_relation_id_from_label


_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_unsupported_intersection_execution_guard_VAR = re.compile(r"V\d+\Z")
_PREFIX = re.compile(
    r"^\s*(?:in|at|on|from|of|for|with|by|to|among|within|through|over|under)\s+"
    r"(?:what|which)\s+(?P<head>[A-Za-z][A-Za-z-]*)\b", re.I,
)
_AUX = frozenset({"a", "an", "the", "is", "was", "are", "were", "has", "have", "had", "date", "time", "of"})
_KINDS = frozenset({"EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL"})


def _words(value: Any) -> set[str]:
    result = set()
    for word in re.findall(r"[A-Za-z]+", str(value).casefold()):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        result.add(word)
    return result


def _name(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _core(graph: Any) -> tuple[Any, ...] | None:
    triples, operators = getattr(graph, "triples", None), getattr(graph, "operators", None)
    if (not isinstance(triples, list) or not isinstance(operators, list)
            or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)):
        return None
    # This lane never changes or deletes any operator. Only empty programs
    # have a closed, independently represented primary/scalar obligation.
    if operators:
        return None
    return tuple(tuple(map(str, t)) for t in triples), str(graph.answer_var), ()


def _unsupported_intersection_execution_guard_complete(execution: Any) -> bool:
    values, count = getattr(execution, "answer_ids", None), getattr(execution, "row_count", None)
    return (isinstance(values, list) and values and isinstance(count, int) and not isinstance(count, bool)
            and count == len(values) and len(set(map(str, values))) == len(values)
            and all(_MID.fullmatch(str(v)) for v in values))


def _raw_core(raw: Mapping[str, Any]) -> tuple[Any, ...] | None:
    triples, ops = raw.get("triples"), raw.get("operators")
    if (not isinstance(triples, list) or ops != [] or not raw.get("answer_var")
            or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)):
        return None
    return tuple(tuple(map(str, t)) for t in triples), str(raw["answer_var"]), ()


def _preview_agrees(raw: Mapping[str, Any], execution: _unsupported_intersection_execution_guard_ExecutedGraph) -> bool:
    values, count, truncated = raw.get("answer_ids"), raw.get("answer_count"), raw.get("answers_truncated")
    if (not isinstance(values, list) or not isinstance(count, int) or isinstance(count, bool)
            or count != len(execution.answer_ids) or len(set(map(str, values))) != len(values)
            or not set(map(str, values)) <= set(map(str, execution.answer_ids))):
        return False
    if truncated is True:
        return len(values) < count
    return truncated is False and len(values) == count and set(map(str, values)) == set(map(str, execution.answer_ids))


def _unsupported_intersection_execution_guard_number(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value).replace(",", ""))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def _literal_matches(surface: str, spec: Mapping[str, Any]) -> bool:
    if spec.get("value_type") == "number":
        value = _unsupported_intersection_execution_guard_number(surface)
        if value is None:
            parsed = parse_english_number(surface)
            value = _unsupported_intersection_execution_guard_number(parsed) if parsed is not None else None
        return value is not None and value == _unsupported_intersection_execution_guard_number(spec.get("value"))
    parsed = parse_english_calendar_date(surface)
    return str(surface).strip() == str(spec.get("value")) or parsed == str(spec.get("value"))


def _signature(spec: Mapping[str, Any]) -> tuple[str, str, str]:
    value = _unsupported_intersection_execution_guard_number(spec.get("value")) if spec.get("value_type") == "number" else None
    normalized = format((value if value else Decimal(0)).normalize(), "f") if value is not None else str(spec.get("value", ""))
    return str(spec.get("type", "")), normalized, str(spec.get("value_type", ""))


def unsupported_intersection_challenger(
    question: str, selected: _unsupported_intersection_execution_guard_ExecutedGraph, executions: Sequence[_unsupported_intersection_execution_guard_ExecutedGraph], ontology: Any,
    prior_selection: Mapping[str, Any] | None, *, stages: Sequence[Mapping[str, Any]],
    question_anchor_contract: Mapping[str, Any] | None = None,
    legacy_plural_request: Callable[[str], bool] | None = None,
    native_spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
    spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> tuple[_unsupported_intersection_execution_guard_ExecutedGraph | None, dict[str, Any]]:
    """Return a complete actual pre-intersection source, or abstain.

    The callbacks must be the caller's actual unchanged legacy plural/native
    parser and its strict internal spec provider. The provided-topic contract
    is question-owned input linking, not a lookup or an alias vocabulary.
    """
    proof: dict[str, Any] = {
        "status": "not_applied", "pure_existing_execution_guard": True,
        "additional_model_calls": 0, "additional_endpoint_queries": 0, "additional_embedding_calls": 0,
        "uses_gold_answers": False, "uses_entity_relation_or_sample_whitelist": False,
        "does_not_change_global_plural_or_model_parsers": True, "answer_ids_synthesized": False,
        "requires_final_post_model_position": True,
    }

    def abstain(reason: str):
        return None, {**proof, "reason": reason}

    if (ontology is None or not callable(legacy_plural_request) or not callable(native_spec_provider)
            or not callable(spec_provider) or not isinstance(prior_selection, Mapping)):
        return abstain("actual_parser_callbacks_or_prior_selection_missing")
    question = str(question)
    prefix = _PREFIX.match(question)
    if (prefix is None or not prefix["head"].casefold().endswith("s")
            or prefix["head"].casefold().endswith("ss") or legacy_plural_request(question)):
        return abstain("not_an_actual_legacy_missed_preposition_wh_plural")
    if native_spec_provider(question) is not None:
        return abstain("native_parser_already_has_obligation")
    spec = _validated_provided_scalar_spec(question, [], spec_provider)
    if (not spec or spec.get("type") not in _KINDS or not str(spec.get("value", "")).strip()
            or spec.get("_normalization_scope") != "internal_post_model_constraint_spec_only"):
        return abstain("one_independently_verified_new_internal_scalar_spec_missing")
    provenance = getattr(selected.graph, "provenance", None)
    marker = provenance.get("selection_answer_intersection") if isinstance(provenance, Mapping) else None
    gate = prior_selection.get("hard_conjunctive_answer_intersection_gate")
    if (not isinstance(marker, Mapping) or not isinstance(gate, Mapping) or gate.get("status") != "applied"
            or "explicit_constraint_obligation" not in gate or gate["explicit_constraint_obligation"] is not None
            or provenance.get("selection_answer_union")
            or str(provenance.get("original_question", "")) != question):
        return abstain("actual_question_owned_intersection_and_applied_explicit_none_gate_missing")
    left_id, right_id = str(marker.get("left_graph_id", "")), str(marker.get("right_graph_id", ""))
    source_id = str(prior_selection.get("hard_conjunctive_answer_intersection_gate_source_graph_id", ""))
    if (not left_id or not right_id or left_id == right_id or source_id != left_id
            or str(gate.get("source_graph_id", "")) != source_id
            or str(gate.get("left_graph_id", "")) != left_id or str(gate.get("right_graph_id", "")) != right_id
            or str(gate.get("selected_graph_id", "")) != left_id or str(selected.graph.graph_id) != left_id
            or str(prior_selection.get("selected_graph_id", "")) != left_id
            or marker.get("score_margin") != gate.get("score_margin")):
        return abstain("actual_prior_marker_and_pregate_source_identity_conflict")
    actual = {}
    for graph_id in (left_id, right_id):
        matches = [e for e in executions if str(e.graph.graph_id) == graph_id]
        if len(matches) != 1 or not _unsupported_intersection_execution_guard_complete(matches[0]) or _core(matches[0].graph) is None:
            return abstain("unique_complete_operator_free_actual_operand_execution_missing")
        execution = matches[0]
        owned = execution.graph.provenance
        if (not isinstance(owned, Mapping) or str(owned.get("original_question", "")) != question
                or owned.get("selection_answer_intersection") or owned.get("selection_answer_union")):
            return abstain("operand_is_postprocessed_or_not_current_question_owned")
        actual[graph_id] = execution
    left, right = actual[left_id], actual[right_id]
    if not _unsupported_intersection_execution_guard_complete(selected) or _core(selected.graph) != _core(left.graph):
        return abstain("current_graph_or_complete_intersection_execution_drifted")
    intersection = set(left.answer_ids) & set(right.answer_ids)
    if (set(selected.answer_ids) != intersection or not intersection < set(left.answer_ids)
            or gate.get("source_answer_count") != len(left.answer_ids)
            or gate.get("selected_answer_count") != len(selected.answer_ids) or gate.get("operator_count") != 0):
        return abstain("current_answers_not_the_attested_complete_actual_intersection")
    snapshots = prior_selection.get("candidates")
    if not isinstance(snapshots, list):
        return abstain("actual_captured_operand_graph_previews_missing")
    for graph_id, execution in actual.items():
        records = [c for c in snapshots if isinstance(c, Mapping) and str(c.get("graph_id")) == graph_id]
        if (len(records) != 1 or _raw_core(records[0]) != _core(execution.graph)
                or not _preview_agrees(records[0], execution)):
            return abstain("captured_preview_or_graph_conflicts_with_actual_full_operand")
        score_key = "left_rule_score" if graph_id == left_id else "right_rule_score"
        if gate.get(score_key) != execution.graph.score:
            return abstain("actual_operand_score_and_captured_gate_conflict")
    counts: dict[str, set[tuple[int, int]]] = {left_id: set(), right_id: set()}
    semantics = []
    linked = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        if stage.get("stage") == "linked_entity_context" and isinstance(stage.get("output"), list):
            linked.extend(stage["output"])
        elif stage.get("stage") == "semantic_path_generation":
            if not isinstance(stage.get("input"), Mapping) or stage["input"].get("question") != question:
                return abstain("semantic_stage_does_not_attest_same_invocation_question")
            output = stage.get("output")
            if isinstance(output, list):
                semantics.extend(e for e in output if isinstance(e, Mapping) and e.get("valid") is True
                                 and isinstance(e.get("output"), Mapping))
        elif stage.get("stage") == "deterministic_sparql_lowering_and_execution":
            for row in stage.get("output", []) if isinstance(stage.get("output"), list) else []:
                if isinstance(row, Mapping) and str(row.get("graph_id")) in counts:
                    pair = row.get("row_count"), row.get("answer_count")
                    if all(isinstance(v, int) and not isinstance(v, bool) for v in pair):
                        counts[str(row["graph_id"])].add(pair)
    if any(counts[graph_id] != {(len(e.answer_ids), len(e.answer_ids))} for graph_id, e in actual.items()):
        return abstain("same_invocation_endpoint_counts_missing_or_conflicting")
    if not semantics or any(len(e["output"].get("semantic_paths", [])) != 1 for e in semantics):
        return abstain("actual_original_semantics_missing_or_have_other_substantive_paths")
    source_triples = left.graph.triples
    if len(source_triples) != 1:
        return abstain("source_not_one_preserved_primary_entity_prefix")
    s, primary, o = map(str, source_triples[0])
    if _MID.fullmatch(s) and _unsupported_intersection_execution_guard_VAR.fullmatch(o) and left.graph.answer_var == o:
        anchor, owner, direction = s, o, "forward"
    elif _MID.fullmatch(o) and _unsupported_intersection_execution_guard_VAR.fullmatch(s) and left.graph.answer_var == s:
        anchor, owner, direction = o, s, "backward"
    else:
        return abstain("source_primary_anchor_owner_direction_not_unique")
    bindings = left.graph.provenance.get("anchor_bindings")
    bound = [(k, v) for k, v in bindings.items() if isinstance(v, Mapping) and str(v.get("id")) == anchor] if isinstance(bindings, Mapping) else []
    if len(bound) != 1 or len(bindings) != 1 or not str(bound[0][1].get("label", "")):
        return abstain("source_does_not_have_one_actual_named_topic_binding")
    root, binding = bound[0]
    label = str(binding["label"])
    provided = question_anchor_contract.get("provided_topic_entities") if isinstance(question_anchor_contract, Mapping) else None
    named_contract = (isinstance(question_anchor_contract, Mapping) and question_anchor_contract.get("question") == question
                      and isinstance(provided, Mapping) and set(map(str, provided)) == {anchor}
                      and _name(provided.get(anchor)) == _name(label))
    literal_topic = bool(re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", question, re.I))
    if not literal_topic and not named_contract:
        return abstain("nonliteral_topic_requires_exact_current_question_provided_mid_contract")
    for context in linked:
        if not isinstance(context, Mapping) or not _MID.fullmatch(str(context.get("id", ""))):
            continue
        if str(context["id"]) != anchor:
            return abstain("another_actual_linked_topic_or_unknown_kind_cannot_be_discarded")
    candidates = []
    for entry in semantics:
        sem = entry["output"]
        path = sem["semantic_paths"][0]
        steps = path.get("steps", []) if isinstance(path, Mapping) else []
        if not steps or not all(isinstance(step, Mapping) for step in steps):
            return abstain("same_invocation_semantic_path_malformed")
        first = steps[0]
        if (path.get("anchor_ref") != root or first.get("from") != root or first.get("direction") != direction
                or _unsupported_intersection_execution_guard_relation_id_from_label(first.get("relation_label")) != primary):
            return abstain("another_validated_primary_scope_conflicts_with_actual_source")
        roots = [a for a in sem.get("anchors", []) if isinstance(a, Mapping) and a.get("id") == root]
        if len(roots) != 1 or _name(roots[0].get("surface")) != _name(label):
            return abstain("actual_original_root_surface_and_bound_topic_contract_conflict")
        if len(steps) == 1:
            continue  # A later fallback cannot supply the scalar witness.
        if len(steps) != 2:
            return abstain("original_semantic_has_unpreserved_additional_joins")
        last = steps[1]
        attribute = _unsupported_intersection_execution_guard_relation_id_from_label(last.get("relation_label"))
        domain, datatype = str(ontology.domain_for_relation(attribute)), str(ontology.range_for_relation(attribute))
        entity_type = str(ontology.range_for_relation(primary) if direction == "forward" else ontology.domain_for_relation(primary))
        if (not first.get("to") or not last.get("to") or first.get("to") == last.get("to")
                or not first.get("id") or not last.get("id") or first.get("id") == last.get("id")
                or first.get("to") != last.get("from") or last.get("direction") != "forward"
                or path.get("path_output_var") != last.get("to") or not attribute or not domain
                or "common.topic" not in ontology.supertypes(domain)
                or schema_type_overlap(entity_type, domain, ontology) != "compatible"
                or _words(prefix["head"]) != _words(domain.rsplit(".", 1)[-1])):
            return abstain("original_scalar_owner_variables_or_positive_entity_schema_unproved")
        numeric = datatype in NUMERIC_TYPES | STRING_TYPES | {"type.enumeration"} and spec.get("value_type") == "number"
        temporal = datatype == "type.datetime" and spec.get("value_type") in {"date", "dateTime", "year"}
        if not numeric and not temporal:
            return abstain("original_attribute_datatype_does_not_match_normalized_obligation")
        goal = str(path.get("goal", ""))
        goal_spec = _validated_provided_scalar_spec(goal, [label], spec_provider)
        if (not goal_spec or _signature(goal_spec) != _signature(spec)
                or not _words(domain.rsplit(".", 1)[-1]) <= _words(goal)
                or not (_words(attribute.rsplit(".", 1)[-1]) - _AUX) <= _words(goal)
                or not isinstance(entry.get("decomposition"), list) or goal not in entry["decomposition"]
                or goal not in left.graph.provenance.get("decomposition", [])):
            return abstain("independent_original_goal_and_source_decomposition_scalar_obligation_missing")
        candidates.append((attribute, domain, datatype, path, entry))
    if not candidates or len({(a, d, t) for a, d, t, _, _ in candidates}) != 1:
        return abstain("no_unique_earlier_complete_primary_plus_scalar_semantic_witness")
    attribute, domain, datatype, path, _ = candidates[0]
    head_words, field_words = _words(prefix["head"]), _words(attribute.rsplit(".", 1)[-1]) - _AUX
    span, cue = spec.get("_source_literal_span"), str(spec.get("_source_comparison_cue", ""))
    if (not isinstance(span, (list, tuple)) or len(span) != 2
            or not all(isinstance(v, int) for v in span) or not cue or not field_words
            or not 0 <= span[0] < span[1] <= len(question)
            or not re.fullmatch(r"\s*[?.!]*\s*", question[span[1]:])):
        return abstain("question_not_one_closed_scalar_conjunct_with_attested_literal_span")
    conjunctions = [m for m in re.finditer(r"\band\b", question, re.I)
                    if not span[0] <= m.start() < span[1]]
    if len(conjunctions) != 1 or conjunctions[0].end() >= span[0]:
        return abstain("question_has_other_conjuncts_outside_the_attested_scalar_literal")
    right_prefix = question[conjunctions[0].end():span[0]].strip()
    if not re.search(re.escape(cue) + r"\s*$", right_prefix, re.I):
        return abstain("scalar_comparator_is_not_attached_to_final_literal")
    property_phrase = re.sub(re.escape(cue) + r"\s*$", "", right_prefix, flags=re.I)
    if _words(property_phrase) - _AUX != head_words | field_words:
        return abstain("question_scalar_conjunct_names_other_owner_or_unaccounted_scope")
    left_body = question[prefix.end():conjunctions[0].start()]
    if re.search(r"\b(?:who|whom|whose|which|that|where|when|while|although|without|except|before|after|during|since|until|between|only|exclusively|not|never|or|either)\b", left_body, re.I):
        return abstain("primary_question_clause_contains_other_unrepresented_scope")
    unused = []
    property_words = head_words | field_words
    for entry in semantics:
        for a in entry["output"].get("anchors", []):
            if not isinstance(a, Mapping) or a.get("id") == root:
                continue
            surface = str(a.get("surface", ""))
            if a.get("id") in bindings:
                return abstain("unused_original_anchor_is_an_actual_extra_bound_topic")
            if _literal_matches(surface, spec):
                reason = "actual_question_boundary_literal_equals_unique_normalized_obligation"
            elif _words(surface) == head_words:
                reason = "question_plural_head_equals_positive_scalar_owner_domain_kind"
            elif _words(surface) - _AUX == property_words:
                reason = "unused_schema_property_phrase_equals_exact_owner_and_scalar_field"
            else:
                return abstain("unreferenced_original_topic_or_kind_has_no_independent_positive_certificate")
            unused.append({"anchor_id": str(a.get("id", "")), "surface": surface, "reason": reason})
    rt = [tuple(map(str, t)) for t in right.graph.triples]
    if len(rt) != 2 or direction != "forward":
        return abstain("secondary_not_one_extra_unexpressed_same_root_entity_level")
    bridges = [t for t in rt if t[0] == anchor and _unsupported_intersection_execution_guard_VAR.fullmatch(t[2])]
    if len(bridges) != 1:
        return abstain("secondary_root_entity_bridge_not_unique")
    _, extra, bridge = bridges[0]
    if (extra == primary or extra in set(ontology.reverse_for_relation(primary))
            or rt.count((bridge, primary, str(right.graph.answer_var))) != 1
            or not _unsupported_intersection_execution_guard_VAR.fullmatch(str(right.graph.answer_var)) or bridge == right.graph.answer_var):
        return abstain("secondary_is_not_an_unsupported_extra_prefix_before_the_same_primary_role")
    root_type = str(ontology.domain_for_relation(primary))
    if (not root_type or "common.topic" not in ontology.supertypes(root_type)
            or schema_type_overlap(str(ontology.domain_for_relation(extra)), root_type, ontology) != "compatible"
            or schema_type_overlap(str(ontology.range_for_relation(extra)), root_type, ontology) != "compatible"):
        return abstain("extra_secondary_level_is_not_a_positive_same_root_entity_type")
    extra_words = _words(extra.rsplit(".", 1)[-1]) - _AUX
    expressed = _words(question)
    for entry in semantics:
        expressed |= _words(entry["output"]["semantic_paths"][0].get("goal", ""))
        for step in entry["output"]["semantic_paths"][0].get("steps", []):
            expressed |= _words(_unsupported_intersection_execution_guard_relation_id_from_label(step.get("relation_label")))
    if not extra_words or extra_words & expressed:
        return abstain("secondary_extra_level_has_question_or_original_semantic_support")
    proof.update(
        status="applied_partial_restore", reason="unsupported_extra_entity_level_intersection_after_missed_word_scalar_and_plural",
        source_graph_id=source_id, secondary_graph_id=right_id, source_answer_count=len(left.answer_ids),
        previous_answer_count=len(selected.answer_ids), source_endpoint_and_preview_counts_consistent=True,
        restores_actual_complete_execution_without_synthesizing_ids=True,
        normalized_scalar_obligation=deepcopy(spec), original_scalar_relation=attribute,
        scalar_owner_type=domain, scalar_datatype=datatype, original_complete_path_id=str(path.get("id", "")),
        unsupported_secondary_relation=extra, original_unused_anchor_certificates=unused,
        current_intersection_matches_actual_full_operand_ids=True, missing_scalar_constraint_not_implemented=True,
        partial_restoration_only=True, f1_improvement_or_nonregression_not_proven_by_compilation=True,
        topic_identity_from_current_question_provided_mid_contract=bool(named_contract and not literal_topic),
        prior_selection_does_not_override_numeric_or_date_precision=True,
    )
    return left, proof



# ---- model_entity_role_safety ----
"""Zero-call final protection of a captured model-selected entity answer.

This only reverses a proven local rule-score entity-to-primitive override. It
does not choose a new model/candidate, rebind constraints, or assert temporal
fidelity. Positive head vocabulary is generated from the loaded ontology.
"""


from collections.abc import Mapping
import json
import re
from typing import Any

from .evaluation import canonicalize_answer_id as _model_entity_role_safety_canonicalize_answer_id
_model_entity_role_safety_anchors = _nested_name_schema_guard_anchors
_model_entity_role_safety_complete = _nested_name_schema_guard_complete
_model_entity_role_safety_entity_answers = _nested_name_schema_guard_entity_answers
_model_entity_role_safety_schema_status = _nested_name_schema_guard_schema_status
_model_entity_role_safety_singular = _nested_name_schema_guard_singular
_model_entity_role_safety_substantive_operators = _nested_name_schema_guard_substantive_operators
_model_entity_role_safety_type_index = _nested_name_schema_guard_type_index


_model_entity_role_safety_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_model_entity_role_safety_HEAD = re.compile(r"^\s*(?:what|which)\s+([A-Za-z]+)(.*)\Z", re.I | re.S)
_BOUNDARY = re.compile(
    r"^(?:that|which|who|whose|with|without|where|from|in|of|by|for|at|on|having|"
    r"is|was|are|were|did|does|do|has|have|had|can|could|will|would|should)\b", re.I)
# These are generic scalar-question grammar exclusions, not positive entity
# head vocabulary; every accepted entity head must come from the ontology.
_SCALAR_HEAD = re.compile(r"(?:date|time|year|month|day|hour|minute|second|number)\Z", re.I)
_PRIMITIVE_TYPES = frozenset({"type.int", "type.float", "type.datetime", "type.text", "type.boolean", "type.rawstring", "type.uri"})


def model_entity_head(question: str) -> str:
    """Parse only one immediate noun, without reducing compound/quoted heads."""
    match = _model_entity_role_safety_HEAD.match(str(question))
    if not match:
        return ""
    head, tail = _model_entity_role_safety_singular(match.group(1)), match.group(2)
    if _SCALAR_HEAD.fullmatch(head):
        return ""
    # No apostrophe/hyphen/digit or adjacent word can be part of a simple head.
    if tail and not tail[0].isspace() and tail[0] not in "?.!":
        return ""
    rest = tail.strip()
    if rest and not re.fullmatch(r"[?.!]+", rest) and not _BOUNDARY.match(rest):
        return ""
    return head


def _model_entity_role_safety_stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _model_entity_role_safety_graph_state(graph: Any) -> str:
    triples = getattr(graph, "triples", None)
    operators = getattr(graph, "operators", None)
    if not isinstance(triples, list) or not triples or not isinstance(operators, list):
        raise ValueError("Missing graph state")
    if any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples):
        raise ValueError("Malformed graph triple")
    if any(not isinstance(o, dict) for o in operators):
        raise ValueError("Malformed graph operator")
    return _model_entity_role_safety_stable({"triples": sorted(tuple(map(str, t)) for t in triples),
                    "answer_var": str(graph.answer_var), "operators": operators})


def _captured_complete(candidate: Mapping[str, Any]) -> bool:
    values = candidate.get("answer_ids")
    count = candidate.get("answer_count")
    return (isinstance(values, list) and bool(values)
            and isinstance(count, int) and not isinstance(count, bool)
            and count == len(values) and not candidate.get("answers_truncated", True)
            and len(set(map(str, values))) == len(values))


def _model_entity_role_safety_candidate_graph(candidate: Mapping[str, Any]) -> Any:
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
        expected = set(_model_entity_role_safety_type_index(ontology).get(head, ()))
        evidence["expected_types"] = sorted(expected)
        if not expected:
            return abstain("head_not_in_concrete_ontology_vocabulary")
        if not _model_entity_role_safety_complete(selected):
            return abstain("source_denotation_incomplete_or_unproven", True)
        source_ids = set(_model_entity_role_safety_canonicalize_answer_id(v) for v in selected.answer_ids)
        if not source_ids or not all(source_ids) or any(_model_entity_role_safety_ENTITY.fullmatch(v) for v in source_ids):
            return abstain("source_not_pure_primitive_answers")
        source_state = _model_entity_role_safety_graph_state(selected.graph)
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
        if (_model_entity_role_safety_graph_state(_model_entity_role_safety_candidate_graph(rule)) != source_state
                or {_model_entity_role_safety_canonicalize_answer_id(v) for v in rule["answer_ids"]} != source_ids):
            return abstain("late_source_state_differs_from_captured_rule_execution")
        model_ids = {_model_entity_role_safety_canonicalize_answer_id(v) for v in model["answer_ids"]}
        if not all(_model_entity_role_safety_ENTITY.fullmatch(v) for v in model_ids):
            return abstain("captured_model_not_pure_entity_answers")
        matching = [e for e in executions if str(e.graph.graph_id) == model_id]
        if not matching:
            return abstain("captured_model_execution_missing", True)
        model_state = _model_entity_role_safety_graph_state(_model_entity_role_safety_candidate_graph(model))
        states = set()
        for execution in matching:
            ids = _model_entity_role_safety_entity_answers(execution)
            if ids is None or not _model_entity_role_safety_complete(execution):
                return abstain("model_execution_incomplete_or_wrong_kind", True)
            states.add((_model_entity_role_safety_graph_state(execution.graph), tuple(sorted(ids))))
        if len(states) != 1 or states != {(model_state, tuple(sorted(model_ids)))}:
            return abstain("model_execution_denotation_or_graph_ambiguous", True)
        replacement = matching[0]
        schema_status, schema = _model_entity_role_safety_schema_status(replacement.graph, expected, ontology)
        evidence["model_schema"] = schema
        if schema_status != "compatible":
            return abstain("model_schema_unknown_generic_or_wrong_head")
        source_anchors = _model_entity_role_safety_anchors(selected.graph)
        if not source_anchors or not source_anchors <= _model_entity_role_safety_anchors(replacement.graph):
            return abstain("source_anchor_not_preserved")
        source_operators = _model_entity_role_safety_substantive_operators(selected.graph)
        model_operators = _model_entity_role_safety_substantive_operators(replacement.graph)
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



# ---- source_role_execution_safety ----
"""Zero-call preservation of an explicitly requested source relation role.

This is a final-only guard: its result is always an existing execution. It
neither lowers a query nor asks a model, endpoint, or embedding ranker anything.
Positive vocabulary is generated from cached ontology endpoint types. Unknown
schema, ambiguous source state and incomplete denotations are abstentions.
"""


from collections import Counter
from collections.abc import Mapping
import json
import re
from typing import Any

from .evaluation import canonicalize_answer_id as _source_role_execution_safety_canonicalize_answer_id
_source_role_execution_safety_candidate_graph = _model_entity_role_safety_candidate_graph
_source_role_execution_safety_graph_state = _model_entity_role_safety_graph_state
_source_role_execution_safety_anchors = _nested_name_schema_guard_anchors
_source_role_execution_safety_complete = _nested_name_schema_guard_complete
_source_role_execution_safety_concrete_type = _nested_name_schema_guard_concrete_type
_source_role_execution_safety_entity_answers = _nested_name_schema_guard_entity_answers
_source_role_execution_safety_schema_status = _nested_name_schema_guard_schema_status
_source_role_execution_safety_singular = _nested_name_schema_guard_singular
_source_role_execution_safety_substantive_operators = _nested_name_schema_guard_substantive_operators
_source_role_execution_safety_type_index = _nested_name_schema_guard_type_index
from .ontology import relation_id_from_label as _source_role_execution_safety_relation_id_from_label


_source_role_execution_safety_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_source_role_execution_safety_WORD = re.compile(r"[A-Za-z]+")
_source_role_execution_safety_HEAD = re.compile(
    r"^\s*(?:what(?:['’]s)?|which)\s+(?:(?:is|was|are|were)\s+)?"
    r"(?:the\s+)?([A-Za-z]+(?:\s+[A-Za-z]+){0,3})", re.I)
_HEAD_BOUNDARY = frozenset({"of", "from", "in", "for", "by", "with", "that",
                            "which", "who", "whose", "where", "did", "does",
                            "do", "is", "was", "are", "were", "has", "have"})


def _source_role_execution_safety_tokens(value: str) -> tuple[str, ...]:
    return tuple(_source_role_execution_safety_singular(word) for word in _source_role_execution_safety_WORD.findall(str(value)))


def _word_forms(value: str) -> set[str]:
    """Exact generic inflections/agent suffixes, never arbitrary prefix match."""
    word = _source_role_execution_safety_singular(value)
    forms = {word}
    if len(word) >= 5 and word.endswith(("or", "er")):
        root = word[:-2]
        forms.update((root, root + "e"))
    if len(word) >= 6 and word.endswith("ed"):
        forms.update((word[:-2], word[:-1]))
    if len(word) >= 6 and word.endswith("ing"):
        forms.update((word[:-3], word[:-3] + "e"))
    return forms


def _role_mentioned(role: tuple[str, ...], words: tuple[str, ...]) -> bool:
    if not role or len(role) > len(words):
        return False
    # Preserve a compound role: 'executive producer' is not just 'producer'.
    return any(all(_word_forms(left) & _word_forms(right)
                   for left, right in zip(role, words[offset:offset + len(role)]))
               for offset in range(len(words) - len(role) + 1))


def _question_words(question: str, context: Any) -> tuple[str, ...] | None:
    # Quoted descriptions/titles may contain an occupation; they cannot prove
    # that the role is a question obligation. Unbalanced quotations abstain.
    text = str(question)
    if text.count('"') % 2:
        return None
    text = re.sub(r'"[^"\n]*"', ' ', text)
    text = re.sub(r"(?<!\w)'[^'\n]*'(?!\w)|“[^”\n]*”|‘[^’\n]*’", ' ', text)
    if isinstance(context, (list, tuple)):
        for entity in context:
            if isinstance(entity, Mapping):
                name = str(entity.get("name") or entity.get("label") or "").strip()
                if name:
                    text = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", " ", text, flags=re.I)
    return _source_role_execution_safety_tokens(text)


def _source_role_execution_safety_source_entity_head(question: str) -> str:
    """Strict noun phrase head; no movie/film or other synonym routing list."""
    match = _source_role_execution_safety_HEAD.match(str(question))
    if not match:
        return ""
    words = []
    for word in match.group(1).split():
        if word.casefold() in _HEAD_BOUNDARY:
            break
        words.append(_source_role_execution_safety_singular(word))
    if not words:
        return ""
    phrase = " ".join(words)
    # A apostrophe/hyphen/number must not turn a compound head into a prefix.
    end = match.start(1) + len(" ".join(match.group(1).split()[:len(words)]))
    tail = str(question)[end:]
    if tail and not tail[0].isspace() and tail[0] not in "?,.;:!":
        return ""
    return phrase


def _source_role_execution_safety_stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _definite_role_conflict(graph: Any, expected: set[str], ontology: Any):
    status, schema = _source_role_execution_safety_schema_status(graph, expected, ontology)
    if status == "conflict":
        # A broad declared endpoint (location vs region, person vs politician)
        # cannot prove a wrong role. Lack of a leaf match is not disjointness.
        expected_ancestors = {str(value) for type_id in expected
                              for value in ontology.supertypes(type_id)}
        ancestor_overlap = expected_ancestors & set(schema.get("declared_types", []))
        schema["declared_type_is_expected_ancestor"] = sorted(ancestor_overlap)
        if ancestor_overlap:
            status = "unknown"
    return status, schema


def _captured_matches(raw: Mapping[str, Any], execution: Any, *, full: bool) -> bool:
    if _source_role_execution_safety_graph_state(_source_role_execution_safety_candidate_graph(raw)) != _source_role_execution_safety_graph_state(execution.graph):
        return False
    preview, count = raw.get("answer_ids"), raw.get("answer_count")
    if (not isinstance(preview, list) or not isinstance(count, int)
            or isinstance(count, bool) or count != len(execution.answer_ids)):
        return False
    ids = {_source_role_execution_safety_canonicalize_answer_id(value) for value in execution.answer_ids}
    seen = {_source_role_execution_safety_canonicalize_answer_id(value) for value in preview}
    if len(seen) != len(preview) or not seen or not seen <= ids:
        return False
    if full:
        return not raw.get("answers_truncated", True) and seen == ids
    return (seen == ids and not raw.get("answers_truncated", True)
            or len(seen) < count and raw.get("answers_truncated") is True)


def _one_execution(executions: list[Any], graph_id: str) -> Any | None:
    matches = [execution for execution in executions
               if str(execution.graph.graph_id) == graph_id]
    if not matches or any(not _source_role_execution_safety_complete(execution) or _source_role_execution_safety_entity_answers(execution) is None
                          for execution in matches):
        return None
    states = {(_source_role_execution_safety_graph_state(execution.graph), tuple(sorted(_source_role_execution_safety_entity_answers(execution))))
              for execution in matches}
    return matches[0] if len(states) == 1 else None


def _anchor_paths(graph: Any, ontology: Any) -> dict[str, list[tuple[str, str, str]]] | None:
    """Require a fully known anchored tree and unique short path to its answer."""
    edges = [tuple(map(str, triple)) for triple in graph.triples]
    if len(set(edges)) != len(edges):
        return None
    adjacency: dict[str, list[tuple[str, tuple[str, str, str]]]] = {}
    for edge in edges:
        subject, relation, object_ = edge
        if subject == object_:
            return None
        if not all(_source_role_execution_safety_concrete_type(str(getattr(ontology, method)(relation) or ""))
                   for method in ("domain_for_relation", "range_for_relation")):
            return None
        adjacency.setdefault(subject, []).append((object_, edge))
        adjacency.setdefault(object_, []).append((subject, edge))
    anchors = _source_role_execution_safety_anchors(graph)
    if (not anchors or not all(_source_role_execution_safety_ENTITY.fullmatch(anchor) for anchor in anchors)
            or str(graph.answer_var) not in adjacency
            or any(len(adjacency.get(anchor, [])) != 1 for anchor in anchors)):
        return None
    # The entire graph must be a connected tree; an unused side branch cannot
    # be silently dropped when preserving the captured source execution.
    answer = str(graph.answer_var)
    parents = {answer: ("", None)}
    pending = [answer]
    while pending:
        node = pending.pop()
        for neighbor, edge in adjacency[node]:
            if neighbor == parents[node][0]:
                continue
            if neighbor in parents:
                return None
            parents[neighbor] = (node, edge)
            pending.append(neighbor)
    if len(parents) != len(adjacency) or len(edges) != len(adjacency) - 1:
        return None
    paths = {}
    used = set()
    for anchor in anchors:
        path, node = [], anchor
        while node != answer:
            node, edge = parents[node]
            path.append(edge)
            used.add(edge)
        if not 1 <= len(path) <= 3:
            return None
        paths[anchor] = path
    return paths if used == set(edges) else None


def _endpoint_role(edge: tuple[str, str, str], anchor: str, ontology: Any) -> tuple[str, ...]:
    method = "domain_for_relation" if edge[0] == anchor else "range_for_relation"
    type_id = str(getattr(ontology, method)(edge[1]) or "")
    return _source_role_execution_safety_tokens(type_id.rsplit(".", 1)[-1]) if _source_role_execution_safety_concrete_type(type_id) else ()


def _source_role_execution_safety_semantic_graphs(value: Any):
    if isinstance(value, Mapping):
        if isinstance(value.get("anchors"), list) and isinstance(value.get("semantic_paths"), list):
            yield value
        else:
            for key in ("output", "semantic_graph", "source_semantic", "semantic_graphs"):
                if key in value:
                    yield from _source_role_execution_safety_semantic_graphs(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _source_role_execution_safety_semantic_graphs(item)
    else:
        graph = getattr(value, "semantic_graph", None)
        if graph is not None:
            yield from _source_role_execution_safety_semantic_graphs(graph)


def _semantic_support(edge: tuple[str, str, str], anchor: str, semantic_graphs: Any,
                      linked_context: Any) -> bool:
    names = {str(item.get("name") or item.get("label") or "").strip().casefold()
             for item in linked_context if isinstance(item, Mapping)
             and _source_role_execution_safety_canonicalize_answer_id(item.get("id", "")) == anchor}
    names.discard("")
    if len(names) != 1:
        return False
    direction = "forward" if edge[0] == anchor else "backward"
    for semantic in _source_role_execution_safety_semantic_graphs(semantic_graphs):
        references = {str(item.get("id")) for item in semantic["anchors"]
                      if isinstance(item, Mapping) and str(item.get("surface", "")).strip().casefold() in names}
        for path in semantic["semantic_paths"]:
            if not isinstance(path, Mapping) or str(path.get("anchor_ref")) not in references:
                continue
            steps = path.get("steps")
            if not isinstance(steps, list) or not steps or not isinstance(steps[0], Mapping):
                continue
            first = steps[0]
            if (str(first.get("from")) == str(path.get("anchor_ref"))
                    and str(first.get("direction")) == direction
                    and _source_role_execution_safety_relation_id_from_label(first.get("relation_label")) == edge[1]):
                return True
    return False


def _context_role_support(role: tuple[str, ...], anchor: str, context: Any) -> bool:
    matches = [item for item in context if isinstance(item, Mapping)
               and _source_role_execution_safety_canonicalize_answer_id(item.get("id", "")) == anchor]
    return (len(matches) == 1 and isinstance(matches[0].get("types"), list)
            and role in {_source_role_execution_safety_tokens(str(value).replace("_", " "))
                         for value in matches[0]["types"]})


def _source_role_execution_safety_decomposition_lines(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _source_role_execution_safety_decomposition_lines(item)
    elif isinstance(value, Mapping):
        for key in ("decomposition", "final_decomposition", "final_candidates", "candidates"):
            if key in value:
                yield from _source_role_execution_safety_decomposition_lines(value[key])


def _last_who_role(value: str) -> str:
    clauses = re.findall(r"\bwho\s+(?:is|are|was|were)\s+(?:the\s+)?"
                         r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)",
                         str(value), re.I)
    return " ".join(_source_role_execution_safety_tokens(" ".join(clauses[-1]))) if clauses else ""


def _semantic_path_nodes(candidate: Any, semantic: Mapping[str, Any], path: Mapping[str, Any],
                         linked_context: Any) -> set[str] | None:
    anchors = {str(item.get("id")): str(item.get("surface", "")).strip().casefold()
               for item in semantic["anchors"] if isinstance(item, Mapping)}
    name = anchors.get(str(path.get("anchor_ref")), "")
    constants = {_source_role_execution_safety_canonicalize_answer_id(item.get("id", "")) for item in linked_context
                 if isinstance(item, Mapping)
                 and str(item.get("name") or item.get("label") or "").strip().casefold() == name}
    constants &= _source_role_execution_safety_anchors(candidate.graph)
    if len(constants) != 1:
        return None
    states = [(constant, {str(path.get("anchor_ref")): constant}) for constant in constants]
    edges = [tuple(map(str, triple)) for triple in candidate.graph.triples]
    graph_anchors = _source_role_execution_safety_anchors(candidate.graph)
    for step in path.get("steps", []):
        if not isinstance(step, Mapping):
            return None
        relation = _source_role_execution_safety_relation_id_from_label(step.get("relation_label"))
        forward = step.get("direction") == "forward"
        if not relation or step.get("direction") not in {"forward", "backward"}:
            return None
        next_states = []
        for node, bindings in states:
            if bindings.get(str(step.get("from"))) != node:
                continue
            for edge in edges:
                target = edge[2 if forward else 0]
                reference = str(step.get("to"))
                if edge[1] != relation or edge[0 if forward else 2] != node:
                    continue
                if (reference not in bindings and target in bindings.values()
                        or target in graph_anchors):
                    continue
                if reference in bindings and bindings[reference] != target:
                    continue
                next_states.append((target, {**bindings, reference: target}))
        states = next_states
        if not states:
            return set()
    return {node for node, _ in states}


def _semantic_path_matches(candidate: Any, semantic: Mapping[str, Any], path: Mapping[str, Any],
                           linked_context: Any) -> bool:
    return _semantic_path_nodes(candidate, semantic, path, linked_context) == {str(candidate.graph.answer_var)}


def _semantic_who_projection(question: str, selected: Any, executions: list[Any], ontology: Any,
                             semantic_graphs: Any, decompositions: Any, linked_context: Any,
                             evidence: dict[str, Any]) -> tuple[Any | None, str]:
    """Use corroborated terminal role, never infer a type from 'Who' alone."""
    if not re.match(r"^\s*who\b", str(question), re.I):
        return None, "not_who_question_with_corroborated_terminal_role"
    if not isinstance(linked_context, (list, tuple)):
        return None, "linked_entity_context_missing_for_semantic_projection"
    supported_roles = {_last_who_role(line) for line in _source_role_execution_safety_decomposition_lines(decompositions)} - {""}
    if not supported_roles:
        return None, "reviewed_decomposition_terminal_role_missing"
    projection = [tuple(map(str, triple)) for triple in selected.graph.triples
                  if str(triple[2]) == str(selected.graph.answer_var)]
    if len(projection) != 1 or not _source_role_execution_safety_complete(selected) or _source_role_execution_safety_entity_answers(selected) is None:
        return None, "selected_projection_or_entity_denotation_unqualified"
    source_edge = projection[0]
    context = Counter(tuple(map(str, triple)) for triple in selected.graph.triples
                      if tuple(map(str, triple)) != source_edge)
    operators, anchors = _source_role_execution_safety_substantive_operators(selected.graph), _source_role_execution_safety_anchors(selected.graph)
    if not context or operators is None or not anchors:
        return None, "source_context_anchor_or_operator_missing"
    source_owner_type = str(ontology.domain_for_relation(source_edge[1]) or "")
    if not _source_role_execution_safety_concrete_type(source_owner_type):
        return None, "source_projection_owner_schema_unknown_or_generic"
    requests, observed_requests, excluded_requests, unknown_requests = [], set(), [], []
    for semantic in _source_role_execution_safety_semantic_graphs(semantic_graphs):
        for path in semantic["semantic_paths"]:
            if not isinstance(path, Mapping):
                continue
            steps = path.get("steps")
            role = _last_who_role(str(path.get("goal", "")))
            if role not in supported_roles or not isinstance(steps, list) or not steps:
                continue
            last = steps[-1]
            if not isinstance(last, Mapping) or last.get("direction") != "forward":
                continue
            relation = _source_role_execution_safety_relation_id_from_label(last.get("relation_label"))
            # Exact ontology predicate leaf + owner phrase. No paraphrase or
            # synonym list maps 'rules' to 'politician' or 'current president'.
            parts = relation.split(".")
            canonical_role = " ".join(_source_role_execution_safety_tokens(" ".join(parts[-1:] + parts[-2:-1])))
            target_type = str(ontology.range_for_relation(relation) or "")
            if role != canonical_role:
                continue
            observed_requests.add(relation)
            requested_owner_type = str(ontology.domain_for_relation(relation) or "")
            prefix = {**path, "steps": steps[:-1], "path_output_var": str(last.get("from", ""))}
            owner_nodes = _semantic_path_nodes(selected, semantic, prefix, linked_context)
            detail = {"relation_id": relation, "requested_owner_type": requested_owner_type,
                      "source_projection_owner": source_edge[0], "source_owner_type": source_owner_type,
                      "semantic_owner_nodes": sorted(owner_nodes) if owner_nodes is not None else None}
            # Scope role obligations to the *actual* source projection owner.
            # Different cached type names are not asserted to be disjoint;
            # a known prefix binding to another graph variable is the evidence
            # that a different branch's terminal cannot replace this field.
            if owner_nodes and source_edge[0] not in owner_nodes:
                excluded_requests.append({**detail, "reason": "semantic_terminal_bound_to_other_actual_owner"})
                continue
            if (owner_nodes != {source_edge[0]} or not _source_role_execution_safety_concrete_type(requested_owner_type)
                    or not _source_role_execution_safety_concrete_type(target_type)):
                unknown_requests.append({**detail, "reason": "semantic_owner_or_terminal_schema_unproven"})
                continue
            if requested_owner_type != source_owner_type:
                unknown_requests.append({**detail, "reason": "same_variable_declared_owner_signature_not_identical"})
                continue
            requests.append((relation, target_type, semantic, path))
    unique = {(relation, target_type) for relation, target_type, _, _ in requests}
    evidence["all_corroborated_terminal_relation_ids"] = sorted(observed_requests)
    evidence["corroborated_terminal_relation_ids"] = sorted({relation for relation, _ in unique})
    evidence["projection_owner_type"] = source_owner_type
    evidence["projection_owner_variable"] = source_edge[0]
    evidence["excluded_other_owner_role_requests"] = list({_source_role_execution_safety_stable(value): value for value in excluded_requests}.values())
    evidence["unproven_owner_role_requests"] = list({_source_role_execution_safety_stable(value): value for value in unknown_requests}.values())
    if unknown_requests:
        return None, "requested_role_owner_evidence_incomplete_or_ambiguous"
    if len(unique) != 1:
        return None, "semantic_terminal_role_missing_or_ambiguous"
    relation, target_type = next(iter(unique))
    status, source_schema = _definite_role_conflict(selected.graph, {target_type}, ontology)
    evidence["semantic_source_schema"] = source_schema
    if status != "conflict":
        return None, "selected_terminal_role_compatible_or_unknown"
    eligible, uncertain = [], []
    for candidate in executions:
        if candidate is selected or _source_role_execution_safety_entity_answers(candidate) is None:
            continue
        graph = candidate.graph
        if (_source_role_execution_safety_anchors(graph) != anchors or graph.answer_var != selected.graph.answer_var
                or _source_role_execution_safety_substantive_operators(graph) != operators):
            continue
        edges = [tuple(map(str, triple)) for triple in graph.triples]
        terminal = [edge for edge in edges if edge[2] == str(graph.answer_var)]
        if (len(terminal) != 1 or terminal[0] != (source_edge[0], relation, source_edge[2])
                or Counter(edge for edge in edges if edge != terminal[0]) != context):
            continue
        if not any(_semantic_path_matches(candidate, semantic, path, linked_context)
                   for requested, _, semantic, path in requests if requested == relation):
            continue
        candidate_status, _ = _source_role_execution_safety_schema_status(graph, {target_type}, ontology)
        if candidate_status != "compatible" or not _source_role_execution_safety_complete(candidate):
            uncertain.append(str(graph.graph_id))
        else:
            eligible.append(candidate)
    evidence["semantic_projection_eligible_graph_ids"] = [str(candidate.graph.graph_id) for candidate in eligible]
    evidence["semantic_projection_uncertain_graph_ids"] = uncertain
    denotations = {tuple(sorted(_source_role_execution_safety_entity_answers(candidate))) for candidate in eligible}
    if uncertain or len(denotations) != 1:
        return None, "semantic_projection_denotation_unknown_missing_or_ambiguous"
    # This restores a declared endpoint, not an implicit latest/current claim.
    # In particular an ARGMAX candidate cannot bypass operator equality above.
    replacement = eligible[0]
    evidence.update(requested_terminal_relation_id=relation,
                    requested_terminal_type=target_type,
                    preserved_context_edge_count=sum(context.values()),
                    preserved_source_operator_count=len(operators),
                    cardinality_expansion_permitted_only_for_corroborated_wrong_role=True,
                    implicit_temporal_ordering_added=False)
    return replacement, "corroborated_same_context_semantic_terminal_role"


def _source_override(prior: Mapping[str, Any], final_id: str) -> str:
    reasons, gate = prior.get("reason_codes"), prior.get("path_alignment_gate")
    if (not isinstance(reasons, (list, tuple)) or "path_alignment_gate" not in reasons
            or not isinstance(gate, Mapping) or gate.get("status") != "applied"
            or str(prior.get("selected_graph_id", "")) != final_id):
        return ""
    source = str(prior.get("path_alignment_gate_source_graph_id") or "")
    proposal = gate.get("proposal")
    if (not source or source == final_id or not isinstance(proposal, Mapping)
            or str(proposal.get("source_graph_id", "")) != source
            or str(proposal.get("graph_id", "")) != final_id):
        return ""
    return source


def _restore_explicit_role(question: str, selected: Any, executions: list[Any], ontology: Any,
                           prior: Any, semantic_graphs: Any, linked_context: Any,
                           evidence: dict[str, Any]) -> tuple[Any | None, str]:
    if not isinstance(prior, Mapping):
        return None, "captured_prior_selection_missing"
    source_id = _source_override(prior, str(selected.graph.graph_id))
    if not source_id:
        return None, "not_proven_path_alignment_override"
    evidence["captured_source_graph_id"] = source_id
    if (prior.get("answer_union_graph_ids")
            or isinstance(getattr(selected.graph, "provenance", None), Mapping)
            and selected.graph.provenance.get("selection_answer_union")):
        return None, "captured_or_current_union_not_single_execution"
    replacement = _one_execution(executions, source_id)
    if replacement is None or not _source_role_execution_safety_complete(selected) or _source_role_execution_safety_entity_answers(selected) is None:
        return None, "source_or_final_execution_incomplete_or_ambiguous"
    captured = prior.get("candidates")
    if not isinstance(captured, list):
        return None, "captured_candidate_state_missing"
    rows = [[item for item in captured if isinstance(item, Mapping)
             and str(item.get("graph_id")) == graph_id]
            for graph_id in (source_id, str(selected.graph.graph_id))]
    if not all(len(matches) == 1 for matches in rows):
        return None, "captured_choice_identity_missing_or_ambiguous"
    if (not _captured_matches(rows[0][0], replacement, full=True)
            or not _captured_matches(rows[1][0], selected, full=False)):
        return None, "captured_and_current_execution_state_conflict"
    if not isinstance(linked_context, (list, tuple)):
        return None, "linked_entity_context_missing"
    words = _question_words(question, linked_context)
    if words is None:
        return None, "question_quotation_ambiguous"
    before = _anchor_paths(replacement.graph, ontology)
    after = _anchor_paths(selected.graph, ontology)
    if before is None or after is None or set(before) != set(after):
        return None, "anchor_or_fully_known_tree_context_not_preserved"
    role_losses = []
    for anchor, path in before.items():
        other = after[anchor]
        role = _endpoint_role(path[0], anchor, ontology)
        alternative_role = _endpoint_role(other[0], anchor, ontology)
        if role != alternative_role:
            if (_role_mentioned(role, words) and not _role_mentioned(alternative_role, words)
                    and _semantic_support(path[0], anchor, semantic_graphs, linked_context)):
                role_losses.append((anchor, role, alternative_role))
            else:
                # Constraint anchors must retain their full requested type,
                # or their exact common ontology leaf role (e.g. genre). The
                # actual cached source execution remains the binding authority.
                if (len(path) != 1 or len(other) != 1
                        or not role or not alternative_role or role[-1] != alternative_role[-1]
                        or not (_semantic_support(path[0], anchor, semantic_graphs, linked_context)
                                or _semantic_support(other[0], anchor, semantic_graphs, linked_context))
                        or not _context_role_support(role, anchor, linked_context)
                        or not _context_role_support(alternative_role, anchor, linked_context)):
                    return None, "unproven_changed_constraint_anchor_role"
        elif (len(path) != len(other)
              or [edge[1] for edge in path] != [edge[1] for edge in other]):
            return None, "same_role_anchor_context_changed"
    if len(role_losses) != 1:
        return None, "no_unique_question_and_semantic_supported_role_loss"
    anchor, role, alternative_role = role_losses[0]
    if len(before[anchor]) != 1:
        return None, "captured_requested_role_not_direct_answer_path"
    original_ops = _source_role_execution_safety_substantive_operators(replacement.graph)
    final_ops = _source_role_execution_safety_substantive_operators(selected.graph)
    if original_ops is None or final_ops is None or not original_ops <= final_ops:
        return None, "captured_source_operator_not_preserved_by_override"
    # Newly generated self-exclusion may be dropped only when the complete
    # restored denotation already satisfies it. Other added constraints abstain.
    for operator in selected.graph.operators:
        if _source_role_execution_safety_stable(operator) in original_ops:
            continue
        excluded = _source_role_execution_safety_canonicalize_answer_id(operator.get("_value_entity_id", ""))
        if (str(operator.get("type", "")).upper() != "NO_EQUAL"
                or str(operator.get("input_var", "")) != str(selected.graph.answer_var)
                or str(operator.get("value_type", "")) != "entity"
                or operator.get("inputs") or operator.get("attribute_relation_label")
                or operator.get("attribute_relation_labels")
                or excluded not in before or excluded in _source_role_execution_safety_entity_answers(replacement)):
            return None, "substantive_added_constraint_not_attested_by_source_answers"
    evidence.update(requested_anchor_id=anchor, ontology_source_role=list(role),
                    ontology_override_role=list(alternative_role),
                    preserved_anchor_ids=sorted(before),
                    preserved_source_operator_count=len(original_ops),
                    captured_override_source_restored=True)
    return replacement, "restore_captured_source_requested_relation_role"


def _head_projection(question: str, selected: Any, executions: list[Any], ontology: Any,
                     evidence: dict[str, Any]) -> tuple[Any | None, str]:
    head = _source_role_execution_safety_source_entity_head(question)
    evidence["head"] = head
    if not head:
        return None, "not_strict_ontology_noun_head"
    expected = set(_source_role_execution_safety_type_index(ontology).get(head, ()))
    evidence["expected_types"] = sorted(expected)
    if not expected:
        return None, "head_not_in_ontology_vocabulary"
    if _source_role_execution_safety_entity_answers(selected) is None or not _source_role_execution_safety_complete(selected):
        return None, "selected_entity_denotation_incomplete_or_unproven"
    status, schema = _definite_role_conflict(selected.graph, expected, ontology)
    evidence["selected_schema"] = schema
    if status != "conflict":
        return None, "selected_head_schema_compatible_or_unknown"
    incoming = [tuple(map(str, triple)) for triple in selected.graph.triples
                if str(triple[2]) == str(selected.graph.answer_var)]
    if len(incoming) != 1:
        return None, "selected_terminal_projection_not_unique"
    projection = incoming[0]
    # Preserve every source operator; variable names are actual graph bindings,
    # not semantic equivalence evidence across arbitrary new graphs.
    operators = _source_role_execution_safety_substantive_operators(selected.graph)
    anchors = _source_role_execution_safety_anchors(selected.graph)
    context = Counter(tuple(map(str, triple)) for triple in selected.graph.triples
                      if tuple(map(str, triple)) != projection)
    if not context or not anchors or operators is None:
        return None, "source_context_anchor_or_operator_missing"
    compatible, unknown = [], []
    for candidate in executions:
        if candidate is selected or _source_role_execution_safety_entity_answers(candidate) is None:
            continue
        graph = candidate.graph
        if _source_role_execution_safety_anchors(graph) != anchors or _source_role_execution_safety_substantive_operators(graph) != operators:
            continue
        edges = [tuple(map(str, triple)) for triple in graph.triples]
        terminal = [edge for edge in edges if edge[2] == str(graph.answer_var)]
        if len(terminal) != 1 or terminal[0][0] != projection[0]:
            continue
        remainder = Counter(edge for edge in edges if edge != terminal[0])
        if remainder != context:
            continue
        candidate_status, candidate_schema = _source_role_execution_safety_schema_status(graph, expected, ontology)
        if candidate_status == "unknown" or candidate_status == "compatible" and not _source_role_execution_safety_complete(candidate):
            unknown.append(str(graph.graph_id))
        elif candidate_status == "compatible":
            compatible.append((candidate, candidate_schema))
    evidence["compatible_graph_ids"] = [str(item.graph.graph_id) for item, _ in compatible]
    evidence["unknown_or_incomplete_graph_ids"] = unknown
    denotations = {tuple(sorted(_source_role_execution_safety_entity_answers(item))) for item, _ in compatible}
    if unknown or len(denotations) != 1:
        return None, "projection_denotation_unknown_missing_or_ambiguous"
    replacement, schema = compatible[0]
    if len(replacement.answer_ids) > len(selected.answer_ids):
        return None, "projection_role_correction_would_expand_cardinality"
    evidence.update(replacement_schema=schema,
                    preserved_context_edge_count=sum(context.values()),
                    preserved_source_operator_count=len(operators))
    return replacement, "unique_same_context_ontology_head_projection"


def source_role_challenger(
    question: str, selected: Any, executions: list[Any], ontology: Any, *,
    prior_selection: Mapping[str, Any] | None = None, semantic_graphs: Any = None,
    decompositions: Any = None, linked_entity_context: Any = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Return an existing source execution when local role-loss is proven.

    ``semantic_graphs`` is source Semantic output, not new model output. The
    optional decomposition is accepted for the caller's common context API but
    is not treated as independent proof of the question's answer role. No Gold,
    IDs, relation/head synonym list, score threshold, or external calls are used.
    """
    evidence = {"status": "not_applied", "source_graph_id": str(selected.graph.graph_id),
                "uses_existing_executions_only": True, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "vocabulary_source": "cached_ontology_endpoint_type_leaf",
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0,
                "decomposition_alone_is_not_role_proof": True}
    try:
        if (isinstance(getattr(selected.graph, "provenance", None), Mapping)
                and selected.graph.provenance.get("selection_answer_union")):
            evidence["reason"] = "current_union_scope_requires_independent_binding_proof"
            return None, evidence
        replacement, override_reason = _restore_explicit_role(
            question, selected, executions, ontology, prior_selection,
            semantic_graphs, linked_entity_context, evidence)
        evidence["captured_override_reason"] = override_reason
        if replacement is None:
            # A captured override with a failed obligation must not be rescued
            # through a weaker, unrelated projection heuristic.
            if evidence.get("captured_source_graph_id"):
                evidence["reason"] = override_reason
                return None, evidence
            replacement, reason = _head_projection(question, selected, executions, ontology, evidence)
            if replacement is None:
                evidence["head_projection_reason"] = reason
                replacement, semantic_reason = _semantic_who_projection(
                    question, selected, executions, ontology, semantic_graphs,
                    decompositions, linked_entity_context, evidence)
                evidence["semantic_projection_reason"] = semantic_reason
                if replacement is not None:
                    reason = semantic_reason
        else:
            reason = override_reason
        evidence["reason"] = reason
        if replacement is not None:
            if _source_role_execution_safety_entity_answers(replacement) == _source_role_execution_safety_entity_answers(selected):
                evidence["reason"] = "source_role_denotation_already_selected"
                return None, evidence
            evidence.update(status="applied", selected_graph_id=str(replacement.graph.graph_id),
                            selected_answer_count=len(replacement.answer_ids))
        return replacement, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="schema_or_execution_evidence_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence



# ---- mediator_terminal_role_scope ----
"""Pure scope gate for an existing mediator's outgoing projection fields.

No queries, embeddings, models, or answer choices are made. Only a complete
existing field pool with one concrete requested target range is narrowed. The
actual CVT variable and every original graph constraint remain untouched.
Shared schema ancestors are not type-disjointness or target-role evidence.
"""


from collections.abc import Mapping
import re
from typing import Any

_mediator_terminal_role_scope_concrete_type = _nested_name_schema_guard_concrete_type
_mediator_terminal_role_scope_type_index = _nested_name_schema_guard_type_index
_mediator_terminal_role_scope_decomposition_lines = _source_role_execution_safety_decomposition_lines
_mediator_terminal_role_scope_tokens = _source_role_execution_safety_tokens
_mediator_terminal_role_scope_source_entity_head = _source_role_execution_safety_source_entity_head


_WH_FIELD = re.compile(r"\b(?:what|which)\s+(?:is|was|are|were)\s+(?:the\s+)?"
                       r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)", re.I)


def _mediator_terminal_role_scope_phrase(value: str) -> str:
    return " ".join(_mediator_terminal_role_scope_tokens(str(value).replace("_", " ")))


def mediator_terminal_role_scope(
    question: str, source_graph: Any, mediator_type: str, candidate_relations: list[str], ontology: Any,
    *, decompositions: Any = None,
) -> tuple[list[str] | None, dict[str, Any]]:
    """Return a strict one-field subset, or None to preserve the existing lane.

    Call before the existing BGE/projection loop. The caller must already have
    established its ordinary mediator eligibility. An exact current-question
    noun head, original decomposition's same-owner field declaration, known
    actual incoming CVT binding, and a complete cached schema field pool are
    all required. Unknown schema and multiple matching fields abstain.
    """
    evidence = {"status": "not_applied", "source_graph_id": str(source_graph.graph_id),
                "actual_mediator_variable": str(source_graph.answer_var),
                "actual_mediator_type": str(mediator_type),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "changes_source_graph_or_operators": False, "selects_answer_ids": False,
                "schema_sibling_disjointness_claimed": False}

    def abstain(reason):
        evidence["reason"] = reason
        return None, evidence

    try:
        head = _mediator_terminal_role_scope_source_entity_head(question)
        evidence["question_head"] = head
        expected = set(_mediator_terminal_role_scope_type_index(ontology).get(head, ())) if head else set()
        evidence["expected_concrete_target_types"] = sorted(expected)
        if not expected:
            return abstain("question_head_unknown_compound_modifier_or_not_entity_noun")
        if not _mediator_terminal_role_scope_concrete_type(str(mediator_type)):
            return abstain("actual_mediator_type_unknown_or_generic")
        provenance = getattr(source_graph, "provenance", None)
        if not isinstance(provenance, Mapping) or provenance.get("selection_answer_union"):
            return abstain("source_current_provenance_missing_or_union_scope_unproven")
        if provenance.get("original_question") is not None and str(provenance["original_question"]).strip() != str(question).strip():
            return abstain("current_question_and_source_provenance_conflict")
        incoming = [str(relation) for _, relation, object_ in source_graph.triples
                    if str(object_) == str(source_graph.answer_var)]
        types = {str(ontology.range_for_relation(relation) or "") for relation in incoming}
        evidence["actual_incoming_range_types"] = sorted(types)
        if not incoming or types != {str(mediator_type)}:
            return abstain("actual_mediator_node_binding_unknown_or_not_unique")
        if (not isinstance(candidate_relations, list) or not candidate_relations
                or not all(isinstance(value, str) and value for value in candidate_relations)
                or len(set(candidate_relations)) != len(candidate_relations)):
            return abstain("candidate_field_pool_missing_or_malformed")
        lookup = getattr(ontology, "relations_for_domain", None)
        if not callable(lookup):
            return abstain("complete_cached_mediator_field_pool_unavailable")
        complete, unknown = [], []
        for relation in lookup(str(mediator_type)):
            relation = str(relation)
            domain = str(ontology.domain_for_relation(relation) or "")
            range_ = str(ontology.range_for_relation(relation) or "")
            if domain != str(mediator_type) or not range_:
                unknown.append(relation)
            elif not range_.startswith("type."):
                complete.append(relation)
        evidence["complete_cached_entity_projection_fields"] = sorted(complete)
        if unknown:
            evidence["unknown_schema_fields"] = sorted(unknown)
            return abstain("cached_outgoing_field_schema_unknown_or_wrong_owner")
        if set(complete) != set(candidate_relations):
            return abstain("candidate_projection_pool_not_complete_cached_schema_pool")
        compatible, uncertain = [], []
        schema = []
        expected_ancestors = {str(value) for type_id in expected for value in ontology.supertypes(type_id)}
        for relation in candidate_relations:
            domain = str(ontology.domain_for_relation(relation) or "")
            range_ = str(ontology.range_for_relation(relation) or "")
            if domain != str(mediator_type) or not _mediator_terminal_role_scope_concrete_type(range_):
                uncertain.append(relation)
                schema.append({"relation_id": relation, "domain": domain, "range": range_, "status": "unknown"})
                continue
            closure = {range_, *(str(value) for value in ontology.supertypes(range_))}
            status = "positive_target_range" if closure & expected else "different_declared_role"
            if status == "positive_target_range":
                compatible.append(relation)
            elif range_ in expected_ancestors:
                uncertain.append(relation)
                status = "generic_expected_ancestor_not_role_proof"
            schema.append({"relation_id": relation, "domain": domain, "range": range_, "status": status})
        evidence.update(candidate_schema=schema, compatible_projection_fields=compatible,
                        uncertain_projection_fields=uncertain)
        if uncertain or len(compatible) != 1:
            return abstain("target_projection_field_unknown_missing_or_multiple")
        selected = compatible[0]
        leaf, owner = selected.rsplit(".", 1)[-1], str(mediator_type).rsplit(".", 1)[-1]
        declared = {(_mediator_terminal_role_scope_phrase(property_), _mediator_terminal_role_scope_phrase(owner_)) for line in _mediator_terminal_role_scope_decomposition_lines(
            decompositions if decompositions is not None else provenance.get("decomposition"))
            for property_, owner_ in _WH_FIELD.findall(line)}
        requested = (_mediator_terminal_role_scope_phrase(leaf), _mediator_terminal_role_scope_phrase(owner))
        evidence["expected_same_owner_decomposition_declaration"] = requested
        if requested not in declared:
            return abstain("original_decomposition_same_mediator_target_field_unproven")
        if len(candidate_relations) == 1:
            return abstain("existing_projection_field_already_unique")
        evidence.update(status="applied", reason="unique_exact_question_and_declared_mediator_target_range",
                        allowed_projection_fields=[selected],
                        excluded_projection_fields=[relation for relation in candidate_relations if relation != selected],
                        preserved_original_constraint_count=len(source_graph.operators),
                        preserved_source_mediator_variable=str(source_graph.answer_var),
                        no_new_projection_field=True)
        return [selected], evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="current_mediator_or_schema_context_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence



# ---- zero_graph_role_safety ----
"""Pure veto of a late challenger that only changes a requested answer role.

Cached schema metadata is role evidence, not proof that sibling Freebase types
are disjoint. In particular no agent/non-agent root or movie/film synonym is
used. An opaque question head needs a uniquely quoted obligation whose original
Semantic and decomposition explicitly corroborate the source terminal role.
"""


from collections import Counter
from collections.abc import Mapping
import json
import re
from types import SimpleNamespace
from typing import Any

from .evaluation import canonicalize_answer_id as _zero_graph_role_safety_canonicalize_answer_id
_zero_graph_role_safety_anchors = _nested_name_schema_guard_anchors
_zero_graph_role_safety_complete = _nested_name_schema_guard_complete
_zero_graph_role_safety_concrete_type = _nested_name_schema_guard_concrete_type
_zero_graph_role_safety_entity_answers = _nested_name_schema_guard_entity_answers
_zero_graph_role_safety_schema_status = _nested_name_schema_guard_schema_status
_zero_graph_role_safety_singular = _nested_name_schema_guard_singular
_zero_graph_role_safety_substantive_operators = _nested_name_schema_guard_substantive_operators
_zero_graph_role_safety_type_index = _nested_name_schema_guard_type_index
from .ontology import relation_id_from_label as _zero_graph_role_safety_relation_id_from_label
_zero_graph_role_safety_decomposition_lines = _source_role_execution_safety_decomposition_lines
_zero_graph_role_safety_semantic_graphs = _source_role_execution_safety_semantic_graphs
_zero_graph_role_safety_tokens = _source_role_execution_safety_tokens


_zero_graph_role_safety_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_zero_graph_role_safety_HEAD = re.compile(r"^\s*(?:what|which)\s+(?:(?:is|was|are|were)\s+)?"
                   r"(?:the\s+)?([A-Za-z]+)\b", re.I)
_WH_ROLE = re.compile(r"\b(?:what|which)\s+(?:is|are|was|were)\s+(?:the\s+)?"
                      r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)", re.I)
_QUOTED = re.compile(r'"([^"\n]+)"')


def _zero_graph_role_safety_phrase(value: str) -> str:
    return " ".join(_zero_graph_role_safety_tokens(str(value).replace("_", " ")))


def _type_role(type_id: str) -> str:
    """Remove only a repeated namespace prefix, not arbitrary type modifiers."""
    namespace, _, leaf = str(type_id).rpartition(".")
    words, prefix = _zero_graph_role_safety_tokens(leaf.replace("_", " ")), _zero_graph_role_safety_tokens(namespace.rsplit(".", 1)[-1])
    if prefix and words[:len(prefix)] == prefix and len(words) > len(prefix):
        words = words[len(prefix):]
    return " ".join(words)


def _last_role(line: str) -> tuple[str, str] | None:
    values = _WH_ROLE.findall(str(line))
    return (_zero_graph_role_safety_phrase(values[-1][0]), _zero_graph_role_safety_phrase(values[-1][1])) if values else None


def _projection(graph: Any) -> tuple[str, str, str] | None:
    triples = getattr(graph, "triples", None)
    if (not isinstance(triples, list) or not triples
            or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in triples)):
        return None
    incident = [tuple(map(str, row)) for row in triples
                if str(row[0]) == str(graph.answer_var) or str(row[2]) == str(graph.answer_var)]
    return incident[0] if len(incident) == 1 and incident[0][2] == str(graph.answer_var) else None


def _question_head(question: str) -> str:
    match = _zero_graph_role_safety_HEAD.match(str(question))
    if not match:
        return ""
    head = _zero_graph_role_safety_singular(match.group(1))
    # Do not treat an apostrophe, hyphen or digit as a head-word boundary.
    tail = str(question)[match.end():]
    if tail and not tail[0].isspace() and tail[0] not in "?,.;:!":
        return ""
    return head


def _quoted_obligation(question: str, semantic: Mapping[str, Any], path: Mapping[str, Any],
                       goal: str, ontology: Any) -> tuple[str, str] | None:
    if str(question).count('"') != 2:
        return None
    quoted = _QUOTED.findall(str(question))
    if len(quoted) != 1 or quoted[0] not in str(goal):
        return None
    anchors = {str(anchor.get("id")): str(anchor.get("surface", ""))
               for anchor in semantic["anchors"] if isinstance(anchor, Mapping)}
    if anchors.get(str(path.get("anchor_ref"))) != quoted[0]:
        return None
    steps = path.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[0], Mapping):
        return None
    first = steps[0]
    relation = _zero_graph_role_safety_relation_id_from_label(first.get("relation_label"))
    field = _zero_graph_role_safety_phrase(relation.rsplit(".", 1)[-1])
    # The phrase must be the requested field, not an unrelated entity title.
    # Missing original field metadata is retained only as textual role context;
    # the separate complete-literal lane requires its actual known text schema.
    question_without_quote = _zero_graph_role_safety_phrase(_QUOTED.sub(" ", str(question)))
    datatype = str(ontology.range_for_relation(relation) or "")
    if (first.get("direction") != "backward"
            or str(first.get("from")) != str(path.get("anchor_ref"))
            or datatype and datatype not in {"type.text", "type.rawstring"}
            or not field or not re.search(r"(?<!\w)" + re.escape(field) + r"(?!\w)", question_without_quote)):
        return None
    return quoted[0], relation


def _terminal_role_evidence(question: str, source_edge: tuple[str, str, str], ontology: Any,
                            semantic_context: Any, decomposition_context: Any):
    source_type = str(ontology.range_for_relation(source_edge[1]) or "")
    if not _zero_graph_role_safety_concrete_type(source_type):
        return None, {"reason": "source_terminal_schema_unknown_or_generic"}
    source_role = _type_role(source_type)
    head = _question_head(question)
    if not head:
        return None, {"reason": "question_not_single_entity_wh_request"}
    known_head_types = set(_zero_graph_role_safety_type_index(ontology).get(head, ()))
    direct_head = source_role == head
    source_graph = type("RoleGraph", (), {"triples": [source_edge], "answer_var": source_edge[2]})()
    source_status, source_schema = _zero_graph_role_safety_schema_status(source_graph, known_head_types, ontology)
    if known_head_types and source_status != "compatible":
        return None, {"reason": "known_question_head_does_not_support_source_role",
                      "head": head, "source_schema": source_schema}
    decomposition_roles = {_last_role(line) for line in _zero_graph_role_safety_decomposition_lines(decomposition_context)} - {None}
    witnesses = []
    for semantic in _zero_graph_role_safety_semantic_graphs(semantic_context):
        for path in semantic["semantic_paths"]:
            if not isinstance(path, Mapping):
                continue
            steps, goal = path.get("steps"), str(path.get("goal", ""))
            role = _last_role(goal)
            if not isinstance(steps, list) or not steps or role not in decomposition_roles:
                continue
            last = steps[-1]
            if not isinstance(last, Mapping) or last.get("direction") != "forward":
                continue
            requested_relation = _zero_graph_role_safety_relation_id_from_label(last.get("relation_label"))
            parts = requested_relation.split(".")
            if len(parts) < 3:
                continue
            expected_role = _zero_graph_role_safety_phrase(parts[-1])
            expected_owner = _zero_graph_role_safety_phrase(parts[-2])
            if (role != (expected_role, expected_owner) or expected_role != source_role
                    or requested_relation.rsplit(".", 1)[0] != source_edge[1].rsplit(".", 1)[0]):
                continue
            # An exact schema head is direct evidence. An unknown head is not
            # translated: it instead needs the literal obligation chain above.
            obligation = _quoted_obligation(question, semantic, path, goal, ontology)
            if not direct_head and obligation is None:
                continue
            witnesses.append({"requested_role": expected_role, "requested_owner": expected_owner,
                              "requested_semantic_relation_id": requested_relation,
                              "quoted_obligation": obligation})
    signatures = {(row["requested_role"], row["requested_owner"], row["quoted_obligation"])
                  for row in witnesses}
    if len(signatures) != 1:
        return None, {"reason": "question_semantic_decomposition_role_missing_or_ambiguous",
                      "head": head, "source_terminal_role": source_role,
                      "role_witness_count": len(signatures)}
    return source_role, {"head": head, "source_terminal_type": source_type,
                         "source_terminal_role": source_role,
                         "question_head_ontology_known": bool(known_head_types),
                         "no_question_head_synonym_mapping": True,
                         "quoted_field_schema_proven": bool(witnesses[0]["quoted_obligation"]
                             and ontology.range_for_relation(witnesses[0]["quoted_obligation"][1])),
                         "does_not_assert_literal_constraint_satisfied": True,
                         "role_witness": witnesses[0]}


def zero_graph_role_regression(
    question: str, source: Any, challenger_graph: Any, challenger_answer_ids: Any,
    ontology: Any, semantic_context: Any, decomposition_context: Any,
) -> tuple[bool, dict[str, Any]]:
    """Veto a pure requested-terminal switch, without choosing/executing a graph.

    The source is the actual complete selected execution before the late gate.
    Semantic and decomposition must be existing original stage context. A true
    result protects its declared role only; it does not assert that all missing
    question constraints were answered or that schema sibling types are disjoint.
    """
    evidence = {"status": "not_applied", "source_graph_id": str(source.graph.graph_id),
                "challenger_graph_id": str(challenger_graph.graph_id),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "protects_declared_role_only": True, "schema_disjointness_claimed": False}
    try:
        if not _zero_graph_role_safety_complete(source) or _zero_graph_role_safety_entity_answers(source) is None:
            evidence["reason"] = "source_complete_entity_execution_not_attested"
            return False, evidence
        if (isinstance(getattr(source.graph, "provenance", None), Mapping)
                and source.graph.provenance.get("selection_answer_union")):
            evidence["reason"] = "source_union_binding_scope_unproven"
            return False, evidence
        if not isinstance(challenger_answer_ids, list) or not challenger_answer_ids:
            evidence["reason"] = "challenger_entity_answers_missing"
            return False, evidence
        challenger_ids = {_zero_graph_role_safety_canonicalize_answer_id(value) for value in challenger_answer_ids}
        if len(challenger_ids) != len(challenger_answer_ids) or not all(_zero_graph_role_safety_ENTITY.fullmatch(value) for value in challenger_ids):
            evidence["reason"] = "challenger_answers_not_unique_entities"
            return False, evidence
        before, after = _projection(source.graph), _projection(challenger_graph)
        if (before is None or after is None or before[1] == after[1]
                or before[0] != after[0] or before[2] != after[2]
                or str(source.graph.answer_var) != str(challenger_graph.answer_var)):
            evidence["reason"] = "not_single_same_owner_terminal_projection_change"
            return False, evidence
        source_context = Counter(tuple(map(str, row)) for row in source.graph.triples if tuple(map(str, row)) != before)
        challenger_context = Counter(tuple(map(str, row)) for row in challenger_graph.triples if tuple(map(str, row)) != after)
        source_ops, challenger_ops = _zero_graph_role_safety_substantive_operators(source.graph), _zero_graph_role_safety_substantive_operators(challenger_graph)
        if (not source_context or source_context != challenger_context
                or _zero_graph_role_safety_anchors(source.graph) != _zero_graph_role_safety_anchors(challenger_graph)
                or source_ops is None or source_ops != challenger_ops):
            evidence["reason"] = "challenger_changes_context_anchor_or_operator_obligations"
            return False, evidence
        source_owner = str(ontology.domain_for_relation(before[1]) or "")
        challenger_owner = str(ontology.domain_for_relation(after[1]) or "")
        if not _zero_graph_role_safety_concrete_type(source_owner) or source_owner != challenger_owner:
            evidence["reason"] = "terminal_owner_schema_unknown_or_changed"
            return False, evidence
        role, role_evidence = _terminal_role_evidence(
            question, before, ontology, semantic_context, decomposition_context)
        evidence.update(role_evidence)
        if role is None:
            return False, evidence
        alternative_type = str(ontology.range_for_relation(after[1]) or "")
        if not _zero_graph_role_safety_concrete_type(alternative_type):
            evidence["reason"] = "challenger_terminal_schema_unknown_or_generic"
            return False, evidence
        alternative_role = _type_role(alternative_type)
        source_type = role_evidence["source_terminal_type"]
        if (alternative_role == role or source_type in ontology.supertypes(alternative_type)
                or alternative_type in ontology.supertypes(source_type)):
            evidence["reason"] = "terminal_role_same_or_type_hierarchy_compatible"
            return False, evidence
        if challenger_ids <= _zero_graph_role_safety_entity_answers(source):
            evidence["reason"] = "challenger_denotation_does_not_add_different_role_answers"
            return False, evidence
        evidence.update(status="applied", reason="late_challenger_only_changes_corroborated_terminal_role",
                        challenger_terminal_type=alternative_type, challenger_terminal_role=alternative_role,
                        preserved_context_edge_count=sum(source_context.values()),
                        preserved_operator_count=len(source_ops),
                        preserved_terminal_owner_type=source_owner,
                        newly_satisfied_requested_constraint_count=0,
                        source_answer_count=len(source.answer_ids), challenger_answer_count=len(challenger_ids))
        return True, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="role_context_or_schema_error",
                        error=f"{type(exc).__name__}: {exc}")
        return False, evidence


_RDF_TERM = r'(?:\?[A-Za-z_]\w*|[A-Za-z_]\w*:[A-Za-z0-9_.]+|<[^<>\s]+>|"(?:[^"\\]|\\.)*"(?:@[A-Za-z-]+|\^\^(?:<[^<>\s]+>|[A-Za-z_]\w*:[A-Za-z0-9_.]+))?)'
_BGP_TRIPLE = re.compile(r"\s*(?P<s>" + _RDF_TERM + r")\s+(?P<p>" + _RDF_TERM
                         + r")\s+(?P<o>" + _RDF_TERM + r")\s*\.\s*")
_zero_graph_role_safety_VAR = re.compile(r"\?[A-Za-z_]\w*\Z")


def _compiled_template_bgp(execution: Any) -> tuple[Any | None, str]:
    """Parse a deliberately small, fully covered SELECT-DISTINCT BGP subset.

    Unknown constructs fail closed. A LIMIT is qualified only below its cap;
    existing row count plus a truncated capped result is not completeness proof.
    The parser never executes/lowers a query or imports a SPARQL/network package.
    """
    text = str(getattr(execution.graph, "sparql", "") or "")
    prefixes = {}
    while True:
        match = re.match(r"\s*PREFIX\s+([A-Za-z_]\w*):\s*<([^>]+)>\s*", text, re.I)
        if not match:
            break
        prefixes[match.group(1)] = match.group(2)
        text = text[match.end():]
    query = re.fullmatch(r"\s*SELECT\s+DISTINCT\s+(\?[A-Za-z_]\w*)\s+WHERE\s*\{([\s\S]*)\}"
                         r"\s*(?:LIMIT\s+(\d+))?\s*", text, re.I)
    if not query:
        return None, "compiled_template_not_supported_select_distinct_bgp"
    answer, body, limit = query.groups()
    if answer.lstrip("?") != str(execution.graph.answer_var).lstrip("?"):
        return None, "compiled_template_answer_variable_conflict"
    if limit and execution.row_count >= int(limit):
        return None, "compiled_template_limit_completeness_unproven"

    def term(value):
        if _zero_graph_role_safety_VAR.fullmatch(value):
            return value
        if value.startswith('"'):
            return value
        if value.startswith("<"):
            uri = value[1:-1]
        elif ":" in value:
            prefix, suffix = value.split(":", 1)
            if prefix not in prefixes:
                raise ValueError("Unknown template prefix")
            uri = prefixes[prefix] + suffix
        else:
            raise ValueError("Unknown template term")
        if not uri.startswith("http://rdf.freebase.com/ns/"):
            raise ValueError("Non-Freebase template term")
        return _zero_graph_role_safety_canonicalize_answer_id(uri)

    # These two common template filters are proven true for the existing pure
    # entity denotation. Arbitrary filters, OR/UNION branches and ordering fail.
    position = 0
    triples = []
    while position < len(body):
        whitespace = re.match(r"\s*", body[position:])
        position += whitespace.end()
        if position == len(body):
            break
        if re.match(r"FILTER\b", body[position:], re.I):
            filter_match = re.match(r"FILTER\s*\(", body[position:], re.I)
            if not filter_match:
                return None, "compiled_template_filter_unsupported"
            begin = position + filter_match.end()
            cursor, depth, quote, escaped = begin, 1, "", False
            while cursor < len(body) and depth:
                char = body[cursor]
                if quote:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = ""
                elif char in "\"'":
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                cursor += 1
            if depth or quote:
                return None, "compiled_template_filter_unbalanced"
            expression = body[begin:cursor-1].strip()
            inequality = re.fullmatch(re.escape(answer) + r"\s*!=\s*(" + _RDF_TERM + r")", expression)
            language = re.fullmatch(r"!\s*isLiteral\s*\(\s*" + re.escape(answer)
                + r"\s*\)\s*(?:OR|\|\|)\s*lang\s*\(\s*" + re.escape(answer)
                + r"\s*\)\s*=\s*''\s*(?:OR|\|\|)\s*langMatches\s*\(\s*lang\s*\(\s*"
                + re.escape(answer) + r"\s*\)\s*,\s*'[A-Za-z-]+'\s*\)", expression, re.I)
            if inequality:
                excluded = term(inequality.group(1))
                if not _zero_graph_role_safety_ENTITY.fullmatch(excluded) or excluded in _zero_graph_role_safety_entity_answers(execution):
                    return None, "compiled_template_exclusion_not_attested"
            elif not language:
                return None, "compiled_template_filter_unsupported"
            position = cursor
            continue
        triple = _BGP_TRIPLE.match(body, position)
        if not triple:
            return None, "compiled_template_body_not_fully_covered_bgp"
        subject, predicate, object_ = map(term, triple.groups())
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", predicate):
            return None, "compiled_template_predicate_not_canonical"
        triples.append((subject, predicate, object_))
        position = triple.end()
    if not triples or len(set(triples)) != len(triples):
        return None, "compiled_template_empty_or_duplicate_bgp"
    return SimpleNamespace(triples=triples, answer_var=answer, operators=[]), "compiled_template_bgp_qualified"


def _literal_text(value: str) -> str | None:
    match = re.fullmatch(r'("(?:[^"\\]|\\.)*")(?:@[A-Za-z-]+|\^\^\S+)?', value)
    if not match:
        return None
    try:
        return str(json.loads(match.group(1)))
    except (TypeError, ValueError):
        return None


def _source_embedding(source_graph: Any, candidate_graph: Any):
    """Injective, answer-bound alpha mapping, preserving all source triples."""
    source = [tuple(map(str, row)) for row in source_graph.triples]
    candidate = [tuple(map(str, row)) for row in candidate_graph.triples]
    constants = _zero_graph_role_safety_anchors(source_graph)
    source_vars = {node for row in source for node in (row[0], row[2]) if node not in constants}
    states = [{str(source_graph.answer_var): str(candidate_graph.answer_var)}]
    for subject, relation, object_ in source:
        next_states = []
        for state in states:
            for c_subject, c_relation, c_object in candidate:
                if relation != c_relation:
                    continue
                updated = dict(state)
                valid = True
                for left, right in ((subject, c_subject), (object_, c_object)):
                    if left not in source_vars:
                        valid &= left == right
                    elif not _zero_graph_role_safety_VAR.fullmatch(right) or left in updated and updated[left] != right:
                        valid = False
                    elif left not in updated:
                        if right in updated.values():
                            valid = False
                        else:
                            updated[left] = right
                if valid:
                    next_states.append(updated)
        states = next_states
        if not states or len(states) > 64:
            return []
    distinct = {json.dumps(value, sort_keys=True): value for value in states}
    return list(distinct.values())


def quoted_constraint_execution_challenger(
    question: str, source: Any, executions: list[Any], ontology: Any,
    semantic_context: Any, decomposition_context: Any,
) -> tuple[Any | None, dict[str, Any]]:
    """A complete already-executed template may refine even an entity source.

    Require a strict answer subset, exact typed quoted literal, source-core
    preservation under a unique alpha mapping, and the original literal owner
    path to the declared answer. No rank, singleton preference, Gold, or query
    execution is involved. A different description owner is an abstention.
    """
    evidence = {"status": "not_applied", "source_graph_id": str(source.graph.graph_id),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "uses_existing_executions_only": True}
    try:
        if (not _zero_graph_role_safety_complete(source) or _zero_graph_role_safety_entity_answers(source) is None
                or source.graph.operators or _projection(source.graph) is None):
            evidence["reason"] = "source_denotation_operator_or_projection_unqualified"
            return None, evidence
        if (isinstance(getattr(source.graph, "provenance", None), Mapping)
                and source.graph.provenance.get("selection_answer_union")):
            evidence["reason"] = "source_union_scope_unproven"
            return None, evidence
        edge = _projection(source.graph)
        role, proof = _terminal_role_evidence(question, edge, ontology, semantic_context, decomposition_context)
        evidence["role_evidence"] = proof
        if role is None or not proof.get("quoted_field_schema_proven"):
            evidence["reason"] = "original_typed_quoted_obligation_and_owner_unproven"
            return None, evidence
        quoted, field = proof["role_witness"]["quoted_obligation"]
        field_type = str(ontology.range_for_relation(field) or "")
        if field_type not in {"type.text", "type.rawstring"}:
            evidence["reason"] = "original_quoted_field_not_known_text"
            return None, evidence
        source_ids = _zero_graph_role_safety_entity_answers(source)
        eligible, excluded = [], []
        for candidate in executions:
            ids = _zero_graph_role_safety_entity_answers(candidate)
            if candidate is source or ids is None or not _zero_graph_role_safety_complete(candidate) or not ids < source_ids:
                continue
            parsed, reason = _compiled_template_bgp(candidate)
            if parsed is None:
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": reason})
                continue
            literal_edges = [row for row in parsed.triples if row[1] == field and _literal_text(row[2]) == quoted]
            if len(literal_edges) != 1:
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "exact_original_literal_field_missing_or_ambiguous"})
                continue
            bindings = _source_embedding(source.graph, parsed)
            if len(bindings) != 1:
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "source_core_binding_missing_or_ambiguous"})
                continue
            # The original Semantic literal path must reach the exact source
            # projection owner. A literal on an unrelated/disconnected entity
            # cannot supply the requested obligation.
            mapping = bindings[0]
            if literal_edges[0][0] != mapping.get(edge[0]):
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "quoted_constraint_attached_to_different_owner"})
                continue
            mapped_source = {(mapping.get(str(row[0]), str(row[0])), str(row[1]),
                              mapping.get(str(row[2]), str(row[2]))) for row in source.graph.triples}
            if set(parsed.triples) != mapped_source | {literal_edges[0]}:
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "template_adds_other_unproven_obligations"})
                continue
            # This lane is intentionally limited to a direct owner-to-answer
            # literal obligation. Longer paths need their own variable proof.
            literal_owner = str(ontology.domain_for_relation(field) or "")
            if not _zero_graph_role_safety_concrete_type(literal_owner) or literal_owner != str(ontology.domain_for_relation(edge[1]) or ""):
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "original_literal_and_answer_owner_types_differ"})
                continue
            eligible.append(candidate)
        evidence.update(eligible_graph_ids=[str(candidate.graph.graph_id) for candidate in eligible],
                        excluded_candidates=excluded)
        denotations = {tuple(sorted(_zero_graph_role_safety_entity_answers(candidate))) for candidate in eligible}
        if len(denotations) != 1:
            evidence["reason"] = "no_unique_complete_quoted_constraint_denotation"
            return None, evidence
        chosen = eligible[0]
        evidence.update(status="applied", reason="existing_strict_subset_fulfils_original_typed_literal_same_owner",
                        selected_graph_id=str(chosen.graph.graph_id), selected_answer_count=len(chosen.answer_ids),
                        exact_quoted_field=field, exact_quoted_value=quoted, original_owner_preserved=True)
        return chosen, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="compiled_template_or_context_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence



__all__ = ['_source_role_execution_safety_source_entity_head', 'bound_subject_self_inequality', 'mediator_terminal_role_scope', 'model_entity_head', 'model_entity_role_challenger', 'nested_name_head', 'nested_name_schema_challenger', 'quoted_constraint_execution_challenger', 'scalar_owner_union_challenger', 'source_role_challenger', 'terminal_union_scope_challenger', 'unsupported_intersection_challenger', 'zero_graph_role_regression']
