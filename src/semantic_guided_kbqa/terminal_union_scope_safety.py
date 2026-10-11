"""Final-only removal of an unsupported terminal-role interpretation.

This is intentionally not a selector or an answer generator. It can return only
the complete, already executed primary of an actual two-source answer union.
Question text, graph identity, full denotations and concrete ontology types must
independently establish the narrow scope. No labels, gold, learned scores,
endpoint requests or model/embedding calls are used.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .contracts import ExecutedGraph, QueryGraphCandidate


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


def _tokens(value: str, *, informative: bool = False) -> list[str]:
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


def _complete_entities(execution: ExecutedGraph) -> bool:
    ids = execution.answer_ids
    return (isinstance(ids, list) and bool(ids)
            and isinstance(execution.row_count, int)
            and not isinstance(execution.row_count, bool)
            and execution.row_count == len(ids)
            and all(isinstance(value, str) and _ENTITY_ID.fullmatch(value) for value in ids)
            and len(set(ids)) == len(ids))


def _terminal_edge(graph: QueryGraphCandidate) -> tuple[str, str, str] | None:
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


def _concrete_type(value: str) -> bool:
    # "type" contains primitive scalars and the generic object root;
    # "common" contains cross-domain topic/taggable scaffolding. Neither is
    # sufficient evidence for changing an entity-denoting terminal role.
    return bool(value and "." in value and value.split(".", 1)[0] not in {"type", "common"})


def _compatible_concrete(left: str, right: str, ontology: Any) -> bool:
    if not _concrete_type(left) or not _concrete_type(right):
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
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
    ontology: Any,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
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
    p_tokens = _tokens(p_edge[1].rsplit(".", 1)[-1], informative=True)
    s_tokens = _tokens(s_edge[1].rsplit(".", 1)[-1], informative=True)
    q_tokens = _tokens(text)
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
