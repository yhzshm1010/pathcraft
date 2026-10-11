"""Zero-call preservation of an explicitly requested source relation role.

This is a final-only guard: its result is always an existing execution. It
neither lowers a query nor asks a model, endpoint, or embedding ranker anything.
Positive vocabulary is generated from cached ontology endpoint types. Unknown
schema, ambiguous source state and incomplete denotations are abstentions.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import re
from typing import Any

from .evaluation import canonicalize_answer_id
from .model_entity_role_safety import _candidate_graph, _graph_state
from .nested_name_schema_guard import (
    _anchors, _complete, _concrete_type, _entity_answers, _schema_status,
    _singular, _substantive_operators, _type_index,
)
from .ontology import relation_id_from_label


_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_WORD = re.compile(r"[A-Za-z]+")
_HEAD = re.compile(
    r"^\s*(?:what(?:['’]s)?|which)\s+(?:(?:is|was|are|were)\s+)?"
    r"(?:the\s+)?([A-Za-z]+(?:\s+[A-Za-z]+){0,3})", re.I)
_HEAD_BOUNDARY = frozenset({"of", "from", "in", "for", "by", "with", "that",
                            "which", "who", "whose", "where", "did", "does",
                            "do", "is", "was", "are", "were", "has", "have"})


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_singular(word) for word in _WORD.findall(str(value)))


def _word_forms(value: str) -> set[str]:
    """Exact generic inflections/agent suffixes, never arbitrary prefix match."""
    word = _singular(value)
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
    return _tokens(text)


def source_entity_head(question: str) -> str:
    """Strict noun phrase head; no movie/film or other synonym routing list."""
    match = _HEAD.match(str(question))
    if not match:
        return ""
    words = []
    for word in match.group(1).split():
        if word.casefold() in _HEAD_BOUNDARY:
            break
        words.append(_singular(word))
    if not words:
        return ""
    phrase = " ".join(words)
    # A apostrophe/hyphen/number must not turn a compound head into a prefix.
    end = match.start(1) + len(" ".join(match.group(1).split()[:len(words)]))
    tail = str(question)[end:]
    if tail and not tail[0].isspace() and tail[0] not in "?,.;:!":
        return ""
    return phrase


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _definite_role_conflict(graph: Any, expected: set[str], ontology: Any):
    status, schema = _schema_status(graph, expected, ontology)
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
    if _graph_state(_candidate_graph(raw)) != _graph_state(execution.graph):
        return False
    preview, count = raw.get("answer_ids"), raw.get("answer_count")
    if (not isinstance(preview, list) or not isinstance(count, int)
            or isinstance(count, bool) or count != len(execution.answer_ids)):
        return False
    ids = {canonicalize_answer_id(value) for value in execution.answer_ids}
    seen = {canonicalize_answer_id(value) for value in preview}
    if len(seen) != len(preview) or not seen or not seen <= ids:
        return False
    if full:
        return not raw.get("answers_truncated", True) and seen == ids
    return (seen == ids and not raw.get("answers_truncated", True)
            or len(seen) < count and raw.get("answers_truncated") is True)


def _one_execution(executions: list[Any], graph_id: str) -> Any | None:
    matches = [execution for execution in executions
               if str(execution.graph.graph_id) == graph_id]
    if not matches or any(not _complete(execution) or _entity_answers(execution) is None
                          for execution in matches):
        return None
    states = {(_graph_state(execution.graph), tuple(sorted(_entity_answers(execution))))
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
        if not all(_concrete_type(str(getattr(ontology, method)(relation) or ""))
                   for method in ("domain_for_relation", "range_for_relation")):
            return None
        adjacency.setdefault(subject, []).append((object_, edge))
        adjacency.setdefault(object_, []).append((subject, edge))
    anchors = _anchors(graph)
    if (not anchors or not all(_ENTITY.fullmatch(anchor) for anchor in anchors)
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
    return _tokens(type_id.rsplit(".", 1)[-1]) if _concrete_type(type_id) else ()


def _semantic_graphs(value: Any):
    if isinstance(value, Mapping):
        if isinstance(value.get("anchors"), list) and isinstance(value.get("semantic_paths"), list):
            yield value
        else:
            for key in ("output", "semantic_graph", "source_semantic", "semantic_graphs"):
                if key in value:
                    yield from _semantic_graphs(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _semantic_graphs(item)
    else:
        graph = getattr(value, "semantic_graph", None)
        if graph is not None:
            yield from _semantic_graphs(graph)


def _semantic_support(edge: tuple[str, str, str], anchor: str, semantic_graphs: Any,
                      linked_context: Any) -> bool:
    names = {str(item.get("name") or item.get("label") or "").strip().casefold()
             for item in linked_context if isinstance(item, Mapping)
             and canonicalize_answer_id(item.get("id", "")) == anchor}
    names.discard("")
    if len(names) != 1:
        return False
    direction = "forward" if edge[0] == anchor else "backward"
    for semantic in _semantic_graphs(semantic_graphs):
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
                    and relation_id_from_label(first.get("relation_label")) == edge[1]):
                return True
    return False


def _context_role_support(role: tuple[str, ...], anchor: str, context: Any) -> bool:
    matches = [item for item in context if isinstance(item, Mapping)
               and canonicalize_answer_id(item.get("id", "")) == anchor]
    return (len(matches) == 1 and isinstance(matches[0].get("types"), list)
            and role in {_tokens(str(value).replace("_", " "))
                         for value in matches[0]["types"]})


def _decomposition_lines(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _decomposition_lines(item)
    elif isinstance(value, Mapping):
        for key in ("decomposition", "final_decomposition", "final_candidates", "candidates"):
            if key in value:
                yield from _decomposition_lines(value[key])


def _last_who_role(value: str) -> str:
    clauses = re.findall(r"\bwho\s+(?:is|are|was|were)\s+(?:the\s+)?"
                         r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)",
                         str(value), re.I)
    return " ".join(_tokens(" ".join(clauses[-1]))) if clauses else ""


def _semantic_path_nodes(candidate: Any, semantic: Mapping[str, Any], path: Mapping[str, Any],
                         linked_context: Any) -> set[str] | None:
    anchors = {str(item.get("id")): str(item.get("surface", "")).strip().casefold()
               for item in semantic["anchors"] if isinstance(item, Mapping)}
    name = anchors.get(str(path.get("anchor_ref")), "")
    constants = {canonicalize_answer_id(item.get("id", "")) for item in linked_context
                 if isinstance(item, Mapping)
                 and str(item.get("name") or item.get("label") or "").strip().casefold() == name}
    constants &= _anchors(candidate.graph)
    if len(constants) != 1:
        return None
    states = [(constant, {str(path.get("anchor_ref")): constant}) for constant in constants]
    edges = [tuple(map(str, triple)) for triple in candidate.graph.triples]
    graph_anchors = _anchors(candidate.graph)
    for step in path.get("steps", []):
        if not isinstance(step, Mapping):
            return None
        relation = relation_id_from_label(step.get("relation_label"))
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
    supported_roles = {_last_who_role(line) for line in _decomposition_lines(decompositions)} - {""}
    if not supported_roles:
        return None, "reviewed_decomposition_terminal_role_missing"
    projection = [tuple(map(str, triple)) for triple in selected.graph.triples
                  if str(triple[2]) == str(selected.graph.answer_var)]
    if len(projection) != 1 or not _complete(selected) or _entity_answers(selected) is None:
        return None, "selected_projection_or_entity_denotation_unqualified"
    source_edge = projection[0]
    context = Counter(tuple(map(str, triple)) for triple in selected.graph.triples
                      if tuple(map(str, triple)) != source_edge)
    operators, anchors = _substantive_operators(selected.graph), _anchors(selected.graph)
    if not context or operators is None or not anchors:
        return None, "source_context_anchor_or_operator_missing"
    source_owner_type = str(ontology.domain_for_relation(source_edge[1]) or "")
    if not _concrete_type(source_owner_type):
        return None, "source_projection_owner_schema_unknown_or_generic"
    requests, observed_requests, excluded_requests, unknown_requests = [], set(), [], []
    for semantic in _semantic_graphs(semantic_graphs):
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
            relation = relation_id_from_label(last.get("relation_label"))
            # Exact ontology predicate leaf + owner phrase. No paraphrase or
            # synonym list maps 'rules' to 'politician' or 'current president'.
            parts = relation.split(".")
            canonical_role = " ".join(_tokens(" ".join(parts[-1:] + parts[-2:-1])))
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
            if (owner_nodes != {source_edge[0]} or not _concrete_type(requested_owner_type)
                    or not _concrete_type(target_type)):
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
    evidence["excluded_other_owner_role_requests"] = list({_stable(value): value for value in excluded_requests}.values())
    evidence["unproven_owner_role_requests"] = list({_stable(value): value for value in unknown_requests}.values())
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
        if candidate is selected or _entity_answers(candidate) is None:
            continue
        graph = candidate.graph
        if (_anchors(graph) != anchors or graph.answer_var != selected.graph.answer_var
                or _substantive_operators(graph) != operators):
            continue
        edges = [tuple(map(str, triple)) for triple in graph.triples]
        terminal = [edge for edge in edges if edge[2] == str(graph.answer_var)]
        if (len(terminal) != 1 or terminal[0] != (source_edge[0], relation, source_edge[2])
                or Counter(edge for edge in edges if edge != terminal[0]) != context):
            continue
        if not any(_semantic_path_matches(candidate, semantic, path, linked_context)
                   for requested, _, semantic, path in requests if requested == relation):
            continue
        candidate_status, _ = _schema_status(graph, {target_type}, ontology)
        if candidate_status != "compatible" or not _complete(candidate):
            uncertain.append(str(graph.graph_id))
        else:
            eligible.append(candidate)
    evidence["semantic_projection_eligible_graph_ids"] = [str(candidate.graph.graph_id) for candidate in eligible]
    evidence["semantic_projection_uncertain_graph_ids"] = uncertain
    denotations = {tuple(sorted(_entity_answers(candidate))) for candidate in eligible}
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
    if replacement is None or not _complete(selected) or _entity_answers(selected) is None:
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
    original_ops = _substantive_operators(replacement.graph)
    final_ops = _substantive_operators(selected.graph)
    if original_ops is None or final_ops is None or not original_ops <= final_ops:
        return None, "captured_source_operator_not_preserved_by_override"
    # Newly generated self-exclusion may be dropped only when the complete
    # restored denotation already satisfies it. Other added constraints abstain.
    for operator in selected.graph.operators:
        if _stable(operator) in original_ops:
            continue
        excluded = canonicalize_answer_id(operator.get("_value_entity_id", ""))
        if (str(operator.get("type", "")).upper() != "NO_EQUAL"
                or str(operator.get("input_var", "")) != str(selected.graph.answer_var)
                or str(operator.get("value_type", "")) != "entity"
                or operator.get("inputs") or operator.get("attribute_relation_label")
                or operator.get("attribute_relation_labels")
                or excluded not in before or excluded in _entity_answers(replacement)):
            return None, "substantive_added_constraint_not_attested_by_source_answers"
    evidence.update(requested_anchor_id=anchor, ontology_source_role=list(role),
                    ontology_override_role=list(alternative_role),
                    preserved_anchor_ids=sorted(before),
                    preserved_source_operator_count=len(original_ops),
                    captured_override_source_restored=True)
    return replacement, "restore_captured_source_requested_relation_role"


def _head_projection(question: str, selected: Any, executions: list[Any], ontology: Any,
                     evidence: dict[str, Any]) -> tuple[Any | None, str]:
    head = source_entity_head(question)
    evidence["head"] = head
    if not head:
        return None, "not_strict_ontology_noun_head"
    expected = set(_type_index(ontology).get(head, ()))
    evidence["expected_types"] = sorted(expected)
    if not expected:
        return None, "head_not_in_ontology_vocabulary"
    if _entity_answers(selected) is None or not _complete(selected):
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
    operators = _substantive_operators(selected.graph)
    anchors = _anchors(selected.graph)
    context = Counter(tuple(map(str, triple)) for triple in selected.graph.triples
                      if tuple(map(str, triple)) != projection)
    if not context or not anchors or operators is None:
        return None, "source_context_anchor_or_operator_missing"
    compatible, unknown = [], []
    for candidate in executions:
        if candidate is selected or _entity_answers(candidate) is None:
            continue
        graph = candidate.graph
        if _anchors(graph) != anchors or _substantive_operators(graph) != operators:
            continue
        edges = [tuple(map(str, triple)) for triple in graph.triples]
        terminal = [edge for edge in edges if edge[2] == str(graph.answer_var)]
        if len(terminal) != 1 or terminal[0][0] != projection[0]:
            continue
        remainder = Counter(edge for edge in edges if edge != terminal[0])
        if remainder != context:
            continue
        candidate_status, candidate_schema = _schema_status(graph, expected, ontology)
        if candidate_status == "unknown" or candidate_status == "compatible" and not _complete(candidate):
            unknown.append(str(graph.graph_id))
        elif candidate_status == "compatible":
            compatible.append((candidate, candidate_schema))
    evidence["compatible_graph_ids"] = [str(item.graph.graph_id) for item, _ in compatible]
    evidence["unknown_or_incomplete_graph_ids"] = unknown
    denotations = {tuple(sorted(_entity_answers(item))) for item, _ in compatible}
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
            if _entity_answers(replacement) == _entity_answers(selected):
                evidence["reason"] = "source_role_denotation_already_selected"
                return None, evidence
            evidence.update(status="applied", selected_graph_id=str(replacement.graph.graph_id),
                            selected_answer_count=len(replacement.answer_ids))
        return replacement, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="schema_or_execution_evidence_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence
