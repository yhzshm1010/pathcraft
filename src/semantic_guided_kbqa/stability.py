"""Consolidated stability; original rules and budgets retained."""
from __future__ import annotations


# ---- stability_answer_role ----
"""Zero-call protection against a demonstrable answer-role bounce.

This module does not generate facts, execute queries, or read evaluation labels.
In particular, refusing a late mascot/participant answer for an explicitly
corroborated date request does not make the incumbent answer correct. A final
replacement is possible only when an already executed graph has the *identical*
full conjunctive scope, with one additional requested terminal projection.

Vocabulary for positive entity roles comes from the loaded ontology. Temporal
words below are datatype grammar, not entity/relation/sample routing lists.
"""


from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import json
import re
from types import SimpleNamespace
from typing import Any

from .evaluation import canonicalize_answer_id as _stability_answer_role_canonicalize_answer_id
from .ontology import FreebaseOntology, _search_tokens, relation_id_from_label as _stability_answer_role_relation_id_from_label, relation_label_from_id
from .safety import _source_role_execution_safety_decomposition_lines as _decomposition_lines


_stability_answer_role_VAR = re.compile(r"V\d+\Z")
_stability_answer_role_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_TEMPORAL_HEAD = re.compile(
    r"^\s*(?:(?:in|on)\s+)?(?:what|which)\s+(year|date|day|month)\b", re.I)
_WH_ROLE = re.compile(
    r"\b(?:what|which|who)\s+(?:is|are|was|were)\s+(?:the\s+)?"
    r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)", re.I)
_TEMPORAL_WORDS = frozenset({"year", "date", "day", "month"})
_PRIMITIVE = frozenset({"type.int", "type.float", "type.datetime", "type.text",
                        "type.boolean", "type.rawstring", "type.uri"})


def _stability_answer_role_words(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", str(value).replace("_", " ").casefold()))


def _last_role(value: Any) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    roles = _WH_ROLE.findall(str(value))
    return (_stability_answer_role_words(roles[-1][0]), _stability_answer_role_words(roles[-1][1])) if roles else None


def _validated_semantic_graphs(value: Any, question: str):
    """Ignore rejected/stale stage outputs and require the actual graph contract."""
    if isinstance(value, Mapping):
        if value.get("valid") is False or value.get("question", question) != question:
            return
        input_ = value.get("input")
        if isinstance(input_, Mapping) and input_.get("question", question) != question:
            return
        if isinstance(value.get("anchors"), list) and isinstance(value.get("semantic_paths"), list):
            from .contracts import validate_semantic_graph
            try:
                graph = validate_semantic_graph({"anchors": value["anchors"],
                                                 "semantic_paths": value["semantic_paths"]}, max_hops=8)
            except (ValueError, TypeError, KeyError):
                return
            yield graph
        else:
            for key in ("output", "semantic_graph", "source_semantic", "semantic_graphs"):
                if key in value:
                    yield from _validated_semantic_graphs(value[key], question)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _validated_semantic_graphs(item, question)


def _base_evidence(graph: Any) -> dict[str, Any]:
    return {"status": "not_applied", "graph_id": str(getattr(graph, "graph_id", "")),
            "additional_model_calls": 0, "additional_endpoint_queries": 0,
            "additional_embedding_calls": 0, "uses_gold_answers": False,
            "entity_relation_or_sample_whitelist": False,
            "proves_answer_correctness": False}


def _triples(graph: Any) -> list[tuple[str, str, str]] | None:
    triples = getattr(graph, "triples", None)
    if (not isinstance(triples, list) or not triples
            or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)):
        return None
    return [tuple(map(str, t)) for t in triples]


def _owned(question: str, context_question: str, graph: Any) -> bool:
    if not question or question != context_question:
        return False
    provenance = getattr(graph, "provenance", None)
    owner = provenance.get("original_question") if isinstance(provenance, Mapping) else None
    return not owner or owner == question


def temporal_role_witness(
    question: str, semantic_context: Any, decomposition_context: Any,
    *, context_question: str,
) -> tuple[dict[str, Any] | None, str]:
    """Require an explicit temporal head and the same declared stage goal.

    A bare ``when`` or ``who`` cannot activate this lane. Semantic *goal* is a
    statement of requested role, not proof its actual steps contain the right
    relation. Accordingly this function neither invents a date predicate nor
    asserts that missing question constraints have been fulfilled.
    """
    if question != context_question:
        return None, "question_context_owner_mismatch"
    match = _TEMPORAL_HEAD.match(str(question))
    if not match:
        return None, "not_explicit_single_temporal_question_head"
    tail = str(question)[match.end():]
    if tail and not tail[0].isspace() and tail[0] not in "?.!":
        return None, "compound_temporal_question_head"
    decomposition_roles = {_last_role(line) for line in _decomposition_lines(decomposition_context)} - {None}
    witnessed = set()
    for semantic in _validated_semantic_graphs(semantic_context, question):
        for path in semantic.get("semantic_paths", []):
            if not isinstance(path, Mapping):
                continue
            role = _last_role(path.get("goal", ""))
            if (role in decomposition_roles and len(role[0]) == 1
                    and role[0][0] in _TEMPORAL_WORDS and role[1]):
                witnessed.add(role)
    if len(witnessed) != 1:
        return None, "temporal_semantic_decomposition_goal_missing_or_ambiguous"
    field, owner = next(iter(witnessed))
    # Year/date is an allowed temporal granularity normalization. Day/month
    # should not activate from an unrelated year request or vice versa.
    head = match.group(1).casefold()
    if field[0] != head and {field[0], head} != {"year", "date"}:
        return None, "temporal_head_goal_granularity_conflict"
    return {"question_temporal_head": head, "declared_temporal_field": field[0],
            "declared_owner": " ".join(owner),
            "witness_sources": ["question", "decomposition", "semantic_goal"],
            "does_not_invent_missing_date_relation": True}, ""


def temporal_source_role_bounce_veto(
    question: str, challenger_graph: Any, ontology: Any,
    semantic_context: Any, decomposition_context: Any,
    *, context_question: str,
) -> tuple[bool, dict[str, Any]]:
    """Reject only a temporal request bouncing back to a constrained topic role.

    Suitable *before* challenger execution. The answer must be reached by the
    same owner/predicate used to constrain a concrete entity anchor. Unknown
    schema, genuinely temporal relation ranges, a different answer path, and
    non-temporal questions abstain. The incumbent is left unchanged, not given
    fabricated answers or an NA result.
    """
    evidence = _base_evidence(challenger_graph)
    if not _owned(question, context_question, challenger_graph):
        evidence["reason"] = "question_context_owner_mismatch"
        return False, evidence
    witness, reason = temporal_role_witness(
        question, semantic_context, decomposition_context, context_question=context_question)
    if witness is None:
        evidence["reason"] = reason
        return False, evidence
    evidence.update(witness)
    triples = _triples(challenger_graph)
    answer = str(getattr(challenger_graph, "answer_var", ""))
    if triples is None or not _stability_answer_role_VAR.fullmatch(answer):
        evidence["reason"] = "invalid_graph_or_answer_variable"
        return False, evidence
    incident = [t for t in triples if answer in (t[0], t[2])]
    if len(incident) != 1 or incident[0][2] != answer:
        evidence["reason"] = "answer_not_unique_forward_terminal"
        return False, evidence
    subject, predicate, _ = incident[0]
    anchor_edges = [t for t in triples if t[0] == subject and t[1] == predicate and _stability_answer_role_MID.fullmatch(t[2])]
    if not anchor_edges:
        evidence["reason"] = "not_same_owner_constrained_topic_role_bounce"
        return False, evidence
    range_method = getattr(ontology, "range_for_relation", None)
    declared_range = str(range_method(predicate) or "") if callable(range_method) else ""
    range_words = set(_stability_answer_role_words(declared_range.rsplit(".", 1)[-1]))
    if (not declared_range or declared_range in _PRIMITIVE
            or declared_range.startswith(("type.", "common.", "freebase.", "base."))
            or range_words & _TEMPORAL_WORDS):
        evidence["reason"] = "range_unknown_generic_primitive_or_temporal"
        return False, evidence
    evidence.update(status="vetoed", reason="temporal_request_returns_constrained_topic_role",
                    answer_relation=predicate, declared_answer_range=declared_range,
                    constrained_topic_ids=sorted({t[2] for t in anchor_edges}),
                    protects_role_only_not_incumbent_f1=True)
    return True, evidence


def _terminal_relation_witness(
    question: str, ontology: Any, semantic_context: Any, decomposition_context: Any,
) -> tuple[tuple[str, str] | None, str]:
    decomposition_roles = {_last_role(line) for line in _decomposition_lines(decomposition_context)} - {None}
    roles = set()
    for semantic in _validated_semantic_graphs(semantic_context, question):
        for path in semantic.get("semantic_paths", []):
            if isinstance(path, Mapping) and (role := _last_role(path.get("goal", ""))) in decomposition_roles:
                roles.add(role)
    if len(roles) != 1:
        return None, "declared_terminal_role_missing_or_ambiguous"
    field, owner = next(iter(roles))
    # Positive entity role is not translated from an unknown verb. A WHO
    # request can use a corroborated declared role but must remain an entity
    # projection under a concrete schema, checked below.
    qwords = _stability_answer_role_words(question)
    who_head = bool(re.match(r"^\s*who\b", question, re.I))
    mentioned = any(qwords[i:i + len(field)] == field for i in range(len(qwords)))
    if not field or not owner or not (who_head or mentioned):
        return None, "question_does_not_corroborate_declared_entity_role"
    domains = getattr(ontology, "relation_domains", {})
    ranges = getattr(ontology, "relation_ranges", {})
    matches = set()
    if isinstance(domains, Mapping) and isinstance(ranges, Mapping):
        # FreebaseOntology.load already builds a complete token->relation
        # index. Every exact leaf match necessarily contains every normalized
        # token of the requested field, even when its relation namespace does
        # not mirror its domain. Do not use owner tokens as an index filter:
        # a predicate alias can have an actual domain absent from its ID.
        # This replaces repeated full schema scans, not the exact verification
        # or uniqueness check. No question/role-specific cache is introduced.
        if isinstance(ontology, FreebaseOntology):
            tokens = set(_search_tokens(field))
            pools = [set(ontology.relation_token_index.get(token, ())) for token in tokens]
            candidates = set.intersection(*pools) if pools else set()
            relation_domains = ((relation, domains.get(relation, "")) for relation in sorted(candidates))
        else:
            # Small test/provider contracts may lack an authoritative complete
            # index. Never guess uniqueness from an incomplete ad hoc index,
            # or perform unbounded work for an unknown schema implementation.
            if len(domains) > 256:
                return None, "unknown_ontology_without_bounded_complete_role_index"
            relation_domains = domains.items()
        for relation, domain in relation_domains:
            if (_stability_answer_role_words(str(relation).rsplit(".", 1)[-1]) == field
                    and _stability_answer_role_words(str(domain).rsplit(".", 1)[-1]) == owner
                    and ranges.get(relation) and ranges[relation] not in _PRIMITIVE
                    and not str(ranges[relation]).startswith(("type.", "common.", "freebase.", "base."))):
                matches.add((str(relation), str(domain)))
    if len(matches) != 1:
        return None, "requested_terminal_ontology_role_unknown_or_ambiguous"
    return next(iter(matches)), ""


def compile_same_owner_declared_projection(
    question: str, graph: Any, ontology: Any, semantic_context: Any,
    decomposition_context: Any, *, context_question: str,
) -> tuple[Any | None, dict[str, Any]]:
    """Pure candidate compiler; preserves every old triple/operator and anchor.

    The old projected field must be under the exact same schema owner as the
    uniquely corroborated requested field. The requested projection is added;
    no wrong-looking spouse, film title, exclusion, or other constraint is
    removed. A returned graph does not imply executable/nonempty/correct.
    """
    evidence = _base_evidence(graph)
    if not _owned(question, context_question, graph):
        evidence["reason"] = "question_context_owner_mismatch"
        return None, evidence
    witness, reason = _terminal_relation_witness(question, ontology, semantic_context, decomposition_context)
    if witness is None:
        evidence["reason"] = reason
        return None, evidence
    requested, owner_type = witness
    triples = _triples(graph)
    answer = str(getattr(graph, "answer_var", ""))
    operators = getattr(graph, "operators", None)
    if triples is None or not isinstance(operators, list) or any(not isinstance(o, dict) for o in operators):
        evidence["reason"] = "invalid_graph_or_operator_scope"
        return None, evidence
    incident = [t for t in triples if answer in (t[0], t[2])]
    if len(incident) != 1 or incident[0][2] != answer or incident[0][1] == requested:
        evidence["reason"] = "not_unique_wrong_same_owner_terminal"
        return None, evidence
    old = incident[0]
    domain_method = getattr(ontology, "domain_for_relation", None)
    old_owner = str(domain_method(old[1]) or "") if callable(domain_method) else ""
    if old_owner != owner_type or not _stability_answer_role_VAR.fullmatch(old[0]):
        evidence["reason"] = "old_and_requested_terminal_owners_differ_or_unknown"
        return None, evidence
    # Require a real anchored owner context, not a floating endpoint field.
    nonprojection = [t for t in triples if t != old]
    if not nonprojection or not any(old[0] in (t[0], t[2]) for t in nonprojection):
        evidence["reason"] = "requested_owner_not_bound_by_existing_context"
        return None, evidence
    variables = {term for t in triples for term in (t[0], t[2]) if _stability_answer_role_VAR.fullmatch(term)}
    new_answer = f"V{max(int(v[1:]) for v in variables) + 1}"
    candidate = deepcopy(graph)
    candidate.graph_id = str(graph.graph_id) + "_DECLARED_PROJECTION"
    candidate.triples = [list(t) for t in triples] + [[old[0], requested, new_answer]]
    candidate.answer_var = new_answer
    candidate.sparql = ""
    candidate.provenance = deepcopy(getattr(graph, "provenance", {}) or {})
    candidate.provenance["stability_declared_projection"] = {
        "source_graph_id": str(graph.graph_id), "added_relation": requested,
        "old_projection_preserved": True, "every_operator_preserved": True,
        "model_calls": 0, "endpoint_queries": 0, "uses_gold_answers": False}
    evidence.update(status="compiled_only", reason="corroborated_same_owner_terminal_projection",
                    requested_relation=requested, owner_type=owner_type,
                    candidate_graph_id=candidate.graph_id,
                    every_old_triple_and_operator_preserved=True)
    return candidate, evidence


def same_scope_declared_projection_execution(
    question: str, selected: Any, executions: list[Any], ontology: Any,
    semantic_context: Any, decomposition_context: Any, *, context_question: str,
) -> tuple[Any | None, dict[str, Any]]:
    """Return only a complete already executed, identical-scope projection.

    Variable spelling is deliberately not alpha-renamed: absent a recorded
    binding identity, a superficially similar graph is not an adequate witness.
    Ambiguous denotations, truncated previews and removed old constraints abstain.
    """
    candidate, evidence = compile_same_owner_declared_projection(
        question, selected.graph, ontology, semantic_context, decomposition_context,
        context_question=context_question)
    if candidate is None:
        return None, evidence
    evidence["status"] = "not_applied"
    target_triples = Counter(tuple(map(str, t)) for t in candidate.triples)
    target_ops = json.dumps(candidate.operators, ensure_ascii=False, sort_keys=True)
    matches = []
    denotations = set()
    for execution in executions:
        graph = getattr(execution, "graph", None)
        triples = _triples(graph)
        if (triples is None or Counter(triples) != target_triples
                or str(getattr(graph, "answer_var", "")) != candidate.answer_var
                or json.dumps(getattr(graph, "operators", None), ensure_ascii=False, sort_keys=True) != target_ops
                or not _owned(question, context_question, graph)):
            continue
        ids, count = getattr(execution, "answer_ids", None), getattr(execution, "row_count", None)
        if (not isinstance(ids, list) or not ids or isinstance(count, bool)
                or not isinstance(count, int) or count != len(ids)):
            evidence["reason"] = "matching_existing_projection_denotation_incomplete"
            return None, evidence
        canonical = [_stability_answer_role_canonicalize_answer_id(value) for value in ids]
        if len(set(canonical)) != len(canonical) or not all(_stability_answer_role_MID.fullmatch(v) for v in canonical):
            evidence["reason"] = "matching_existing_projection_not_unique_entities"
            return None, evidence
        denotations.add(tuple(sorted(canonical)))
        matches.append(execution)
    if not matches or len(denotations) != 1:
        evidence["reason"] = "identical_scope_existing_projection_missing_or_ambiguous"
        return None, evidence
    evidence.update(status="existing_execution_selected", reason="identical_full_scope_requested_projection",
                    selected_graph_id=str(matches[0].graph.graph_id), uses_existing_executions_only=True)
    return matches[0], evidence


def certify_declared_terminal_compose(
    question: str, semantic_graph: Mapping[str, Any], compose_output: Mapping[str, Any],
    ontology: Any, decomposition_context: Any, *, context_question: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Certify one grounded terminal-label drift in an actual rejected Compose.

    Only a terminal predicate in a *validation view* of Semantic is corrected.
    The real Compose payload, all of its triples, and all of its entity anchors
    stay byte-for-byte unchanged. The requested predicate must already exist in
    another current two-hop path to the same anchored mediator; no relation is
    guessed or retrieved. Both paths must fit the complete real Compose graph,
    with exactly one joint binding assignment covering every raw BGP.

    The caller must use its existing fixed Operator/candidate-call budget if it
    admits the saved Compose. This function does not supply an Operator output.
    Nor is this a license to copy/remove constraints from another Compose rank.
    """
    evidence = {"status": "not_applied", "additional_model_calls": 0,
                "additional_endpoint_queries": 0, "additional_embedding_calls": 0,
                "uses_gold_answers": False, "entity_relation_or_sample_whitelist": False,
                "proves_answer_correctness": False, "real_compose_output_unchanged": True}

    def abstain(reason: str):
        evidence["reason"] = reason
        return None, evidence

    if question != context_question:
        return abstain("question_context_owner_mismatch")
    if not re.match(r"^\s*who\b", str(question), re.I):
        return abstain("not_declared_entity_who_request")
    if (not isinstance(semantic_graph, Mapping) or not isinstance(compose_output, Mapping)
            or semantic_graph.get("question", question) != question):
        return abstain("malformed_or_stale_semantic_compose_input")
    anchors = semantic_graph.get("anchors")
    paths = semantic_graph.get("semantic_paths")
    if not isinstance(anchors, list) or not isinstance(paths, list) or not 2 <= len(paths) <= 8:
        return abstain("independent_grounded_path_witness_missing")
    try:
        from .contracts import validate_compose_graph_output
        raw = validate_compose_graph_output(compose_output)
    except (ValueError, TypeError, KeyError) as exc:
        return abstain("invalid_real_compose_contract:" + type(exc).__name__)
    if not 2 <= len(raw["triples"]) <= 20:
        return abstain("compose_scope_size_outside_bounded_certificate")
    # Exact one-to-one current anchor surfaces, never dropped named anchors or
    # entity synonym/canonical-name guessing. This excludes another rank's
    # extra named entity even when dropping it would improve the metric.
    if any(not isinstance(a, Mapping) or not a.get("id") or not a.get("surface") for a in anchors):
        return abstain("malformed_current_anchor_context")
    anchor_surfaces = [str(a["surface"]) for a in anchors]
    raw_surfaces = [str(e["surface"]) for e in raw["entities"]]
    if (len(set(anchor_surfaces)) != len(anchor_surfaces)
            or len(set(raw_surfaces)) != len(raw_surfaces)
            or set(anchor_surfaces) != set(raw_surfaces)):
        return abstain("named_anchor_context_not_exactly_preserved")
    surface_to_entity = {str(e["surface"]): str(e["id"]) for e in raw["entities"]}
    anchor_to_entity = {str(a["id"]): surface_to_entity[str(a["surface"])] for a in anchors}
    requested, reason = _terminal_relation_witness(question, ontology, semantic_graph, decomposition_context)
    if requested is None:
        return abstain(reason)
    requested_relation, owner_type = requested
    domain_method = getattr(ontology, "domain_for_relation", None)
    range_method = getattr(ontology, "range_for_relation", None)
    if not callable(domain_method) or not callable(range_method):
        return abstain("ontology_domain_range_interface_missing")
    decomposition_roles = {_last_role(line) for line in _decomposition_lines(decomposition_context)} - {None}
    drift_paths = []
    for index, path in enumerate(paths):
        if not isinstance(path, Mapping) or not isinstance(path.get("steps"), list) or not path["steps"]:
            return abstain("malformed_semantic_path")
        role = _last_role(path.get("goal", ""))
        last = path["steps"][-1]
        if not isinstance(last, Mapping):
            return abstain("malformed_semantic_terminal")
        old_relation = _stability_answer_role_relation_id_from_label(last.get("relation_label"))
        if (role in decomposition_roles
                and _stability_answer_role_words(requested_relation.rsplit(".", 1)[-1]) == role[0]
                and _stability_answer_role_words(owner_type.rsplit(".", 1)[-1]) == role[1]
                and old_relation != requested_relation):
            if (last.get("direction") != "forward" or domain_method(old_relation) != owner_type
                    or not range_method(old_relation) or range_method(old_relation) in _PRIMITIVE):
                return abstain("declared_terminal_drift_not_same_concrete_owner")
            drift_paths.append((index, path, old_relation))
    if len(drift_paths) != 1:
        return abstain("terminal_role_drift_missing_or_not_unique")
    drift_index, drift, old_relation = drift_paths[0]
    first = drift["steps"][0]
    root = str(drift.get("anchor_ref", ""))
    if (root not in anchor_to_entity or not isinstance(first, Mapping)
            or first.get("direction") != "backward"
            or _stability_answer_role_relation_id_from_label(first.get("relation_label")) != old_relation
            or domain_method(old_relation) != owner_type):
        return abstain("drifted_terminal_owner_not_attested_by_same_anchor_field")
    supporting_paths = []
    for index, path in enumerate(paths):
        steps = path["steps"]
        if (index != drift_index and str(path.get("anchor_ref", "")) == root and len(steps) == 2
                and all(isinstance(s, Mapping) for s in steps)
                and steps[0].get("direction") == "backward" and steps[1].get("direction") == "forward"
                and _stability_answer_role_relation_id_from_label(steps[0].get("relation_label")) == old_relation
                and _stability_answer_role_relation_id_from_label(steps[1].get("relation_label")) == requested_relation):
            supporting_paths.append(index)
    if not supporting_paths:
        return abstain("current_independent_same_mediator_requested_relation_missing")
    corrected = deepcopy(dict(semantic_graph))
    corrected["semantic_paths"][drift_index]["steps"][-1]["relation_label"] = relation_label_from_id(requested_relation)
    raw_triples = [
        (str(t["subject"]), _stability_answer_role_relation_id_from_label(t["relation_label"]), str(t["object"]))
        for t in raw["triples"]]
    answer = str(raw["answer_var"])
    if sum(answer in (t[0], t[2]) for t in raw_triples) != 1:
        return abstain("real_compose_answer_not_unique_terminal")

    # Each Semantic local-variable mapping is path scoped. The complete joint
    # certificate requires coverage of *all* real BGPs, not merely the head
    # path. Bounds cap the pure search; exceeding them abstains.
    all_options = []
    for index, path in enumerate(corrected["semantic_paths"]):
        anchor = str(path.get("anchor_ref", ""))
        if anchor not in anchor_to_entity:
            return abstain("path_root_anchor_missing_from_real_compose")
        states = [({anchor: anchor_to_entity[anchor]}, frozenset())]
        expected_from = anchor
        for step in path["steps"]:
            if (not isinstance(step, Mapping) or str(step.get("from")) != expected_from
                    or step.get("direction") not in {"forward", "backward"}):
                return abstain("semantic_path_connectivity_unproven")
            relation = _stability_answer_role_relation_id_from_label(step.get("relation_label"))
            next_states = []
            for binding, covered in states:
                current = binding.get(str(step["from"]))
                for edge_index, edge in enumerate(raw_triples):
                    left, predicate, right = edge
                    if predicate != relation:
                        continue
                    source, target = (left, right) if step["direction"] == "forward" else (right, left)
                    if source != current or not _stability_answer_role_VAR.fullmatch(target):
                        continue
                    previous = binding.get(str(step["to"]))
                    if previous is not None and previous != target:
                        continue
                    new_binding = {**binding, str(step["to"]): target}
                    next_states.append((new_binding, covered | {edge_index}))
            if len(next_states) > 128:
                return abstain("path_mapping_certificate_budget_exceeded")
            states = next_states
            expected_from = str(step["to"])
        if str(path.get("path_output_var")) != expected_from:
            return abstain("semantic_path_terminal_identity_unproven")
        if index == drift_index or index in supporting_paths:
            states = [s for s in states if s[0].get(expected_from) == answer]
        if not states:
            return abstain("real_compose_drops_current_path_or_requested_role")
        all_options.append(states)
    combinations = [((), frozenset())]
    for options in all_options:
        combinations = [(bindings + (tuple(sorted(binding.items())),), covered | edges)
                        for bindings, covered in combinations for binding, edges in options]
        if len(combinations) > 256:
            return abstain("joint_scope_certificate_budget_exceeded")
    certificates = {bindings for bindings, covered in combinations if len(covered) == len(raw_triples)}
    if len(certificates) != 1:
        return abstain("complete_joint_scope_binding_missing_or_ambiguous")
    evidence.update(status="certified_validation_view_only", reason="single_declared_terminal_role_drift",
                    drift_path_id=str(drift.get("id", "")), drift_step_id=str(drift["steps"][-1].get("id", "")),
                    old_relation_id=old_relation, requested_relation_id=requested_relation,
                    same_owner_type=owner_type, supporting_path_ids=[str(paths[i].get("id", "")) for i in supporting_paths],
                    complete_real_compose_bgp_count=len(raw_triples), complete_joint_binding_count=1,
                    no_compose_triple_or_named_anchor_removed=True,
                    requires_existing_fixed_operator_model_call_budget=True)
    return corrected, evidence


_OPERATOR_LANGUAGE = re.compile(
    r"\b(?:and|or|nor|neither|not|except|excluding|besides|without|only|before|after|between|during|"
    r"first|last|earliest|latest|newest|oldest|recent|most|least|highest|lowest|"
    r"largest|smallest|longest|shortest|more|less|over|under|than|maximum|minimum|"
    r"previous|previously|former|currently|current|different|other|another|instead|rather|"
    r"all|any|every|each|either|both|whose|that|which|while|where|when|with|"
    r"year|date|month|day|age|number|count|total|sum|average|rank|ranking)\b", re.I)


def _explicit_terminal_request(prefix: str, field: tuple[str, ...]) -> str | None:
    """Require an explicit voice/character request or a requested role noun.

    A review's actor goal alone cannot turn every WHO verb into a voice-actor
    request. These are closed English question forms, not entity/predicate ID
    routing. The actual requested predicate remains independently grounded and
    ontology verified. Merely mentioning an actor as the *object* of 'married'
    does not satisfy the copular requested-role form.
    """
    voice = (r"\s*Who\s+(?:(?:did|does|do|is|was|are|were)\s+(?:the\s+)?voice\s+(?:of|for)"
             r"|(?:voiced|voices)\s*|(?:provided|supplied|performed)\s+(?:the\s+)?voice\s+(?:of|for))"
             r"\s+(?:the\s+)?character\s+")
    if re.fullmatch(voice, prefix, re.I):
        return "explicit_voice_character_grammar"
    if field:
        role = r"\s+".join(re.escape(word) for word in field)
        direct = (rf"\s*Who\s+(?:is|was|are|were)\s+(?:the\s+)?{role}"
                  r"\s+(?:of|for)\s+(?:the\s+)?character\s+")
        if re.fullmatch(direct, prefix, re.I):
            return "explicit_copular_declared_role_character_grammar"
    return None


def _stability_answer_role_compile_rejected_compose_terminal_restore(
    question: str, selected_graph: Any, traces: list[dict[str, Any]], ontology: Any,
    *, context_question: str, rewrite_guard: Mapping[str, Any] | None,
) -> tuple[Any | None, dict[str, Any]]:
    """Compile a late zero-model candidate from the current rejected raw Compose.

    Trigger is an actual EMPTY-result rewrite fallback to the very decomposition
    that this question's recorded reviewer rejected. The real reviewed raw
    Compose has a certified complete same-mediator terminal-role correction and
    no need for an additional Operator: a closed single-entity WHO request with
    no numeric, temporal, conjunction, exclusion, or extremum language. Full
    current Grounding payload identity and all relation/anchor bindings must be
    present. The caller may execute at most one candidate only under its real
    saved-request/time budget, and must preserve the incumbent on empty/error.

    This is not the pre-Operator acceptance route. Existing model payloads and
    call counts stay unchanged; no other rank's Operator output is borrowed.
    """
    if isinstance(selected_graph, Mapping):
        selected_graph = SimpleNamespace(**selected_graph)
    evidence = _base_evidence(selected_graph)

    def abstain(reason: str):
        evidence["reason"] = reason
        return None, evidence

    if not _owned(question, context_question, selected_graph):
        return abstain("question_context_owner_mismatch")
    if (not isinstance(rewrite_guard, Mapping) or rewrite_guard.get("fallback_used") is not True
            or rewrite_guard.get("rewritten_failure") != "EMPTY_RESULT"
            or rewrite_guard.get("original_failure") is not None):
        return abstain("actual_successful_original_fallback_from_empty_rewrite_not_attested")
    if not isinstance(traces, list):
        return abstain("current_stage_traces_missing")
    provenance = getattr(selected_graph, "provenance", None)
    source_decomp = provenance.get("decomposition") if isinstance(provenance, Mapping) else None
    if not isinstance(source_decomp, list) or not source_decomp:
        return abstain("fallback_source_decomposition_missing")
    source_triples = _triples(selected_graph)
    source_answer = str(getattr(selected_graph, "answer_var", ""))
    source_incident = [t for t in source_triples or [] if source_answer in (t[0], t[2])]
    if len(source_incident) != 1 or source_incident[0][2] != source_answer:
        return abstain("fallback_source_terminal_role_not_unique")
    source_relation = source_incident[0][1]
    source_domain_method = getattr(ontology, "domain_for_relation", None)
    source_domain = str(source_domain_method(source_relation) or "") if callable(source_domain_method) else ""
    source_role = (_stability_answer_role_words(source_relation.rsplit(".", 1)[-1]), _stability_answer_role_words(source_domain.rsplit(".", 1)[-1]))
    if (not source_domain or source_role not in {_last_role(line) for line in source_decomp}):
        return abstain("selected_source_no_longer_projects_the_rejected_original_role")
    reviewed = []
    for trace in traces:
        if not isinstance(trace, Mapping) or trace.get("stage") != "decomposition_review_and_rewrite":
            continue
        if trace.get("input", {}).get("question") != question:
            return abstain("review_stage_question_owner_mismatch")
        output = trace.get("output")
        for record in output.get("candidates", []) if isinstance(output, Mapping) else []:
            if (not isinstance(record, Mapping) or record.get("status") != "rewritten"
                    or record.get("original_decomposition") != source_decomp
                    or not isinstance(record.get("final_decomposition"), list)):
                continue
            events = record.get("events")
            if not isinstance(events, list):
                continue
            rejected = [e for e in events if isinstance(e, Mapping) and e.get("action") == "review"
                        and e.get("status") == "valid" and e.get("input", {}).get("question") == question
                        and e.get("input", {}).get("decomposition") == source_decomp
                        and isinstance(e.get("model_output"), Mapping)
                        and e["model_output"].get("is_reasonable") is False]
            rewritten = [e for e in events if isinstance(e, Mapping) and e.get("action") == "rewrite"
                         and e.get("status") == "valid" and e.get("input", {}).get("question") == question
                         and isinstance(e.get("model_output"), Mapping)
                         and e["model_output"].get("decomposition") == record["final_decomposition"]]
            if len(rejected) == 1 and len(rewritten) == 1:
                reviewed.append(record["final_decomposition"])
    reviewed_signatures = {tuple(map(str, d)) for d in reviewed}
    if len(reviewed_signatures) != 1:
        return abstain("unique_current_rejected_original_and_rewritten_scope_missing")
    reviewed_decomp = list(next(iter(reviewed_signatures)))
    raw_compose_records = []
    grounding_records = []
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        output = trace.get("output")
        if trace.get("stage") == "compose_graph_generation" and isinstance(output, list):
            raw_compose_records.extend(output)
        if trace.get("stage") == "grounded_semantic_candidates" and isinstance(output, list):
            grounding_records.extend(output)
    compiled = []
    refusals = Counter()
    for record in raw_compose_records:
        if (not isinstance(record, Mapping) or record.get("valid") is not False
                or record.get("error") != "compose output omitted one or more grounded semantic steps"
                or not isinstance(record.get("input"), Mapping)
                or record["input"].get("question") != question):
            continue
        semantic = record["input"]
        corrected, certificate = certify_declared_terminal_compose(
            question, semantic, record.get("output"), ontology, reviewed_decomp, context_question=context_question)
        if corrected is None:
            refusals[str(certificate.get("reason", "uncertified"))] += 1
            continue
        anchors = semantic["anchors"]
        if len(anchors) != 1:
            refusals["closed_single_topic_question_required"] += 1
            continue
        surface = str(anchors[0]["surface"])
        # Mask the actual entity string, so e.g. dates in a movie title do not
        # become invented operator obligations. Require exactly one occurrence
        # and that it is the complete question tail.
        matches = list(re.finditer(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", question, re.I))
        if len(matches) != 1 or not re.fullmatch(r"\s*[?.!]?\s*", question[matches[0].end():]):
            refusals["question_not_closed_single_topic_tail"] += 1
            continue
        prefix = question[:matches[0].start()]
        if (not re.fullmatch(r"\s*Who\s+[A-Za-z ]+\s", prefix, re.I)
                or _OPERATOR_LANGUAGE.search(prefix) or re.search(r"\d", prefix)):
            refusals["question_has_unprovided_operator_or_nonclosed_scope"] += 1
            continue
        requested_field = _stability_answer_role_words(certificate["requested_relation_id"].rsplit(".", 1)[-1])
        question_role_gate = _explicit_terminal_request(prefix, requested_field)
        if question_role_gate is None:
            refusals["question_does_not_explicitly_request_voice_or_declared_terminal_role"] += 1
            continue
        prefix_words = _stability_answer_role_words(prefix)
        if any(prefix_words[i:i + len(source_role[0])] == source_role[0] for i in range(len(prefix_words))):
            refusals["question_explicitly_requests_current_original_role"] += 1
            continue
        matching = [g for g in grounding_records if isinstance(g, Mapping)
                    and g.get("compose_input") == semantic]
        if not matching:
            refusals["exact_current_complete_grounding_payload_missing"] += 1
            continue
        anchor_bindings = []
        relation_bindings = []
        for grounded in matching:
            bindings, relations = grounded.get("anchor_bindings"), grounded.get("relation_bindings")
            if (not isinstance(bindings, Mapping) or set(bindings) != {str(anchors[0]["id"])}
                    or not isinstance(relations, Mapping)):
                return abstain("matching_grounding_binding_scope_incomplete")
            value = bindings[str(anchors[0]["id"])]
            value = value.get("id") if isinstance(value, Mapping) else value
            if not _stability_answer_role_MID.fullmatch(str(value)):
                return abstain("matching_grounding_topic_not_verified_entity")
            expected_relations = {str(s["id"]): _stability_answer_role_relation_id_from_label(s["relation_label"])
                                  for path in semantic["semantic_paths"] for s in path["steps"]}
            if dict(relations) != expected_relations:
                return abstain("matching_grounding_predicates_not_exact_current_input")
            grounded_provenance = grounded.get("provenance")
            grounded_hops = grounded_provenance.get("hops") if isinstance(grounded_provenance, Mapping) else None
            support_indexes = [i for i, path in enumerate(semantic["semantic_paths"])
                               if str(path["id"]) in certificate["supporting_path_ids"]]
            if (not isinstance(grounded_hops, list)
                    or any(i >= len(grounded_hops) or not isinstance(grounded_hops[i], list)
                           or not grounded_hops[i] or not isinstance(grounded_hops[i][0], Mapping)
                           or grounded_hops[i][0].get("reaches_cvt") is not True
                           or grounded_hops[i][0].get("cvt_detection_status") != "resolved"
                           for i in support_indexes)):
                return abstain("current_supporting_mediator_kind_not_resolved")
            anchor_bindings.append(str(value))
            relation_bindings.append(dict(relations))
        if len(set(anchor_bindings)) != 1:
            return abstain("matching_grounding_topic_binding_ambiguous")
        raw = record["output"]
        entity_map = {str(e["id"]): anchor_bindings[0] for e in raw["entities"]}
        triples = [[entity_map.get(str(t["subject"]), str(t["subject"])),
                    _stability_answer_role_relation_id_from_label(t["relation_label"]),
                    entity_map.get(str(t["object"]), str(t["object"]))] for t in raw["triples"]]
        # Source-role override is attested by this question's rejected original
        # decomposition. It is not performed if the original already requests
        # this same terminal role (another failure mechanism must be diagnosed).
        requested_role = _last_role(next(path["goal"] for path in semantic["semantic_paths"]
                                         if str(path["id"]) == certificate["drift_path_id"]))
        if requested_role in {_last_role(line) for line in source_decomp}:
            refusals["fallback_original_already_requests_same_terminal_role"] += 1
            continue
        from .contracts import QueryGraphCandidate
        graph = QueryGraphCandidate(
            graph_id=str(getattr(selected_graph, "graph_id", "")) + "_RAW_DECLARED_TERMINAL",
            triples=triples, answer_var=str(raw["answer_var"]), operators=[],
            score=float(getattr(selected_graph, "score", 0.0)), compose_output=deepcopy(raw),
            provenance={"original_question": question, "decomposition": reviewed_decomp,
                        "stability_rejected_compose_restore": {
                            "source_graph_id": str(getattr(selected_graph, "graph_id", "")),
                            "actual_semantic_rank": record.get("semantic_rank"),
                            "actual_compose_rank": record.get("compose_rank"),
                            "complete_grounding_payload_identity_attested": True,
                            "operator_output_not_borrowed": True,
                            "closed_question_requires_no_additional_operator": True,
                            "question_requested_role_gate": question_role_gate,
                            "certificate": certificate},
                        "anchor_bindings": {str(anchors[0]["id"]): {"id": anchor_bindings[0], "label": surface}}})
        compiled.append((graph, certificate))
    signatures = {json.dumps({"triples": g.triples, "answer_var": g.answer_var, "operators": g.operators},
                             ensure_ascii=False, sort_keys=True) for g, _ in compiled}
    if not compiled or len(signatures) != 1:
        evidence["certificate_refusals"] = dict(refusals)
        return abstain("unique_certified_raw_compose_candidate_missing")
    graph, certificate = compiled[0]
    evidence.update(status="compiled_only", reason="attested_rejected_original_role_restored_from_full_raw_compose",
                    candidate_graph_id=graph.graph_id, complete_raw_bgp_count=len(graph.triples),
                    no_operator_model_call_added=True, operator_output_not_borrowed=True,
                    source_projected_rejected_original_role=" ".join(source_role[0]),
                    requires_real_saved_request_and_time_budget=True, certificate=certificate)
    return graph, evidence



# ---- stability_extrema_scope ----
"""Restore a *recorded* temporal extrema suffix without inventing a path.

This is a pure compiler: no model, embedding, endpoint, training example,
evaluation field or old answer is read.  Callers must run the one proposal
only within saved per-question endpoint/time credits, and accept it only as
a nonempty strict subset of a complete current source execution.

The source is the current executed BGP, not a prior successful answer.  Every
originating Semantic step must be schema-certified and survive in the actual
grounding and rejected Compose output.  The only permitted extension is an
already-produced forward record/date suffix attached to the same answer
entity.  ``latest`` alone does not license guessing either ``from`` or ``to``.
"""


from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
import re
from typing import Any

from .budgeted_schema_repair import (
    SCALAR_TYPES, _compile_grounded, _scalar_property_surface_supported,
    _unique_embedding, _regular_role_stems, schema_type_overlap,
)
from .constraint_repairs import _compose_scalar_scope_safety_original_semantic_scope_certificate as original_semantic_scope_certificate
from .contracts import EntityCandidate as _stability_extrema_scope_EntityCandidate, GroundedSemanticCandidate as _stability_extrema_scope_GroundedSemanticCandidate, QueryGraphCandidate as _stability_extrema_scope_QueryGraphCandidate
from .ontology import relation_id_from_label as _stability_extrema_scope_relation_id_from_label

_stability_extrema_scope_VAR = re.compile(r"V\d+\Z")
_MAX = re.compile(r"\b(?:latest|last|most recent|newest)\b", re.I)
_MIN = re.compile(r"\b(?:earliest|first|oldest)\b", re.I)
_START = re.compile(r"\b(?:from|start(?:ed|ing|s)?|began|begin(?:s|ning)?|since)\b", re.I)
_END = re.compile(r"\b(?:to|until|end(?:ed|ing|s)?|finish(?:ed|ing|es)?)\b", re.I)
_START_LEAVES = frozenset({"from", "from_date", "start", "start_date", "begin", "begin_date"})
_END_LEAVES = frozenset({"to", "to_date", "end", "end_date", "finish", "finish_date"})


def _evidence(reason: str) -> dict[str, Any]:
    return {"status": "abstained", "reason": reason,
            "additional_model_calls": 0, "additional_embedding_calls": 0,
            "additional_endpoint_queries": 0, "uses_evaluation_gold": False,
            "uses_whitelist": False}


def _temporal_extrema(question: str) -> tuple[str, str] | None:
    text = re.sub(r'["“][^"”]*["”]', " ", question)
    text = re.sub(r"\b(?:first|last)\s+name\b", " ", text, flags=re.I)
    maximum, minimum = bool(_MAX.search(text)), bool(_MIN.search(text))
    start, end = bool(_START.search(text)), bool(_END.search(text))
    temporal_extreme = r"(?:latest|last|most\s+recent|newest|earliest|first|oldest)"
    start_phase = r"(?:from|start(?:ed|ing|s)?|began|begin(?:s|ning)?|since)"
    end_phase = r"(?:to|until|end(?:ed|ing|s)?|finish(?:ed|ing|es)?)"
    phase = start_phase if start else end_phase
    # "from City" is a topic preposition, not a start-date obligation.
    phase_joint = bool(re.search(rf"\b(?:{phase}\s+(?:the\s+)?{temporal_extreme}|"
                               rf"{temporal_extreme}\s+(?:date\s+)?{phase})\b", text, re.I))
    # A separately dated/conjoined/counting obligation has no closed proof
    # in this lane.  It must remain unchanged, rather than be silently lost.
    if (maximum == minimum or start == end or not phase_joint
            or re.search(r"\d|\b(?:and|or|not|before|after|during|between|count|how many)\b", text, re.I)):
        return None
    return ("ARGMAX" if maximum else "ARGMIN", "start" if start else "end")


def _closed_question_scope(question, grounded, triples):
    """Reject a named/relational question clause absent from this full scope.

    The small fixed set is grammar/operator vocabulary, not entity or
    predicate routing.  Content vocabulary is derived exclusively from the
    actual topic surfaces and the predicates already present in Compose.
    """
    text = question
    used = {p.get("anchor_ref") for p in grounded.compose_input.get("semantic_paths", [])}
    for anchor in grounded.compose_input.get("anchors", []):
        if anchor.get("id") not in used:
            continue
        pattern = re.escape(str(anchor.get("surface", "")))
        if not pattern or not re.search(pattern, text, re.I):
            return False
        text = re.sub(pattern, " ", text, flags=re.I)
    grammar = {
        "a", "an", "the", "his", "her", "their", "its", "who", "whose", "which", "what",
        "is", "are", "was", "were", "has", "have", "had", "do", "does", "did", "of", "for",
        "by", "with", "from", "to", "in", "at", "among", "that", "holding", "held", "hold",
        "holds", "return", "date", "time", "latest", "earliest", "last", "first", "most",
        "recent", "newest", "oldest", "start", "started", "begin", "began", "end", "ended",
        "until", "since",
    }
    schema_words = re.findall(r"[A-Za-z]+", " ".join(p for _, p, _ in triples).replace("_", " "))
    schema_roots = set().union(*(_regular_role_stems(w) for w in schema_words))
    for word in re.findall(r"[A-Za-z]+", text.casefold()):
        if word in grammar:
            continue
        forms = set(_regular_role_stems(word))
        if len(word) > 6 and word.endswith("al"):
            forms.update(_regular_role_stems(word[:-2]))
        if not forms & schema_roots:
            return False
    return True


def compile_saved_extrema_extension(
    question: str, source_graph: _stability_extrema_scope_QueryGraphCandidate,
    grounded: _stability_extrema_scope_GroundedSemanticCandidate, compose_record: Mapping[str, Any],
    source_semantic: Mapping[str, Any], ontology: Any, *, graph_id: str,
) -> tuple[_stability_extrema_scope_QueryGraphCandidate | None, dict[str, Any]]:
    """Compile one actual failed Compose's date-ordering suffix, or abstain.

    ``source_semantic`` must be the authentic originating Semantic output
    matched to this grounding, not an arbitrarily chosen alternate review.
    All source operators are preserved.  Any existing substantive constraint
    protects the incumbent; no answer replacement is licensed by this helper.
    """
    intent = _temporal_extrema(str(question))
    if intent is None:
        return None, _evidence("temporal_extrema_or_phase_not_unique_and_explicit")
    if (ontology is None or grounded.compose_input.get("question") != question
            or source_graph.provenance.get("original_question") != question
            or grounded.provenance.get("dropped_paths")
            or compose_record.get("valid") is not False
            or not str(compose_record.get("error", "")).startswith("unable to transfer grounded relation for ")
            or compose_record.get("input") != grounded.compose_input):
        return None, _evidence("missing_same_flow_failed_compose_or_question_ownership")
    if (not source_graph.triples or len(source_graph.triples) > 12
            or any(not isinstance(op, dict) or str(op.get("type", "")).upper() != "NO_EQUAL"
                   for op in source_graph.operators)):
        return None, _evidence("preserve_existing_substantive_incumbent_constraint")
    scope_ok, scope = original_semantic_scope_certificate(grounded, source_semantic, ontology=ontology)
    if not scope_ok:
        return None, {**_evidence("original_semantic_scope_not_preserved"), "scope_evidence": scope}
    core, names = _compile_grounded(grounded)
    if not core or len(core) != len(source_graph.triples):
        return None, _evidence("source_is_not_complete_current_grounded_core")
    output = compose_record.get("output")
    if not isinstance(output, Mapping):
        return None, _evidence("actual_compose_output_missing")
    bindings: dict[str, str] = {}
    by_surface: dict[str, str] = {}
    for anchor in grounded.compose_input.get("anchors", []):
        bound = grounded.anchor_bindings.get(str(anchor.get("id", "")))
        if bound is not None:
            for label in (anchor.get("surface", ""), bound.label):
                key = str(label).strip().casefold()
                if key in by_surface and by_surface[key] != bound.entity_id:
                    return None, _evidence("ambiguous_actual_topic_binding")
                if key:
                    by_surface[key] = bound.entity_id
    entities, raw = output.get("entities"), output.get("triples")
    if not isinstance(entities, list) or not isinstance(raw, list) or len(raw) > 14:
        return None, _evidence("invalid_or_unbounded_actual_compose")
    for entity in entities:
        if (not isinstance(entity, Mapping) or str(entity.get("id", "")) in bindings
                or str(entity.get("surface", "")).strip().casefold() not in by_surface):
            return None, _evidence("compose_added_unproved_topic_or_literal")
        bindings[str(entity["id"])] = by_surface[str(entity["surface"]).strip().casefold()]
    triples: list[list[str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None, _evidence("invalid_actual_compose_triple")
        a, b = str(item.get("subject", "")), str(item.get("object", ""))
        p = _stability_extrema_scope_relation_id_from_label(item.get("relation_label", []))
        if (any(v not in bindings and not _stability_extrema_scope_VAR.fullmatch(v) for v in (a, b))
                or not p or not ontology.domain_for_relation(p) or not ontology.range_for_relation(p)):
            return None, _evidence("compose_node_or_predicate_not_schema_proved")
        triples.append([bindings.get(a, a), p, bindings.get(b, b)])
    answer = str(output.get("answer_var", ""))
    core_answer = names.get(str(grounded.compose_input["semantic_paths"][0]["path_output_var"]))
    core_graph = _stability_extrema_scope_QueryGraphCandidate("current_grounded_core", core, core_answer or "", [], grounded.score, {})
    core_mapping = _unique_embedding(core_graph, triples)
    source_mapping = _unique_embedding(source_graph, triples)
    if core_mapping is None or source_mapping is None or source_mapping.get(source_graph.answer_var) != answer:
        return None, _evidence("source_scope_or_answer_role_not_uniquely_preserved")
    embedded_core = Counter((core_mapping.get(a, a), p, core_mapping.get(b, b)) for a, p, b in core)
    embedded_source = Counter((source_mapping.get(a, a), p, source_mapping.get(b, b))
                              for a, p, b in source_graph.triples)
    if embedded_source != embedded_core:
        return None, _evidence("source_and_grounded_core_have_different_scope")
    extra = list((Counter(map(tuple, triples)) - embedded_core).elements())
    if len(extra) != 2:
        return None, _evidence("compose_extension_not_exactly_one_record_date_chain")
    scalar = [t for t in extra if ontology.range_for_relation(t[1]) == "type.datetime"]
    if len(scalar) != 1:
        return None, _evidence("compose_extension_has_no_unique_datetime_leaf")
    owner, attribute, date_var = scalar[0]
    record = [t for t in extra if t != scalar[0]]
    if (len(record) != 1 or record[0][0] != answer or record[0][2] != owner
            or not _stability_extrema_scope_VAR.fullmatch(date_var) or date_var == answer
            or any(a == date_var for a, _, _ in triples)
            or owner in {v for t in embedded_core for v in (t[0], t[2])}):
        return None, _evidence("date_owner_not_unique_new_record_of_same_answer_entity")
    parent = record[0][1]
    if ontology.range_for_relation(parent) != ontology.domain_for_relation(attribute):
        return None, _evidence("date_record_owner_schema_does_not_match")
    answer_types = {ontology.domain_for_relation(p) for a, p, _ in embedded_core if a == answer}
    answer_types.update(ontology.range_for_relation(p) for _, p, b in embedded_core if b == answer)
    parent_domain = ontology.domain_for_relation(parent)
    overlap = {schema_type_overlap(t, parent_domain, ontology) for t in answer_types}
    if (not answer_types or not any(t not in SCALAR_TYPES and "common.topic" in ontology.supertypes(t)
                                    for t in answer_types)
            or ("hard_conflict" in overlap and "compatible" not in overlap)):
        return None, _evidence("existing_answer_entity_or_record_attachment_type_unproved")
    # Ordinary role classes without declared inheritance are not disjoint.
    # Unknown is not a proof of compatibility either: endpoint types in the
    # same paid evidence query must positively settle the owner relation.
    attachment_schema_proved = "compatible" in overlap
    leaf = attribute.rsplit(".", 1)[-1].casefold()
    phase_leaves = _START_LEAVES if intent[1] == "start" else _END_LEAVES
    if leaf not in phase_leaves:
        return None, _evidence("actual_compose_date_phase_differs_from_explicit_question")
    if (not _scalar_property_surface_supported(question, parent)
            or not _scalar_property_surface_supported(question, attribute)):
        return None, _evidence("actual_record_and_date_not_independently_named_by_question")
    old_anchors = {v for t in source_graph.triples for v in (t[0], t[2]) if not _stability_extrema_scope_VAR.fullmatch(v)}
    new_anchors = {v for t in triples for v in (t[0], t[2]) if not _stability_extrema_scope_VAR.fullmatch(v)}
    if not old_anchors or old_anchors != new_anchors:
        return None, _evidence("actual_compose_changed_topic_anchor_scope")
    if not _closed_question_scope(question, grounded, triples):
        return None, _evidence("question_has_unrepresented_topic_or_relation_scope")
    operators = deepcopy(source_graph.operators)
    for op in operators:
        op["input_var"] = source_mapping.get(str(op.get("input_var", "")), str(op.get("input_var", "")))
        if isinstance(op.get("inputs"), list):
            op["inputs"] = [source_mapping.get(str(v), str(v)) for v in op["inputs"]]
    operators.append({"type": intent[0], "inputs": [], "input_var": date_var,
                      "attribute_relation_label": [], "attribute_relation_labels": [],
                      "attribute_relation_id": "", "value": "", "value_type": "",
                      "_attribute_value_already_bound": True, "_order_bound_value": True,
                      "_order_temporal_lexical": True})
    result = _stability_extrema_scope_QueryGraphCandidate(graph_id, triples, answer, operators, float(source_graph.score),
                                 deepcopy(dict(output)), deepcopy(source_graph.provenance))
    certificate = {**_evidence("actual_failed_compose_temporal_suffix_preserves_complete_source"),
                   "status": "compiled", "source_graph_id": source_graph.graph_id,
                   "graph_id": graph_id, "source_variable_mapping": source_mapping,
                   "grounded_core_variable_mapping": core_mapping,
                   "original_semantic_scope": scope, "actual_compose_output_reused": True,
                   "extrema_type": intent[0], "explicit_temporal_phase": intent[1],
                   "record_relation_id": parent, "date_relation_id": attribute,
                   "record_attachment_domain": parent_domain,
                   "record_attachment_schema_proved": attachment_schema_proved,
                   "requires_same_query_actual_owner_type_certificate": not attachment_schema_proved,
                   "date_owner_var": owner, "date_value_var": date_var,
                   "source_bgp_and_operators_preserved": True,
                   "requires_saved_execution_count_and_time_credit": True,
                   "requires_nonempty_strict_subset_of_complete_current_source": True,
                   "requires_complete_same_query_datetime_and_tie_certificate": True,
                   "new_repair_denotation_verified": False}
    result.provenance["stability_extrema_scope"] = deepcopy(certificate)
    return result, certificate


def _stability_extrema_scope_compile_stability_extrema_from_current_stages(
    question: str, source_graph: _stability_extrema_scope_QueryGraphCandidate,
    traces: Sequence[Mapping[str, Any]], ontology: Any, *, graph_id: str,
) -> tuple[_stability_extrema_scope_QueryGraphCandidate | None, dict[str, Any]]:
    """Closed current-stage adapter; never borrow other questions' traces."""
    if _temporal_extrema(question) is None:
        return None, _evidence("temporal_extrema_or_phase_not_unique_and_explicit")
    def latest(stage: str) -> Any:
        values = [t.get("output") for t in traces if isinstance(t, Mapping) and t.get("stage") == stage]
        return values[-1] if values else None
    semantics = latest("semantic_path_generation")
    groundings = latest("grounded_semantic_candidates")
    compositions = latest("compose_graph_generation")
    if not all(isinstance(v, list) for v in (semantics, groundings, compositions)):
        return None, _evidence("current_stage_corpus_missing")
    valid_semantics = [s for s in semantics if isinstance(s, Mapping) and s.get("valid") is True]
    proposals: dict[str, tuple[_stability_extrema_scope_QueryGraphCandidate, dict[str, Any]]] = {}
    for raw in groundings:
        if not isinstance(raw, Mapping) or raw.get("compose_input", {}).get("question") != question:
            continue
        provenance = raw.get("provenance", {})
        index = provenance.get("semantic_graph_index") if isinstance(provenance, Mapping) else None
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(valid_semantics):
            continue
        semantic = valid_semantics[index].get("output")
        if not isinstance(semantic, Mapping):
            continue
        bound = raw.get("anchor_bindings", {})
        anchors = raw.get("compose_input", {}).get("anchors", [])
        if not isinstance(bound, Mapping) or not isinstance(anchors, list):
            continue
        entity_bindings = {str(a.get("id")): _stability_extrema_scope_EntityCandidate(str(bound[str(a.get("id"))]),
                           str(a.get("surface", "")), 0.0)
                           for a in anchors if isinstance(a, Mapping) and str(a.get("id")) in bound
                           and isinstance(bound[str(a.get("id"))], str)}
        grounded = _stability_extrema_scope_GroundedSemanticCandidate(deepcopy(dict(raw["compose_input"])), entity_bindings,
                    deepcopy(dict(raw.get("relation_bindings", {}))), float(raw.get("score", 0)),
                    deepcopy(dict(provenance)))
        for record in compositions:
            if not isinstance(record, Mapping) or record.get("input") != grounded.compose_input:
                continue
            candidate, evidence = compile_saved_extrema_extension(
                question, source_graph, grounded, record, semantic, ontology, graph_id=graph_id)
            if candidate is not None:
                from .lowering import lower_sparql
                proposals[lower_sparql(candidate, limit=None)] = (candidate, evidence)
    if len(proposals) != 1:
        return None, {**_evidence("no_unique_scope_certified_actual_compose_extrema"),
                      "compiled_unique_proposal_count": len(proposals)}
    return next(iter(proposals.values()))


def _stability_extrema_scope_lower_stability_extrema_evidence_query(candidate: _stability_extrema_scope_QueryGraphCandidate) -> str:
    """Read all actual dates in one query; never break ties with LIMIT 1.

    This is an evidence query, not the final answer projection.  The caller
    must use ``resolve_stability_extrema_rows`` before changing its result.
    """
    from .lowering import lower_sparql
    proof = candidate.provenance.get("stability_extrema_scope")
    if not isinstance(proof, Mapping) or proof.get("status") != "compiled":
        raise ValueError("Missing compiled temporal extrema certificate")
    value_var = str(proof.get("date_value_var", ""))
    if not _stability_extrema_scope_VAR.fullmatch(value_var) or not candidate.operators:
        raise ValueError("Missing bound temporal value variable")
    operator = candidate.operators[-1]
    if (operator.get("type") != proof.get("extrema_type") or operator.get("input_var") != value_var
            or not operator.get("_order_bound_value")):
        raise ValueError("Compiled temporal extrema operator changed")
    ordinary = deepcopy(candidate)
    ordinary.operators = ordinary.operators[:-1]
    owner = str(proof.get("date_owner_var", ""))
    parent = str(proof.get("record_relation_id", ""))
    attribute = str(proof.get("date_relation_id", ""))
    suffix = [[ordinary.answer_var, parent, owner], [owner, attribute, value_var]]
    if any(ordinary.triples.count(t) != 1 for t in suffix):
        raise ValueError("Compiled temporal suffix is no longer unique")
    for triple in suffix:
        ordinary.triples.remove(triple)
    query = lower_sparql(ordinary, limit=None)
    expected = "SELECT DISTINCT ?" + ordinary.answer_var
    if not query.startswith(expected + "\nWHERE {") or query.count("\n}") != 1:
        raise ValueError("Evidence query requires one unlimited basic source graph")
    query = query.replace(expected, expected + " ?" + value_var + " ?" + owner + " ?stability_owner_type (DATATYPE(?" + value_var
                          + ") AS ?stability_datetime_type)", 1)
    query = query.replace("\n}", "\n  OPTIONAL { ?" + ordinary.answer_var
                          + " <http://rdf.freebase.com/ns/" + parent + "> ?" + owner + " ."
                          + " OPTIONAL { ?" + owner + " <http://rdf.freebase.com/ns/" + attribute + "> ?"
                          + value_var + " . } }\n  OPTIONAL { ?" + ordinary.answer_var
                          + " <http://rdf.freebase.com/ns/type.object.type> ?stability_owner_type . }\n}", 1)
    return query


def _datetime_value(value: str, datatype: str) -> datetime | None:
    datatype = datatype.rsplit("#", 1)[-1]
    if datatype == "date":
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(Z|[+-]\d{2}:\d{2})?", value)
        if match is None:
            return None
        text = match[1] + "T00:00:00" + (match[2] or "")
    elif datatype == "dateTime":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?", value):
            return None
        text = value
    else:
        # In particular no gYear, signed-year, zero-year or unknown cast.
        return None
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if result.utcoffset() is not None and abs(result.utcoffset()) > timedelta(hours=14):
            return None
        return result
    except (ValueError, OverflowError):
        return None


def _stability_extrema_scope_resolve_stability_extrema_rows(
    candidate: _stability_extrema_scope_QueryGraphCandidate, rows: Sequence[Mapping[str, Any]], *,
    current_source_ids: Sequence[str], current_source_row_count: int,
    query_result_complete: bool, ontology: Any = None,
) -> tuple[list[str] | None, dict[str, Any]]:
    """Use complete same-query dates; retain every tied answer entity.

    All source entities must appear.  A source with no position record is
    ineligible under the explicit question obligation.  Any existing record
    with no date is unknown, not permission to discard that entity.
    Source and date row caps, invalid dates, mixed
    datatypes/precision or timezone-presence all cause abstention.
    """
    proof = candidate.provenance.get("stability_extrema_scope")
    if not isinstance(proof, Mapping) or proof.get("status") != "compiled":
        return None, _evidence("compiled_extrema_certificate_missing")
    source = list(current_source_ids)
    if (not query_result_complete or current_source_row_count != len(source)
            or len(source) < 2 or len(set(source)) != len(source)
            or not all(re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", str(v)) for v in source)
            or not isinstance(rows, Sequence) or not rows or len(rows) > 4096):
        return None, _evidence("source_or_datetime_query_not_complete")
    values: list[tuple[str, datetime]] = []
    timezone_present: set[bool] = set()
    datatypes: set[str] = set()
    owner_types: dict[str, set[str]] = {}
    actual_source_seen: set[str] = set()
    eligible_source: set[str] = set()
    eligible_records: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return None, _evidence("invalid_datetime_query_row")
        identifier = str(row.get(candidate.answer_var, ""))
        identifier = identifier.removeprefix("http://rdf.freebase.com/ns/")
        value = str(row.get(str(proof.get("date_value_var")), ""))
        datatype = str(row.get("stability_datetime_type", ""))
        if identifier not in source:
            return None, _evidence("actual_source_entity_not_certified")
        actual_source_seen.add(identifier)
        actual_type = str(row.get("stability_owner_type", ""))
        if actual_type.startswith("http://rdf.freebase.com/ns/"):
            actual_type = actual_type.removeprefix("http://rdf.freebase.com/ns/")
        if actual_type:
            owner_types.setdefault(identifier, set()).add(actual_type)
        record = str(row.get(str(proof.get("date_owner_var")), "")).removeprefix("http://rdf.freebase.com/ns/")
        if not record:
            if value or datatype:
                return None, _evidence("date_without_actual_position_record")
            continue
        if not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", record):
            return None, _evidence("position_record_is_not_an_actual_resource")
        eligible_source.add(identifier)
        eligible_records.add((identifier, record))
        if not value or not datatype:
            return None, _evidence("actual_eligible_position_record_has_no_date")
        if not datatype.startswith("http://www.w3.org/2001/XMLSchema#"):
            return None, _evidence("datetime_datatype_not_certified")
        parsed = _datetime_value(value, datatype)
        if parsed is None:
            return None, _evidence("datetime_value_not_exactly_normalizable")
        datatypes.add(datatype)
        timezone_present.add(parsed.tzinfo is not None)
        values.append((identifier, parsed))
    if (actual_source_seen != set(source) or not eligible_source
            or len(datatypes) != 1 or len(timezone_present) != 1
            or {v for v, _ in values} != eligible_source):
        return None, _evidence("date_precision_timezone_or_source_date_coverage_unproved")
    if proof.get("requires_same_query_actual_owner_type_certificate"):
        wanted = str(proof.get("record_attachment_domain", ""))
        if not all(any(actual == wanted or (ontology is not None
                       and wanted in ontology.supertypes(actual))
                       for actual in owner_types.get(identifier, set())) for identifier in eligible_source):
            return None, _evidence("actual_source_answer_record_attachment_type_unproved")
    try:
        extreme = (max if proof["extrema_type"] == "ARGMAX" else min)(v for _, v in values)
        accepted_set = {identifier for identifier, value in values if value == extreme}
    except (ValueError, TypeError, OverflowError):
        return None, _evidence("datetime_values_not_comparable")
    if not accepted_set or not accepted_set < set(source):
        return None, _evidence("no_nonempty_strict_subset_of_complete_source")
    accepted = [v for v in source if v in accepted_set]
    return accepted, {**_evidence("complete_same_query_datetime_extrema_with_all_ties"),
                      "status": "qualified", "source_answer_count": len(source),
                      "date_row_count": len(rows), "extremal_entity_count": len(accepted),
                      "all_tied_entities_retained": True,
                      "all_source_entities_covered": True,
                      "eligible_position_entity_count": len(eligible_source),
                      "eligible_position_record_count": len(eligible_records),
                      "all_eligible_position_records_have_actual_date": True,
                      "source_entities_without_any_position_record": [v for v in source if v not in eligible_source],
                      "actual_datetime_datatype": next(iter(datatypes)),
                      "same_query_record_attachment_type_verified": True,
                      "timezone_presence_uniform": True,
                      "new_repair_denotation_verified": True}



# ---- stability_named_constraint ----
"""Compile a closed named-relative scope lost before Compose.

This module performs no retrieval, execution, ranking or answer selection.
It recovers a *whole* question-owned two-path scope, only when the incumbent
has positive schema evidence for a different non-generic terminal role.
Different roles are not an ontology disjointness certificate. The caller must pay
for at most one bounded verification from existing same-question credits.
No entity, predicate, question index, or evaluation answer is stored here.
"""


from collections.abc import Mapping, Sequence
from copy import deepcopy
import re

from .contracts import EntityCandidate as _stability_named_constraint_EntityCandidate, GroundedSemanticCandidate as _stability_named_constraint_GroundedSemanticCandidate, QueryGraphCandidate as _stability_named_constraint_QueryGraphCandidate, validate_semantic_graph
from .lowering import lower_sparql as _stability_named_constraint_lower_sparql
from .ontology import relation_id_from_label as _stability_named_constraint_relation_id_from_label

_stability_named_constraint_MID = re.compile(r"(?:m|g)\.[A-Za-z0-9_]+")


def _plain(value):
    return " ".join(str(value).split()).casefold()


def _stability_named_constraint_words(value):
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(value).casefold())
    return tuple(word[:-1] if len(word) > 3 and word.endswith("s")
                 and not word.endswith(("ss", "'s")) else word for word in words)


def _type_words(value):
    return _stability_named_constraint_words(str(value).rsplit(".", 1)[-1].replace("_", " "))


def _prefix(value, prefix):
    return _plain(value) == _plain(prefix) or _plain(value).startswith(_plain(prefix) + " ")


def _exact_relation(label, ontology):
    relation = _stability_named_constraint_relation_id_from_label(label)
    return relation if relation in ontology.relation_ids else ""


def _terminal_types(graph, ontology):
    result = set()
    for triple in graph.triples:
        if not isinstance(triple, (tuple, list)) or len(triple) != 3:
            return set()
        left, relation, right = map(str, triple)
        if right == graph.answer_var:
            value = ontology.range_for_relation(relation)
        elif left == graph.answer_var:
            value = ontology.domain_for_relation(relation)
        else:
            continue
        if not value:
            return set()
        result.add(value)
    return result


def _stability_named_constraint_compile_named_constraint_scope(
    question, incumbent, original_semantic, grounded_primary, linked_context,
    ontology, *, decomposition, context_question, graph_id="stability_named_constraint",
):
    """Return a whole-scope graph or fail closed with a diagnostic reason.

    ``linked_context`` is the already-provided entity-linker contract for this
    exact question, not a nearest-name lookup. A model-only suffix that is
    absent from the question does not become an alias: the authoritative
    named span must uniquely identify a current provided MID, including any
    role modifiers explicitly present in the question.

    This narrow compiler recognizes a named ``with`` relative and one
    attendance clause. All predicates come from the *original* Semantic
    labels and exact ontology mappings, never an attendance-relation table.
    It does not authorize replacing a correctly typed incumbent.
    """
    evidence = {
        "status": "abstained", "version": "closed-named-relative-scope-v1",
        "additional_model_calls": 0, "additional_embedding_calls": 0,
        "additional_endpoint_queries": 0, "uses_gold_answers": False,
        "uses_gold_sparql": False, "entity_or_relation_whitelist": False,
    }

    def fail(reason, **details):
        return None, {**evidence, "reason": reason, **details}

    graph = getattr(incumbent, "graph", incumbent)
    if (ontology is None or not isinstance(question, str)
            or context_question != question or not isinstance(linked_context, list)
            or not isinstance(graph, _stability_named_constraint_QueryGraphCandidate)
            or not isinstance(graph.provenance, Mapping)
            or not isinstance(graph.triples, list) or not isinstance(graph.operators, list)
            or not isinstance(grounded_primary, _stability_named_constraint_GroundedSemanticCandidate)
            or not isinstance(grounded_primary.provenance, Mapping)
            or not isinstance(grounded_primary.anchor_bindings, Mapping)
            or not isinstance(grounded_primary.relation_bindings, Mapping)):
        return fail("missing_same_question_runtime_contract")
    if (graph.provenance.get("original_question") != question
            or any(key in graph.provenance for key in
                   ("selection_answer_union", "selection_answer_intersection"))):
        return fail("incumbent_is_not_one_owned_execution")
    try:
        semantic = validate_semantic_graph(original_semantic, max_hops=3)
    except (ValueError, TypeError, KeyError):
        return fail("invalid_original_semantic_contract")
    paths = semantic["semantic_paths"]
    if (len(paths) != 2 or len(semantic["anchors"]) != 2
            or not isinstance(decomposition, Sequence) or isinstance(decomposition, str)
            or list(decomposition) != [path["goal"] for path in paths]
            or graph.provenance.get("decomposition") != list(decomposition)):
        return fail("whole_original_decomposition_is_not_two_owned_paths")
    original_anchors = {item["id"]: item["surface"] for item in semantic["anchors"]}
    actual = getattr(grounded_primary, "compose_input", None)
    if (not isinstance(actual, Mapping) or actual.get("question") != question
            or not isinstance(actual.get("semantic_paths"), list)
            or not isinstance(actual.get("anchors"), list)
            or len(actual.get("semantic_paths", [])) != 1
            or len(actual.get("anchors", [])) != 1):
        return fail("primary_grounding_is_not_one_owned_complete_path")
    primary = actual["semantic_paths"][0]
    originals = [path for path in paths if path == primary]
    if len(originals) != 1:
        return fail("primary_semantic_scope_changed_during_grounding")
    primary = originals[0]
    secondary = next(path for path in paths if path is not primary)
    root = primary["anchor_ref"]
    other = secondary["anchor_ref"]
    if (root == other or {root, other} != set(original_anchors)
            or len(primary["steps"]) != 2
            or any(step["direction"] != "forward" for step in primary["steps"])
            or len(secondary["steps"]) != 1
            or secondary["steps"][0]["direction"] != "backward"):
        return fail("not_one_continuous_primary_and_one_inverse_named_constraint")
    binding = grounded_primary.anchor_bindings.get(root)
    if (set(grounded_primary.anchor_bindings) != {root} or not isinstance(binding, _stability_named_constraint_EntityCandidate)
            or not _stability_named_constraint_MID.fullmatch(str(binding.entity_id))
            or _plain(binding.label) != _plain(original_anchors[root])
            or actual["anchors"] != [{"id": root, "surface": original_anchors[root]}]):
        return fail("primary_root_binding_changed_or_unknown")
    source_bindings = graph.provenance.get("anchor_bindings")
    if (not isinstance(source_bindings, Mapping) or set(source_bindings) != {root}
            or not isinstance(source_bindings[root], Mapping)
            or source_bindings[root].get("id") != binding.entity_id
            or _plain(source_bindings[root].get("label")) != _plain(binding.label)):
        return fail("incumbent_is_not_owned_by_the_same_primary_root")
    required_drop = {"path_id": secondary["id"], "anchor_id": other,
                     "reason": "NO_GROUNDED_VARIANTS"}
    if grounded_primary.provenance.get("dropped_paths") != [required_drop]:
        return fail("secondary_path_loss_is_not_explicit_and_unique")
    relations = [_exact_relation(step["relation_label"], ontology)
                 for step in primary["steps"]]
    secondary_relation = _exact_relation(secondary["steps"][0]["relation_label"], ontology)
    if (not all(relations) or not secondary_relation
            or set(grounded_primary.relation_bindings) != {step["id"] for step in primary["steps"]}
            or any(grounded_primary.relation_bindings.get(step["id"]) != relation
                   for step, relation in zip(primary["steps"], relations))):
        return fail("original_labels_do_not_have_identical_exact_grounded_predicates")
    first_range = ontology.range_for_relation(relations[0])
    if ontology.domain_for_relation(relations[1]) not in ontology.supertypes(first_range):
        return fail("original_primary_schema_is_not_continuous")
    expected_type = ontology.range_for_relation(relations[-1])
    if (not expected_type or ontology.domain_for_relation(secondary_relation) != expected_type
            or _type_words(expected_type) != _stability_named_constraint_words(secondary["steps"][0]["relation_label"][-2])):
        return fail("named_constraint_is_not_attached_to_the_same_requested_owner")
    terminal_types = _terminal_types(graph, ontology)
    expected_closure = set(ontology.supertypes(expected_type))
    terminal_closures = {value: set(ontology.supertypes(value)) for value in terminal_types}
    # A declared upper class can still return the requested kind. A broad
    # namespace root (location.location, organization.organization, etc.) is
    # not evidence of a wrong kind even when the snapshot misses inheritance.
    # This is a conservative schema-shape rule, not a list of routed classes.
    def generic_type(value):
        segments = str(value).split(".")
        return (not value or segments[0] in {"type", "common", "freebase", "base"}
                or (len(segments) == 2 and segments[0] == segments[1]))

    if (not terminal_types or "common.topic" not in expected_closure
            or any(generic_type(value) or "common.topic" not in closure
                   or expected_type in closure or value in expected_closure
                   for value, closure in terminal_closures.items())):
        return fail("incumbent_type_is_requested_or_unknown_do_not_replace")
    root_surface = original_anchors[root]
    pattern = (r"^\s*(?:what|which)\s+(.+?)\s+with\s+(.+?)\s+named\s+(.+?)\s+did\s+"
               + re.escape(root_surface) + r"\s+(?:go\s+to|attend)\s*\?\s*$")
    match = re.fullmatch(pattern, question, re.I)
    if not match or _stability_named_constraint_words(match.group(1)) != _type_words(expected_type):
        return fail("question_is_not_a_closed_named_relative_of_exact_requested_type")
    role, named_span = match.group(2).strip(), match.group(3).strip()
    if re.search(r"\b(?:and|or|not|except|before|after|where|who|which|that)\b",
                 " ".join(match.groups()), re.I):
        return fail("question_has_an_unaccounted_extra_obligation")
    constraint_words = _stability_named_constraint_words(secondary["steps"][0]["relation_label"][-1])
    role_words = _stability_named_constraint_words(role)
    if not constraint_words or role_words[-len(constraint_words):] != constraint_words:
        return fail("question_named_role_is_not_original_constraint_leaf")
    modifiers = role_words[:-len(constraint_words)]
    if not named_span or not _prefix(original_anchors[other], named_span):
        return fail("original_secondary_anchor_does_not_own_question_named_span")
    context = [item for item in linked_context if isinstance(item, Mapping)]
    if len(context) != len(linked_context) or any(not _stability_named_constraint_MID.fullmatch(str(item.get("id", "")))
                                               or not str(item.get("name", "")).strip() for item in context):
        return fail("current_linker_context_contract_unknown")
    named = [item for item in context if _prefix(item["name"], named_span)]
    if (len(named) != 1 or len({item["id"] for item in context}) != len(context)
            or named[0]["id"] == binding.entity_id):
        return fail("question_named_span_does_not_identify_one_current_provided_mid")
    secondary_entity = named[0]
    provided_type = ontology.range_for_relation(secondary_relation)
    if (not provided_type or not isinstance(secondary_entity.get("types"), list)
            or _type_words(provided_type) not in {_stability_named_constraint_words(value) for value in secondary_entity["types"]}
            or any(modifier not in _stability_named_constraint_words(secondary_entity["name"]) for modifier in modifiers)):
        return fail("provided_named_entity_does_not_prove_constraint_type_or_question_modifiers")
    roots = [item for item in context if item["id"] == binding.entity_id]
    if len(roots) != 1 or _plain(roots[0]["name"]) != _plain(root_surface):
        return fail("current_provided_root_contract_disagrees")
    if (not isinstance(roots[0].get("types"), list)
            or _type_words(ontology.domain_for_relation(relations[0]))
            not in {_stability_named_constraint_words(value) for value in roots[0]["types"]}):
        return fail("current_provided_root_does_not_prove_primary_domain_type")
    for item in context:
        if item["id"] in {binding.entity_id, secondary_entity["id"]}:
            continue
        if not isinstance(item.get("types"), list) or "Type" not in item["types"]:
            return fail("extra_provided_entity_obligation_unaccounted")
    first, last = primary["steps"]
    expected_goal = (
        f"What is the {first['relation_label'][-1]} of {first['relation_label'][-2]} {root_surface}? "
        f"What is the {last['relation_label'][-1]} of the {first['relation_label'][-1]}?"
    )
    expected_constraint = (
        f"{original_anchors[other]} is the {secondary['steps'][0]['relation_label'][-1]} "
        f"of the {secondary['steps'][0]['relation_label'][-2]}."
    )
    if (_plain(primary["goal"]) != _plain(expected_goal)
            or _plain(secondary["goal"]) != _plain(expected_constraint)):
        return fail("original_goals_do_not_close_primary_and_named_relative_scope")
    named_tail = _plain(original_anchors[other])[len(_plain(named_span)):].strip()
    if named_tail and any(word in _stability_named_constraint_words(question) for word in _stability_named_constraint_words(named_tail)):
        return fail("model_anchor_suffix_is_question_required_do_not_discard")
    allowed_names = {_plain(root_surface): binding.entity_id,
                     _plain(secondary_entity["name"]): secondary_entity["id"]}
    for operator in graph.operators:
        if (not isinstance(operator, Mapping) or operator.get("type") != "NO_EQUAL"
                or operator.get("input_var") != graph.answer_var
                or operator.get("inputs", []) or operator.get("attribute_relation_label", [])
                or operator.get("attribute_relation_labels", [])
                or operator.get("value_type") != "entity"
                or _plain(operator.get("value")) not in allowed_names
                or (operator.get("_value_entity_id") is not None
                    and operator["_value_entity_id"] != allowed_names.get(_plain(operator.get("value"))))):
            return fail("incumbent_has_non_echo_operator_scope_that_cannot_be_transferred")
    variables = ["V0", "V1"]
    triples = [[binding.entity_id, relations[0], variables[0]],
               [variables[0], relations[1], variables[1]],
               [variables[1], secondary_relation, secondary_entity["id"]]]
    operators = deepcopy(graph.operators)
    for operator in operators:
        operator["input_var"] = variables[1]
    provenance = deepcopy(graph.provenance)
    provenance.update({
        "anchor_bindings": {
            root: {"id": binding.entity_id, "label": binding.label, "score": binding.score},
            other: {"id": secondary_entity["id"], "label": secondary_entity["name"],
                    "source": "same_question_provided_entity_contract"},
        },
        "relation_bindings": {**grounded_primary.relation_bindings,
                              secondary["steps"][0]["id"]: secondary_relation},
        "grounded_semantic": deepcopy(grounded_primary.provenance),
        "stability_named_constraint": {
            "version": evidence["version"], "source_graph_id": graph.graph_id,
            "contract_origin": "current_question_named_span_and_current_provided_mid",
            "question_named_span": named_span, "original_model_anchor": original_anchors[other],
            "provided_entity_name": secondary_entity["name"],
            "model_only_suffix_rejected": named_tail,
            "entity_alias_rule_created": False,
            "whole_question_owned_scope_preserved": True,
            "incumbent_wrong_terminal_types": sorted(terminal_types),
            "ontology_type_disjointness_proven": False,
            "requested_type": expected_type,
        },
    })
    candidate = _stability_named_constraint_QueryGraphCandidate(
        graph_id=graph_id, triples=triples, answer_var=variables[1],
        operators=operators, score=graph.score, compose_output={}, provenance=provenance,
    )
    candidate.sparql = _stability_named_constraint_lower_sparql(candidate)
    return candidate, {
        **evidence, "status": "compiled_requires_paid_endpoint_verification",
        "reason": "whole_named_relative_scope_restored",
        **provenance["stability_named_constraint"],
    }



# ---- stability_runtime ----
"""Late, bounded stability repairs from this question's actual stage evidence.

Nothing in this module reads a question index, a baseline file or gold answers.
All models have finished before it runs. Unknown/incomplete evidence leaves
the existing result untouched. At most one new repair query is attempted.
"""
from copy import deepcopy
import re
import time

from .contracts import EntityCandidate as _stability_runtime_EntityCandidate, ExecutedGraph, GroundedSemanticCandidate as _stability_runtime_GroundedSemanticCandidate, QueryGraphCandidate as _stability_runtime_QueryGraphCandidate
from .evaluation import canonicalize_answer_id as _stability_runtime_canonicalize_answer_id
from .lowering import lower_sparql as _stability_runtime_lower_sparql
from .runtime_context import has_saved_repair_budget, execute_repair_from_saved_budget, charge_repair_preparation, remember_kg_label_rows, current_query_budget
_stability_runtime_compile_named_constraint_scope = _stability_named_constraint_compile_named_constraint_scope
_stability_runtime_compile_stability_extrema_from_current_stages = _stability_extrema_scope_compile_stability_extrema_from_current_stages
_stability_runtime_lower_stability_extrema_evidence_query = _stability_extrema_scope_lower_stability_extrema_evidence_query
_stability_runtime_resolve_stability_extrema_rows = _stability_extrema_scope_resolve_stability_extrema_rows
_stability_runtime_compile_rejected_compose_terminal_restore = _stability_answer_role_compile_rejected_compose_terminal_restore

_ENTITY = re.compile(r'[mg]\.[A-Za-z0-9_]+')
_NAMED_SHAPE = re.compile(r'^\s*(?:what|which)\b.*\bwith\b.*\bnamed\b.*\bdid\b.*(?:go\s+to|attend)\s*\?\s*$',re.I)
_DATE_PHASE = re.compile(r'\b(?:from|start|end|to)\b.*\b(?:earliest|latest|first|last)\b|\b(?:earliest|latest|first|last)\b.*\b(?:from|start|end|to)\b',re.I)


def _source_execution(question, result, kg):
    raw = result.get('selected_graph')
    ids = result.get('answer_ids')
    traces = result.get('traces')
    if (result.get('failure') or not isinstance(raw,dict) or not isinstance(ids,list)
            or not ids or not isinstance(traces,list) or len(ids)!=len(set(ids))
            or any(not isinstance(v,str) or not _ENTITY.fullmatch(v) for v in ids)
            or (raw.get('provenance') or {}).get('original_question')!=question):
        return None
    try:
        graph=_stability_runtime_QueryGraphCandidate(raw['graph_id'],deepcopy(raw['triples']),raw['answer_var'],
              deepcopy(raw['operators']),float(raw.get('score',0)),{},deepcopy(raw.get('provenance',{})))
        query=_stability_runtime_lower_sparql(graph,limit=None)
    except (KeyError,TypeError,ValueError):
        return None
    state=current_query_budget()
    saved=state.cache.get((id(kg),query)) if state is not None and state.question==question else None
    if not isinstance(saved,tuple) or len(saved)!=2 or not isinstance(saved[0],list):return None
    source_rows=saved[0]
    if (len(source_rows)!=len(ids) or any(not isinstance(row,dict) for row in source_rows)
            or {_stability_runtime_canonicalize_answer_id(row.get(graph.answer_var,'')) for row in source_rows}!=set(ids)):
        return None
    # An ID/signature match is insufficient: prove the unlimited current source
    # query and its entire returned ID count, with the exact final projection.
    for trace in reversed(traces):
        if not isinstance(trace,dict):continue
        output=trace.get('output')
        if trace.get('stage')!='deterministic_sparql_lowering_and_execution' or not isinstance(output,list):
            continue
        for item in output:
            if (isinstance(item,dict) and item.get('graph_id')==graph.graph_id
                    and item.get('sparql')==query and not item.get('error')
                    and type(item.get('row_count')) is int and type(item.get('answer_count')) is int
                    and item.get('row_count')==item.get('answer_count')==len(ids)):
                return ExecutedGraph(graph,list(ids),deepcopy(result.get('answers',[])),item['row_count'])
    return None


def _named_proposal(question, source, traces, linked_context, ontology):
    if not _NAMED_SHAPE.fullmatch(question):return None,None
    semantics=[];diagnostics=None;proposals={}
    for trace in traces:
        stage=trace.get('stage');output=trace.get('output')
        if stage=='semantic_path_generation':
            semantics=[x for x in output if isinstance(x,dict) and x.get('valid') is True] if isinstance(output,list) else []
            diagnostics=None
        elif stage=='complete_semantic_graph_guard' and isinstance(output,dict) and output.get('rejected_count',0):
            # The compact trace does not record the post-filter index mapping.
            semantics=[]
        elif stage=='semantic_subgraph_grounding_and_ranking':diagnostics=output
        elif stage=='grounded_semantic_candidates' and isinstance(output,list) and isinstance(diagnostics,dict):
            entities={}
            for record in diagnostics.get('semantic_graphs',[]):
                for anchor in record.get('anchors',[]):
                    for entity in anchor.get('candidates',[]):
                        entities[(record.get('graph_index'),anchor.get('anchor_id'),entity.get('id'))]=entity
            for raw in output:
                if not isinstance(raw,dict):continue
                provenance=raw.get('provenance') or {};index=provenance.get('semantic_graph_index')
                if type(index) is not int or not 0<=index<len(semantics):continue
                bindings={};valid=True
                for anchor,entity_id in (raw.get('anchor_bindings') or {}).items():
                    evidence=entities.get((index,anchor,entity_id))
                    if not isinstance(evidence,dict) or 'label' not in evidence or 'score' not in evidence:
                        valid=False;break
                    bindings[anchor]=_stability_runtime_EntityCandidate(entity_id,evidence['label'],evidence['score'])
                if not valid:continue
                grounded=_stability_runtime_GroundedSemanticCandidate(deepcopy(raw['compose_input']),bindings,
                         deepcopy(raw['relation_bindings']),raw['score'],deepcopy(provenance))
                entry=semantics[index]
                proposal,proof=_stability_runtime_compile_named_constraint_scope(question,source,entry.get('output'),grounded,
                    linked_context,ontology,decomposition=entry.get('decomposition'),context_question=question)
                if proposal is not None:proposals[proposal.sparql]=(proposal,proof)
    return next(iter(proposals.values())) if len(proposals)==1 else (None,None)


def _with_names_and_sentinel(graph, limit=257):
    query=_stability_runtime_lower_sparql(graph,limit=None)
    prefix='SELECT DISTINCT ?'+graph.answer_var
    if not query.startswith(prefix+'\nWHERE {') or 'stability_name' in query:
        raise ValueError('Unknown final projection for same-query labels')
    query=query.replace(prefix,prefix+' ?stability_name',1)
    end=query.rfind('\n}')
    if end<0:raise ValueError('Missing source query boundary')
    optional=('\n  OPTIONAL { ?'+graph.answer_var+' <http://rdf.freebase.com/ns/type.object.name> '
              '?stability_name . FILTER(LANG(?stability_name) = "en") }')
    return query[:end]+optional+query[end:]+f'\nLIMIT {limit}'


def _certified_query_names(rows,answer_var):
    name_sets={}
    for row in rows:
        label=row.get('stability_name') if isinstance(row,dict) else None
        if isinstance(label,str) and label.strip():
            identifier=_stability_runtime_canonicalize_answer_id(row.get(answer_var,''))
            key=' '.join(label.casefold().split())
            name_sets.setdefault(identifier,{}).setdefault(key,set()).add(label)
    names={i:sorted(next(iter(values.values())))[0] for i,values in name_sets.items() if len(values)==1}
    conflicts=sorted(i for i,values in name_sets.items() if len(values)>1)
    return names,conflicts


def _apply_stability_repairs(question,result,*,ontology,kg,linked_context):
    if not has_saved_repair_budget():return result
    # Keep the common path cheap: most questions do not satisfy any closed form.
    actor_possible=isinstance(result.get('rewrite_guard'),dict) and result['rewrite_guard'].get('fallback_used') is True
    if not (_NAMED_SHAPE.fullmatch(question) or _DATE_PHASE.search(question) or actor_possible):return result
    started=time.monotonic();source=_source_execution(question,result,kg)
    if source is None:
        charge_repair_preparation(started);return result
    traces=result['traces'];proposal=None;proof=None;mode=None
    if _NAMED_SHAPE.fullmatch(question):
        proposal,proof=_named_proposal(question,source,traces,linked_context,ontology)
        mode='named_relative'
    if proposal is None and _DATE_PHASE.search(question):
        proposal,proof=_stability_runtime_compile_stability_extrema_from_current_stages(question,source.graph,traces,ontology,
                        graph_id='stability_temporal_extrema')
        mode='temporal_extrema'
    if proposal is None and actor_possible:
        proposal,proof=_stability_runtime_compile_rejected_compose_terminal_restore(question,source.graph,traces,ontology,
                        context_question=question,rewrite_guard=result.get('rewrite_guard'))
        mode='declared_terminal_role'
    try:
        query=(_stability_runtime_lower_stability_extrema_evidence_query(proposal) if mode=='temporal_extrema'
               else _with_names_and_sentinel(proposal)) if proposal is not None else None
    except (ValueError,TypeError,KeyError):query=None
    charge_repair_preparation(started)
    if proposal is None or query is None:return result
    rows,receipt=execute_repair_from_saved_budget(kg,query)
    finalization=time.monotonic()
    audit={'mode':mode,'compile':proof,'execution':receipt,'status':'original_preserved',
           'source_graph_id':source.graph.graph_id,'source_answer_ids':list(source.answer_ids),
           'external_model_calls':0,'sparql':query}
    ids=None
    if rows is not None:
        if mode=='temporal_extrema':
            audit['raw_evidence_rows']=deepcopy(rows)
            audit['answer_filter_mode']='local_extrema_from_complete_same_query_source_record_date_and_type'
            ids,accepted=_stability_runtime_resolve_stability_extrema_rows(proposal,rows,current_source_ids=source.answer_ids,
                          current_source_row_count=source.row_count,query_result_complete=True,ontology=ontology)
            audit['acceptance']=accepted
        elif 0<len(rows)<257:
            values=[_stability_runtime_canonicalize_answer_id(r.get(proposal.answer_var,'')) for r in rows]
            if all(_ENTITY.fullmatch(v) for v in values):
                ids=list(dict.fromkeys(values))
                remember_kg_label_rows([{'entity':r.get(proposal.answer_var),'label':r.get('stability_name','')} for r in rows])
    if ids:
        # Rewrite only after a complete positive certificate. Preserve the real
        # query as executed (including evidence filtering for temporal extrema).
        prior_names={a.get('id'):a.get('label') for a in source.answers if isinstance(a,dict)
                     and isinstance(a.get('label'),str) and a['label'].strip()
                     and _stability_runtime_canonicalize_answer_id(a['label'])!=a.get('id')}
        row_names,conflicts=_certified_query_names(rows,proposal.answer_var)
        audit['same_query_label_conflict_ids']=conflicts
        answers=[{'id':i,'label':prior_names.get(i) or row_names.get(i) or i} for i in ids]
        proposal.sparql=query
        graph={'graph_id':proposal.graph_id,'triples':proposal.triples,'answer_var':proposal.answer_var,
               'operators':proposal.operators,'score':proposal.score,'provenance':proposal.provenance}
        graph['provenance']['stability_executed_query']=query
        if mode=='temporal_extrema':
            graph['provenance']['stability_local_extrema_from_complete_evidence']=audit['acceptance']
        result=dict(result,answer_ids=ids,answers=answers,selected_graph=graph)
        audit.update(status='applied',answer_ids=ids,answer_count=len(ids),raw_row_count=len(rows))
    traces.append({'stage':'late_stability_scope_repair','input':{'question':question},'output':audit})
    charge_repair_preparation(finalization)
    return result


def apply_stability_repairs(question,result,*,ontology,kg,linked_context):
    """Optional repair failures must never turn an existing success into error."""
    state=current_query_budget()
    charged_before=state.repair_seconds if state is not None else 0.0
    started=time.monotonic()
    try:
        return _apply_stability_repairs(question,result,ontology=ontology,kg=kg,linked_context=linked_context)
    except Exception as exc:
        if isinstance(result,dict) and isinstance(result.get('traces'),list):
            result['traces'].append({'stage':'late_stability_scope_repair','input':{'question':question},
                'output':{'status':'original_preserved','reason':'optional_stability_evidence_unavailable',
                          'error':f'{type(exc).__name__}: {str(exc)[:240]}','external_model_calls':0}})
        return result
    finally:
        if state is not None:
            elapsed=time.monotonic()-started
            already_charged=state.repair_seconds-charged_before
            unbilled=max(0.0,elapsed-already_charged)
            state.repair_seconds+=unbilled
            state.repair_preparation_seconds+=unbilled



__all__ = ['_stability_answer_role_compile_rejected_compose_terminal_restore', '_stability_extrema_scope_compile_stability_extrema_from_current_stages', '_stability_extrema_scope_lower_stability_extrema_evidence_query', '_stability_extrema_scope_resolve_stability_extrema_rows', '_stability_named_constraint_compile_named_constraint_scope', 'apply_stability_repairs', 'certify_declared_terminal_compose', 'compile_same_owner_declared_projection', 'compile_saved_extrema_extension', 'same_scope_declared_projection_execution', 'temporal_role_witness', 'temporal_source_role_bounce_veto']
