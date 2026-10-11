"""Zero-call protection against a demonstrable answer-role bounce.

This module does not generate facts, execute queries, or read evaluation labels.
In particular, refusing a late mascot/participant answer for an explicitly
corroborated date request does not make the incumbent answer correct. A final
replacement is possible only when an already executed graph has the *identical*
full conjunctive scope, with one additional requested terminal projection.

Vocabulary for positive entity roles comes from the loaded ontology. Temporal
words below are datatype grammar, not entity/relation/sample routing lists.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import json
import re
from types import SimpleNamespace
from typing import Any

from .evaluation import canonicalize_answer_id
from .ontology import FreebaseOntology, _search_tokens, relation_id_from_label, relation_label_from_id
from .source_role_execution_safety import _decomposition_lines


_VAR = re.compile(r"V\d+\Z")
_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_TEMPORAL_HEAD = re.compile(
    r"^\s*(?:(?:in|on)\s+)?(?:what|which)\s+(year|date|day|month)\b", re.I)
_WH_ROLE = re.compile(
    r"\b(?:what|which|who)\s+(?:is|are|was|were)\s+(?:the\s+)?"
    r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)", re.I)
_TEMPORAL_WORDS = frozenset({"year", "date", "day", "month"})
_PRIMITIVE = frozenset({"type.int", "type.float", "type.datetime", "type.text",
                        "type.boolean", "type.rawstring", "type.uri"})


def _words(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", str(value).replace("_", " ").casefold()))


def _last_role(value: Any) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    roles = _WH_ROLE.findall(str(value))
    return (_words(roles[-1][0]), _words(roles[-1][1])) if roles else None


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
    if triples is None or not _VAR.fullmatch(answer):
        evidence["reason"] = "invalid_graph_or_answer_variable"
        return False, evidence
    incident = [t for t in triples if answer in (t[0], t[2])]
    if len(incident) != 1 or incident[0][2] != answer:
        evidence["reason"] = "answer_not_unique_forward_terminal"
        return False, evidence
    subject, predicate, _ = incident[0]
    anchor_edges = [t for t in triples if t[0] == subject and t[1] == predicate and _MID.fullmatch(t[2])]
    if not anchor_edges:
        evidence["reason"] = "not_same_owner_constrained_topic_role_bounce"
        return False, evidence
    range_method = getattr(ontology, "range_for_relation", None)
    declared_range = str(range_method(predicate) or "") if callable(range_method) else ""
    range_words = set(_words(declared_range.rsplit(".", 1)[-1]))
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
    qwords = _words(question)
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
            if (_words(str(relation).rsplit(".", 1)[-1]) == field
                    and _words(str(domain).rsplit(".", 1)[-1]) == owner
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
    if old_owner != owner_type or not _VAR.fullmatch(old[0]):
        evidence["reason"] = "old_and_requested_terminal_owners_differ_or_unknown"
        return None, evidence
    # Require a real anchored owner context, not a floating endpoint field.
    nonprojection = [t for t in triples if t != old]
    if not nonprojection or not any(old[0] in (t[0], t[2]) for t in nonprojection):
        evidence["reason"] = "requested_owner_not_bound_by_existing_context"
        return None, evidence
    variables = {term for t in triples for term in (t[0], t[2]) if _VAR.fullmatch(term)}
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
        canonical = [canonicalize_answer_id(value) for value in ids]
        if len(set(canonical)) != len(canonical) or not all(_MID.fullmatch(v) for v in canonical):
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
        old_relation = relation_id_from_label(last.get("relation_label"))
        if (role in decomposition_roles
                and _words(requested_relation.rsplit(".", 1)[-1]) == role[0]
                and _words(owner_type.rsplit(".", 1)[-1]) == role[1]
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
            or relation_id_from_label(first.get("relation_label")) != old_relation
            or domain_method(old_relation) != owner_type):
        return abstain("drifted_terminal_owner_not_attested_by_same_anchor_field")
    supporting_paths = []
    for index, path in enumerate(paths):
        steps = path["steps"]
        if (index != drift_index and str(path.get("anchor_ref", "")) == root and len(steps) == 2
                and all(isinstance(s, Mapping) for s in steps)
                and steps[0].get("direction") == "backward" and steps[1].get("direction") == "forward"
                and relation_id_from_label(steps[0].get("relation_label")) == old_relation
                and relation_id_from_label(steps[1].get("relation_label")) == requested_relation):
            supporting_paths.append(index)
    if not supporting_paths:
        return abstain("current_independent_same_mediator_requested_relation_missing")
    corrected = deepcopy(dict(semantic_graph))
    corrected["semantic_paths"][drift_index]["steps"][-1]["relation_label"] = relation_label_from_id(requested_relation)
    raw_triples = [
        (str(t["subject"]), relation_id_from_label(t["relation_label"]), str(t["object"]))
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
            relation = relation_id_from_label(step.get("relation_label"))
            next_states = []
            for binding, covered in states:
                current = binding.get(str(step["from"]))
                for edge_index, edge in enumerate(raw_triples):
                    left, predicate, right = edge
                    if predicate != relation:
                        continue
                    source, target = (left, right) if step["direction"] == "forward" else (right, left)
                    if source != current or not _VAR.fullmatch(target):
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


def compile_rejected_compose_terminal_restore(
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
    source_role = (_words(source_relation.rsplit(".", 1)[-1]), _words(source_domain.rsplit(".", 1)[-1]))
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
        requested_field = _words(certificate["requested_relation_id"].rsplit(".", 1)[-1])
        question_role_gate = _explicit_terminal_request(prefix, requested_field)
        if question_role_gate is None:
            refusals["question_does_not_explicitly_request_voice_or_declared_terminal_role"] += 1
            continue
        prefix_words = _words(prefix)
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
            if not _MID.fullmatch(str(value)):
                return abstain("matching_grounding_topic_not_verified_entity")
            expected_relations = {str(s["id"]): relation_id_from_label(s["relation_label"])
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
                    relation_id_from_label(t["relation_label"]),
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
