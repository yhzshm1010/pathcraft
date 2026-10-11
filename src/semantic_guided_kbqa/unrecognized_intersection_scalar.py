"""Pure shadow compiler for an intersection that missed a scalar obligation.

Requires actual prior applied-intersection metadata with an explicitly null
scalar spec, and a full reviewed single path from the same current flow.
Existing source BGP/operators are preserved; one schema-exact scalar edge and
the independently normalized comparator are appended. No models, execution,
file/Gold access, answer selection, sample routing or global guard change.
"""
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import re
from typing import Any

from .budgeted_schema_repair import (
    NUMERIC_TYPES, STRING_TYPES, _existing_exact_scalar_intent, _number,
    _one_explicit_scalar_spec, lower_budgeted_scalar_query,
)
from .comparison_number_normalization import normalize_comparison_constraint_spec, parse_english_number
from .contracts import ExecutedGraph, QueryGraphCandidate
from .ontology import relation_id_from_label


_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_VAR = re.compile(r"V\d+\Z")


def _words(value):
    words = set()
    for token in re.findall(r"[a-z]+", str(value).replace("_", " ").casefold()):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        words.add(token)
    return words


def _head(question):
    match = re.match(r"\s*(?:(?:in|of|among)\s+)?(?:what|which)\s+([A-Za-z]+)\b", question, re.I)
    return _words(match[1]) if match else set()


def _entity(value):
    if isinstance(value, Mapping):
        return str(value.get("id", value.get("entity_id", ""))), str(value.get("label", value.get("name", "")))
    return str(getattr(value, "entity_id", "")), str(getattr(value, "label", ""))


def _evidence(reason):
    return {"status": "not_compiled", "reason": reason, "shadow_only": True,
            "model_calls": 0, "endpoint_queries": 0, "embedding_calls": 0,
            "uses_gold_answers_or_sparql": False, "uses_entity_relation_or_sample_whitelist": False,
            "production_activation_authorized": False, "compiler_selects_answers": False,
            "requires_actual_full_source_execution_and_saved_query_time_budget": True}


def intersection_scalar_question_necessary(question):
    """Question-only necessary shape; never supplies missing prior authority."""
    if re.search(r'["“”]|\b(?:not|never|or|either|except|approximately|about|between|most|least|first|last|latest|earliest)\b', str(question), re.I):
        return _evidence("question_not_one_unquoted_numeric_comparison")
    spec = _one_explicit_scalar_spec(str(question), [], "type.float")
    if spec is None:
        spec = normalize_comparison_constraint_spec(str(question))
    if not spec or spec.get("value_type") != "number" or spec.get("type") not in {"LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"}:
        return _evidence("prior_missing_spec_is_not_now_one_proven_numeric_comparison")
    if len(re.findall(r"\band\b", str(question), re.I)) != 1:
        return _evidence("question_has_other_conjuncts_or_no_same_owner_and_clause")
    return {**_evidence("necessary_question_shape_only_actual_prior_and_full_scope_still_required"), "necessary_gate": True,
            "normalized_spec": deepcopy(spec)}


def intersection_scalar_obligation_necessary(question, prior):
    if (not isinstance(prior, Mapping) or prior.get("status") != "applied"
            or "explicit_constraint_obligation" not in prior or prior["explicit_constraint_obligation"] is not None
            or prior.get("grounded_anchor_obligation_count") != 1):
        return _evidence("not_proven_applied_intersection_with_explicitly_unrecognized_scalar")
    proof = intersection_scalar_question_necessary(question)
    if proof.get("necessary_gate"):
        proof["reason"] = "necessary_prior_and_number_only_full_scope_still_required"
    return proof


def _program_core(graph):
    read = graph.get if isinstance(graph, Mapping) else lambda key, default=None: getattr(graph, key, default)
    triples, operators, answer = read("triples"), read("operators"), read("answer_var")
    if (not isinstance(triples, list) or not triples or not isinstance(operators, list)
            or any(not isinstance(t, (list, tuple)) or len(t) != 3
                   or any(not isinstance(v, str) or not v for v in t) for t in triples)
            or any(not isinstance(op, Mapping) for op in operators)
            or not isinstance(answer, str) or not _VAR.fullmatch(answer)):
        return None
    return tuple(tuple(t) for t in triples), answer, deepcopy(operators)


def _complete_entity_execution(execution):
    values, count = getattr(execution, "answer_ids", None), getattr(execution, "row_count", None)
    return (isinstance(values, list) and bool(values) and isinstance(count, int) and not isinstance(count, bool)
            and count == len(values) and all(isinstance(v, str) and _MID.fullmatch(v) for v in values)
            and len(set(values)) == len(values))


def _finite_score(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def resolve_unrecognized_intersection_source(
    question: str, selected: ExecutedGraph, executions: Sequence[ExecutedGraph],
    prior_selection: Mapping[str, Any] | None, stages: Sequence[Mapping[str, Any]],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Attest the actual full source before any paid query; never completes IDs.

    Runtime operands, the complete passed selector record, its latest actual
    stage, captured previews/programs, and endpoint counts must all agree.
    Uniqueness is tested over *all* matching executions, before completeness.
    No artifacts, ontology, models, endpoint calls, or answer synthesis.
    """
    proof = {**_evidence("current_source_not_attested"), "pure_current_execution_attestation": True,
             "answer_ids_synthesized": False, "additional_endpoint_queries": 0,
             "additional_model_calls": 0, "additional_embedding_calls": 0}

    def abstain(reason):
        return None, {**proof, "reason": reason}

    if (not isinstance(question, str) or not question or not isinstance(prior_selection, Mapping)
            or not isinstance(stages, Sequence) or isinstance(stages, (str, bytes))
            or not isinstance(executions, Sequence) or isinstance(executions, (str, bytes))):
        return abstain("actual_passed_prior_or_current_runtime_contract_missing")
    rankings = [s for s in stages if isinstance(s, Mapping) and s.get("stage") == "glm_final_graph_ranking"]
    if (not rankings or not isinstance(rankings[-1].get("output"), Mapping)
            or rankings[-1]["output"] != prior_selection):
        return abstain("passed_prior_not_identical_to_latest_actual_final_ranking_record")
    ranking_input = rankings[-1].get("input", {})
    if (not isinstance(ranking_input, Mapping)
            or "question" in ranking_input and ranking_input["question"] != question):
        return abstain("latest_final_ranking_not_owned_by_current_question")
    graph = getattr(selected, "graph", None)
    provenance = getattr(graph, "provenance", None)
    marker = provenance.get("selection_answer_intersection") if isinstance(provenance, Mapping) else None
    gate = prior_selection.get("hard_conjunctive_answer_intersection_gate")
    if (not isinstance(marker, Mapping) or not isinstance(gate, Mapping)
            or not intersection_scalar_obligation_necessary(question, gate).get("necessary_gate")
            or provenance.get("original_question") != question or "selection_answer_union" in provenance):
        return abstain("current_question_owned_intersection_and_explicit_null_applied_prior_missing")
    left_id, right_id = marker.get("left_graph_id"), marker.get("right_graph_id")
    if (not isinstance(left_id, str) or not left_id or not isinstance(right_id, str) or not right_id or left_id == right_id
            or getattr(graph, "graph_id", None) != left_id or prior_selection.get("selected_graph_id") != left_id
            or prior_selection.get("hard_conjunctive_answer_intersection_gate_source_graph_id") != left_id
            or gate.get("source_graph_id") != left_id or gate.get("selected_graph_id") != left_id
            or gate.get("left_graph_id") != left_id or gate.get("right_graph_id") != right_id
            or not _finite_score(marker.get("score_margin")) or not _finite_score(gate.get("score_margin"))
            or marker["score_margin"] != gate["score_margin"]):
        return abstain("marker_gate_and_current_source_graph_identities_or_margin_conflict")
    actual = {}
    for graph_id in (left_id, right_id):
        matches = [e for e in executions if getattr(getattr(e, "graph", None), "graph_id", None) == graph_id]
        if len(matches) != 1 or not _complete_entity_execution(matches[0]) or _program_core(matches[0].graph) is None:
            return abstain("unique_complete_actual_operand_execution_missing")
        execution = matches[0]
        owned = getattr(execution.graph, "provenance", None)
        if (not isinstance(owned, Mapping) or owned.get("original_question") != question
                or "selection_answer_intersection" in owned or "selection_answer_union" in owned):
            return abstain("actual_operand_not_question_owned_or_is_postprocessed")
        actual[graph_id] = execution
    left, right = actual[left_id], actual[right_id]
    if (not _complete_entity_execution(selected) or _program_core(graph) != _program_core(left.graph)
            or getattr(graph, "score", None) != left.graph.score
            or provenance.get("anchor_bindings") != left.graph.provenance.get("anchor_bindings")):
        return abstain("current_complete_intersection_core_score_or_root_binding_drifted")
    intersected = set(left.answer_ids) & set(right.answer_ids)
    if (not intersected or set(selected.answer_ids) != intersected or not intersected < set(left.answer_ids)
            or gate.get("source_answer_count") != len(left.answer_ids)
            or gate.get("selected_answer_count") != len(selected.answer_ids)
            or any(not isinstance(gate.get(k), int) or isinstance(gate.get(k), bool)
                   for k in ("source_answer_count", "selected_answer_count", "operator_count"))
            or gate.get("operator_count") != len(left.graph.operators)
            or gate.get("operator_count") != len(right.graph.operators) or isinstance(gate.get("operator_count"), bool)):
        return abstain("current_ids_or_gate_counts_not_the_exact_complete_actual_intersection")
    snapshots = prior_selection.get("candidates")
    if not isinstance(snapshots, list):
        return abstain("actual_captured_selector_operand_contracts_missing")
    for graph_id, execution in actual.items():
        records = [c for c in snapshots if isinstance(c, Mapping) and c.get("graph_id") == graph_id]
        if len(records) != 1 or _program_core(records[0]) != _program_core(execution.graph):
            return abstain("captured_selector_operand_core_missing_duplicate_or_conflicting")
        record = records[0]
        values, count, truncated = record.get("answer_ids"), record.get("answer_count"), record.get("answers_truncated")
        if (not isinstance(values, list) or not values or any(not isinstance(v, str) or not _MID.fullmatch(v) for v in values)
                or len(set(values)) != len(values) or not isinstance(count, int) or isinstance(count, bool)
                or count != len(execution.answer_ids) or not set(values) <= set(execution.answer_ids)
                or not (truncated is True and len(values) < count
                        or truncated is False and len(values) == count and set(values) == set(execution.answer_ids))):
            return abstain("captured_selector_preview_count_or_truncation_conflicts_with_actual_full_ids")
        answers = record.get("answers")
        if answers is not None and (not isinstance(answers, list) or any(not isinstance(a, Mapping) for a in answers)
                                    or [a.get("id") for a in answers] != values):
            return abstain("captured_selector_answer_objects_conflict_with_preview_ids")
        score_key = "left_rule_score" if graph_id == left_id else "right_rule_score"
        if (not _finite_score(execution.graph.score) or not _finite_score(record.get("rule_score"))
                or not _finite_score(gate.get(score_key))
                or record["rule_score"] != execution.graph.score or gate[score_key] != execution.graph.score):
            return abstain("captured_selector_actual_operand_and_gate_scores_conflict")
    execution_rows = {left_id: [], right_id: []}
    for stage in stages:
        if not isinstance(stage, Mapping) or stage.get("stage") != "deterministic_sparql_lowering_and_execution":
            continue
        stage_input, output = stage.get("input", {}), stage.get("output")
        if (not isinstance(stage_input, Mapping) or "question" in stage_input and stage_input["question"] != question
                or not isinstance(output, list)):
            return abstain("same_flow_actual_execution_trace_malformed_or_other_question")
        for row in output:
            if isinstance(row, Mapping) and row.get("graph_id") in execution_rows:
                execution_rows[row["graph_id"]].append(row)
    for graph_id, execution in actual.items():
        rows = execution_rows[graph_id]
        if len(rows) != 1:
            return abstain("same_flow_actual_operand_execution_trace_missing_or_duplicate")
        row = rows[0]
        if (any(not isinstance(row.get(k), int) or isinstance(row.get(k), bool) for k in ("row_count", "answer_count"))
                or row["row_count"] != execution.row_count or row["answer_count"] != len(execution.answer_ids)):
            return abstain("same_flow_actual_operand_row_and_answer_counts_conflict")
    proof.update(status="resolved_actual_complete_source", reason="all_current_operand_selector_and_execution_identities_attested",
                 source_graph_id=left_id, secondary_graph_id=right_id, source_answer_count=len(left.answer_ids),
                 current_intersection_answer_count=len(selected.answer_ids), prior_and_latest_actual_ranking_full_record_identical=True,
                 all_matching_executions_counted_before_completeness=True, current_ids_exact_actual_operand_intersection=True,
                 captured_programs_previews_counts_scores_and_actual_execution_counts_agree=True,
                 operands_question_owned_and_not_postprocessed=True, actual_source_returned_without_synthesis=True)
    return left, proof


def compile_unrecognized_intersection_scalar(
    question: str, source_graph: Any, prior: Mapping[str, Any], reviewed_semantic: Mapping[str, Any],
    original_anchor_candidates: Mapping[str, Sequence[Mapping[str, Any]]], topic_contract: Mapping[str, Any],
    ontology: Any, *, graph_id: str = "unrecognized_intersection_scalar_shadow",
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Topic authority is explicit, not an invented canonical-label alias.

    ``topic_contract`` must be supplied from the complete current question's
    already-provided/linker MID inputs. A selected graph's anchor subset does
    not create that contract. No existing name-sensitive guard is relaxed.
    """
    proof = intersection_scalar_obligation_necessary(question, prior)
    if not proof.get("necessary_gate"):
        return None, proof

    def abstain(reason):
        proof.update(status="not_compiled", reason=reason)
        return None, proof

    paths = reviewed_semantic.get("semantic_paths", [])
    if ontology is None or len(paths) != 1 or not isinstance(paths[0], Mapping):
        return abstain("reviewed_original_has_other_paths_or_unknown_scope")
    path = paths[0]
    steps = path.get("steps", [])
    if len(steps) != 2 or any(not isinstance(s, Mapping) for s in steps):
        return abstain("reviewed_path_not_exact_entity_prefix_plus_scalar_field")
    first, scalar_step = steps
    root = str(path.get("anchor_ref", ""))
    if (first.get("from") != root or first.get("to") != scalar_step.get("from")
            or scalar_step.get("to") != path.get("path_output_var") or scalar_step.get("direction") != "forward"
            or first.get("direction") not in {"forward", "backward"}):
        return abstain("reviewed_original_variable_owner_or_direction_not_connected")
    primary = relation_id_from_label(first.get("relation_label"))
    field = relation_id_from_label(scalar_step.get("relation_label"))
    owner_type = str(ontology.range_for_relation(primary) if first["direction"] == "forward" else ontology.domain_for_relation(primary))
    field_domain, field_range = str(ontology.domain_for_relation(field)), str(ontology.range_for_relation(field))
    numeric_string = field_range in STRING_TYPES | {"type.enumeration"}
    if (not owner_type or owner_type != field_domain or "common.topic" not in set(ontology.supertypes(owner_type))
            or field_range not in NUMERIC_TYPES and not numeric_string):
        return abstain("source_entity_owner_and_exact_numeric_field_not_same_declared_type")
    if _head(question) != _words(owner_type.rsplit(".", 1)[-1]):
        return abstain("question_answer_kind_does_not_match_exact_declared_scalar_owner")
    if (not isinstance(topic_contract, Mapping) or topic_contract.get("question") != question
            or topic_contract.get("binding_source") != "current_question_topic_mid_contract"
            or not isinstance(topic_contract.get("topics"), list) or len(topic_contract["topics"]) != 1):
        return abstain("complete_current_question_topic_mid_contract_missing_or_more_than_one_topic")
    anchor, label = _entity(topic_contract["topics"][0])
    original_roots = [a for a in reviewed_semantic.get("anchors", []) if isinstance(a, Mapping) and str(a.get("id")) == root]
    provenance = getattr(source_graph, "provenance", {})
    actual_topics = provenance.get("anchor_bindings", {}) if isinstance(provenance, Mapping) else {}
    actual_id, actual_label = _entity(actual_topics.get(root)) if isinstance(actual_topics, Mapping) else ("", "")
    root_candidates = original_anchor_candidates.get(root)
    if (not _MID.fullmatch(anchor) or not label or actual_id != anchor or actual_label.casefold() != label.casefold()
            or len(original_roots) != 1 or str(original_roots[0].get("surface", "")).casefold() != label.casefold()
            or not isinstance(root_candidates, (list, tuple)) or len(root_candidates) != 1
            or _entity(root_candidates[0]) != (anchor, label)):
        return abstain("actual_source_reviewed_linker_and_input_topic_identity_do_not_align")
    if provenance.get("original_question") != question:
        return abstain("source_graph_not_owned_by_this_exact_current_question")
    owner_words = _words(owner_type.rsplit(".", 1)[-1])
    field_words = _words(field.rsplit(".", 1)[-1])
    clause = re.search(r"\band\s+(?:the\s+)?(?P<body>[A-Za-z ]+?)\s+(?:is|has|have)\s+(?:greater|more|higher|larger|less|fewer|lower|smaller|at\s+most|at\s+least|no\s+more|no\s+less)\s+(?:than\s+)?", question, re.I)
    if not clause or _words(clause["body"]) != owner_words | field_words:
        return abstain("question_does_not_explicitly_conjoin_same_owner_and_exact_field")
    # An extra capitalized topic in the entity prefix cannot be hidden by
    # the authoritative contract. This does not rewrite names or infer MIDs.
    prefix = question[:clause.start()].strip()
    caps = list(re.finditer(r"\b[A-Z][A-Za-z]*\b", prefix))
    for token in caps:
        if token.start() == 0:
            continue
        if not _words(token[0]) <= _words(label):
            return abstain("question_prefix_has_unrepresented_named_topic_token")
    spec = proof["normalized_spec"]
    goal = str(path.get("goal", ""))
    goal_spec = _one_explicit_scalar_spec(goal, [label], "type.float")
    if goal_spec is None:
        goal_spec = normalize_comparison_constraint_spec(goal)
    if (not goal_spec or goal_spec.get("type") != spec["type"] or _number(goal_spec.get("value")) != _number(spec["value"])
            or not owner_words | field_words <= _words(goal)):
        return abstain("reviewed_goal_does_not_attest_same_owner_field_numeric_comparator_literal")
    scope = []
    referenced = {str(path.get("anchor_ref"))} | {str(s.get(k)) for s in steps for k in ("from", "to")}
    for item in reviewed_semantic.get("anchors", []):
        if not isinstance(item, Mapping):
            return abstain("original_anchor_contract_invalid")
        key, surface = str(item.get("id", "")), str(item.get("surface", ""))
        if key == root:
            continue
        if key in referenced or original_anchor_candidates.get(key) != []:
            return abstain("extra_original_anchor_is_path_referenced_bound_or_unknown")
        literal = _number(surface)
        if literal is None:
            literal = _number(parse_english_number(surface))
        if literal is not None and literal == _number(spec["value"]):
            scope.append({"anchor_id": key, "surface": surface, "kind": "exact_normalized_scalar_threshold"})
        elif _words(surface) == owner_words and _head(question) == owner_words:
            scope.append({"anchor_id": key, "surface": surface, "kind": "unbound_unused_kind_proved_by_qhead_and_same_declared_owner"})
        else:
            return abstain("unused_original_topic_kind_or_literal_not_positively_accounted_for")
    if (len([x for x in scope if x["kind"] == "exact_normalized_scalar_threshold"]) > 1
            or len([x for x in scope if x["kind"].startswith("unbound_unused_kind")]) > 1):
        return abstain("duplicate_kind_or_literal_scope_ambiguous")
    if _existing_exact_scalar_intent(source_graph, field, spec, ontology):
        return abstain("source_already_expresses_same_owner_field_comparator_literal")
    triples, operators = getattr(source_graph, "triples", None), getattr(source_graph, "operators", None)
    if (not isinstance(triples, list) or len(triples) != 1 or not isinstance(operators, list)
            or any(op.get("type") != "NO_EQUAL" for op in operators)
            or prior.get("operator_count") != len(operators)):
        return abstain("source_has_other_bgp_or_substantive_operators_not_this_prefix")
    answer = str(source_graph.answer_var)
    expected = [anchor, primary, answer] if first["direction"] == "forward" else [answer, primary, anchor]
    if not _VAR.fullmatch(answer) or triples[0] != expected or source_graph.graph_id != prior.get("source_graph_id"):
        return abstain("actual_source_primary_direction_id_or_entity_answer_does_not_match_reviewed_prefix")
    intersection = provenance.get("selection_answer_intersection")
    if (not isinstance(intersection, Mapping) or intersection.get("left_graph_id") != prior.get("left_graph_id")
            or intersection.get("right_graph_id") != prior.get("right_graph_id")
            or source_graph.graph_id not in {prior.get("left_graph_id"), prior.get("right_graph_id")}):
        return abstain("prior_applied_intersection_not_the_current_graph_postprocessing")
    result = deepcopy(source_graph)
    result.graph_id, result.sparql = graph_id, ""
    # This is a new query denotation, not the old answer-ID intersection.
    # Keep the actual prior marker as evidence, without claiming that its
    # postprocessing applies to answers returned by the new scalar query.
    archived_intersection = result.provenance.pop("selection_answer_intersection")
    value_var = "V" + str(max(int(v[1:]) for t in triples for v in (str(t[0]), str(t[2])) if _VAR.fullmatch(v)) + 1)
    result.triples.append([answer, field, value_var])
    op = {"type": spec["type"], "inputs": [], "input_var": value_var, "attribute_relation_id": "",
          "attribute_relation_label": [], "attribute_relation_labels": [], "value": spec["value"], "value_type": "number",
          "_attribute_value_already_bound": True, "_numeric_comparison_cast": True,
          "_numeric_cast_kind": "float" if numeric_string or field_range == "type.float" else "integer"}
    if numeric_string:
        op["_budgeted_numeric_string_value_guard"] = True
    result.operators.append(op)
    proof.update(status="compiled_shadow_not_executed", reason="actual_intersection_missed_same_owner_reviewed_numeric_obligation",
                 source_graph_id=source_graph.graph_id, reviewed_path_id=path["id"], scalar_relation_id=field,
                 primary_relation_id=primary, owner_type=owner_type, owner_var=answer, scalar_var=value_var,
                 prior_full_source_answer_count=prior.get("source_answer_count"),
                 prior_selection_answer_intersection_archived=deepcopy(archived_intersection),
                 old_selection_answer_intersection_marker_removed=True,
                 original_unused_anchor_scope=scope, topic_mid_contract_source=topic_contract["binding_source"],
                 actual_topic_id=anchor, canonical_topic_label=label,
                 canonical_label_fully_present_in_question=bool(re.search(re.escape(label), question, re.I)),
                 canonical_label_alias_inferred_or_name_changed=False,
                 provided_current_question_mid_authority_explicitly_used=True,
                 no_global_name_sensitive_guard_or_other_repair_trigger_relaxed=True,
                 source_bgp_and_operators_exactly_preserved=True, source_entity_answer_projection_unchanged=True,
                 restored_scalar_obligation_from_single_original_reviewed_path=True,
                 require_actual_uncapped_nonempty_strict_subset_of_full_source_before_replacement=True,
                 strict_subset_of_postprocessed_incumbent_not_required_or_safety_proof=True)
    result.provenance["unrecognized_intersection_scalar_shadow"] = deepcopy(proof)
    return result, proof


def compile_unrecognized_intersection_from_current_stages(
    question: str, source_graph: Any, stages: Sequence[Mapping[str, Any]], ontology: Any,
    *, graph_id: str = "unrecognized_intersection_scalar_shadow",
    linked_entity_context: Sequence[Mapping[str, Any]] | None = None,
    linked_entity_context_question: str | None = None,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Current stages or explicitly question-owned existing context cache.

    A cache is not a fabricated stage and cannot override actual stage scope.
    Its owner must be the exact question used for the caller's cache lookup.
    Never reads historical artifacts or fetches missing context.
    """
    prior = None
    context = None
    reviewed_decompositions = set()
    pending = None
    source_program_verified = False
    proposals = []
    corpus_failure = None
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        if stage.get("stage") == "linked_entity_context":
            next_context = stage.get("output")
            if context is not None and next_context != context:
                corpus_failure = "same_flow_linked_context_contracts_conflict"
            context = next_context
        elif stage.get("stage") == "decomposition_review_and_rewrite":
            output = stage.get("output", {})
            review_input = stage.get("input", {})
            if (not isinstance(review_input, Mapping) or review_input.get("question") != question
                    or not isinstance(output, Mapping) or not isinstance(output.get("final_candidates"), list)):
                corpus_failure = "same_flow_current_question_review_contract_missing_or_conflicting"
                continue
            if any(isinstance(c, Mapping) and c.get("source") == "review_rewrite"
                   and (not isinstance(c.get("decomposition"), list) or not c["decomposition"]
                        or any(not isinstance(s, str) or not s for s in c["decomposition"]))
                   for c in output["final_candidates"]):
                corpus_failure = "reviewed_rewrite_proposal_contract_malformed"
            reviewed_decompositions.update(tuple(c["decomposition"]) for c in output["final_candidates"]
                                           if isinstance(c, Mapping) and c.get("source") == "review_rewrite"
                                           and isinstance(c.get("decomposition"), list)
                                           and all(isinstance(s, str) for s in c["decomposition"]))
        elif stage.get("stage") == "semantic_path_generation":
            output = stage.get("output", [])
            if pending is not None:
                corpus_failure = "earlier_reviewed_semantic_corpus_has_no_complete_grounding_contract"
            semantic_input = stage.get("input", {})
            if (not isinstance(semantic_input, Mapping) or semantic_input.get("question") != question
                    or not isinstance(output, list)):
                corpus_failure = "semantic_stage_not_this_complete_current_question_flow"
                pending = None
            else:
                # The actual pipeline grounds the valid semantic entries,
                # including original fallbacks, and graph_index refers to
                # that valid-only list. Do not reindex after review filtering.
                # Every validated reviewed proposal enters the closed corpus,
                # including competing scope that cannot compile or ground.
                valid_entries = [entry for entry in output if isinstance(entry, Mapping) and entry.get("valid") is True]
                pending = [(index, entry) for index, entry in enumerate(valid_entries)
                           if isinstance(entry.get("decomposition"), list)
                           and all(isinstance(s, str) for s in entry["decomposition"])
                           and tuple(entry["decomposition"]) in reviewed_decompositions]
        elif stage.get("stage") == "semantic_subgraph_grounding_and_ranking" and pending is not None:
            output = stage.get("output", {})
            graphs = output.get("semantic_graphs") if isinstance(output, Mapping) else None
            if not isinstance(graphs, list):
                corpus_failure = "reviewed_semantic_corpus_grounding_contract_missing"
            else:
                if any(not isinstance(g, Mapping) or not isinstance(g.get("graph_index"), int)
                       or isinstance(g.get("graph_index"), bool) for g in graphs):
                    corpus_failure = "reviewed_semantic_corpus_grounding_index_contract_malformed"
                for index, entry in pending:
                    matches = [g for g in graphs if isinstance(g, Mapping) and g.get("graph_index") == index
                               and not isinstance(g.get("graph_index"), bool)]
                    if (len(matches) != 1 or not isinstance(entry.get("output"), Mapping)
                            or matches[0].get("dropped_paths") != [] or not isinstance(matches[0].get("anchors"), list)):
                        corpus_failure = "validated_reviewed_scope_missing_duplicate_or_dropped_grounding_contract"
                        continue
                    graph, semantic = matches[0], entry["output"]
                    anchors = graph["anchors"]
                    if (any(not isinstance(a, Mapping) or not isinstance(a.get("anchor_id"), str)
                            or not isinstance(a.get("candidates"), list) for a in anchors)
                            or len({a["anchor_id"] for a in anchors}) != len(anchors)):
                        corpus_failure = "validated_reviewed_scope_anchor_grounding_contract_incomplete"
                        continue
                    paths = semantic.get("semantic_paths")
                    if (not isinstance(paths, list) or any(not isinstance(p, Mapping)
                                                          or p.get("goal") not in entry["decomposition"] for p in paths)):
                        corpus_failure = "validated_reviewed_scope_goal_not_owned_by_reviewed_decomposition"
                        continue
                    candidates = {a["anchor_id"]: a["candidates"] for a in anchors}
                    semantic_anchors = semantic.get("anchors")
                    if (not isinstance(semantic_anchors, list)
                            or any(not isinstance(a, Mapping) or not isinstance(a.get("id"), str) for a in semantic_anchors)
                            or len({a["id"] for a in semantic_anchors}) != len(semantic_anchors)
                            or set(candidates) != {a["id"] for a in semantic_anchors}):
                        corpus_failure = "reviewed_semantic_and_complete_grounding_anchor_scopes_conflict"
                        continue
                    proposals.append((semantic, candidates))
            pending = None
        elif stage.get("stage") == "glm_final_graph_ranking":
            output = stage.get("output", {})
            prior = output.get("hard_conjunctive_answer_intersection_gate") if isinstance(output, Mapping) else None
            actual = [g for g in output.get("candidates", []) if isinstance(g, Mapping)
                      and g.get("graph_id") == source_graph.graph_id] if isinstance(output, Mapping) else []
            source_program_verified = bool(len(actual) == 1 and actual[0].get("triples") == source_graph.triples
                                           and actual[0].get("operators") == source_graph.operators
                                           and actual[0].get("answer_var") == source_graph.answer_var)
    if pending is not None:
        corpus_failure = "reviewed_semantic_corpus_has_no_complete_current_grounding_contract"
    if corpus_failure is not None:
        return None, _evidence(corpus_failure)
    context_origin = "actual_stage"
    if linked_entity_context is not None:
        if (linked_entity_context_question != question or not isinstance(linked_entity_context, list)
                or any(not isinstance(c, Mapping) for c in linked_entity_context)):
            return None, _evidence("supplied_current_question_cache_context_owner_or_contract_conflicts")
        if context is not None and context != linked_entity_context:
            return None, _evidence("actual_stage_and_current_question_cache_context_conflict")
        if context is None:
            context = linked_entity_context
            context_origin = "current_question_cache"
    if not isinstance(context, list) or any(not isinstance(c, Mapping) for c in context):
        return None, _evidence("complete_current_question_topic_contract_context_missing")
    if not source_program_verified:
        return None, _evidence("actual_final_selector_source_query_program_not_verified")
    topic_contract = {"question": question, "binding_source": "current_question_topic_mid_contract", "topics": context}
    candidates, rejections = [], []
    for semantic, anchors in proposals:
        candidate, proof = compile_unrecognized_intersection_scalar(question, source_graph, prior, semantic, anchors,
                                                                    topic_contract, ontology, graph_id=graph_id)
        (candidates if candidate is not None else rejections).append((candidate, proof))
    if not candidates:
        return rejections[0] if rejections else (None, _evidence("same_current_flow_reviewed_full_scalar_semantic_missing"))
    if rejections:
        return None, {**_evidence("competing_validated_reviewed_scope_not_fully_proven"),
                      "reviewed_proposal_count": len(proposals), "rejected_reviewed_scope_proofs": [p for _, p in rejections]}
    keys = {(tuple(tuple(t) for t in c.triples), c.answer_var, str(c.operators)) for c, _ in candidates}
    if len(keys) != 1:
        return None, _evidence("multiple_reviewed_scalar_restoration_programs_are_ambiguous")
    candidate, proof = candidates[0]
    proof["same_current_flow_reviewed_semantic_linker_and_actual_prior_verified"] = True
    proof["actual_final_selector_source_query_program_verified"] = True
    proof["entire_validated_reviewed_same_flow_proposal_corpus_checked"] = True
    proof["validated_reviewed_proposal_count"] = len(proposals)
    proof["context_origin"] = context_origin
    proof["current_question_cache_owner_explicitly_verified"] = context_origin == "current_question_cache"
    candidate.provenance["unrecognized_intersection_scalar_shadow"] = deepcopy(proof)
    return candidate, proof


def lower_unrecognized_intersection_scalar(candidate: QueryGraphCandidate, *, limit: int = 256) -> str:
    return lower_budgeted_scalar_query(candidate, limit=limit)
