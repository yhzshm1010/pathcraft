"""Pure, final restoration of a proved unsupported answer intersection.

This is deliberately not a plural-parser change and not a scalar repair.
It may return only the actual complete pre-intersection execution. A missing
explicit scalar condition is still missing after restoration. No query, model,
file, Gold, relinking, or answer-ID synthesis is performed here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .budgeted_schema_repair import (
    NUMERIC_TYPES, STRING_TYPES, _validated_provided_scalar_spec, schema_type_overlap,
)
from .comparison_number_normalization import parse_english_calendar_date, parse_english_number
from .contracts import ExecutedGraph
from .ontology import relation_id_from_label


_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_VAR = re.compile(r"V\d+\Z")
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


def _complete(execution: Any) -> bool:
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


def _preview_agrees(raw: Mapping[str, Any], execution: ExecutedGraph) -> bool:
    values, count, truncated = raw.get("answer_ids"), raw.get("answer_count"), raw.get("answers_truncated")
    if (not isinstance(values, list) or not isinstance(count, int) or isinstance(count, bool)
            or count != len(execution.answer_ids) or len(set(map(str, values))) != len(values)
            or not set(map(str, values)) <= set(map(str, execution.answer_ids))):
        return False
    if truncated is True:
        return len(values) < count
    return truncated is False and len(values) == count and set(map(str, values)) == set(map(str, execution.answer_ids))


def _number(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value).replace(",", ""))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def _literal_matches(surface: str, spec: Mapping[str, Any]) -> bool:
    if spec.get("value_type") == "number":
        value = _number(surface)
        if value is None:
            parsed = parse_english_number(surface)
            value = _number(parsed) if parsed is not None else None
        return value is not None and value == _number(spec.get("value"))
    parsed = parse_english_calendar_date(surface)
    return str(surface).strip() == str(spec.get("value")) or parsed == str(spec.get("value"))


def _signature(spec: Mapping[str, Any]) -> tuple[str, str, str]:
    value = _number(spec.get("value")) if spec.get("value_type") == "number" else None
    normalized = format((value if value else Decimal(0)).normalize(), "f") if value is not None else str(spec.get("value", ""))
    return str(spec.get("type", "")), normalized, str(spec.get("value_type", ""))


def unsupported_intersection_challenger(
    question: str, selected: ExecutedGraph, executions: Sequence[ExecutedGraph], ontology: Any,
    prior_selection: Mapping[str, Any] | None, *, stages: Sequence[Mapping[str, Any]],
    question_anchor_contract: Mapping[str, Any] | None = None,
    legacy_plural_request: Callable[[str], bool] | None = None,
    native_spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
    spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
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
        if len(matches) != 1 or not _complete(matches[0]) or _core(matches[0].graph) is None:
            return abstain("unique_complete_operator_free_actual_operand_execution_missing")
        execution = matches[0]
        owned = execution.graph.provenance
        if (not isinstance(owned, Mapping) or str(owned.get("original_question", "")) != question
                or owned.get("selection_answer_intersection") or owned.get("selection_answer_union")):
            return abstain("operand_is_postprocessed_or_not_current_question_owned")
        actual[graph_id] = execution
    left, right = actual[left_id], actual[right_id]
    if not _complete(selected) or _core(selected.graph) != _core(left.graph):
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
    if _MID.fullmatch(s) and _VAR.fullmatch(o) and left.graph.answer_var == o:
        anchor, owner, direction = s, o, "forward"
    elif _MID.fullmatch(o) and _VAR.fullmatch(s) and left.graph.answer_var == s:
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
                or relation_id_from_label(first.get("relation_label")) != primary):
            return abstain("another_validated_primary_scope_conflicts_with_actual_source")
        roots = [a for a in sem.get("anchors", []) if isinstance(a, Mapping) and a.get("id") == root]
        if len(roots) != 1 or _name(roots[0].get("surface")) != _name(label):
            return abstain("actual_original_root_surface_and_bound_topic_contract_conflict")
        if len(steps) == 1:
            continue  # A later fallback cannot supply the scalar witness.
        if len(steps) != 2:
            return abstain("original_semantic_has_unpreserved_additional_joins")
        last = steps[1]
        attribute = relation_id_from_label(last.get("relation_label"))
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
    bridges = [t for t in rt if t[0] == anchor and _VAR.fullmatch(t[2])]
    if len(bridges) != 1:
        return abstain("secondary_root_entity_bridge_not_unique")
    _, extra, bridge = bridges[0]
    if (extra == primary or extra in set(ontology.reverse_for_relation(primary))
            or rt.count((bridge, primary, str(right.graph.answer_var))) != 1
            or not _VAR.fullmatch(str(right.graph.answer_var)) or bridge == right.graph.answer_var):
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
            expressed |= _words(relation_id_from_label(step.get("relation_label")))
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
