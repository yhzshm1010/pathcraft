"""Shadow-only completion of a missing named participant on an existing CVT.

This is NOT restoration of the original Semantic: its middle owner/relation
may have changed during actual grounding, and its unused topic was never
attached. An explicit question participant, a unique original-hop inverse and
declared CVT schema jointly propose a new, fully scoped shadow graph. No
execution, model, file, Gold or production selection lives in this module.
"""
from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
from typing import Any

from .contracts import QueryGraphCandidate
from .budgeted_schema_repair import schema_type_overlap
from .ontology import relation_id_from_label


def _entity(value):
    if isinstance(value, Mapping):
        return str(value.get("id", value.get("entity_id", ""))), str(value.get("label", value.get("name", "")))
    return str(getattr(value, "entity_id", "")), str(getattr(value, "label", ""))


def _proof(reason):
    return {"status": "not_compiled", "reason": reason, "shadow_only": True,
            "is_faithful_original_semantic_restoration": False,
            "model_calls": 0, "endpoint_queries": 0, "embedding_calls": 0,
            "uses_gold_answers_or_gold_sparql": False, "uses_sample_entity_or_relation_whitelist": False,
            "production_activation_authorized": False, "compiler_selects_answers": False,
            "future_runtime_requires_actual_failure_or_empty_and_saved_budget": True}


def compile_shadow_missing_cvt_participant(
    question: str, original: Mapping[str, Any], grounded: Mapping[str, Any],
    rejection: Mapping[str, Any], topic_bindings: Mapping[str, Any],
    linked_context: Sequence[Mapping[str, Any]], ontology: Any, *, graph_id: str = "missing_cvt_participant_shadow",
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    proof = _proof("missing_scope_not_uniquely_proven")

    def abstain(reason):
        proof["reason"] = reason
        return None, proof

    if ontology is None or rejection.get("reason") != "pure_ontology_expansion_schema_discontinuity":
        return abstain("requires_actual_pure_ontology_schema_rejection")
    original_paths = original.get("semantic_paths", [])
    composition = grounded.get("compose_input", {})
    paths = composition.get("semantic_paths", []) if isinstance(composition, Mapping) else []
    metadata = grounded.get("provenance", {})
    if (len(original_paths) != 1 or len(paths) != 1 or not isinstance(metadata, Mapping)
            or metadata.get("dropped_paths") or not isinstance(original_paths[0], Mapping) or not isinstance(paths[0], Mapping)):
        return abstain("original_and_actual_grounding_not_one_complete_path")
    path, original_path = paths[0], original_paths[0]
    steps, original_steps = path.get("steps", []), original_path.get("steps", [])
    if (len(steps) != 4 or len(original_steps) != 4
            or any(not isinstance(s, Mapping) for s in [*steps, *original_steps])
            or any(path.get(k) != original_path.get(k) for k in ("id", "anchor_ref", "goal", "path_output_var"))
            or any(any(a.get(k) != b.get(k) for k in ("id", "from", "to", "direction")) for a, b in zip(steps, original_steps))):
        return abstain("original_actual_full_path_ids_variables_directions_goal_mismatch")
    root = str(path["anchor_ref"])
    relations = grounded.get("relation_bindings", {})
    ground_bindings = grounded.get("anchor_bindings", {})
    if not isinstance(relations, Mapping) or not isinstance(ground_bindings, Mapping) or set(ground_bindings) != {root}:
        return abstain("grounded_path_anchor_scope_or_relation_bindings_unknown")
    current = root
    ids = []
    for step in steps:
        predicate = str(relations.get(str(step["id"]), ""))
        if (step.get("from") != current or step.get("direction") not in {"forward", "backward"}
                or relation_id_from_label(step.get("relation_label")) != predicate
                or not ontology.domain_for_relation(predicate) or not ontology.range_for_relation(predicate)):
            return abstain("actual_path_not_connected_exact_ontology_ids")
        current = str(step["to"])
        ids.append(predicate)
    if current != str(path["path_output_var"]):
        return abstain("actual_terminal_projection_not_complete_path_output")
    edge = rejection.get("edge_index")
    if (rejection.get("path_index") != 0 or not isinstance(edge, int) or not 0 <= edge < len(ids) - 1
            or rejection.get("left_relation_id") != ids[edge] or rejection.get("right_relation_id") != ids[edge + 1]
            or rejection.get("semantic_rank") != metadata.get("rank")):
        return abstain("actual_guard_rejection_not_this_grounded_path_boundary")
    for index in range(len(ids) - 1):
        left = ontology.range_for_relation(ids[index]) if steps[index]["direction"] == "forward" else ontology.domain_for_relation(ids[index])
        right = ontology.domain_for_relation(ids[index + 1]) if steps[index + 1]["direction"] == "forward" else ontology.range_for_relation(ids[index + 1])
        overlap = schema_type_overlap(left, right, ontology)
        if overlap == "disjoint" or (index == edge and overlap != "unknown"):
            return abstain("schema_has_hard_disjoint_or_rejection_not_unproven_entity_role")
    hops = metadata.get("hops", [])
    if (len(hops) != 1 or len(hops[0]) != len(steps)
            or any(h.get("retrieval_mode") != "ontology_expansion" or h.get("relation_id") != rid
                   or h.get("direction") != step["direction"] for h, rid, step in zip(hops[0], ids, steps))):
        return abstain("actual_retained_grounding_not_full_pure_ontology_expansion")

    originals = original.get("anchors", [])
    original_topics = {str(a.get("id")): str(a.get("surface", "")) for a in originals if isinstance(a, Mapping)}
    if len(original_topics) != 2 or root not in original_topics or set(topic_bindings) != set(original_topics):
        return abstain("requires_exactly_root_and_one_actual_unused_named_topic")
    participant = next(a for a in original_topics if a != root)
    if participant in {str(s.get(k)) for s in original_steps for k in ("from", "to")}:
        return abstain("participant_already_has_original_path_scope_not_missing")
    root_id, root_name = _entity(topic_bindings[root])
    person_id, person_name = _entity(topic_bindings[participant])
    if (not re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", root_id) or not re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", person_id)
            or root_id == person_id or ground_bindings[root] != root_id
            or original_topics[root].casefold() != root_name.casefold()
            or original_topics[participant].casefold() != person_name.casefold()):
        return abstain("actual_topic_ids_names_do_not_match_original_unused_anchor")
    grounded_roots = [a for a in composition.get("anchors", []) if isinstance(a, Mapping) and a.get("id") == root]
    if (composition.get("question") != question or len(grounded_roots) != 1
            or str(grounded_roots[0].get("surface", "")).casefold() != root_name.casefold()
            or not re.search(r"(?<!\w)" + re.escape(root_name) + r"(?!\w)", str(path.get("goal", "")), re.I)):
        return abstain("actual_grounding_question_root_surface_and_goal_identity_mismatch")
    names = {str(x.get("id")): str(x.get("name", x.get("label", ""))) for x in linked_context if isinstance(x, Mapping)}
    if names.get(root_id, "").casefold() != root_name.casefold() or names.get(person_id, "").casefold() != person_name.casefold():
        return abstain("actual_linked_context_does_not_confirm_both_topic_id_names")
    grammar = re.fullmatch(r"\s*what\s+did\s+" + re.escape(person_name) + r"\s+play\s+in\s+(?:the\s+)?"
                           r"(?P<qualification>[A-Za-z -]+?)\s+" + re.escape(root_name) + r"\s*\?\s*", str(question), re.I)
    if grammar is None:
        return abstain("question_does_not_uniquely_attach_named_play_subject_to_named_work")
    # The qualification must name the first retained predicate completely;
    # no extra numeric/topic qualifiers can be silently omitted.
    qualification = set(re.findall(r"[a-z]+", grammar["qualification"].casefold()))
    first_terms = set(re.findall(r"[a-z]+", ids[0].replace("_", " ").casefold()))
    if not qualification or not qualification <= first_terms:
        return abstain("question_has_unrepresented_qualification_or_extra_scope")

    before_cvt, last = steps[-2:]
    if before_cvt["direction"] != "forward" or last["direction"] != "forward":
        return abstain("not_one_forward_work_to_record_to_terminal_role")
    cvt = str(ontology.range_for_relation(ids[-2]))
    role_domain = str(ontology.domain_for_relation(ids[-1]))
    terminal_type = str(ontology.range_for_relation(ids[-1]))
    if cvt != role_domain or "common.topic" in set(ontology.supertypes(cvt)) or "common.topic" not in set(ontology.supertypes(terminal_type)):
        return abstain("actual_terminal_record_role_types_not_proven")
    original_participant_hop = relation_id_from_label(original_steps[-2].get("relation_label"))
    participant_type = str(ontology.domain_for_relation(original_participant_hop))
    if (str(ontology.range_for_relation(original_participant_hop)) != cvt
            or "base.type_ontology.agent" not in set(ontology.supertypes(participant_type))):
        return abstain("original_replaced_hop_does_not_independently_attest_agent_record_role")
    fields = [p for p in ontology.relations_for_domain(cvt)
              if str(ontology.range_for_relation(p)) == participant_type
              and p in set(ontology.reverse_for_relation(original_participant_hop))]
    if len(fields) != 1:
        return abstain("participant_field_not_unique_declared_exact_original_hop_inverse")
    participant_field = fields[0]
    work_type = str(ontology.domain_for_relation(ids[-2]))
    if "common.topic" not in set(ontology.supertypes(work_type)) or "base.type_ontology.non_agent" not in set(ontology.supertypes(work_type)):
        return abstain("actual_material_work_role_is_not_schema_proven")
    if (steps[0]["direction"] != steps[1]["direction"] or ids[1] not in set(ontology.reverse_for_relation(ids[0]))
            or steps[1]["to"] != before_cvt["from"]):
        return abstain("material_work_owner_not_the_declared_inverse_return_to_question_root")
    local_to_global = {root: root_id}
    for index, step in enumerate(steps):
        local_to_global[str(step["to"])] = "V" + str(index)
    triples = []
    for step, predicate in zip(steps, ids):
        a, b = local_to_global[str(step["from"])], local_to_global[str(step["to"])]
        triples.append([a, predicate, b] if step["direction"] == "forward" else [b, predicate, a])
    cvt_var, work_var = local_to_global[str(before_cvt["to"])], local_to_global[str(before_cvt["from"])]
    triples.extend([[cvt_var, participant_field, person_id],
                    [root_id, "type.object.type", work_type], [person_id, "type.object.type", participant_type]])
    operator = {"type": "EQUAL", "inputs": [], "input_var": work_var, "attribute_relation_id": "",
                "attribute_relation_label": [], "attribute_relation_labels": [], "value": root_id, "value_type": "mid"}
    changes = [{"step_id": s["id"], "original_relation": relation_id_from_label(o.get("relation_label")), "actual_grounded_relation": p}
               for s, o, p in zip(steps, original_steps, ids) if relation_id_from_label(o.get("relation_label")) != p]
    proof.update(status="compiled_shadow_not_executed", reason="new_explicit_question_participant_scope_on_unique_actual_record_field",
                 original_semantic_role_changes=changes, participant_anchor_id=participant, participant_id=person_id,
                 participant_name=person_name, root_id=root_id, root_name=root_name, participant_type=participant_type,
                 work_type=work_type, actual_record_type=cvt, terminal_type=terminal_type,
                 participant_field=participant_field, original_hop_inverse_witness=original_participant_hop,
                 root_work_identity_filter_is_new_question_scope_not_assumed_functionality=True,
                 all_actual_grounded_core_edges_preserved=True, terminal_projection_preserved=True,
                 participant_and_work_type_id_validation_deferred_to_same_query=True,
                 linked_context_has_human_type_labels_not_canonical_type_id_certificate=True,
                 question_covered_topic_ids=[root_id, person_id], missing_original_participant_scope_is_real=True,
                 current_nonempty_sample_cannot_activate_failure_empty_runtime_gate=True)
    provenance = {"anchor_bindings": {root: {"id": root_id, "label": root_name}, participant: {"id": person_id, "label": person_name}},
                  "shadow_missing_cvt_participant": deepcopy(proof)}
    candidate = QueryGraphCandidate(graph_id, triples, local_to_global[str(path["path_output_var"])], [operator],
                                    float(grounded.get("score", 0.0)), None, provenance)
    return candidate, proof
