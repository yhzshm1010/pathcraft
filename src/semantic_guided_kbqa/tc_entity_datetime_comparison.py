"""Pure correction of a date comparison mistakenly applied to an entity.

The only field used is the current, already-present TC datetime property.
All BGP and every other operator are preserved. Full current-flow review,
grounding, topic and captured source programs authorize day equality. A
subtype/same-leaf certificate does not assert equality of property values.
No models, execution, file/Gold access, field search, aliases or activation.
"""
from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
from typing import Any

from .budgeted_literal_constraint_repair import _spec
from .comparison_number_normalization import parse_english_calendar_date
from .contracts import QueryGraphCandidate
from .lowering import lower_sparql
from .ontology import relation_id_from_label


_VAR = re.compile(r"V\d+\Z")
_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_COMPARISONS = {"EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL"}
_XSD = "http://www.w3.org/2001/XMLSchema#"


def _proof(reason):
    return {"status": "not_compiled", "reason": reason, "shadow_only": True,
            "model_calls": 0, "endpoint_queries": 0, "embedding_calls": 0,
            "uses_gold_answers_or_sparql": False, "uses_entity_relation_or_sample_whitelist": False,
            "compiler_selects_answers": False, "production_activation_authorized": False,
            "requires_actual_saved_request_and_remaining_measured_time": True}


def _words(value):
    return {word[:-1] if word.endswith("s") and len(word) > 4 else word
            for word in re.findall(r"[a-z]+", str(value).replace("_", " ").casefold())}


def _entity(value):
    if isinstance(value, Mapping):
        return str(value.get("id", "")), str(value.get("label", value.get("name", "")))
    return str(value), ""


def _read(graph, key, default=None):
    return graph.get(key, default) if isinstance(graph, Mapping) else getattr(graph, key, default)


def _field(operator):
    explicit = str(operator.get("attribute_relation_id", ""))
    labelled = relation_id_from_label(operator.get("attribute_relation_label", []))
    if explicit and labelled and explicit != labelled:
        return ""
    return explicit or labelled


def tc_entity_datetime_necessary(question: str, graph: Any, ontology: Any):
    spec = _spec(question)
    if not spec or spec.get("value_type") != "date" or spec.get("type") != "EQUAL":
        return _proof("question_not_one_exact_unquoted_on_calendar_day")
    triples, operators, answer = (_read(graph, key) for key in ("triples", "operators", "answer_var"))
    if (not isinstance(triples, list) or not triples or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)
            or not isinstance(operators, list) or any(not isinstance(op, Mapping) for op in operators)
            or not isinstance(answer, str) or not _VAR.fullmatch(answer)):
        return _proof("source_program_contract_missing")
    substantive = [(i, op) for i, op in enumerate(operators) if op.get("type") != "NO_EQUAL"]
    tcs = [(i, op) for i, op in substantive if op.get("type") == "TC"]
    comparisons = [(i, op) for i, op in substantive if op.get("type") in _COMPARISONS]
    if len(substantive) != 2 or len(tcs) != 1 or len(comparisons) != 1:
        return _proof("not_one_existing_tc_and_one_impossible_comparison")
    tc_index, tc = tcs[0]
    comparison_index, comparison = comparisons[0]
    field = _field(tc)
    if (comparison.get("input_var") != answer or tc.get("input_var") != answer
            or comparison.get("inputs") not in (None, []) or tc.get("inputs") not in (None, [])
            or _field(comparison) or comparison.get("attribute_relation_labels") not in (None, [])
            or comparison.get("_attribute_value_already_bound")
            or any(comparison.get(k) for k in ("_numeric_cast_compare", "_numeric_comparison_cast"))
            or comparison.get("value") != spec["value"] or comparison.get("value_type") not in {"date", "dateTime"}
            or not field or tc.get("value_type") != "year" or str(tc.get("value")) != spec["value"][:4]):
        return _proof("comparison_or_existing_tc_not_exact_same_owner_and_calendar_literal")
    provenance = _read(graph, "provenance")
    if (not isinstance(provenance, Mapping) or any(k in provenance for k in ("selection_answer_intersection", "selection_answer_union"))
            or "original_question" in provenance and provenance["original_question"] != question):
        return _proof("source_postprocessed_or_explicit_question_provenance_conflicts")
    if ontology is None or ontology.range_for_relation(field) != "type.datetime":
        return _proof("existing_tc_field_not_declared_datetime")
    incoming = {str(ontology.range_for_relation(str(p))) for s, p, o in triples if o == answer}
    head = re.match(r"\s*(?:what|which)\s+([A-Za-z]+)\b", question, re.I)
    owners = [t for t in incoming if t and "common.topic" in ontology.supertypes(t)
              and head and _words(t.rsplit(".", 1)[-1]) == _words(head[1])]
    if len(owners) != 1 or any(not t or "common.topic" not in ontology.supertypes(t) for t in incoming):
        return _proof("comparison_input_not_positive_exact_question_head_entity_owner")
    owner_type, field_domain = owners[0], str(ontology.domain_for_relation(field))
    if (not field_domain or owner_type not in ontology.supertypes(field_domain)
            or "common.topic" not in ontology.supertypes(field_domain)):
        return _proof("existing_tc_datetime_domain_not_exact_owner_or_declared_subtype")
    return {**_proof("necessary_entity_datetime_error_only_full_current_scope_required"), "necessary_gate": True,
            "obligation": spec, "tc_operator_index": tc_index, "comparison_operator_index": comparison_index,
            "existing_tc_field": field, "owner_var": answer, "owner_type": owner_type, "tc_field_domain": field_domain,
            "entity_valued_comparison_proved_by_schema": True}


def compile_tc_entity_datetime_from_current_stages(
    question: str, graph: Any, stages: Sequence[Mapping[str, Any]], ontology: Any,
    *, graph_id: str = "tc_entity_datetime_shadow",
):
    proof = tc_entity_datetime_necessary(question, graph, ontology)
    if not proof.get("necessary_gate"):
        return None, proof

    def abstain(reason):
        return None, {**proof, "status": "not_compiled", "reason": reason}

    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return abstain("complete_current_stages_missing")
    contexts, reviews, generations, linkers, groundings, beams, rankings = [], [], [], [], [], [], []
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        kind, output = stage.get("stage"), stage.get("output")
        if kind == "linked_entity_context":
            contexts.append(output)
        elif kind in {"decomposition_review_and_rewrite", "semantic_path_generation"}:
            if not isinstance(stage.get("input"), Mapping) or stage["input"].get("question") != question:
                return abstain("review_or_semantic_stage_not_exact_current_question")
            (reviews if kind == "decomposition_review_and_rewrite" else generations).append(output)
        elif kind == "grounded_semantic_candidates":
            groundings.append(output)
        elif kind == "semantic_subgraph_grounding_and_ranking":
            linkers.append(output)
        elif kind == "final_graph_beam":
            beams.append(output)
        elif kind == "glm_final_graph_ranking":
            rankings.append(output)
    if (len(contexts) != 1 or not isinstance(contexts[0], list) or len(contexts[0]) != 1
            or not isinstance(contexts[0][0], Mapping) or len(reviews) != 1 or len(generations) != 1
            or len(linkers) != 1 or not isinstance(linkers[0], Mapping) or not isinstance(linkers[0].get("semantic_graphs"), list)
            or len(groundings) != 1 or not isinstance(groundings[0], list) or len(beams) != 1 or not isinstance(beams[0], list)
            or not rankings or not isinstance(rankings[-1], Mapping)):
        return abstain("actual_topic_review_semantic_grounding_and_source_corpus_not_unique_complete")
    ranking = rankings[-1]
    captured = [c for c in ranking.get("candidates", []) if isinstance(c, Mapping) and c.get("graph_id") == _read(graph, "graph_id")]
    if (ranking.get("selected_graph_id") != _read(graph, "graph_id") or len(captured) != 1
            or any(captured[0].get(k) != _read(graph, k) for k in ("triples", "operators", "answer_var"))):
        return abstain("actual_final_selected_source_full_program_not_attested")
    provenance = _read(graph, "provenance")
    bindings, relations, meta = (provenance.get(k) for k in ("anchor_bindings", "relation_bindings", "grounded_semantic"))
    if (not isinstance(bindings, Mapping) or len(bindings) != 1 or not isinstance(relations, Mapping)
            or not isinstance(meta, Mapping) or not isinstance(meta.get("semantic_graph_index"), int)
            or isinstance(meta.get("semantic_graph_index"), bool) or meta.get("dropped_paths") != []):
        return abstain("actual_source_origin_and_single_topic_binding_contract_missing")
    root, bound = next(iter(bindings.items()))
    anchor, label = _entity(bound)
    context_id, context_label = _entity(contexts[0][0])
    if (not _MID.fullmatch(anchor) or not label or (anchor, label) != (context_id, context_label)
            or not re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", question, re.I)):
        return abstain("actual_topic_mid_label_and_literal_current_question_do_not_align")
    review = reviews[0]
    if not isinstance(review, Mapping) or not isinstance(review.get("final_candidates"), list) or not isinstance(generations[0], list):
        return abstain("reviewed_rewrite_and_valid_semantic_corpus_missing")
    reviewed = {tuple(c["decomposition"]) for c in review["final_candidates"] if isinstance(c, Mapping)
                and c.get("source") == "review_rewrite" and isinstance(c.get("decomposition"), list)
                and all(isinstance(s, str) for s in c["decomposition"])}
    valid_entries = [e for e in generations[0] if isinstance(e, Mapping) and e.get("valid") is True]
    index = meta["semantic_graph_index"]
    if not 0 <= index < len(valid_entries):
        return abstain("actual_originating_valid_semantic_index_not_attested")
    original = valid_entries[index]
    if not isinstance(original.get("decomposition"), list) or tuple(original["decomposition"]) not in reviewed:
        return abstain("actual_source_not_from_current_reviewed_complete_decomposition")
    origins = [g for g in groundings[0] if isinstance(g, Mapping) and g.get("relation_bindings") == relations
               and isinstance(g.get("provenance"), Mapping) and g["provenance"].get("semantic_graph_index") == index
               and g["provenance"].get("rank") == meta.get("rank") and g["provenance"].get("dropped_paths") == []
               and isinstance(g.get("anchor_bindings"), Mapping) and set(g["anchor_bindings"]) == {root}
               and _entity(g["anchor_bindings"][root])[0] == anchor]
    if len(origins) != 1 or not isinstance(origins[0].get("compose_input"), Mapping):
        return abstain("actual_same_flow_source_grounding_not_unique")
    composed = origins[0]["compose_input"]
    paths = composed.get("semantic_paths")
    if composed.get("question") != question or not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], Mapping):
        return abstain("actual_grounded_full_path_not_this_question_or_has_other_scope")
    grounded_path = paths[0]
    steps = grounded_path.get("steps")
    if not isinstance(steps, list) or not steps or grounded_path.get("anchor_ref") != root:
        return abstain("actual_source_grounded_path_contract_missing")
    current, mapping, consumed = anchor, {root: anchor}, []
    for step in steps:
        if not isinstance(step, Mapping) or step.get("from") not in mapping or not step.get("to"):
            return abstain("actual_grounded_path_variable_chain_invalid")
        predicate = relations.get(step.get("id"))
        if predicate != relation_id_from_label(step.get("relation_label")):
            return abstain("actual_grounded_path_relation_binding_conflict")
        edges = [t for t in _read(graph, "triples") if t[1] == predicate and
                 (step.get("direction") == "forward" and t[0] == current or step.get("direction") == "backward" and t[2] == current)]
        if len(edges) != 1:
            return abstain("all_source_bgp_not_unique_complete_original_path")
        consumed.append(edges[0])
        current = edges[0][2] if step["direction"] == "forward" else edges[0][0]
        mapping[step["to"]] = current
    if (consumed != _read(graph, "triples") or current != _read(graph, "answer_var")
            or mapping.get(grounded_path.get("path_output_var")) != current):
        return abstain("source_has_unaccounted_bgp_or_other_answer_projection")
    field = proof["existing_tc_field"]
    original_field = field
    ancestry = provenance.get("operator_attribute_repair")
    if ancestry is not None:
        if (not isinstance(ancestry, Mapping) or ancestry.get("operator_index") != proof["tc_operator_index"]
                or ancestry.get("owner_var") != proof["owner_var"] or ancestry.get("first_relation") != field
                or ancestry.get("second_relation") != ""):
            return abstain("actual_tc_attribute_repair_ancestry_conflicts")
        parents = [g for g in beams[0] if isinstance(g, Mapping) and g.get("graph_id") == ancestry.get("source_graph_id")]
        if len(parents) != 1 or any(parents[0].get(k) != _read(graph, k) for k in ("triples", "answer_var")):
            return abstain("actual_pre_attribute_source_full_bgp_not_unique")
        parent_ops = parents[0].get("operators")
        if not isinstance(parent_ops, list) or len(parent_ops) != len(_read(graph, "operators")):
            return abstain("actual_pre_attribute_source_operator_contract_missing")
        for i, (old, now) in enumerate(zip(parent_ops, _read(graph, "operators"))):
            if not isinstance(old, Mapping):
                return abstain("actual_pre_attribute_source_operator_contract_invalid")
            keys = {"attribute_relation_id", "attribute_relation_label", "attribute_relation_labels"} if i == proof["tc_operator_index"] else set()
            if {k: v for k, v in old.items() if k not in keys} != {k: v for k, v in now.items() if k not in keys}:
                return abstain("tc_ancestry_changed_other_conditions_or_owner")
        original_field = _field(parent_ops[proof["tc_operator_index"]])
    original_domain = str(ontology.domain_for_relation(original_field))
    if (not original_field or ontology.range_for_relation(original_field) != "type.datetime"
            or original_domain != proof["owner_type"] or original_field.rsplit(".", 1)[-1] != field.rsplit(".", 1)[-1]
            or original_domain not in ontology.supertypes(proof["tc_field_domain"])):
        return abstain("existing_tc_not_exact_or_declared_subtype_same_leaf_original_field")
    spec = proof["obligation"]
    goals = []
    for semantic_index, entry in enumerate(valid_entries):
        if not isinstance(entry.get("decomposition"), list) or tuple(entry["decomposition"]) not in reviewed:
            continue
        semantic = entry.get("output")
        original_paths = semantic.get("semantic_paths") if isinstance(semantic, Mapping) else None
        anchors = semantic.get("anchors") if isinstance(semantic, Mapping) else None
        if not isinstance(original_paths, list) or len(original_paths) != 1 or not isinstance(original_paths[0], Mapping) or not isinstance(anchors, list):
            return abstain("competing_reviewed_original_has_other_path_or_unknown_scope")
        links = [g for g in linkers[0]["semantic_graphs"] if isinstance(g, Mapping) and g.get("graph_index") == semantic_index
                 and not isinstance(g.get("graph_index"), bool)]
        if (len(links) != 1 or links[0].get("dropped_paths") != [] or not isinstance(links[0].get("anchors"), list)
                or any(not isinstance(a, Mapping) or not isinstance(a.get("id"), str) for a in anchors)
                or len({a["id"] for a in anchors}) != len(anchors)):
            return abstain("reviewed_original_complete_anchor_linker_scope_missing_or_ambiguous")
        linked_anchors = links[0]["anchors"]
        if (any(not isinstance(a, Mapping) or not isinstance(a.get("anchor_id"), str)
                or not isinstance(a.get("candidates"), list) for a in linked_anchors)
                or len({a["anchor_id"] for a in linked_anchors}) != len(linked_anchors)
                or {a["anchor_id"] for a in linked_anchors} != {a["id"] for a in anchors}):
            return abstain("reviewed_original_and_complete_linker_anchor_scopes_conflict")
        actual_links = {a["anchor_id"]: a["candidates"] for a in linked_anchors}
        path = original_paths[0]
        original_steps = path.get("steps")
        if (path.get("id") != grounded_path.get("id") or path.get("anchor_ref") != root or path.get("path_output_var") != grounded_path.get("path_output_var")
                or not isinstance(original_steps, list) or len(original_steps) != len(steps)
                or any(not isinstance(a, Mapping) or any(a.get(k) != b.get(k) for k in ("id", "from", "to", "direction"))
                       for a, b in zip(original_steps, steps))):
            return abstain("competing_reviewed_original_path_topology_or_owner_conflict")
        for a in anchors:
            if not isinstance(a, Mapping):
                return abstain("original_anchor_scope_invalid")
            if a.get("id") == root:
                roots = actual_links[root]
                if a.get("surface") != label or len(roots) != 1 or _entity(roots[0]) != (anchor, label):
                    return abstain("original_topic_surface_conflicts_with_actual_mid_label")
            elif actual_links[a["id"]] != [] or parse_english_calendar_date(str(a.get("surface", ""))) != spec["value"]:
                return abstain("original_extra_topic_or_literal_not_exact_question_date")
        goal = str(path.get("goal", ""))
        match = re.search(r"\breturn\s+(?:the\s+)?([A-Za-z]+)\s+whose\s+([A-Za-z ]+?)\s+(?:is|equals|is equal to)\s+([^?.!]+)[?.!]*\s*$", goal, re.I)
        if (goal not in entry["decomposition"] or not match or _words(match[1]) != _words(proof["owner_type"].rsplit(".", 1)[-1])
                or _words(match[2]) != _words(original_field.rsplit(".", 1)[-1])
                or parse_english_calendar_date(match[3].strip()) != spec["value"]):
            return abstain("reviewed_goal_does_not_explicitly_prove_unique_owner_field_day_equality")
        goals.append(goal)
    if not goals or grounded_path.get("goal") not in goals:
        return abstain("actual_grounded_source_goal_not_in_closed_reviewed_equality_corpus")
    span = spec.get("normalization", {}).get("_source_literal_span")
    if not isinstance(span, list) or len(span) != 2:
        return abstain("question_calendar_literal_span_not_proven")
    before, after = question[:span[0]], question[span[1]:]
    head = re.escape(proof["owner_type"].rsplit(".", 1)[-1])
    event = re.fullmatch(rf"\s*(?:what|which)\s+{head}(?:\s+that)?(?:\s+(?:is|was|were|are|has|have|had|been)){{0,2}}\s+([A-Za-z]+)\s+on\s+", before, re.I)
    if not event or not re.fullmatch(r"\s+[A-Za-z]+\s+" + re.escape(label) + r"\s*[?.!]*\s*", after, re.I):
        return abstain("question_owner_date_and_single_topic_attachment_not_closed")
    event_word = event[1].casefold()
    variants = {event_word, event_word[:-1] if event_word.endswith("ed") else event_word,
                event_word[:-2] if event_word.endswith("ed") else event_word}
    if len(variants & (_words(original_field.rsplit(".", 1)[-1]) - {"date", "initial"})) != 1:
        return abstain("question_event_not_morphologically_the_reviewed_original_field")
    candidate = deepcopy(graph)
    candidate.graph_id, candidate.sparql = graph_id, ""
    old_comparison = deepcopy(candidate.operators[proof["comparison_operator_index"]])
    comparison = deepcopy(old_comparison)
    comparison.update(type="EQUAL", input_var=proof["owner_var"], attribute_relation_id=field,
                      attribute_relation_label=[], attribute_relation_labels=[], value=spec["value"], value_type="dateTime",
                      _comparison_date_precision="day", _calendar_day_boundary=True, _tc_entity_datetime_typed_guard=True)
    candidate.operators[proof["comparison_operator_index"]] = comparison
    proof.update(status="compiled_shadow_not_executed", reason="impossible_entity_date_comparison_bound_to_same_existing_tc_datetime_field",
                 source_graph_id=_read(graph, "graph_id"), original_tc_field=original_field,
                 original_tc_domain=original_domain, reviewed_goals=goals, actual_topic_id=anchor,
                 source_question_attested_by_actual_grounded_compose_input=True,
                 all_source_bgp_and_all_other_operators_exactly_preserved=True, answer_projection_unchanged=True,
                 selected_tc_field_not_substituted=True, subtype_same_leaf_does_not_assert_property_value_equivalence=True,
                 original_impossible_comparison=old_comparison, runtime_typed_datetime_guard_required=True,
                 original_source_grounding_index=index, original_source_grounding_rank=meta["rank"])
    candidate.provenance["tc_entity_datetime_comparison"] = deepcopy(proof)
    return candidate, proof


def lower_tc_entity_datetime(candidate: QueryGraphCandidate, *, limit: int = 256):
    marked = [(i, op) for i, op in enumerate(candidate.operators) if op.get("_tc_entity_datetime_typed_guard")]
    if len(marked) != 1 or marked[0][1].get("type") != "EQUAL" or not _field(marked[0][1]):
        raise ValueError("One same-existing-TC field day equality is required")
    # Only TC/comparison/NO_EQUAL operators are accepted by the compiler;
    # unchanged lowering increments its filter index once for each of them.
    variable = "?equal" + str(marked[0][0])
    query = lower_sparql(candidate, limit=limit)
    guard = f'  FILTER(isLiteral({variable}) && LANG({variable}) = "")\n'
    guard += f'  FILTER(DATATYPE({variable}) IN (<{_XSD}date>, <{_XSD}dateTime>))\n'
    return query.replace("\n}\n", "\n" + guard + "}\n", 1)
