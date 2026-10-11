"""Pure compilation from one complete original Semantic scalar path.

No models, endpoint calls, Gold access or sample routing inside this compiler.
An ordinary changed predicate is not a contradiction.  Only an official
inverse used without reversing direction, or an extra same-predicate hop
that replaces the explicitly identified scalar owner's answer role, can
qualify.  Unproved original topic scope is always rejected.
"""
from copy import deepcopy
import re
from typing import Any, Callable, Mapping, Sequence

from .budgeted_schema_repair import (
    NUMERIC_TYPES, SCALAR_TYPES, STRING_TYPES, _VAR, _normal_anchor_scope,
    _native_scalar_obligation, _number, _validated_provided_scalar_spec,
    schema_type_overlap,
)
from .contracts import EntityCandidate, ExecutedGraph, GroundedSemanticCandidate, QueryGraphCandidate
from .ontology import relation_id_from_label


def _evidence(reason):
    return {"status": "not_qualified", "reason": reason, "pure_compiler_only": True,
            "model_calls": 0, "endpoint_queries": 0, "embedding_calls": 0,
            "uses_gold": False, "new_denotation_verified": False}


def _tokens(value):
    tokens = set()
    for word in re.findall(r"[A-Za-z]+", str(value).casefold()):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.add(word)
    return tokens


def _question_head(question):
    match = re.match(r"^\s*(?:(?:in|at|on)\s+)?(?:which|what)\s+(?:the\s+)?([A-Za-z]+)", question, re.I)
    return _tokens(match.group(1)) if match else set()


def compile_original_scalar_path_restore(
    question: str, selected: ExecutedGraph, original_semantic: Mapping[str, Any],
    topic_bindings: Mapping[str, EntityCandidate], ontology: Any, *, graph_id: str,
    linked_entity_context: Sequence[Mapping[str, Any]] | None = None,
    spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    evidence = _evidence("original_scalar_path_not_completely_proved")
    paths = original_semantic.get("semantic_paths", [])
    if ontology is None or len(paths) != 1:
        return None, _evidence("requires_one_original_path_no_dropped_original_branches")
    path = paths[0]
    steps = path.get("steps", [])
    if len(steps) != 2 or not all(isinstance(s, dict) for s in steps):
        return None, _evidence("requires_one_direct_primary_edge_and_one_scalar_attribute")
    first, last = steps
    root = str(path.get("anchor_ref", ""))
    owner_local, scalar_local = str(first.get("to", "")), str(last.get("to", ""))
    if (str(first.get("from", "")) != root or str(last.get("from", "")) != owner_local
            or str(path.get("path_output_var", "")) != scalar_local
            or last.get("direction") != "forward" or first.get("direction") not in {"forward", "backward"}
            or root not in topic_bindings):
        return None, _evidence("original_variable_owner_or_direction_is_ambiguous")
    surfaces = [str(a.get("surface", "")) for a in original_semantic.get("anchors", [])
                if isinstance(a, Mapping) and str(a.get("id", "")) == root]
    actual_label = str(topic_bindings[root].label)
    if (len(surfaces) != 1 or not surfaces[0]
            or " ".join(surfaces[0].casefold().split()) != " ".join(actual_label.casefold().split())
            or not re.search(r"(?<!\w)" + re.escape(surfaces[0]) + r"(?!\w)", question, re.I)):
        return None, _evidence("actual_bound_topic_label_does_not_match_original_question_semantic_root")
    owned_question = getattr(selected.graph, "provenance", {}).get("original_question")
    if owned_question is not None and str(owned_question).strip() != str(question).strip():
        return None, _evidence("current_selected_graph_question_contract_conflict")
    # This adapter is solely an anchor-scope representation of ORIGINAL
    # Semantic plus actual existing topic bindings, not fabricated grounding.
    adapter = GroundedSemanticCandidate(dict(original_semantic), dict(topic_bindings), {}, 0.0,
                                         {"source": "original_semantic_scope_only"})
    scope_ok, labels, scope = _normal_anchor_scope(adapter, original_semantic, question, linked_entity_context)
    if not scope_ok:
        return None, {**_evidence("original_extra_topic_or_schema_kind_scope_unknown"), "scope": scope}
    anchor = str(topic_bindings[root].entity_id)
    if not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", anchor):
        return None, _evidence("actual_topic_binding_is_not_a_resolved_freebase_entity")
    for context in linked_entity_context or []:
        name = str(context.get("name", context.get("label", "")))
        tags = _tokens(" ".join(map(str, context.get("types", []))))
        if (name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, re.I)
                and not tags & {"type", "property", "domain", "namespace"}
                and str(context.get("id", "")) not in {str(b.entity_id) for b in topic_bindings.values()}):
            return None, _evidence("question_has_an_existing_linked_topic_not_in_original_scope")
    primary = relation_id_from_label(first.get("relation_label", []))
    attribute = relation_id_from_label(last.get("relation_label", []))
    p_domain, p_range = str(ontology.domain_for_relation(primary)), str(ontology.range_for_relation(primary))
    a_domain, a_range = str(ontology.domain_for_relation(attribute)), str(ontology.range_for_relation(attribute))
    if not all((p_domain, p_range, a_domain, a_range)):
        return None, _evidence("original_exact_predicate_not_completely_in_ontology")
    output_type = p_range if first["direction"] == "forward" else p_domain
    numeric_string = a_range in (STRING_TYPES | {"type.enumeration"})
    if (a_range not in NUMERIC_TYPES and not numeric_string
            or schema_type_overlap(output_type, a_domain, ontology) != "compatible"
            or "common.topic" not in set(ontology.supertypes(a_domain))):
        return None, _evidence("original_scalar_owner_not_a_positively_compatible_entity")
    head = _question_head(question)
    if not head or head != _tokens(a_domain.rsplit(".", 1)[-1]):
        return None, _evidence("question_head_does_not_uniquely_prove_the_original_attribute_owner_role")
    if not _tokens(attribute.rsplit(".", 1)[-1]) <= _tokens(question):
        return None, _evidence("exact_original_scalar_property_not_explicit_in_question")
    spec = _native_scalar_obligation(question, labels, "type.float" if numeric_string else a_range)
    if spec is None and spec_provider is not None:
        spec = _validated_provided_scalar_spec(question, labels, spec_provider)
    if not spec or spec.get("value_type") != "number" or spec.get("type") not in {
        "GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "EQUAL",
    }:
        return None, _evidence("single_exact_numeric_obligation_not_proved")
    current = selected.graph
    constants = {str(v) for t in current.triples for v in (t[0], t[2]) if not _VAR.fullmatch(str(v))}
    if constants != {anchor}:
        return None, _evidence("incumbent_topic_anchor_scope_differs")
    roots = [(str(p), str(b), "forward") for a, p, b in current.triples if str(a) == anchor and _VAR.fullmatch(str(b))]
    roots += [(str(p), str(a), "backward") for a, p, b in current.triples if str(b) == anchor and _VAR.fullmatch(str(a))]
    if len(roots) != 1:
        return None, _evidence("incumbent_primary_edge_not_unique")
    actual_relation, first_owner, actual_direction = roots[0]
    inverse = set(ontology.reverse_for_relation(primary))
    desired_direction = str(first["direction"])
    contradiction = None
    if actual_relation in inverse and actual_direction == desired_direction:
        # Preserve all actual source conditions; arbitrary extra clauses are
        # not authorized to disappear during this narrowly proved inversion.
        if len(current.triples) == 1 and not current.operators and current.answer_var == first_owner:
            contradiction = {"kind": "official_inverse_substituted_without_direction_flip", "actual_relation": actual_relation,
                             "required_relation": primary, "unchanged_direction": desired_direction}
    elif actual_relation == primary and actual_direction == desired_direction and current.answer_var != first_owner:
        if len(current.triples) == 2 and len(current.operators) == 1:
            extra = next((tuple(map(str, t)) for t in current.triples if anchor not in (str(t[0]), str(t[2]))), None)
            expected = (first_owner, primary, str(current.answer_var)) if desired_direction == "forward" else (str(current.answer_var), primary, first_owner)
            op = current.operators[0]
            actual_attribute = str(op.get("attribute_relation_id", "")) or relation_id_from_label(op.get("attribute_relation_label", []))
            if (extra == expected and op.get("input_var") == current.answer_var and actual_attribute == attribute
                    and op.get("type") == spec["type"] and _number(op.get("value")) == _number(spec["value"])):
                contradiction = {"kind": "extra_same_predicate_hop_overrode_original_unique_entity_owner", "original_owner_var": first_owner,
                                 "actual_owner_and_answer_var": current.answer_var, "extra_triple": list(extra)}
    if contradiction is None:
        return None, _evidence("no_proved_inverse_error_or_unique_owner_projection_override")
    first_triple = [anchor, primary, "V0"] if desired_direction == "forward" else ["V0", primary, anchor]
    op = {"type": spec["type"], "input_var": "V1", "inputs": [], "attribute_relation_id": "",
          "attribute_relation_label": [], "attribute_relation_labels": [], "value": str(spec["value"]),
          "value_type": "number", "_attribute_value_already_bound": True,
          "_numeric_cast_kind": "float" if numeric_string or a_range == "type.float" else "integer"}
    op["_numeric_cast_compare" if spec["type"] == "EQUAL" else "_numeric_comparison_cast"] = True
    if numeric_string:
        op["_budgeted_numeric_string_value_guard"] = True
    proposed = QueryGraphCandidate(graph_id, [first_triple, ["V0", attribute, "V1"]], "V0", [op], float(current.score), {},
        {"original_scalar_path_restore_shadow": {"original_path_id": str(path.get("id", "")), "original_owner_local_var": owner_local,
         "original_value_local_var": scalar_local, "actual_topic_binding": anchor, "contradiction_witness": contradiction,
         "question_owner_schema_type": a_domain, "requires_saved_query_count_and_time_credit": True,
         "uses_gold": False, "caller_owns_execution_and_budget": True}})
    evidence.update({"status": "compiled_not_executed", "reason": "complete_original_path_with_proved_incumbent_direction_or_owner_contradiction",
                     "contradiction": contradiction, "primary_relation": primary, "attribute_relation": attribute,
                     "normalized_spec": dict(spec), "required_owner_type": a_domain, "original_scope": scope,
                     "incumbent_subset_used_as_positive_evidence": False,
                     "remaining_requirements": ["actual paid execution and complete entity rows", "no query truncation",
                                                "fresh and old nonreg influence audit", "safe final selection and labels"]})
    return proposed, evidence
