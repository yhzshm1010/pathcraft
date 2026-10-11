"""Compile a closed named-relative scope lost before Compose.

This module performs no retrieval, execution, ranking or answer selection.
It recovers a *whole* question-owned two-path scope, only when the incumbent
has positive schema evidence for a different non-generic terminal role.
Different roles are not an ontology disjointness certificate. The caller must pay
for at most one bounded verification from existing same-question credits.
No entity, predicate, question index, or evaluation answer is stored here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import re

from .contracts import EntityCandidate, GroundedSemanticCandidate, QueryGraphCandidate, validate_semantic_graph
from .lowering import lower_sparql
from .ontology import relation_id_from_label

_MID = re.compile(r"(?:m|g)\.[A-Za-z0-9_]+")


def _plain(value):
    return " ".join(str(value).split()).casefold()


def _words(value):
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(value).casefold())
    return tuple(word[:-1] if len(word) > 3 and word.endswith("s")
                 and not word.endswith(("ss", "'s")) else word for word in words)


def _type_words(value):
    return _words(str(value).rsplit(".", 1)[-1].replace("_", " "))


def _prefix(value, prefix):
    return _plain(value) == _plain(prefix) or _plain(value).startswith(_plain(prefix) + " ")


def _exact_relation(label, ontology):
    relation = relation_id_from_label(label)
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


def compile_named_constraint_scope(
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
            or not isinstance(graph, QueryGraphCandidate)
            or not isinstance(graph.provenance, Mapping)
            or not isinstance(graph.triples, list) or not isinstance(graph.operators, list)
            or not isinstance(grounded_primary, GroundedSemanticCandidate)
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
    if (set(grounded_primary.anchor_bindings) != {root} or not isinstance(binding, EntityCandidate)
            or not _MID.fullmatch(str(binding.entity_id))
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
            or _type_words(expected_type) != _words(secondary["steps"][0]["relation_label"][-2])):
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
    if not match or _words(match.group(1)) != _type_words(expected_type):
        return fail("question_is_not_a_closed_named_relative_of_exact_requested_type")
    role, named_span = match.group(2).strip(), match.group(3).strip()
    if re.search(r"\b(?:and|or|not|except|before|after|where|who|which|that)\b",
                 " ".join(match.groups()), re.I):
        return fail("question_has_an_unaccounted_extra_obligation")
    constraint_words = _words(secondary["steps"][0]["relation_label"][-1])
    role_words = _words(role)
    if not constraint_words or role_words[-len(constraint_words):] != constraint_words:
        return fail("question_named_role_is_not_original_constraint_leaf")
    modifiers = role_words[:-len(constraint_words)]
    if not named_span or not _prefix(original_anchors[other], named_span):
        return fail("original_secondary_anchor_does_not_own_question_named_span")
    context = [item for item in linked_context if isinstance(item, Mapping)]
    if len(context) != len(linked_context) or any(not _MID.fullmatch(str(item.get("id", "")))
                                               or not str(item.get("name", "")).strip() for item in context):
        return fail("current_linker_context_contract_unknown")
    named = [item for item in context if _prefix(item["name"], named_span)]
    if (len(named) != 1 or len({item["id"] for item in context}) != len(context)
            or named[0]["id"] == binding.entity_id):
        return fail("question_named_span_does_not_identify_one_current_provided_mid")
    secondary_entity = named[0]
    provided_type = ontology.range_for_relation(secondary_relation)
    if (not provided_type or not isinstance(secondary_entity.get("types"), list)
            or _type_words(provided_type) not in {_words(value) for value in secondary_entity["types"]}
            or any(modifier not in _words(secondary_entity["name"]) for modifier in modifiers)):
        return fail("provided_named_entity_does_not_prove_constraint_type_or_question_modifiers")
    roots = [item for item in context if item["id"] == binding.entity_id]
    if len(roots) != 1 or _plain(roots[0]["name"]) != _plain(root_surface):
        return fail("current_provided_root_contract_disagrees")
    if (not isinstance(roots[0].get("types"), list)
            or _type_words(ontology.domain_for_relation(relations[0]))
            not in {_words(value) for value in roots[0]["types"]}):
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
    if named_tail and any(word in _words(question) for word in _words(named_tail)):
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
    candidate = QueryGraphCandidate(
        graph_id=graph_id, triples=triples, answer_var=variables[1],
        operators=operators, score=graph.score, compose_output={}, provenance=provenance,
    )
    candidate.sparql = lower_sparql(candidate)
    return candidate, {
        **evidence, "status": "compiled_requires_paid_endpoint_verification",
        "reason": "whole_named_relative_scope_restored",
        **provenance["stability_named_constraint"],
    }
