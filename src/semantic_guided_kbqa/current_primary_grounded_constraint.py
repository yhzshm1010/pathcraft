"""Independent compile-only witness for a current primary ENTITY source.

This pure helper does not itself authorize a
request or select an answer: the caller owns paid count/time credit and must
prove a complete nonempty strict subset of the real source execution.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from .budgeted_schema_repair import (
    SCALAR_TYPES, _COMPARISON, _ISO_DATE, _NAMED_DATE, _VAR, _evidence, _one_explicit_scalar_spec,
    _regular_role_stems, compile_grounded_constraint_repair,
    grounding_schema_evidence,
)
from .compose_scalar_scope_safety import original_semantic_scope_certificate
from .lowering import lower_sparql


def _tokens(value):
    return tuple(word[:-1] if len(word) > 4 and word.endswith("s") else word
                 for word in re.findall(r"[A-Za-z]+", str(value).casefold()))


def _label(value):
    return " ".join(str(value).split()).casefold()


def _word_forms(value):
    forms = set(_regular_role_stems(value))
    # An adjective formed from a canonical owner noun is regular morphology,
    # not an occupation/property synonym list (government -> governmental).
    for word in tuple(forms):
        if len(word) > 6 and word.endswith("al"):
            forms.add(word[:-2])
    return forms


def _question_relative_owner_tail(tail, owner_words, leaf_words, range_id):
    """Consume the whole relative condition, never just a scalar prefix."""
    comparisons = list(_COMPARISON.finditer(tail))
    if len(comparisons) != 1:
        return False
    literal = tail[comparisons[0].end():].strip().rstrip("?!. ").strip()
    if not ((_ISO_DATE.fullmatch(literal) or _NAMED_DATE.fullmatch(literal))
            if range_id == "type.datetime"
            else re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?", literal)):
        return False
    body = tail[:comparisons[0].start()].strip().lstrip(",").strip()
    if re.search(r"\b(?:and|or|where)\b", body, re.I):
        return False
    words = list(_tokens(body))
    if not words or words.pop(0) not in {"who", "which", "that"}:
        return False
    if words and words[-1] == ("date" if range_id == "type.datetime" else "value"):
        words.pop()
    if not leaf_words or tuple(words[-len(leaf_words):]) != tuple(leaf_words):
        return False
    del words[-len(leaf_words):]
    possessives = [index for index, word in enumerate(words) if word in {"his", "her", "their", "its", "whose"}]
    if len(possessives) != 1 or possessives[0] == len(words) - 1:
        return False
    index = possessives[0]
    auxiliaries = {"is", "are", "was", "were", "has", "have", "had", "does", "do", "did"}
    pre_owner = [word for word in words[:index] if word not in auxiliaries]
    actual_owner = [*pre_owner, *words[index + 1:]]
    if len(actual_owner) != len(owner_words):
        return False
    # Require all owner tokens, without extras or duplicates, in any regular
    # active arrangement. E.g. held + possessive + governmental position is
    # exactly the same canonical owner as government positions held.
    matches = []

    def visit(position, used):
        if position == len(actual_owner):
            matches.append(tuple(sorted(used)))
            return
        for target, word in enumerate(owner_words):
            if target not in used and _word_forms(actual_owner[position]) & _word_forms(word):
                visit(position + 1, used | {target})

    if len(actual_owner) > 6:
        return False
    visit(0, set())
    return bool(matches)


def current_primary_grounded_constraint_witness(
    question, source, grounded, source_semantic, ontology, *, decomposition,
    source_query, graph_id="current_primary_constraint",
):
    """Compile the existing f4 extension only after independent source proof.

    Eligibility is narrow: one rooted, complete, purely ontology-grounded
    original path, a direct regular active WHO entity role, an unmodified
    one-edge actual SELECT, and the same full scalar obligation in question,
    original goal and its independently supplied decomposition. An original
    three-clause goal explicitly returns the primary owner, not the scalar.
    """
    evidence = _evidence("missing_independent_current_primary_source_proof")

    def abstain(reason, **detail):
        return None, {**evidence, "reason": reason, **detail}

    if (not isinstance(decomposition, Sequence) or isinstance(decomposition, str)
            or not decomposition or not all(isinstance(value, str) for value in decomposition)):
        return abstain("original_goal_not_independently_attached_to_same_decomposition")
    graph = getattr(source, "graph", None)
    if (graph is None or not isinstance(getattr(graph, "provenance", None), Mapping)
            or not isinstance(getattr(graph, "triples", None), list)
            or not isinstance(getattr(graph, "operators", None), list)):
        return abstain("actual_source_graph_contract_unknown")
    if (not isinstance(graph.provenance.get("original_question"), str)
            or graph.provenance["original_question"] != question):
        return abstain("actual_source_original_question_not_identically_owned")
    ok, scope = original_semantic_scope_certificate(
        grounded, source_semantic, ontology=ontology,
    )
    if not ok:
        return abstain("original_semantic_scope_not_preserved", scope_certificate=scope)
    schema = grounding_schema_evidence(grounded, ontology)
    if schema.get("status") != "qualified":
        return abstain("not_complete_rejected_entity_role_grounding", schema_evidence=schema)
    paths = source_semantic["semantic_paths"]
    if len(paths) != 1:
        return abstain("current_primary_requires_one_original_path")
    path = paths[0]
    steps = path["steps"]
    if (len(steps) < 2 or steps[0]["direction"] != "forward"
            or steps[0]["from"] != path["anchor_ref"]
            or path["path_output_var"] != steps[-1]["to"]
            or any(step["direction"] != "forward" for step in steps)
            or any(left["to"] != right["from"] for left, right in zip(steps, steps[1:]))):
        return abstain("original_path_is_not_one_unambiguous_forward_owner_chain")
    root = str(path["anchor_ref"])
    anchors = source_semantic.get("anchors")
    actual_anchors = grounded.compose_input.get("anchors")
    if (not isinstance(anchors, list) or not isinstance(actual_anchors, list)
            or any(not isinstance(item, Mapping) or not item.get("id")
                   or not str(item.get("surface", "")).strip() for item in anchors)
            or len({item["id"] for item in anchors}) != len(anchors)):
        return abstain("original_anchor_contract_unknown_or_ambiguous")
    labels = {str(item["id"]): str(item["surface"]).strip() for item in anchors}
    actual_labels = {str(item.get("id")): str(item.get("surface", "")).strip()
                     for item in actual_anchors if isinstance(item, Mapping)}
    binding = grounded.anchor_bindings.get(root)
    label = labels.get(root, "")
    if (binding is None or not label or len(actual_labels) != len(actual_anchors)
            or _label(label) != _label(actual_labels.get(root, ""))
            or _label(label) != _label(binding.label)
            or set(grounded.anchor_bindings) != {root} or set(actual_labels) != {root}):
        return abstain("actual_root_label_or_grounded_anchor_scope_not_identical")
    source_anchors = graph.provenance.get("anchor_bindings")
    if (not isinstance(source_anchors, Mapping) or not source_anchors
            or set(source_anchors) != {root}):
        return abstain("actual_source_anchor_scope_not_exactly_owned_root")
    source_anchor = source_anchors[root]
    if (not isinstance(source_anchor, Mapping)
            or not isinstance(source_anchor.get("id"), str) or not source_anchor["id"]
            or source_anchor["id"] != binding.entity_id
            or not isinstance(source_anchor.get("label"), str) or not source_anchor["label"].strip()
            or _label(source_anchor["label"]) != _label(label)
            or _label(source_anchor["label"]) != _label(binding.label)):
        return abstain("actual_source_root_binding_not_identical_to_original_and_grounded_anchor")
    topic_accounting = []
    for identifier, surface in labels.items():
        if identifier == root:
            continue
        used = any(identifier in (step["from"], step["to"]) for step in steps)
        mentioned = any(re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)",
                                  str(value), re.I) for value in (question, path.get("goal", ""), *decomposition))
        if used or mentioned:
            return abstain("extra_original_topic_is_required_or_unaccounted")
        topic_accounting.append({"id": identifier, "surface": surface,
                                 "disposition": "unused_and_absent_from_question"})
    text = " ".join(str(question).split())
    clause = re.match(r"^Who\s+([A-Za-z]+)\s+(?:the\s+)?[\"“]?" + re.escape(label)
                      + r"[\"”]?(?=\s*[,?]|\s+(?:who|which|that|whose|with|after|before|from|in|on)\b|$)",
                      text, re.I)
    if not clause or not re.fullmatch(r"[A-Za-z]+(?:ed|s)", clause.group(1)):
        return abstain("question_has_no_direct_regular_active_who_role")
    verb = clause.group(1).casefold()
    if verb in {"is", "was", "does", "has"} or (not verb.endswith("ed") and verb.endswith(("ers", "ors"))):
        return abstain("question_primary_token_not_an_unambiguous_regular_active_verb")
    first = steps[0]
    relation = str(grounded.relation_bindings.get(first["id"], ""))
    stems = _regular_role_stems(verb)
    if (not stems & _regular_role_stems(first["relation_label"][-1])
            or not stems & _regular_role_stems(relation.rsplit(".", 1)[-1])):
        return abstain("question_original_primary_and_actual_relation_disagree")
    role_type = str(ontology.range_for_relation(relation))
    if role_type in SCALAR_TYPES or "common.topic" not in set(ontology.supertypes(role_type)):
        return abstain("primary_relation_lacks_positive_entity_schema_proof")
    provenance = graph.provenance
    if any(key in provenance for key in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("postprocessed_source_is_not_one_actual_execution")
    if graph.operators:
        return abstain("source_already_has_scalar_or_other_operator")
    if (len(graph.triples) != 1 or not _VAR.fullmatch(str(graph.answer_var))
            or list(map(str, graph.triples[0])) != [str(binding.entity_id), relation, str(graph.answer_var)]):
        return abstain("actual_source_is_not_exact_direct_primary_entity_role")
    expected_query = lower_sparql(graph)
    if (not isinstance(source_query, str) or not source_query.strip()
            or source_query.strip() != expected_query.strip()
            or str(graph.sparql).strip() != source_query.strip()):
        return abstain("actual_source_owned_query_not_exact_unlimited_lowering")
    ids = source.answer_ids
    if (not isinstance(ids, list) or not ids or not all(isinstance(value, str) for value in ids)
            or not isinstance(source.row_count, int) or source.row_count != len(ids)
            or isinstance(source.row_count, bool) or len(set(ids)) != len(ids)
            or any(not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", str(value)) for value in ids)):
        return abstain("actual_source_entity_denotation_not_complete")
    scalar = [(step, str(grounded.relation_bindings.get(step["id"], ""))) for step in steps
              if str(ontology.range_for_relation(str(grounded.relation_bindings.get(step["id"], "")))) in SCALAR_TYPES]
    if len(scalar) != 1 or scalar[0][0] is not steps[-1]:
        return abstain("not_one_terminal_original_grounded_scalar_obligation")
    terminal, scalar_relation = scalar[0]
    range_id = str(ontology.range_for_relation(scalar_relation))
    spec = _one_explicit_scalar_spec(text, list(labels.values()), range_id)
    if spec is None:
        return abstain("question_scalar_literal_or_comparison_is_ambiguous")
    goal = str(path.get("goal", "")).strip()
    if (not isinstance(decomposition, Sequence) or isinstance(decomposition, str)
            or list(decomposition) != [goal]):
        return abstain("original_goal_not_independently_attached_to_same_decomposition")
    if _one_explicit_scalar_spec(goal, list(labels.values()), range_id) != spec:
        return abstain("question_and_original_goal_scalar_obligations_disagree")
    clauses = [part.strip().rstrip(".").strip() for part in goal.split("?") if part.strip()]
    primary = _tokens(first["relation_label"][-1])
    owner = _tokens(steps[-2]["relation_label"][-1])
    leaf = _tokens(terminal["relation_label"][-1])
    expected_first = ("who", "is", "the", *primary, "of", *_tokens(first["relation_label"][-2]), *_tokens(label))
    expected_second = ("what", "is", "the", *owner, "of", "the", *primary)
    expected_last = ("return", "the", *primary, "whose", *owner, "has", "a", *leaf,
                     "date" if range_id == "type.datetime" else "value")
    if (len(steps) != 3 or len(clauses) != 3 or _tokens(clauses[0]) != expected_first
            or _tokens(clauses[1]) != expected_second
            or _tokens(clauses[2])[:len(expected_last)] != expected_last
            or _one_explicit_scalar_spec(clauses[2], [], range_id) != spec):
        return abstain("original_goal_does_not_explicitly_return_and_constrain_same_primary_owner")
    comparison = list(_COMPARISON.finditer(clauses[2]))
    if len(comparison) != 1 or _tokens(clauses[2][:comparison[0].start()]) != expected_last:
        return abstain("original_goal_has_unaccounted_condition_words")
    literal_text = clauses[2][comparison[0].end():].strip()
    if not ((_ISO_DATE.fullmatch(literal_text) or _NAMED_DATE.fullmatch(literal_text))
            if range_id == "type.datetime"
            else re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?", literal_text)):
        return abstain("original_goal_has_unaccounted_condition_words")
    # Consume the entire relative condition. An added topic/condition outside
    # the original path is not silently ignored even when its date agrees.
    tail = text[clause.end():]
    if not _question_relative_owner_tail(tail, owner, leaf, range_id):
        return abstain("question_relative_owner_or_extra_scope_not_fully_accounted")
    candidate, compilation = compile_grounded_constraint_repair(
        grounded, graph, question, spec, ontology, graph_id=graph_id,
    )
    if candidate is None:
        return abstain("existing_f4_grounded_extension_did_not_compile", compilation=compilation)
    evidence.update(status="compiled_not_executed", reason="independent_current_primary_entity_missing_scalar_witness",
                    source_graph_id=graph.graph_id, source_answer_count=len(ids),
                    primary_relation_id=relation, primary_entity_type=role_type,
                    question_active_verb=verb, normalized_scalar_spec=spec,
                    scope_certificate=scope, extra_original_topics=topic_accounting,
                    original_goal_decomposition_same_primary_owner=True,
                    actual_source_original_question_and_root_binding_verified=True,
                    actual_source_owned_query_verified=True,
                    required_repair_denotation="complete_nonempty_strict_subset_of_actual_source_ids",
                    requires_actual_paid_saved_count_and_time_credit=True,
                    query_executed=False, full_flow_recovery_proven=False,
                    compilation=compilation)
    return candidate, evidence
