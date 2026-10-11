"""Pure compilation of a closed-question primary-node projection repair.

No candidate query is executed, and no answer is selected here. The caller may
spend only an already saved same-question request and measured time credit,
after all model stages. Every original triple/operator stays unchanged. An
ancestor-compatible answer type is not a trigger: exact closed question grammar,
a typed anchored main relation, and original Semantic variable evidence are.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from .evaluation import canonicalize_answer_id
from .nested_name_schema_guard import _anchors, _complete, _concrete_type, _entity_answers, _type_index
from .ontology import relation_id_from_label
from .source_role_execution_safety import _semantic_graphs, _semantic_path_nodes, _tokens, _word_forms


_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_WORD_PHRASE = r"[A-Za-z]+(?:\s+[A-Za-z]+){0,3}"
_ACTIVE = re.compile(r"^(?:what|which)\s+(" + _WORD_PHRASE + r")\s+([A-Za-z]+)(?:\s+the)?\s*$", re.I)
_RELATIVE = re.compile(r"^(?:what|which)\s+(?:is|was)\s+(?:(?:a|an|the)\s+)?("
                       + _WORD_PHRASE + r")\s+(?:where|for\s+which)\s*$", re.I)
_ROLE_SUFFIX = re.compile(r"^\s*(?:is|was)\s+(?:(?:a|an|the)\s+)?(" + _WORD_PHRASE + r")\s*[?.!]*\s*$", re.I)
_GOAL_HEAD = re.compile(r"\b(?:what|which)\s+(?:is|was|are|were)\s+(?:(?:a|an|the)\s+)?("
                        + _WORD_PHRASE + r")\s*(?=[?.!]|$)", re.I)


def _phrase(value: str) -> str:
    return " ".join(_tokens(value.replace("_", " ")))


def _finite_forms(word: str) -> set[str]:
    value = word.casefold()
    forms = {value}
    if len(value) > 4 and value.endswith("ed"):
        forms.update((value[:-2], value[:-1]))
    elif len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        forms.add(value[:-1])
    return forms


def _closed_question(question: str, label: str):
    text = str(question).strip()
    # Full topic span is allowed to contain a year/date; any number or
    # additional wording outside it must be accounted for by the closed grammar.
    words = str(label).strip().split()
    if not words:
        return None, "current_topic_label_missing"
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in words) + r"(?!\w)"
    matches = list(re.finditer(pattern, text, re.I))
    if len(matches) != 1:
        return None, "full_current_topic_label_not_unique_in_question"
    match = matches[0]
    before, after = text[:match.start()].strip(), text[match.end():]
    relative, suffix = _RELATIVE.fullmatch(before), _ROLE_SUFFIX.fullmatch(after)
    if relative and suffix:
        return {"grammar": "closed_relative_primary_role", "head": _phrase(relative[1]),
                "relation_cue": _phrase(suffix[1]), "topic_span": [match.start(), match.end()]}, "closed_question"
    active = _ACTIVE.fullmatch(re.sub(r"\s+the\s*$", "", before, flags=re.I))
    if active and re.fullmatch(r"\s*[?.!]*\s*", after):
        return {"grammar": "closed_active_primary_role", "head": _phrase(active[1]),
                "relation_cue": active[2].casefold(), "topic_span": [match.start(), match.end()]}, "closed_question"
    return None, "question_has_unsupported_extra_clause_modifier_or_scalar"


def _relation_cue_supported(spec: Mapping[str, Any], relation: str) -> bool:
    words = re.findall(r"[A-Za-z]+", relation.rsplit(".", 1)[-1].replace("_", " "))
    cue = str(spec["relation_cue"])
    if spec["grammar"] == "closed_active_primary_role":
        # Do not turn a nominal 'founding figures' field into an active verb
        # 'founded': source/subject voice would be reversed. Only finite lexical
        # forms of an actual predicate word support the active construction.
        return any(not word.casefold().endswith("ing") and _finite_forms(cue) & _finite_forms(word)
                   for word in words)
    cue_words = _tokens(cue)
    schema_words = _tokens(" ".join(words))
    return bool(cue_words) and all(any(_word_forms(word) & _word_forms(schema_word)
                                      for schema_word in schema_words) for word in cue_words)


def _operators_supported(graph: Any, anchor: str) -> bool:
    variables = {str(node) for row in graph.triples for node in (row[0], row[2])
                 if canonicalize_answer_id(node) != anchor}
    for operator in graph.operators:
        if (not isinstance(operator, Mapping) or str(operator.get("type", "")).upper() != "NO_EQUAL"
                or str(operator.get("input_var", "")) not in variables
                or operator.get("inputs") or operator.get("attribute_relation_label")
                or operator.get("attribute_relation_labels") or operator.get("attribute_relation_id")):
            return False
        if operator.get("value_type") == "variable":
            if str(operator.get("value", "")) not in variables:
                return False
        elif operator.get("value_type") in {"entity", "mid"}:
            value = operator.get("_value_entity_id") or operator.get("value")
            if canonicalize_answer_id(value) != anchor:
                return False
        else:
            return False
    return True


def _semantic_primary_witness(source: Any, target: str, relation: str, anchor: str,
                              names: set[str], endpoint_type: str, ontology: Any, semantics: Any,
                              context: list[Mapping[str, Any]]):
    index = _type_index(ontology)
    witnesses = []
    for semantic in _semantic_graphs(semantics):
        references = {str(item.get("id")) for item in semantic["anchors"]
                      if isinstance(item, Mapping) and str(item.get("surface", "")).strip().casefold() in names}
        for path in semantic["semantic_paths"]:
            if not isinstance(path, Mapping) or str(path.get("anchor_ref")) not in references:
                continue
            steps = path.get("steps")
            if not isinstance(steps, list) or not steps or not isinstance(steps[0], Mapping):
                continue
            first = steps[0]
            if (str(first.get("from")) != str(path.get("anchor_ref"))
                    or relation_id_from_label(first.get("relation_label")) != relation):
                continue
            prefix = {**path, "steps": [first], "path_output_var": str(first.get("to", ""))}
            if _semantic_path_nodes(source, semantic, prefix, context) != {target}:
                continue
            heads = [_phrase(match[1]) for match in _GOAL_HEAD.finditer(str(path.get("goal", "")))]
            matching = [head for head in heads if endpoint_type in index.get(head, ())]
            if not matching:
                continue
            witnesses.append({"semantic_anchor_ref": str(path.get("anchor_ref")),
                              "semantic_primary_variable": str(first.get("to", "")),
                              "mapped_actual_primary_variable": target,
                              "semantic_primary_direction": str(first.get("direction", "")),
                              "original_wh_primary_heads": matching})
    return witnesses


def compile_primary_projection_repair(
    question: str, source: Any, ontology: Any, semantic_graphs: Any,
    *, linked_entity_context: Any = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Compile only a different answer_var, never execute or choose answers.

    Protect singleton, incomplete, union, multi-anchor and scalar/extrema cases.
    A nominal adjective not represented by an exact ontology head (e.g. 'major
    religion') is unknown and abstains; there is no descriptor exception list.
    """
    graph = source.graph
    evidence = {"status": "not_compiled", "source_graph_id": str(graph.graph_id),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "requires_saved_request_and_remaining_measured_time": True,
                "requires_post_model_execution": True, "compiler_selects_answers": False,
                "preserves_all_source_triples_and_operators": True,
                "ancestor_schema_compatibility_is_not_trigger": True}

    def abstain(reason):
        evidence["reason"] = reason
        return None, evidence

    try:
        ids = _entity_answers(source)
        if ids is None or not _complete(source) or len(ids) <= 1:
            return abstain("source_singleton_incomplete_or_not_pure_entities")
        provenance = getattr(graph, "provenance", None)
        if not isinstance(provenance, Mapping) or provenance.get("selection_answer_union"):
            return abstain("source_current_bindings_missing_or_union_scope_unproven")
        original_question = provenance.get("original_question")
        if original_question is not None and str(original_question).strip() != str(question).strip():
            return abstain("current_source_question_provenance_conflict")
        anchors = _anchors(graph)
        if len(anchors) != 1 or not all(_ENTITY.fullmatch(value) for value in anchors):
            return abstain("not_single_entity_anchor_current_graph")
        anchor = next(iter(anchors))
        bindings = provenance.get("anchor_bindings")
        if not isinstance(bindings, Mapping):
            return abstain("source_current_anchor_bindings_missing")
        labels = {str(value.get("label", "")).strip() for value in bindings.values()
                  if isinstance(value, Mapping) and canonicalize_answer_id(value.get("id", "")) == anchor}
        labels.discard("")
        if len(labels) != 1:
            return abstain("source_current_anchor_label_missing_or_ambiguous")
        label = next(iter(labels))
        spec, reason = _closed_question(question, label)
        if spec is None:
            return abstain(reason)
        evidence["closed_question"] = spec
        expected = set(_type_index(ontology).get(spec["head"], ()))
        if not expected:
            evidence["unsupported_head_phrase"] = spec["head"]
            evidence["unknown_modifier_or_head"] = True
            return abstain("primary_head_or_modifier_not_exact_ontology_vocabulary")
        if not _operators_supported(graph, anchor):
            return abstain("source_has_unsupported_scalar_extrema_or_external_operator")
        if not isinstance(linked_entity_context, (list, tuple)):
            return abstain("current_entity_context_missing")
        names = {str(item.get("name") or item.get("label") or "").strip().casefold()
                 for item in linked_entity_context if isinstance(item, Mapping)
                 and canonicalize_answer_id(item.get("id", "")) == anchor}
        names.discard("")
        if names != {label.casefold()}:
            return abstain("current_context_and_bound_topic_label_conflict")
        # The whole unchanged BGP must be connected, so another disconnected
        # branch cannot become an unreported obligation of the new projection.
        adjacency = {}
        for subject, relation, object_ in graph.triples:
            subject, object_ = str(subject), str(object_)
            adjacency.setdefault(subject, set()).add(object_)
            adjacency.setdefault(object_, set()).add(subject)
        reached, pending = {anchor}, [anchor]
        while pending:
            for neighbor in adjacency[pending.pop()] - reached:
                reached.add(neighbor)
                pending.append(neighbor)
        if reached != set(adjacency):
            return abstain("source_bgp_disconnected")
        candidates, unknown = [], []
        for subject, relation, object_ in graph.triples:
            subject, relation, object_ = map(str, (subject, relation, object_))
            if anchor not in (subject, object_) or not _relation_cue_supported(spec, relation):
                continue
            target = object_ if subject == anchor else subject
            side = "range_for_relation" if subject == anchor else "domain_for_relation"
            endpoint_type = str(getattr(ontology, side)(relation) or "")
            if not _concrete_type(endpoint_type):
                unknown.append(relation)
                continue
            if endpoint_type not in expected:
                continue
            witnesses = _semantic_primary_witness(source, target, relation, anchor, names,
                endpoint_type, ontology, semantic_graphs, list(linked_entity_context))
            if not witnesses:
                continue
            candidates.append((target, relation, endpoint_type, witnesses))
        targets = {target for target, _, _, _ in candidates}
        evidence.update(primary_candidate_variables=sorted(targets), unknown_primary_relation_schema=sorted(set(unknown)))
        if unknown or len(targets) != 1:
            return abstain("primary_variable_schema_or_original_semantic_not_unique")
        target = next(iter(targets))
        if target == str(graph.answer_var):
            return abstain("closed_question_primary_variable_already_projected")
        target, relation, endpoint_type, witnesses = next(value for value in candidates if value[0] == target)
        repaired = deepcopy(graph)
        repaired.graph_id = str(graph.graph_id) + "__primary_projection"
        repaired.answer_var = target
        repaired.sparql = ""
        repaired.provenance = deepcopy(dict(provenance))
        evidence.update(status="compiled", reason="closed_question_typed_primary_variable_projection",
                        target_answer_var=target, original_answer_var=str(graph.answer_var),
                        primary_relation_id=relation, primary_endpoint_type=endpoint_type,
                        original_semantic_variable_witnesses=witnesses)
        repaired.provenance["budgeted_primary_projection_repair"] = deepcopy(evidence)
        return repaired, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="source_question_or_schema_evidence_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence
