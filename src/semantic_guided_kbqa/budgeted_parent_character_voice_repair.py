"""Pure typed repair of a parent-name query accidentally returning a date.

Only current original Semantic, already resolved source anchors/context, and
cached official schema/inverses are used. All source BGP and operators remain.
The caller alone may execute one query against saved request AND time credit.
Animated/other unsupported entity-head modifiers are not silently ignored.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import re
from typing import Any

from .budgeted_primary_projection_repair import _finite_forms
from .evaluation import canonicalize_answer_id
from .nested_name_schema_guard import _anchors, _complete, _concrete_type
from .ontology import relation_id_from_label
from .source_role_execution_safety import _decomposition_lines, _semantic_graphs, _semantic_path_nodes, _tokens


_QUESTION = re.compile(r"^\s*what\s+is\s+(?:the\s+)?name\s+of\s+(.+?)['’]s\s+"
                       r"([A-Za-z]+)\s+that\s+(.+?)\s+([A-Za-z]+)\s*[?.!]\s*$", re.I)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_VAR = re.compile(r"V\d+\Z")
_WH_FIELD = re.compile(r"\b(?:what|which)\s+(?:is|was|are|were)\s+(?:the\s+)?"
                       r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)", re.I)


def _phrase(value: str) -> str:
    return " ".join(_tokens(str(value).replace("_", " ")))


def _context_has_type(context: Mapping[str, Any], type_id: str) -> bool:
    values = context.get("types")
    return (isinstance(values, list) and _concrete_type(type_id)
            and _phrase(type_id.rsplit(".", 1)[-1]) in {_phrase(str(value)) for value in values})


def compile_parent_character_voice_repair(
    question: str, source: Any, ontology: Any, semantic_graphs: Any, *,
    linked_entity_context: Any = None, decompositions: Any = None,
    question_anchor_contract: Mapping[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Compile a current parent/character/typed performance condition, or abstain.

    Scalar answer contradiction is necessary, so ordinary entity answers are
    untouched. Original resolved IDs are reused, not relinked; the actor name
    must agree with this question and context. The child remains the same
    already resolved original Semantic/source anchor, including supplied mention
    spelling variants; no fuzzy lookup or new entity ID is generated.
    """
    graph = source.graph
    evidence = {"status": "not_compiled", "source_graph_id": str(graph.graph_id),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "requires_saved_request_and_remaining_measured_time": True,
                "requires_post_model_execution": True, "compiler_selects_answers": False,
                "preserves_all_source_bgp_and_operators": True,
                "does_not_relink_or_generate_entity_ids": True}

    def abstain(reason):
        evidence["reason"] = reason
        return None, evidence

    try:
        match = _QUESTION.fullmatch(str(question))
        if not match:
            evidence["unsupported_question_head_or_modifier"] = True
            return abstain("not_closed_parent_name_and_performance_question")
        child_mention, kinship, actor_mention, performance_cue = match.groups()
        if re.search(r"\d", child_mention + actor_mention):
            return abstain("unsupported_numeric_named_clause")
        if not _complete(source) or not source.answer_ids:
            return abstain("actual_source_complete_denotation_missing")
        normalized = [canonicalize_answer_id(value) for value in source.answer_ids]
        if any(_ENTITY.fullmatch(value) or not _DATE.fullmatch(value) for value in normalized):
            return abstain("source_not_pure_calendar_date_role_contradiction")
        for value in normalized:
            date.fromisoformat(value)
        incoming = [str(relation) for _, relation, object_ in graph.triples if str(object_) == str(graph.answer_var)]
        if (not incoming or any(str(ontology.range_for_relation(relation) or "") != "type.datetime" for relation in incoming)
                or any(str(subject) == str(graph.answer_var) for subject, _, _ in graph.triples)):
            return abstain("source_answer_schema_not_known_date_only")
        provenance = getattr(graph, "provenance", None)
        if (not isinstance(provenance, Mapping) or provenance.get("selection_answer_union")
                or str(provenance.get("original_question", "")).strip() != str(question).strip()):
            return abstain("source_question_owned_graph_or_union_scope_unproven")
        anchors = _anchors(graph)
        if len(anchors) != 2 or not all(_ENTITY.fullmatch(value) for value in anchors):
            return abstain("source_not_exact_actor_and_child_entity_anchors")
        bindings = provenance.get("anchor_bindings")
        if not isinstance(bindings, Mapping) or not isinstance(linked_entity_context, (list, tuple)):
            return abstain("current_anchor_bindings_or_typed_context_missing")
        contexts = {}
        for item in linked_entity_context:
            if not isinstance(item, Mapping):
                continue
            entity_id = canonicalize_answer_id(item.get("id", ""))
            contexts.setdefault(entity_id, []).append(item)
        actor_ids = {entity_id for entity_id in anchors for item in contexts.get(entity_id, [])
                     if " ".join(str(item.get("name", "")).casefold().split()) == " ".join(actor_mention.casefold().split())}
        if len(actor_ids) != 1:
            return abstain("question_actor_name_not_uniquely_current_resolved_anchor")
        actor_id = next(iter(actor_ids))
        child_id = next(iter(anchors - {actor_id}))
        if len(contexts.get(actor_id, [])) != 1 or len(contexts.get(child_id, [])) != 1:
            return abstain("current_actor_or_child_context_ambiguous")
        child_name = str(contexts[child_id][0].get("name", "")).strip()
        child_mention_exact = " ".join(child_mention.casefold().split()) == " ".join(child_name.casefold().split())
        if not child_mention_exact:
            contract = question_anchor_contract
            if contract is None:
                contract = getattr(source, "question_anchor_contract", None)
            if contract is None:
                contract = provenance.get("question_anchor_contract")
            provided = contract.get("provided_topic_entities") if isinstance(contract, Mapping) else None
            if (not isinstance(contract, Mapping) or str(contract.get("question", "")) != str(question)
                    or not isinstance(provided, Mapping)
                    or str(provided.get(child_id, "")).strip().casefold() != child_name.casefold()
                    or str(provided.get(actor_id, "")).strip().casefold() != str(contexts[actor_id][0].get("name", "")).strip().casefold()):
                return abstain("nonliteral_child_mention_requires_current_question_provided_linking_contract")
            evidence["uses_current_question_provided_topic_contract"] = True
        else:
            evidence["uses_current_question_provided_topic_contract"] = False
        for operator in graph.operators:
            if (not isinstance(operator, Mapping) or str(operator.get("type", "")).upper() != "NO_EQUAL"
                    or str(operator.get("input_var", "")) != str(graph.answer_var)
                    or operator.get("inputs") or operator.get("attribute_relation_label")
                    or operator.get("attribute_relation_labels") or operator.get("attribute_relation_id")
                    or operator.get("value_type") not in {"mid", "entity"}
                    or canonicalize_answer_id(operator.get("_value_entity_id") or operator.get("value")) not in anchors):
                return abstain("source_other_operator_obligation_not_supported_by_closed_question")
        semantics = list(_semantic_graphs(semantic_graphs))
        actor_witnesses, parent_witnesses = [], []
        for semantic in semantics:
            for path in semantic["semantic_paths"]:
                if not isinstance(path, Mapping):
                    continue
                steps = path.get("steps")
                if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], Mapping):
                    continue
                step = steps[0]
                if step.get("direction") != "backward" or str(step.get("from")) != str(path.get("anchor_ref")):
                    continue
                binding = bindings.get(str(path.get("anchor_ref")))
                if not isinstance(binding, Mapping):
                    continue
                bound = canonicalize_answer_id(binding.get("id", ""))
                relation = relation_id_from_label(step.get("relation_label"))
                domain, range_ = str(ontology.domain_for_relation(relation) or ""), str(ontology.range_for_relation(relation) or "")
                if bound == actor_id and _context_has_type(contexts[actor_id][0], range_) and _concrete_type(domain):
                    nodes = _semantic_path_nodes(source, semantic, path, linked_entity_context)
                    if nodes is not None and len(nodes) == 1:
                        node = next(iter(nodes))
                        if [node, relation, actor_id] in [list(map(str, row)) for row in graph.triples]:
                            actor_witnesses.append((node, domain, relation))
                if bound != child_id or not _context_has_type(contexts[child_id][0], range_) or not _concrete_type(domain):
                    continue
                surfaces = {str(item.get("surface", "")).strip().casefold() for item in semantic["anchors"]
                            if isinstance(item, Mapping) and str(item.get("id")) == str(path.get("anchor_ref"))}
                if surfaces != {child_name.casefold()} or str(binding.get("label", "")).strip().casefold() != child_name.casefold():
                    continue
                for inverse in getattr(ontology, "reverse_relation_ids", {}).get(relation, ()):
                    if (relation not in getattr(ontology, "reverse_relation_ids", {}).get(inverse, ())
                            or _phrase(str(inverse).rsplit(".", 1)[-1]) != _phrase(kinship)
                            or ontology.domain_for_relation(inverse) != range_
                            or ontology.range_for_relation(inverse) != domain):
                        continue
                    parent_witnesses.append((str(inverse), domain, range_, relation))
        actor_unique, parent_unique = set(actor_witnesses), set(parent_witnesses)
        evidence.update(actor_semantic_witness_count=len(actor_unique), official_parent_inverse_witness_count=len(parent_unique))
        if len(actor_unique) != 1 or len(parent_unique) != 1:
            return abstain("original_actor_variable_or_official_parent_inverse_not_unique")
        actor_node, actor_cvt_type, actor_relation = next(iter(actor_unique))
        parent_relation, parent_type, child_type, inverse_relation = next(iter(parent_unique))
        declared = {(_phrase(property_), _phrase(owner_)) for line in _decomposition_lines(
            decompositions if decompositions is not None else provenance.get("decomposition"))
            for property_, owner_ in _WH_FIELD.findall(line)}
        character_fields = []
        for relation in ontology.relations_for_domain(actor_cvt_type):
            range_ = str(ontology.range_for_relation(relation) or "")
            if (ontology.domain_for_relation(relation) == actor_cvt_type and _concrete_type(range_)
                    and parent_type in ontology.supertypes(range_)
                    and (_phrase(str(relation).rsplit(".", 1)[-1]), _phrase(actor_cvt_type.rsplit(".", 1)[-1])) in declared):
                character_fields.append(str(relation))
        if len(character_fields) != 1:
            return abstain("declared_same_actor_cvt_parent_character_field_not_unique")
        character_field = character_fields[0]
        voice_candidates = []
        for entity_id, items in contexts.items():
            if entity_id in anchors or len(items) != 1 or not _ENTITY.fullmatch(entity_id):
                continue
            item = items[0]
            name_words = re.findall(r"[A-Za-z]+", str(item.get("name", "")))
            if len(name_words) != 1 or not (_finite_forms(name_words[0]) & _finite_forms(performance_cue)):
                continue
            for relation in ontology.relations_for_domain(actor_cvt_type):
                if (ontology.domain_for_relation(relation) == actor_cvt_type
                        and _context_has_type(item, str(ontology.range_for_relation(relation) or ""))):
                    voice_candidates.append((str(relation), entity_id))
        voice_unique = set(voice_candidates)
        if len(voice_unique) != 1:
            return abstain("current_performance_value_lemma_and_typed_field_not_unique")
        voice_field, voice_id = next(iter(voice_unique))
        if any(str(subject) == actor_node and str(relation) in {character_field, voice_field}
               for subject, relation, _ in graph.triples):
            return abstain("existing_actor_cvt_character_or_performance_binding_requires_separate_mapping")
        variables = {str(node) for row in graph.triples for node in (row[0], row[2]) if str(node) not in anchors}
        if not all(_VAR.fullmatch(value) for value in variables):
            return abstain("source_global_variable_contract_unproven")
        new_var = "V" + str(max(int(value[1:]) for value in variables) + 1)
        additions = [[actor_node, character_field, new_var], [child_id, parent_relation, new_var],
                     [actor_node, voice_field, voice_id]]
        repaired = deepcopy(graph)
        repaired.graph_id = str(graph.graph_id) + "__typed_parent_voice"
        repaired.answer_var = new_var
        repaired.triples = [*deepcopy(graph.triples), *additions]
        repaired.sparql = ""
        repaired.provenance = deepcopy(dict(provenance))
        evidence.update(status="compiled", reason="date_contradiction_with_original_parent_actor_and_typed_performance_obligations",
                        actor_entity_id=actor_id, child_entity_id=child_id, original_child_mention=child_mention,
                        reuses_original_resolved_child_without_fuzzy_lookup=True,
                        source_date_answer_var=str(graph.answer_var), target_answer_var=new_var,
                        actual_actor_cvt_variable=actor_node, actual_actor_cvt_type=actor_cvt_type,
                        original_actor_relation=actor_relation, character_relation=character_field,
                        parent_relation=parent_relation, original_inverse_relation=inverse_relation,
                        performance_relation=voice_field, current_performance_value_id=voice_id,
                        added_triples=additions, retained_source_triple_count=len(graph.triples),
                        retained_source_operator_count=len(graph.operators))
        repaired.provenance["budgeted_parent_character_voice_repair"] = deepcopy(evidence)
        return repaired, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="current_parent_voice_schema_or_context_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence
