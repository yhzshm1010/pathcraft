"""Pure scope gate for an existing mediator's outgoing projection fields.

No queries, embeddings, models, or answer choices are made. Only a complete
existing field pool with one concrete requested target range is narrowed. The
actual CVT variable and every original graph constraint remain untouched.
Shared schema ancestors are not type-disjointness or target-role evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .nested_name_schema_guard import _concrete_type, _type_index
from .source_role_execution_safety import _decomposition_lines, _tokens, source_entity_head


_WH_FIELD = re.compile(r"\b(?:what|which)\s+(?:is|was|are|were)\s+(?:the\s+)?"
                       r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)", re.I)


def _phrase(value: str) -> str:
    return " ".join(_tokens(str(value).replace("_", " ")))


def mediator_terminal_role_scope(
    question: str, source_graph: Any, mediator_type: str, candidate_relations: list[str], ontology: Any,
    *, decompositions: Any = None,
) -> tuple[list[str] | None, dict[str, Any]]:
    """Return a strict one-field subset, or None to preserve the existing lane.

    Call before the existing BGE/projection loop. The caller must already have
    established its ordinary mediator eligibility. An exact current-question
    noun head, original decomposition's same-owner field declaration, known
    actual incoming CVT binding, and a complete cached schema field pool are
    all required. Unknown schema and multiple matching fields abstain.
    """
    evidence = {"status": "not_applied", "source_graph_id": str(source_graph.graph_id),
                "actual_mediator_variable": str(source_graph.answer_var),
                "actual_mediator_type": str(mediator_type),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "changes_source_graph_or_operators": False, "selects_answer_ids": False,
                "schema_sibling_disjointness_claimed": False}

    def abstain(reason):
        evidence["reason"] = reason
        return None, evidence

    try:
        head = source_entity_head(question)
        evidence["question_head"] = head
        expected = set(_type_index(ontology).get(head, ())) if head else set()
        evidence["expected_concrete_target_types"] = sorted(expected)
        if not expected:
            return abstain("question_head_unknown_compound_modifier_or_not_entity_noun")
        if not _concrete_type(str(mediator_type)):
            return abstain("actual_mediator_type_unknown_or_generic")
        provenance = getattr(source_graph, "provenance", None)
        if not isinstance(provenance, Mapping) or provenance.get("selection_answer_union"):
            return abstain("source_current_provenance_missing_or_union_scope_unproven")
        if provenance.get("original_question") is not None and str(provenance["original_question"]).strip() != str(question).strip():
            return abstain("current_question_and_source_provenance_conflict")
        incoming = [str(relation) for _, relation, object_ in source_graph.triples
                    if str(object_) == str(source_graph.answer_var)]
        types = {str(ontology.range_for_relation(relation) or "") for relation in incoming}
        evidence["actual_incoming_range_types"] = sorted(types)
        if not incoming or types != {str(mediator_type)}:
            return abstain("actual_mediator_node_binding_unknown_or_not_unique")
        if (not isinstance(candidate_relations, list) or not candidate_relations
                or not all(isinstance(value, str) and value for value in candidate_relations)
                or len(set(candidate_relations)) != len(candidate_relations)):
            return abstain("candidate_field_pool_missing_or_malformed")
        lookup = getattr(ontology, "relations_for_domain", None)
        if not callable(lookup):
            return abstain("complete_cached_mediator_field_pool_unavailable")
        complete, unknown = [], []
        for relation in lookup(str(mediator_type)):
            relation = str(relation)
            domain = str(ontology.domain_for_relation(relation) or "")
            range_ = str(ontology.range_for_relation(relation) or "")
            if domain != str(mediator_type) or not range_:
                unknown.append(relation)
            elif not range_.startswith("type."):
                complete.append(relation)
        evidence["complete_cached_entity_projection_fields"] = sorted(complete)
        if unknown:
            evidence["unknown_schema_fields"] = sorted(unknown)
            return abstain("cached_outgoing_field_schema_unknown_or_wrong_owner")
        if set(complete) != set(candidate_relations):
            return abstain("candidate_projection_pool_not_complete_cached_schema_pool")
        compatible, uncertain = [], []
        schema = []
        expected_ancestors = {str(value) for type_id in expected for value in ontology.supertypes(type_id)}
        for relation in candidate_relations:
            domain = str(ontology.domain_for_relation(relation) or "")
            range_ = str(ontology.range_for_relation(relation) or "")
            if domain != str(mediator_type) or not _concrete_type(range_):
                uncertain.append(relation)
                schema.append({"relation_id": relation, "domain": domain, "range": range_, "status": "unknown"})
                continue
            closure = {range_, *(str(value) for value in ontology.supertypes(range_))}
            status = "positive_target_range" if closure & expected else "different_declared_role"
            if status == "positive_target_range":
                compatible.append(relation)
            elif range_ in expected_ancestors:
                uncertain.append(relation)
                status = "generic_expected_ancestor_not_role_proof"
            schema.append({"relation_id": relation, "domain": domain, "range": range_, "status": status})
        evidence.update(candidate_schema=schema, compatible_projection_fields=compatible,
                        uncertain_projection_fields=uncertain)
        if uncertain or len(compatible) != 1:
            return abstain("target_projection_field_unknown_missing_or_multiple")
        selected = compatible[0]
        leaf, owner = selected.rsplit(".", 1)[-1], str(mediator_type).rsplit(".", 1)[-1]
        declared = {(_phrase(property_), _phrase(owner_)) for line in _decomposition_lines(
            decompositions if decompositions is not None else provenance.get("decomposition"))
            for property_, owner_ in _WH_FIELD.findall(line)}
        requested = (_phrase(leaf), _phrase(owner))
        evidence["expected_same_owner_decomposition_declaration"] = requested
        if requested not in declared:
            return abstain("original_decomposition_same_mediator_target_field_unproven")
        if len(candidate_relations) == 1:
            return abstain("existing_projection_field_already_unique")
        evidence.update(status="applied", reason="unique_exact_question_and_declared_mediator_target_range",
                        allowed_projection_fields=[selected],
                        excluded_projection_fields=[relation for relation in candidate_relations if relation != selected],
                        preserved_original_constraint_count=len(source_graph.operators),
                        preserved_source_mediator_variable=str(source_graph.answer_var),
                        no_new_projection_field=True)
        return [selected], evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="current_mediator_or_schema_context_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence
