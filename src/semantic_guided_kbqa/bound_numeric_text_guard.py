"""Pure value-shape guard for existing bound numeric text comparisons.

Only malformed textual/enum values are excluded. Relation paths, projection,
literal, comparator, cast, and all existing operators remain unchanged. The
caller owns complete-source, saved-budget, result-cap and strict-subset gates.
"""
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re

from .budgeted_schema_repair import STRING_TYPES, lower_budgeted_scalar_query


_VAR = re.compile(r"V\d+\Z")
_NUMERIC = re.compile(r"[+-]?\d+(?:\.\d+)?\Z")
_COMPARISONS = frozenset({"LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL", "EQUAL"})


def compile_bound_numeric_text_guard(source_graph, ontology, *, graph_id="bound_numeric_text_guard"):
    proof = {"status": "not_compiled", "pure_compiler_only": True,
             "additional_model_calls": 0, "additional_endpoint_queries": 0,
             "additional_embedding_calls": 0, "uses_gold_answers": False,
             "uses_entity_relation_or_sample_whitelist": False,
             "requires_actual_complete_source_and_saved_budget": True}

    def abstain(reason):
        return None, {**proof, "reason": reason}

    triples = getattr(source_graph, "triples", None)
    operators = getattr(source_graph, "operators", None)
    provenance = getattr(source_graph, "provenance", None)
    if (ontology is None or not isinstance(triples, list) or not triples
            or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)
            or not isinstance(operators, list) or not isinstance(provenance, Mapping)):
        return abstain("source_graph_or_schema_contract_missing")
    if any(provenance.get(k) for k in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("postprocessed_answer_is_not_one_source_query")
    eligible = []
    for index, operator in enumerate(operators):
        if not isinstance(operator, Mapping):
            return abstain("source_operator_contract_malformed")
        kind = str(operator.get("type", "")).upper()
        numeric_cast = operator.get("_numeric_cast_compare") if kind == "EQUAL" else operator.get("_numeric_comparison_cast")
        if kind not in _COMPARISONS or not numeric_cast:
            continue
        variable = str(operator.get("input_var", ""))
        if (not _VAR.fullmatch(variable) or variable == str(source_graph.answer_var)
                or not operator.get("_attribute_value_already_bound")
                or operator.get("attribute_relation_id") or operator.get("attribute_relation_label")
                or operator.get("attribute_relation_labels") or operator.get("inputs")
                or operator.get("_budgeted_numeric_string_value_guard")):
            continue
        value = str(operator.get("value", ""))
        try:
            if not _NUMERIC.fullmatch(value) or not Decimal(value).is_finite():
                continue
        except InvalidOperation:
            continue
        fields = [tuple(map(str, t)) for t in triples if str(t[2]) == variable]
        if (len(fields) != 1 or fields[0][0] != str(source_graph.answer_var)
                or any(str(t[0]) == variable for t in triples)):
            continue
        _, predicate, _ = fields[0]
        domain = str(ontology.domain_for_relation(predicate) or "")
        datatype = str(ontology.range_for_relation(predicate) or "")
        if (not domain or "common.topic" not in set(ontology.supertypes(domain))
                or datatype not in STRING_TYPES | {"type.enumeration"}):
            continue
        eligible.append((index, variable, predicate, datatype))
    if not eligible:
        return abstain("no_unguarded_bound_numeric_text_field_on_entity_answer")
    result = deepcopy(source_graph)
    result.graph_id, result.sparql = graph_id, ""
    for index, _, _, _ in eligible:
        result.operators[index]["_budgeted_numeric_string_value_guard"] = True
    proof.update(status="compiled_not_executed", reason="bound_text_numeric_cast_requires_valid_full_value_shape",
                 source_graph_id=source_graph.graph_id, guarded_variables=[v for _, v, _, _ in eligible],
                 guarded_fields=[{"relation_id": p, "range": t} for _, _, p, t in eligible],
                 source_triples_projection_literals_comparators_and_casts_unchanged=True,
                 no_numeric_cast_replacement_or_truncation_fix=True,
                 require_uncapped_nonempty_strict_subset_of_actual_source=True)
    result.provenance["bound_numeric_text_guard"] = deepcopy(proof)
    return result, proof


def lower_bound_numeric_text_guard(candidate, *, limit=256):
    return lower_budgeted_scalar_query(candidate, limit=limit)
