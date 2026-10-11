"""Pure compiler for an explicit year equality on an existing bound field.

No relation/projection/anchor/other operator changes; no execution, selection,
model, embedding, file or Gold access. A runtime caller owns saved-request/time and nonregression
gates. Source singleton correction is a diagnostic opportunity, not a claim
that an already-perfect singleton can be changed safely.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any


_COMPARISONS = frozenset({"EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"})
_VARIABLE = re.compile(r"V\d+\Z")
_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")


def _words(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", str(value).replace("_", " ").casefold()))


def _attribute_words(predicate: str) -> tuple[str, ...]:
    return tuple(w for w in _words(str(predicate).rsplit(".", 1)[-1]) if w not in {"date", "time", "of"})


def _contains(text: str, phrase: tuple[str, ...]) -> bool:
    words = _words(text)
    return bool(phrase) and any(words[i:i + len(phrase)] == phrase for i in range(len(words) - len(phrase) + 1))


def compile_bound_year_equality(
    question: str, source_graph: Any, ontology: Any, *, graph_id: str = "bound_year_equality_diagnostic",
) -> tuple[Any | None, dict[str, Any]]:
    """Compile one uniquely attested event-year comparison; never select IDs."""
    evidence = {"status": "not_compiled", "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "uses_entity_or_question_whitelist": False, "compiler_selects_answers": False,
                "requires_actual_saved_budget_before_runtime_execution": True,
                "pure_compiler_does_not_execute_or_select_answers": True}
    def abstain(reason):
        evidence["reason"] = reason
        return None, evidence

    question = str(question)
    if (re.search(r'["“”]|\b(?:not|never|before|after|prior|later|earlier|less|more|greater|lower|higher|during|through|until|since|between|or|either)\b', question, re.I)
            or re.findall(r"(?<!\w)\d+(?!\w)", question) == []):
        return abstain("question_is_not_one_unambiguous_event_year_equality")
    years = re.findall(r"(?<!\w)([12]\d{3})(?!\w)", question)
    numbers = re.findall(r"(?<!\w)\d+(?!\w)", question)
    if len(years) != 1 or numbers != years:
        return abstain("question_has_other_numbers_or_multiple_years")
    year = years[0]
    triples = getattr(source_graph, "triples", None)
    operators = getattr(source_graph, "operators", None)
    provenance = getattr(source_graph, "provenance", None)
    if (ontology is None or not isinstance(triples, list) or len(triples) != 2
            or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)
            or not isinstance(operators, list) or any(not isinstance(o, Mapping) for o in operators)
            or not isinstance(provenance, Mapping)):
        return abstain("not_one_existing_entity_path_and_bound_datetime_field")
    if any(provenance.get(key) for key in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("postprocessed_source_not_one_query")
    answer = str(source_graph.answer_var)
    fields = [(str(a), str(p), str(b)) for a, p, b in triples
              if str(a) == answer and str(ontology.range_for_relation(str(p))) == "type.datetime"]
    if len(fields) != 1 or not _VARIABLE.fullmatch(fields[0][2]):
        return abstain("year_field_is_not_unique_or_not_owned_by_answer")
    owner, attribute, value_var = fields[0]
    context = [tuple(map(str, t)) for t in triples if tuple(map(str, t)) != fields[0]]
    if len(context) != 1 or context[0][2] != owner or not _ENTITY.fullmatch(context[0][0]):
        return abstain("context_is_not_one_anchor_to_answer_entity_role")
    attribute_words = _attribute_words(attribute)
    attribute_pattern = r"\s+".join(re.escape(word) for word in attribute_words)
    if not attribute_words or not re.search(r"\b" + attribute_pattern + r"\s+in\s+" + year + r"\b", question, re.I):
        return abstain("question_does_not_explicitly_name_bound_field_in_year")
    entity_type = str(ontology.range_for_relation(context[0][1]) or "")
    domain = str(ontology.domain_for_relation(attribute) or "")
    if entity_type != domain or not entity_type or entity_type.startswith("type."):
        return abstain("entity_role_and_datetime_owner_not_exactly_same_declared_type")
    bindings = provenance.get("anchor_bindings")
    if not isinstance(bindings, Mapping):
        return abstain("actual_anchor_label_binding_missing")
    anchors = [value for value in bindings.values() if isinstance(value, Mapping) and str(value.get("id")) == context[0][0]]
    if len(anchors) != 1 or not str(anchors[0].get("label", "")) or not _contains(question, _words(anchors[0]["label"])):
        return abstain("actual_anchor_name_is_not_unique_in_question")
    decomposition = provenance.get("decomposition")
    if not isinstance(decomposition, list):
        return abstain("independent_decomposition_year_field_attachment_missing")
    entity_words = _words(entity_type.rsplit(".", 1)[-1])
    attached = [text for text in decomposition if isinstance(text, str)
                and _contains(text, attribute_words) and _contains(text, entity_words)
                and re.search(r"\b(?:is|equals?|equal\s+to)\s+" + year + r"\b", text, re.I)]
    if not attached:
        return abstain("decomposition_does_not_confirm_same_entity_field_year_equality")
    comparison = [(i, op) for i, op in enumerate(operators) if str(op.get("type", "")).upper() in _COMPARISONS]
    if len(comparison) != 1:
        return abstain("not_one_existing_comparison_operator")
    index, original = comparison[0]
    if (str(original.get("input_var")) != value_var or original.get("attribute_relation_id") not in (None, "")
            or original.get("attribute_relation_label") not in (None, [])
            or str(original.get("value")) != year or original.get("inputs") not in (None, [])):
        return abstain("source_comparison_is_not_on_that_already_bound_year_field")
    for i, op in enumerate(operators):
        if i == index or op.get("type") == "NO_EQUAL":
            continue
        if (op.get("type") != "TC" or str(op.get("input_var")) != owner
                or str(op.get("attribute_relation_id") or "") not in {"", attribute}
                or str(op.get("value")) != year
                or op.get("attribute_relation_label") not in (None, [], [x.replace("_", " ") for x in attribute.split(".")])):
            return abstain("another_substantive_operator_has_different_owner_field_or_value")
    if original.get("type") == "EQUAL" and original.get("_comparison_date_precision") == "year":
        return abstain("source_already_has_exact_bound_year_equality")
    result = deepcopy(source_graph)
    result.graph_id, result.sparql = graph_id, ""
    replacement = deepcopy(original)
    replacement.update(type="EQUAL", input_var=value_var, value=year, value_type="dateTime",
                       attribute_relation_id="", attribute_relation_label=[], attribute_relation_labels=[],
                       _attribute_value_already_bound=True, _comparison_date_precision="year", _calendar_day_boundary=False)
    for key in ("_numeric_comparison_cast", "_numeric_cast_compare", "_numeric_cast_kind", "_temporal_exists_semantics"):
        replacement.pop(key, None)
    result.operators[index] = replacement
    evidence.update(status="compiled_not_executed", reason="question_and_decomposition_attest_same_bound_event_year",
                    year=year, owner_var=owner, value_var=value_var, scalar_relation_id=attribute,
                    anchor_id=context[0][0], anchor_name=anchors[0]["label"],
                    source_graph_id=source_graph.graph_id, compiled_graph_id=graph_id,
                    preserves_all_source_triples=True, answer_var_unchanged=True, all_other_operators_unchanged=True,
                    independent_decomposition_matches=attached, original_operator=dict(original), compiled_operator=replacement)
    result.provenance["bound_year_equality_diagnostic"] = deepcopy(evidence)
    return result, evidence
