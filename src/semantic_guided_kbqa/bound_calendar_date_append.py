"""Pure append of an explicit day filter to an already bound field.

The compiler adds exactly one EQUAL operator and preserves every source BGP
edge, answer projection and operator. It does not fix unrelated bad joins or
operators, execute queries, choose answers, access files or inspect Gold.
An actual caller still owns full-execution, saved-budget and nonregression
checks. A semantic obligation does not prove that changing an answer is safe.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from .bound_year_equality_safety import _contains, _words
from .budgeted_literal_constraint_repair import _spec
from .comparison_number_normalization import parse_english_calendar_date
from .numeric_temporal_slot_normalization import DATE_RE, lexical_tokens
from .ontology import relation_id_from_label


_VAR = re.compile(r"V\d+\Z")
_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_FILTER_TYPES = frozenset({"NO_EQUAL", "EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL", "TC"})
_AUXILIARY = frozenset({"is", "are", "was", "were", "has", "have", "had", "been", "being", "date", "time", "of", "the", "a", "an"})


def _tokens(value: Any) -> set[str]:
    # Reuse existing scalar morphology. The extra irregular form is lexical,
    # not a relation/entity/question allowlist.
    return {"birth" if token == "born" else token for token in lexical_tokens(str(value))}


def _leaf_tokens(predicate: str) -> set[str]:
    return _tokens(str(predicate).rsplit(".", 1)[-1]) - _AUXILIARY


def _normalized_dates(text: str, *, known_literal: str = "", known_value: str = "") -> list[str | None]:
    # The shared parser may normalize fully spelled-out day/year literals.
    # Recognize that exact attested span, not an arbitrary month substring.
    if known_literal and known_value and known_literal != known_value:
        text = re.sub(re.escape(known_literal), lambda match: known_value, text, flags=re.I)
    values = []
    for match in DATE_RE.finditer(text):
        value = match[0]
        values.append(value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
                      else parse_english_calendar_date(value))
    return values


def compile_bound_calendar_date_equality_append(
    question: str, source_graph: Any, ontology: Any, *, graph_id: str = "bound_day_append_shadow",
) -> tuple[Any | None, dict[str, Any]]:
    """Compile only a positively attached, uniquely represented day equality.

    There is deliberately no source-answer or index argument. Runtime use is
    not authorized merely by successful compilation. A runtime caller must
    require saved budget, a complete source and a nonempty strict subset.
    """
    evidence = {"status": "not_compiled", "pure_compiler_only": True,
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "uses_entity_or_question_whitelist": False, "compiler_selects_answers": False,
                "requires_actual_full_source_execution": True,
                "requires_saved_request_and_remaining_measured_time": True,
                "requires_post_model_execution": True, "compiler_does_not_authorize_answer_replacement": True}

    def abstain(reason: str):
        evidence["reason"] = reason
        return None, evidence

    question = str(question)
    spec = _spec(question)
    if spec is None or spec.get("datatype") != "type.datetime" or spec.get("type") != "EQUAL":
        return abstain("question_not_one_unquoted_explicit_on_full_calendar_day")
    value = str(spec["value"])
    normalization = spec.get("normalization", {})
    span = normalization.get("_source_literal_span")
    literal = str(normalization.get("_source_literal", ""))
    # _spec's ISO fallback does not retain spans; recover the exact literal,
    # never infer an unexpressed date from an operator or decomposition.
    if not isinstance(span, list) or len(span) != 2:
        literal = value
        matches = list(re.finditer(r"\bon\s+(" + re.escape(literal) + r")(?!\w)", question, re.I))
        if len(matches) != 1:
            return abstain("normalized_question_literal_span_not_unique")
        span = [matches[0].start(1), matches[0].end(1)]
    if _normalized_dates(question, known_literal=literal, known_value=value) != [value]:
        return abstain("question_has_another_invalid_or_conflicting_calendar_date")

    triples = getattr(source_graph, "triples", None)
    operators = getattr(source_graph, "operators", None)
    provenance = getattr(source_graph, "provenance", None)
    answer = str(getattr(source_graph, "answer_var", ""))
    if (ontology is None or not _VAR.fullmatch(answer) or not isinstance(triples, list)
            or not triples or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)
            or not isinstance(operators, list) or any(not isinstance(op, Mapping) for op in operators)
            or not isinstance(provenance, Mapping)):
        return abstain("source_graph_contract_or_ontology_missing")
    if any(provenance.get(k) for k in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("postprocessed_incumbent_is_not_one_preserved_query")
    if any(str(op.get("type", "")).upper() not in _FILTER_TYPES for op in operators):
        return abstain("non_filter_operator_cannot_prove_denotation_refinement")

    fields = [tuple(map(str, t)) for t in triples if str(t[0]) == answer
              and str(ontology.range_for_relation(str(t[1]))) == "type.datetime"]
    if len(fields) != 1 or not _VAR.fullmatch(fields[0][2]):
        return abstain("datetime_value_not_uniquely_bound_directly_on_entity_answer")
    owner, attribute, value_var = fields[0]
    if any(str(s) == value_var or (str(o) == value_var and tuple(map(str, (s, p, o))) != fields[0]) for s, p, o in triples):
        return abstain("bound_datetime_is_reused_as_entity_or_another_field")
    primary = [tuple(map(str, t)) for t in triples if str(t[2]) == owner and _MID.fullmatch(str(t[0]))]
    if len(primary) != 1:
        return abstain("entity_answer_primary_anchor_role_not_unique")
    anchor_id, primary_relation, _ = primary[0]
    role_type = str(ontology.range_for_relation(primary_relation) or "")
    domain = str(ontology.domain_for_relation(attribute) or "")
    role_closure = set(ontology.supertypes(role_type))
    domain_closure = set(ontology.supertypes(domain))
    # A specialized owner (e.g. an event-participating person) is allowed only
    # by explicit ontology inheritance, never by ordinary-role similarity.
    if (not role_type or not domain or role_type.startswith("type.") or domain.startswith("type.")
            or "common.topic" not in role_closure or "common.topic" not in domain_closure
            or not (domain in role_closure or role_type in domain_closure)):
        return abstain("primary_entity_role_and_datetime_owner_not_proven_compatible")
    role_words, field_words = _leaf_tokens(primary_relation), _leaf_tokens(attribute)
    if not role_words or not field_words:
        return abstain("primary_role_or_datetime_field_has_no_explicit_lexical_content")

    prefix = question[:span[0]]
    clause = re.search(r"\b(?P<role>[A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:who|which|that)\s+"
                       r"(?P<event>[A-Za-z]+(?:\s+[A-Za-z]+){0,4})\s+on\s*$", prefix, re.I)
    if clause is None or not role_words <= _tokens(clause["role"]):
        return abstain("question_does_not_attach_date_event_to_the_primary_answer_role")
    event_words = _tokens(clause["event"]) - _AUXILIARY
    if event_words != field_words:
        return abstain("question_event_verb_does_not_uniquely_name_existing_datetime_field")
    bindings = provenance.get("anchor_bindings")
    if not isinstance(bindings, Mapping):
        return abstain("actual_topic_bindings_missing")
    anchors = [v for v in bindings.values() if isinstance(v, Mapping) and str(v.get("id")) == anchor_id]
    if len(anchors) != 1 or not str(anchors[0].get("label", "")) or not _contains(question, _words(anchors[0]["label"])):
        return abstain("primary_actual_topic_name_not_unique_in_question")

    decomposition = provenance.get("decomposition")
    attached = []
    if isinstance(decomposition, list):
        for text in decomposition:
            if not isinstance(text, str) or _normalized_dates(text, known_literal=literal, known_value=value) != [value]:
                continue
            words = _tokens(text)
            if (role_words <= words and field_words <= words
                    and re.search(r"\b(?:is|equals?|equal\s+to)\s+(?:" + re.escape(literal) + "|" + re.escape(value) + r")(?!\w)", text, re.I)):
                attached.append(text)
    if not attached:
        return abstain("decomposition_does_not_independently_attest_same_role_field_day_equality")

    for op in operators:
        kind = str(op.get("type", "")).upper()
        if kind == "NO_EQUAL":
            continue
        input_var = str(op.get("input_var", ""))
        op_attribute = str(op.get("attribute_relation_id") or relation_id_from_label(op.get("attribute_relation_label")))
        if input_var == value_var or (input_var == owner and (op_attribute == attribute or (kind == "TC" and not op_attribute))):
            return abstain("same_bound_datetime_already_has_equality_interval_or_comparison")
        op_value = str(op.get("value", ""))
        op_dates = _normalized_dates(op_value)
        date_typed = (str(op.get("value_type", "")).casefold() in {"date", "datetime", "year"}
                      or str(ontology.range_for_relation(op_attribute)) == "type.datetime")
        if (op_dates and op_dates != [value]) or (date_typed and op_value != value):
            return abstain("another_operator_has_conflicting_or_unproved_date_literal")
    result = deepcopy(source_graph)
    result.graph_id, result.sparql = graph_id, ""
    new_operator = {"type": "EQUAL", "inputs": [], "input_var": value_var,
                    "attribute_relation_id": "", "attribute_relation_label": [], "attribute_relation_labels": [],
                    "value": value, "value_type": "dateTime", "_attribute_value_already_bound": True,
                    "_comparison_date_precision": "day", "_calendar_day_boundary": True}
    result.operators.append(new_operator)
    evidence.update(status="compiled_not_executed", reason="explicit_question_and_decomposition_day_missing_on_existing_bound_field",
                    source_graph_id=source_graph.graph_id, compiled_graph_id=graph_id, owner_var=owner,
                    value_var=value_var, datetime_relation=attribute, primary_relation=primary_relation,
                    primary_entity_type=role_type, datetime_owner_type=domain,
                    primary_anchor_id=anchor_id, primary_anchor_name=anchors[0]["label"], normalized_day=value,
                    question_literal_normalization=deepcopy(normalization), question_attachment=clause[0],
                    independent_decomposition_matches=attached, appended_operator=deepcopy(new_operator),
                    preserves_exact_source_triples=True, preserves_all_source_operators=True,
                    answer_var_unchanged=True, added_relations=0, appended_operator_count=1,
                    does_not_correct_unrelated_joins_or_operators=True,
                    subset_is_not_nonregression_or_f1_improvement_proof=True)
    result.provenance["bound_calendar_date_append_shadow"] = deepcopy(evidence)
    return result, evidence
