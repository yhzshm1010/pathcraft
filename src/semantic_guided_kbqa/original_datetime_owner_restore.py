"""Isolated, fail-closed restoration of one originally grounded date owner.

Only a two-edge incumbent that repeats the original primary predicate and
projects its second entity can qualify. The complete original two-step path,
actual complete grounding, topic identity, question grammar and decomposition
must independently attest the primary entity and its existing datetime field.
No models, queries, file/Gold access, answer selection or production hook.
"""
from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
from typing import Any

from .bound_calendar_date_append import _tokens
from .budgeted_literal_constraint_repair import _spec
from .contracts import QueryGraphCandidate
from .lowering import lower_sparql
from .ontology import relation_id_from_label


_VAR = re.compile(r"V\d+\Z")
_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_XSD = "http://www.w3.org/2001/XMLSchema#"
_TEMPORAL_DATATYPES = frozenset({_XSD + "gYear", _XSD + "date", _XSD + "dateTime"})
_YEAR_LITERAL = re.compile(r"(?P<year>[1-9]\d{0,3})\s*(?P<era>BCE|BC|CE|AD)?\Z", re.I)
_YEAR_SHAPE = r"^[+-]?[0-9]+(?:Z|[+-][0-9]{2}:[0-9]{2}|-[0-9]{2}-[0-9]{2}(?:[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:[.][0-9]+)?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?)?$"


def _words(value: Any) -> set[str]:
    return {"child" if word == "children" else word for word in _tokens(value)}


def _leaf(predicate: str) -> set[str]:
    return _words(str(predicate).rsplit(".", 1)[-1]) - {"date", "time", "of"}


def _proof(reason: str) -> dict[str, Any]:
    return {"status": "not_compiled", "reason": reason, "pure_compiler_only": True,
            "model_calls": 0, "endpoint_queries": 0, "embedding_calls": 0,
            "uses_gold_answers": False, "uses_entity_question_or_relation_whitelist": False,
            "compiler_selects_answers": False, "compiler_does_not_authorize_answer_replacement": True,
            "requires_post_model_execution_and_saved_request_time_budget": True,
            "new_denotation_verified": False}


def _literal(value: str) -> dict[str, Any] | None:
    match = _YEAR_LITERAL.fullmatch(value.strip())
    if match and (match["era"] or len(match["year"]) == 4):
        magnitude = int(match["year"])
        era = str(match["era"] or "CE").upper()
        return {"precision": "year", "year": -magnitude if era in {"BC", "BCE"} else magnitude,
                "source_literal": value.strip(), "explicit_era": era,
                "year_zero_adjustment": 0, "normalization_contract": "historical_era_signed_same_magnitude_no_offset"}
    parsed = _spec("An event happened on " + value.strip() + "?")
    if parsed and parsed.get("datatype") == "type.datetime":
        return {"precision": "day", "value": str(parsed["value"]), "source_literal": value.strip(),
                "normalization": deepcopy(parsed["normalization"])}
    return None


def datetime_question_obligation(question: str) -> dict[str, Any] | None:
    """One equality date after 'on'; no hidden extra literal or comparator."""
    text = str(question)
    if re.search(r'["“”]|\b(?:not|never|or|either|except|before|after|prior|later|earlier|less|more|greater|lower|higher|during|until|since|between|about|approximately|most|least|first|last|latest|earliest)\b', text, re.I):
        return None
    cues = list(re.finditer(r"\bon\s+", text, re.I))
    if len(cues) != 1:
        return None
    cue = cues[0]
    tail = text[cue.end():]
    end = re.search(r"\s+and\b|[?;!]", tail, re.I)
    literal = tail[:end.start()] if end else tail
    spec = _literal(literal.strip().rstrip("."))
    if spec is None:
        return None
    spec["literal_span"] = [cue.end(), cue.end() + len(literal.rstrip())]
    outside = text[:spec["literal_span"][0]] + text[spec["literal_span"][1]:]
    if re.search(r"\d", outside):
        return None
    return spec


def datetime_owner_restore_necessary_gate(question: str, current: Any) -> dict[str, Any]:
    """Current-graph-only necessary gates; no original-context substitution."""
    spec = datetime_question_obligation(question)
    if spec is None:
        return _proof("question_not_one_explicit_event_day_or_era_year_equality")
    triples = getattr(current, "triples", None)
    if (not isinstance(triples, list) or len(triples) != 2 or any(not isinstance(t, (list, tuple)) or len(t) != 3 for t in triples)
            or getattr(current, "operators", None) != []):
        return _proof("current_not_one_unconstrained_two_edge_path")
    first, last = [tuple(map(str, t)) for t in triples]
    if (_MID.fullmatch(first[0]) and _VAR.fullmatch(first[2]) and last == (first[2], first[1], str(current.answer_var))
            and _VAR.fullmatch(last[2]) and first[2] != last[2]):
        direction, anchor, owner = "forward", first[0], first[2]
    elif (_MID.fullmatch(first[2]) and _VAR.fullmatch(first[0]) and last == (str(current.answer_var), first[1], first[0])
            and _VAR.fullmatch(last[0]) and first[0] != last[0]):
        direction, anchor, owner = "backward", first[2], first[0]
    else:
        return _proof("current_second_edge_not_exact_repeated_primary_projection_drift")
    return {**_proof("necessary_current_graph_shape_only_original_context_still_required"),
            "necessary_gate": True, "obligation": spec, "primary_relation": first[1],
            "primary_direction": direction, "anchor_id": anchor, "original_owner_var": owner,
            "wrong_entity_projection": str(current.answer_var)}


def _grounding_tuple(grounding: Any):
    if isinstance(grounding, Mapping):
        return grounding.get("compose_input"), grounding.get("relation_bindings"), grounding.get("anchor_bindings"), grounding.get("provenance", {})
    return (getattr(grounding, "compose_input", None), getattr(grounding, "relation_bindings", None),
            getattr(grounding, "anchor_bindings", None), getattr(grounding, "provenance", {}))


def _entity(value: Any) -> tuple[str, str]:
    if isinstance(value, Mapping):
        return str(value.get("id", value.get("entity_id", ""))), str(value.get("label", value.get("name", "")))
    if isinstance(value, str):
        return value, ""
    return str(getattr(value, "entity_id", "")), str(getattr(value, "label", ""))


def compile_original_datetime_owner_restore(
    question: str, current: Any, original_semantic: Mapping[str, Any], actual_groundings: Sequence[Any],
    topic_bindings: Mapping[str, Any], ontology: Any, *, graph_id: str = "original_datetime_owner_shadow",
    linked_entity_context: Sequence[Mapping[str, Any]] | None = None,
    datetime_schema: Mapping[str, Any] | None = None,
    verify_temporal_datatype_in_query: bool = False,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Restore only a proved owner drift; signed years require real datatype."""
    proof = datetime_owner_restore_necessary_gate(question, current)
    if not proof.get("necessary_gate"):
        return None, proof

    def abstain(reason: str):
        proof.update(status="not_compiled", reason=reason)
        return None, proof

    provenance = getattr(current, "provenance", None)
    if not isinstance(provenance, Mapping) or any(provenance.get(k) for k in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("current_not_one_question_owned_query")
    if provenance.get("original_question", question) != question:
        return abstain("current_question_contract_conflict")
    paths = original_semantic.get("semantic_paths", [])
    if ontology is None or len(paths) != 1 or not isinstance(paths[0], Mapping):
        return abstain("original_not_one_complete_path")
    path = paths[0]
    steps = path.get("steps", [])
    if len(steps) != 2 or any(not isinstance(s, Mapping) for s in steps):
        return abstain("original_not_exact_primary_plus_datetime_path")
    first, last = steps
    root = str(path.get("anchor_ref", ""))
    if (first.get("from") != root or first.get("to") != last.get("from") or last.get("to") != path.get("path_output_var")
            or last.get("direction") != "forward" or first.get("direction") != proof["primary_direction"]
            or relation_id_from_label(first.get("relation_label")) != proof["primary_relation"]
            or first.get("id") == last.get("id") or root not in topic_bindings):
        return abstain("original_primary_variable_id_or_direction_does_not_match_current")
    anchor, label = _entity(topic_bindings[root])
    originals = [a for a in original_semantic.get("anchors", []) if isinstance(a, Mapping) and str(a.get("id")) == root]
    actuals = provenance.get("anchor_bindings", {})
    actual_id, actual_label = _entity(actuals.get(root)) if isinstance(actuals, Mapping) else ("", "")
    if (anchor != proof["anchor_id"] or actual_id != anchor or not label or label.casefold() != actual_label.casefold()
            or len(originals) != 1 or str(originals[0].get("surface", "")).casefold() != label.casefold()
            or not re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", question, re.I)):
        return abstain("original_and_actual_topic_id_label_question_identity_not_proven")
    literal = proof["obligation"]["source_literal"]
    for item in original_semantic.get("anchors", []):
        if not isinstance(item, Mapping) or item.get("id") == root:
            continue
        if str(item.get("surface", "")).strip().casefold() != literal.casefold() or item.get("id") in topic_bindings:
            return abstain("extra_original_topic_or_unknown_kind_is_not_a_proven_scalar_literal")
    for item in linked_entity_context or []:
        name = str(item.get("name", item.get("label", "")))
        if (name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, re.I)
                and str(item.get("id", "")) != anchor):
            return abstain("another_explicit_linked_question_topic_not_in_original_path")

    original_field = relation_id_from_label(last.get("relation_label"))
    fields = set()
    witnesses = []
    for grounding in actual_groundings:
        composed, relations, bindings, meta = _grounding_tuple(grounding)
        if not isinstance(composed, Mapping) or not isinstance(relations, Mapping) or not isinstance(bindings, Mapping) or not isinstance(meta, Mapping) or meta.get("dropped_paths"):
            continue
        actual_paths = composed.get("semantic_paths", [])
        if len(actual_paths) != 1 or not isinstance(actual_paths[0], Mapping):
            continue
        actual_path = actual_paths[0]
        actual_steps = actual_path.get("steps", [])
        if (len(actual_steps) != 2 or any(not isinstance(s, Mapping) for s in actual_steps)
                or any(actual_path.get(k) != path.get(k) for k in ("id", "anchor_ref", "path_output_var", "goal"))
                or any(any(a.get(k) != b.get(k) for k in ("id", "from", "to", "direction")) for a, b in zip(actual_steps, steps))):
            continue
        bound_anchor, _ = _entity(bindings.get(root))
        primary = str(relations.get(str(first["id"]), ""))
        field = str(relations.get(str(last["id"]), ""))
        if (bound_anchor != anchor or primary != proof["primary_relation"]
                or relation_id_from_label(actual_steps[0].get("relation_label")) != primary
                or relation_id_from_label(actual_steps[1].get("relation_label")) != field
                or field.split(".", 1)[0] != original_field.split(".", 1)[0]
                or field.rsplit(".", 1)[-1] != original_field.rsplit(".", 1)[-1]
                or str(ontology.range_for_relation(field)) != "type.datetime"):
            continue
        if ontology.domain_for_relation(original_field) and field != original_field:
            continue
        fields.add(field)
        witnesses.append({"actual_relation_bindings": dict(relations), "complete_path_id": path["id"],
                          "actual_grounding_provenance": deepcopy(dict(meta)),
                          "original_and_grounded_step_ids_variables_directions_identical": True})
    if len(fields) != 1:
        return abstain("actual_complete_grounding_does_not_uniquely_prove_original_datetime_rid")
    field = next(iter(fields))
    primary = proof["primary_relation"]
    role_type = str(ontology.range_for_relation(primary) if first["direction"] == "forward" else ontology.domain_for_relation(primary))
    domain = str(ontology.domain_for_relation(field))
    if (not role_type or not domain or "common.topic" not in set(ontology.supertypes(role_type))
            or "common.topic" not in set(ontology.supertypes(domain))
            or not (role_type in set(ontology.supertypes(domain)) or domain in set(ontology.supertypes(role_type)))):
        return abstain("actual_primary_entity_and_datetime_owner_not_ontology_compatible")
    role, event = _leaf(primary), _leaf(field)
    before = question[:proof["obligation"]["literal_span"][0]]
    event_match = re.search(r"\b([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+on\s*$", before, re.I)
    if event_match is None or not event <= _words(event_match[1]):
        return abstain("question_event_does_not_match_the_actual_datetime_field")
    event_prefix = re.sub(r"\s+on\s*$", "", before, flags=re.I)
    event_tokens = list(re.finditer(r"[A-Za-z]+", event_prefix))
    event_start = None
    for size in range(1, min(4, len(event_tokens)) + 1):
        suffix = event_prefix[event_tokens[-size].start():]
        if _words(suffix) - {"is", "was", "were", "are", "has", "have", "had", "been"} == event:
            event_start = event_tokens[-size].start()
            break
    head = event_prefix[:event_start].strip() if event_start is not None else ""
    head = re.sub(r"\s+(?:is|was|were|are|has|have|had|been)\s*$", "", head, flags=re.I)
    if not re.fullmatch(r"(?:who|(?:what|which)\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,3})", head, re.I):
        return abstain("question_does_not_request_an_entity_event_owner")
    after = question[proof["obligation"]["literal_span"][1]:].strip().rstrip("? .!")
    tail = re.fullmatch(r"and\s+(?:was|is)\s+(?:a|an|the)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+of\s+(.+)", after, re.I)
    if tail is None or _words(tail[1]) != role:
        return abstain("question_has_extra_scope_or_primary_role_attachment_not_exact")
    topic_text = tail[2].strip()
    # Titles are not alternate topic identities; require the linked name as
    # the complete trailing identity and allow at most one preceding word.
    if not re.fullmatch(r"(?:[A-Za-z]+\s+)?" + re.escape(label), topic_text, re.I):
        return abstain("question_topic_tail_has_another_unaccounted_entity_or_constraint")
    goal = str(path.get("goal", ""))
    decompositions = provenance.get("decomposition", [])
    independent = [text for text in decompositions if isinstance(text, str) and text.strip() == goal.strip()]
    goal_values = re.findall(r"\b(?:is|equals?|equal\s+to)\s+(.+?)(?=[?;!.]|$)", goal, re.I)
    goal_specs = [_literal(text.strip()) for text in goal_values]
    matching = [s for s in goal_specs if s and s.get("precision") == proof["obligation"]["precision"]
                and s.get("year", s.get("value")) == proof["obligation"].get("year", proof["obligation"].get("value"))]
    if not independent or len(matching) != 1 or not role <= _words(goal) or not event <= _words(goal):
        return abstain("original_goal_and_actual_decomposition_do_not_attest_same_role_field_equal_literal")
    proof.update(original_path_id=path["id"], actual_grounding_witnesses=witnesses,
                 datetime_relation_id=field, primary_entity_type=role_type, datetime_owner_type=domain,
                 original_datetime_label=deepcopy(last["relation_label"]), actual_topic_id=anchor, actual_topic_label=label,
                 original_owner_var=proof["original_owner_var"], original_question_and_decomposition_match=True,
                 wh_entity_subject_supported_by_explicit_primary_relationship_conjunct=True,
                 no_wh_kind_synonym_or_entity_concept_routing=True,
                 contradiction="repeated_primary_entity_hop_replaced_original_bound_datetime_and_owner_projection")

    spec = proof["obligation"]
    if spec["precision"] == "year" and spec["year"] < 0:
        if datetime_schema is None and verify_temporal_datatype_in_query is True:
            proof["datatype_validation_deferred_to_same_query"] = True
            proof["requires_nonempty_typed_literal_validated_query_result"] = True
        elif (not isinstance(datetime_schema, Mapping) or datetime_schema.get("error")
              or datetime_schema.get("scalar_relation_id") != field or datetime_schema.get("primary_relation_id") != primary
              or datetime_schema.get("root_anchor_id") != anchor):
            return abstain("bce_year_requires_actual_same_path_datetime_binding_schema")
        else:
            raw = datetime_schema.get("raw_sparql_json", {})
            bindings = raw.get("results", {}).get("bindings", []) if isinstance(raw, Mapping) else []
            if len(bindings) != 1 or not isinstance(bindings[0], Mapping):
                return abstain("bce_datetime_binding_not_unique_complete_observed_literal")
            term = bindings[0].get("V1", {})
            value = str(term.get("value", "")) if isinstance(term, Mapping) else ""
            datatype = str(term.get("datatype", "")) if isinstance(term, Mapping) else ""
            shape = r"^[+-]?[0-9]+(?:Z|[+-][0-9]{2}:[0-9]{2})?$" if datatype == _XSD + "gYear" else _YEAR_SHAPE
            if (term.get("type") not in {"literal", "typed-literal"} or datatype not in _TEMPORAL_DATATYPES
                    or term.get("xml:lang") or not re.fullmatch(shape, value)):
                return abstain("bce_observed_term_not_proven_temporal_typed_signed_year_shape")
            observed_year = int(re.match(r"^[+-]?\d+", value)[0])
            if observed_year != spec["year"]:
                return abstain("bce_observed_signed_year_does_not_match_question_without_offset")
            proof["actual_datetime_binding_schema"] = {"datatype": datatype, "value": value,
                                                        "observed_signed_year": observed_year,
                                                        "no_padding_or_year_zero_offset_assumed": True}
    owner, scalar = proof["original_owner_var"], str(current.answer_var)
    first_triple = list(map(str, current.triples[0]))
    op = {"type": "EQUAL", "inputs": [], "input_var": scalar, "attribute_relation_id": "",
          "attribute_relation_label": [], "attribute_relation_labels": [], "_attribute_value_already_bound": True,
          "value_type": "dateTime", "_comparison_date_precision": spec["precision"],
          "_verify_temporal_datatype_in_query": True}
    if spec["precision"] == "year":
        op.update(value=str(spec["year"]), _original_datetime_signed_year_equality=True)
    else:
        op.update(value=spec["value"], _calendar_day_boundary=True)
    candidate = QueryGraphCandidate(graph_id, [first_triple, [owner, field, scalar]], owner, [op],
                                    float(getattr(current, "score", 0.0)), None, deepcopy(dict(provenance)))
    # The restored query must not claim the wrong repeated-primary grounding
    # as its own. Preserve that original state in the repair proof and attach
    # the actual complete datetime grounding that authorized the new query.
    proof["original_source_relation_bindings"] = deepcopy(provenance.get("relation_bindings", {}))
    proof["original_source_grounded_semantic"] = deepcopy(provenance.get("grounded_semantic", {}))
    candidate.provenance["relation_bindings"] = deepcopy(witnesses[0]["actual_relation_bindings"])
    candidate.provenance["grounded_semantic"] = deepcopy(witnesses[0]["actual_grounding_provenance"])
    proof.update(status="compiled_not_executed", reason="complete_original_grounded_datetime_owner_restored",
                 source_graph_id=current.graph_id, compiled_graph_id=graph_id, restored_answer_var=owner,
                 removes_only_proven_wrong_repeated_primary_edge=True, preserves_original_complete_path=True,
                 preserved_existing_operators=0, graph_role_change_is_not_subset_evidence=True)
    candidate.provenance["original_datetime_owner_restore"] = deepcopy(proof)
    return candidate, proof


def lower_original_datetime_owner_restore(candidate: QueryGraphCandidate, *, limit: int = 256) -> str:
    """Lower just the isolated signed-year equality, never alter global code."""
    signed = [op for op in candidate.operators if op.get("_original_datetime_signed_year_equality")]
    if not signed:
        query = lower_sparql(candidate, limit=limit)
        exact = [op for op in candidate.operators if op.get("_verify_temporal_datatype_in_query")]
        if len(exact) != 1:
            raise ValueError("Isolated calendar day requires one explicit datatype-verified operator")
        variable = str(exact[0].get("input_var", ""))
        if not _VAR.fullmatch(variable):
            raise ValueError("Calendar day operand invalid")
        guard = f'  FILTER(isLiteral(?{variable}) && LANG(?{variable}) = "")\n'
        guard += f'  FILTER(DATATYPE(?{variable}) IN (<{_XSD}date>, <{_XSD}dateTime>))\n'
        return query.replace("\n}\n", "\n" + guard + "}\n", 1)
    if len(candidate.operators) != 1 or len(signed) != 1:
        raise ValueError("Isolated signed-year lowering requires one and only one operator")
    op = signed[0]
    value = str(op.get("value", ""))
    variable = str(op.get("input_var", ""))
    if not re.fullmatch(r"-?[1-9]\d{0,3}", value) or not _VAR.fullmatch(variable) or op.get("type") != "EQUAL":
        raise ValueError("Signed-year equality contract invalid")
    base = deepcopy(candidate)
    base.operators = []
    query = lower_sparql(base, limit=limit)
    filter_text = (f'  FILTER(isLiteral(?{variable}) && LANG(?{variable}) = "")\n'
                   f'  FILTER(DATATYPE(?{variable}) IN (<{_XSD}gYear>, <{_XSD}date>, <{_XSD}dateTime>))\n'
                   f'  FILTER(REGEX(STR(?{variable}), "{_YEAR_SHAPE}"))\n'
                   f'  FILTER(<{_XSD}integer>(REPLACE(STR(?{variable}), "^([+-]?[0-9]+).*$", "$1")) = {value})\n')
    marker = "\n}\n"
    if marker not in query:
        raise ValueError("Simple source lowering does not contain one outer WHERE terminator")
    return query.replace(marker, "\n" + filter_text + "}\n", 1)


def compile_datetime_owner_restore_from_current_stages(
    question: str, current: Any, stages: Sequence[Mapping[str, Any]], ontology: Any, *,
    graph_id: str = "original_datetime_owner_shadow", datetime_schema: Mapping[str, Any] | None = None,
    verify_temporal_datatype_in_query: bool = False,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Resolve only the actual originating ordered Semantic/Grounding batch.

    ``stages`` is in-memory current-flow stage data supplied by the caller,
    not file/trace access in this module. A source grounding with the current
    semantic index/rank, relation IDs and topic IDs must actually exist in
    that ordered batch before another grounding in the batch can be used.
    """
    meta = getattr(current, "provenance", {}).get("grounded_semantic", {})
    if (not isinstance(meta, Mapping) or not isinstance(meta.get("semantic_graph_index"), int)
            or isinstance(meta.get("semantic_graph_index"), bool)):
        return None, _proof("current_originating_semantic_index_missing")
    index = meta["semantic_graph_index"]
    current_relations = getattr(current, "provenance", {}).get("relation_bindings", {})
    topics = getattr(current, "provenance", {}).get("anchor_bindings", {})
    if (not isinstance(topics, Mapping) or not topics
            or not isinstance(current_relations, Mapping) or not current_relations):
        return None, _proof("current_actual_topic_or_relation_bindings_missing")
    pending, context = None, []
    candidates = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        if stage.get("stage") == "linked_entity_context" and isinstance(stage.get("output"), list):
            context = stage["output"]
        elif stage.get("stage") == "semantic_path_generation":
            output = stage.get("output", [])
            pending = [s["output"] for s in output if isinstance(s, Mapping) and s.get("valid") is True and isinstance(s.get("output"), Mapping)] if isinstance(output, list) else None
        elif stage.get("stage") == "grounded_semantic_candidates" and pending is not None:
            output = stage.get("output", [])
            if not isinstance(output, list) or not 0 <= index < len(pending):
                continue
            actual = [g for g in output if isinstance(g, Mapping) and g.get("provenance", {}).get("semantic_graph_index") == index]
            origins = [g for g in actual if g.get("relation_bindings") == current_relations
                       and g.get("provenance", {}).get("rank") == meta.get("rank")
                       and isinstance(g.get("anchor_bindings"), Mapping)
                       and set(g["anchor_bindings"]) == set(topics)
                       and all(_entity(v)[0] and _entity(topics[k])[0] == _entity(v)[0]
                               for k, v in g["anchor_bindings"].items())]
            if len(origins) != 1:
                continue
            candidate, proof = compile_original_datetime_owner_restore(question, current, pending[index], actual, topics,
                                                                        ontology, graph_id=graph_id,
                                                                        linked_entity_context=context,
                                                                        datetime_schema=datetime_schema,
                                                                        verify_temporal_datatype_in_query=verify_temporal_datatype_in_query)
            candidates.append((candidate, proof))
    positives = [(c, p) for c, p in candidates if c is not None]
    if not positives:
        return candidates[0] if len(candidates) == 1 else (None, _proof("actual_originating_semantic_grounding_batch_not_uniquely_proven"))
    keys = {(tuple(tuple(t) for t in c.triples), c.answer_var, str(c.operators)) for c, _ in positives}
    if len(keys) != 1 or len(candidates) != 1:
        return None, _proof("multiple_current_originating_batches_or_restoration_programs_are_ambiguous")
    candidate, proof = positives[0]
    proof["actual_ordered_originating_batch_and_source_grounding_verified"] = True
    candidate.provenance["original_datetime_owner_restore"] = deepcopy(proof)
    return candidate, proof
