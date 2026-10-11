"""Restore a *recorded* temporal extrema suffix without inventing a path.

This is a pure compiler: no model, embedding, endpoint, training example,
evaluation field or old answer is read.  Callers must run the one proposal
only within saved per-question endpoint/time credits, and accept it only as
a nonempty strict subset of a complete current source execution.

The source is the current executed BGP, not a prior successful answer.  Every
originating Semantic step must be schema-certified and survive in the actual
grounding and rejected Compose output.  The only permitted extension is an
already-produced forward record/date suffix attached to the same answer
entity.  ``latest`` alone does not license guessing either ``from`` or ``to``.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
import re
from typing import Any

from .budgeted_schema_repair import (
    SCALAR_TYPES, _compile_grounded, _scalar_property_surface_supported,
    _unique_embedding, _regular_role_stems, schema_type_overlap,
)
from .compose_scalar_scope_safety import original_semantic_scope_certificate
from .contracts import EntityCandidate, GroundedSemanticCandidate, QueryGraphCandidate
from .ontology import relation_id_from_label

_VAR = re.compile(r"V\d+\Z")
_MAX = re.compile(r"\b(?:latest|last|most recent|newest)\b", re.I)
_MIN = re.compile(r"\b(?:earliest|first|oldest)\b", re.I)
_START = re.compile(r"\b(?:from|start(?:ed|ing|s)?|began|begin(?:s|ning)?|since)\b", re.I)
_END = re.compile(r"\b(?:to|until|end(?:ed|ing|s)?|finish(?:ed|ing|es)?)\b", re.I)
_START_LEAVES = frozenset({"from", "from_date", "start", "start_date", "begin", "begin_date"})
_END_LEAVES = frozenset({"to", "to_date", "end", "end_date", "finish", "finish_date"})


def _evidence(reason: str) -> dict[str, Any]:
    return {"status": "abstained", "reason": reason,
            "additional_model_calls": 0, "additional_embedding_calls": 0,
            "additional_endpoint_queries": 0, "uses_evaluation_gold": False,
            "uses_whitelist": False}


def _temporal_extrema(question: str) -> tuple[str, str] | None:
    text = re.sub(r'["“][^"”]*["”]', " ", question)
    text = re.sub(r"\b(?:first|last)\s+name\b", " ", text, flags=re.I)
    maximum, minimum = bool(_MAX.search(text)), bool(_MIN.search(text))
    start, end = bool(_START.search(text)), bool(_END.search(text))
    temporal_extreme = r"(?:latest|last|most\s+recent|newest|earliest|first|oldest)"
    start_phase = r"(?:from|start(?:ed|ing|s)?|began|begin(?:s|ning)?|since)"
    end_phase = r"(?:to|until|end(?:ed|ing|s)?|finish(?:ed|ing|es)?)"
    phase = start_phase if start else end_phase
    # "from City" is a topic preposition, not a start-date obligation.
    phase_joint = bool(re.search(rf"\b(?:{phase}\s+(?:the\s+)?{temporal_extreme}|"
                               rf"{temporal_extreme}\s+(?:date\s+)?{phase})\b", text, re.I))
    # A separately dated/conjoined/counting obligation has no closed proof
    # in this lane.  It must remain unchanged, rather than be silently lost.
    if (maximum == minimum or start == end or not phase_joint
            or re.search(r"\d|\b(?:and|or|not|before|after|during|between|count|how many)\b", text, re.I)):
        return None
    return ("ARGMAX" if maximum else "ARGMIN", "start" if start else "end")


def _closed_question_scope(question, grounded, triples):
    """Reject a named/relational question clause absent from this full scope.

    The small fixed set is grammar/operator vocabulary, not entity or
    predicate routing.  Content vocabulary is derived exclusively from the
    actual topic surfaces and the predicates already present in Compose.
    """
    text = question
    used = {p.get("anchor_ref") for p in grounded.compose_input.get("semantic_paths", [])}
    for anchor in grounded.compose_input.get("anchors", []):
        if anchor.get("id") not in used:
            continue
        pattern = re.escape(str(anchor.get("surface", "")))
        if not pattern or not re.search(pattern, text, re.I):
            return False
        text = re.sub(pattern, " ", text, flags=re.I)
    grammar = {
        "a", "an", "the", "his", "her", "their", "its", "who", "whose", "which", "what",
        "is", "are", "was", "were", "has", "have", "had", "do", "does", "did", "of", "for",
        "by", "with", "from", "to", "in", "at", "among", "that", "holding", "held", "hold",
        "holds", "return", "date", "time", "latest", "earliest", "last", "first", "most",
        "recent", "newest", "oldest", "start", "started", "begin", "began", "end", "ended",
        "until", "since",
    }
    schema_words = re.findall(r"[A-Za-z]+", " ".join(p for _, p, _ in triples).replace("_", " "))
    schema_roots = set().union(*(_regular_role_stems(w) for w in schema_words))
    for word in re.findall(r"[A-Za-z]+", text.casefold()):
        if word in grammar:
            continue
        forms = set(_regular_role_stems(word))
        if len(word) > 6 and word.endswith("al"):
            forms.update(_regular_role_stems(word[:-2]))
        if not forms & schema_roots:
            return False
    return True


def compile_saved_extrema_extension(
    question: str, source_graph: QueryGraphCandidate,
    grounded: GroundedSemanticCandidate, compose_record: Mapping[str, Any],
    source_semantic: Mapping[str, Any], ontology: Any, *, graph_id: str,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Compile one actual failed Compose's date-ordering suffix, or abstain.

    ``source_semantic`` must be the authentic originating Semantic output
    matched to this grounding, not an arbitrarily chosen alternate review.
    All source operators are preserved.  Any existing substantive constraint
    protects the incumbent; no answer replacement is licensed by this helper.
    """
    intent = _temporal_extrema(str(question))
    if intent is None:
        return None, _evidence("temporal_extrema_or_phase_not_unique_and_explicit")
    if (ontology is None or grounded.compose_input.get("question") != question
            or source_graph.provenance.get("original_question") != question
            or grounded.provenance.get("dropped_paths")
            or compose_record.get("valid") is not False
            or not str(compose_record.get("error", "")).startswith("unable to transfer grounded relation for ")
            or compose_record.get("input") != grounded.compose_input):
        return None, _evidence("missing_same_flow_failed_compose_or_question_ownership")
    if (not source_graph.triples or len(source_graph.triples) > 12
            or any(not isinstance(op, dict) or str(op.get("type", "")).upper() != "NO_EQUAL"
                   for op in source_graph.operators)):
        return None, _evidence("preserve_existing_substantive_incumbent_constraint")
    scope_ok, scope = original_semantic_scope_certificate(grounded, source_semantic, ontology=ontology)
    if not scope_ok:
        return None, {**_evidence("original_semantic_scope_not_preserved"), "scope_evidence": scope}
    core, names = _compile_grounded(grounded)
    if not core or len(core) != len(source_graph.triples):
        return None, _evidence("source_is_not_complete_current_grounded_core")
    output = compose_record.get("output")
    if not isinstance(output, Mapping):
        return None, _evidence("actual_compose_output_missing")
    bindings: dict[str, str] = {}
    by_surface: dict[str, str] = {}
    for anchor in grounded.compose_input.get("anchors", []):
        bound = grounded.anchor_bindings.get(str(anchor.get("id", "")))
        if bound is not None:
            for label in (anchor.get("surface", ""), bound.label):
                key = str(label).strip().casefold()
                if key in by_surface and by_surface[key] != bound.entity_id:
                    return None, _evidence("ambiguous_actual_topic_binding")
                if key:
                    by_surface[key] = bound.entity_id
    entities, raw = output.get("entities"), output.get("triples")
    if not isinstance(entities, list) or not isinstance(raw, list) or len(raw) > 14:
        return None, _evidence("invalid_or_unbounded_actual_compose")
    for entity in entities:
        if (not isinstance(entity, Mapping) or str(entity.get("id", "")) in bindings
                or str(entity.get("surface", "")).strip().casefold() not in by_surface):
            return None, _evidence("compose_added_unproved_topic_or_literal")
        bindings[str(entity["id"])] = by_surface[str(entity["surface"]).strip().casefold()]
    triples: list[list[str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None, _evidence("invalid_actual_compose_triple")
        a, b = str(item.get("subject", "")), str(item.get("object", ""))
        p = relation_id_from_label(item.get("relation_label", []))
        if (any(v not in bindings and not _VAR.fullmatch(v) for v in (a, b))
                or not p or not ontology.domain_for_relation(p) or not ontology.range_for_relation(p)):
            return None, _evidence("compose_node_or_predicate_not_schema_proved")
        triples.append([bindings.get(a, a), p, bindings.get(b, b)])
    answer = str(output.get("answer_var", ""))
    core_answer = names.get(str(grounded.compose_input["semantic_paths"][0]["path_output_var"]))
    core_graph = QueryGraphCandidate("current_grounded_core", core, core_answer or "", [], grounded.score, {})
    core_mapping = _unique_embedding(core_graph, triples)
    source_mapping = _unique_embedding(source_graph, triples)
    if core_mapping is None or source_mapping is None or source_mapping.get(source_graph.answer_var) != answer:
        return None, _evidence("source_scope_or_answer_role_not_uniquely_preserved")
    embedded_core = Counter((core_mapping.get(a, a), p, core_mapping.get(b, b)) for a, p, b in core)
    embedded_source = Counter((source_mapping.get(a, a), p, source_mapping.get(b, b))
                              for a, p, b in source_graph.triples)
    if embedded_source != embedded_core:
        return None, _evidence("source_and_grounded_core_have_different_scope")
    extra = list((Counter(map(tuple, triples)) - embedded_core).elements())
    if len(extra) != 2:
        return None, _evidence("compose_extension_not_exactly_one_record_date_chain")
    scalar = [t for t in extra if ontology.range_for_relation(t[1]) == "type.datetime"]
    if len(scalar) != 1:
        return None, _evidence("compose_extension_has_no_unique_datetime_leaf")
    owner, attribute, date_var = scalar[0]
    record = [t for t in extra if t != scalar[0]]
    if (len(record) != 1 or record[0][0] != answer or record[0][2] != owner
            or not _VAR.fullmatch(date_var) or date_var == answer
            or any(a == date_var for a, _, _ in triples)
            or owner in {v for t in embedded_core for v in (t[0], t[2])}):
        return None, _evidence("date_owner_not_unique_new_record_of_same_answer_entity")
    parent = record[0][1]
    if ontology.range_for_relation(parent) != ontology.domain_for_relation(attribute):
        return None, _evidence("date_record_owner_schema_does_not_match")
    answer_types = {ontology.domain_for_relation(p) for a, p, _ in embedded_core if a == answer}
    answer_types.update(ontology.range_for_relation(p) for _, p, b in embedded_core if b == answer)
    parent_domain = ontology.domain_for_relation(parent)
    overlap = {schema_type_overlap(t, parent_domain, ontology) for t in answer_types}
    if (not answer_types or not any(t not in SCALAR_TYPES and "common.topic" in ontology.supertypes(t)
                                    for t in answer_types)
            or ("hard_conflict" in overlap and "compatible" not in overlap)):
        return None, _evidence("existing_answer_entity_or_record_attachment_type_unproved")
    # Ordinary role classes without declared inheritance are not disjoint.
    # Unknown is not a proof of compatibility either: endpoint types in the
    # same paid evidence query must positively settle the owner relation.
    attachment_schema_proved = "compatible" in overlap
    leaf = attribute.rsplit(".", 1)[-1].casefold()
    phase_leaves = _START_LEAVES if intent[1] == "start" else _END_LEAVES
    if leaf not in phase_leaves:
        return None, _evidence("actual_compose_date_phase_differs_from_explicit_question")
    if (not _scalar_property_surface_supported(question, parent)
            or not _scalar_property_surface_supported(question, attribute)):
        return None, _evidence("actual_record_and_date_not_independently_named_by_question")
    old_anchors = {v for t in source_graph.triples for v in (t[0], t[2]) if not _VAR.fullmatch(v)}
    new_anchors = {v for t in triples for v in (t[0], t[2]) if not _VAR.fullmatch(v)}
    if not old_anchors or old_anchors != new_anchors:
        return None, _evidence("actual_compose_changed_topic_anchor_scope")
    if not _closed_question_scope(question, grounded, triples):
        return None, _evidence("question_has_unrepresented_topic_or_relation_scope")
    operators = deepcopy(source_graph.operators)
    for op in operators:
        op["input_var"] = source_mapping.get(str(op.get("input_var", "")), str(op.get("input_var", "")))
        if isinstance(op.get("inputs"), list):
            op["inputs"] = [source_mapping.get(str(v), str(v)) for v in op["inputs"]]
    operators.append({"type": intent[0], "inputs": [], "input_var": date_var,
                      "attribute_relation_label": [], "attribute_relation_labels": [],
                      "attribute_relation_id": "", "value": "", "value_type": "",
                      "_attribute_value_already_bound": True, "_order_bound_value": True,
                      "_order_temporal_lexical": True})
    result = QueryGraphCandidate(graph_id, triples, answer, operators, float(source_graph.score),
                                 deepcopy(dict(output)), deepcopy(source_graph.provenance))
    certificate = {**_evidence("actual_failed_compose_temporal_suffix_preserves_complete_source"),
                   "status": "compiled", "source_graph_id": source_graph.graph_id,
                   "graph_id": graph_id, "source_variable_mapping": source_mapping,
                   "grounded_core_variable_mapping": core_mapping,
                   "original_semantic_scope": scope, "actual_compose_output_reused": True,
                   "extrema_type": intent[0], "explicit_temporal_phase": intent[1],
                   "record_relation_id": parent, "date_relation_id": attribute,
                   "record_attachment_domain": parent_domain,
                   "record_attachment_schema_proved": attachment_schema_proved,
                   "requires_same_query_actual_owner_type_certificate": not attachment_schema_proved,
                   "date_owner_var": owner, "date_value_var": date_var,
                   "source_bgp_and_operators_preserved": True,
                   "requires_saved_execution_count_and_time_credit": True,
                   "requires_nonempty_strict_subset_of_complete_current_source": True,
                   "requires_complete_same_query_datetime_and_tie_certificate": True,
                   "new_repair_denotation_verified": False}
    result.provenance["stability_extrema_scope"] = deepcopy(certificate)
    return result, certificate


def compile_stability_extrema_from_current_stages(
    question: str, source_graph: QueryGraphCandidate,
    traces: Sequence[Mapping[str, Any]], ontology: Any, *, graph_id: str,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Closed current-stage adapter; never borrow other questions' traces."""
    if _temporal_extrema(question) is None:
        return None, _evidence("temporal_extrema_or_phase_not_unique_and_explicit")
    def latest(stage: str) -> Any:
        values = [t.get("output") for t in traces if isinstance(t, Mapping) and t.get("stage") == stage]
        return values[-1] if values else None
    semantics = latest("semantic_path_generation")
    groundings = latest("grounded_semantic_candidates")
    compositions = latest("compose_graph_generation")
    if not all(isinstance(v, list) for v in (semantics, groundings, compositions)):
        return None, _evidence("current_stage_corpus_missing")
    valid_semantics = [s for s in semantics if isinstance(s, Mapping) and s.get("valid") is True]
    proposals: dict[str, tuple[QueryGraphCandidate, dict[str, Any]]] = {}
    for raw in groundings:
        if not isinstance(raw, Mapping) or raw.get("compose_input", {}).get("question") != question:
            continue
        provenance = raw.get("provenance", {})
        index = provenance.get("semantic_graph_index") if isinstance(provenance, Mapping) else None
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(valid_semantics):
            continue
        semantic = valid_semantics[index].get("output")
        if not isinstance(semantic, Mapping):
            continue
        bound = raw.get("anchor_bindings", {})
        anchors = raw.get("compose_input", {}).get("anchors", [])
        if not isinstance(bound, Mapping) or not isinstance(anchors, list):
            continue
        entity_bindings = {str(a.get("id")): EntityCandidate(str(bound[str(a.get("id"))]),
                           str(a.get("surface", "")), 0.0)
                           for a in anchors if isinstance(a, Mapping) and str(a.get("id")) in bound
                           and isinstance(bound[str(a.get("id"))], str)}
        grounded = GroundedSemanticCandidate(deepcopy(dict(raw["compose_input"])), entity_bindings,
                    deepcopy(dict(raw.get("relation_bindings", {}))), float(raw.get("score", 0)),
                    deepcopy(dict(provenance)))
        for record in compositions:
            if not isinstance(record, Mapping) or record.get("input") != grounded.compose_input:
                continue
            candidate, evidence = compile_saved_extrema_extension(
                question, source_graph, grounded, record, semantic, ontology, graph_id=graph_id)
            if candidate is not None:
                from .lowering import lower_sparql
                proposals[lower_sparql(candidate, limit=None)] = (candidate, evidence)
    if len(proposals) != 1:
        return None, {**_evidence("no_unique_scope_certified_actual_compose_extrema"),
                      "compiled_unique_proposal_count": len(proposals)}
    return next(iter(proposals.values()))


def lower_stability_extrema_evidence_query(candidate: QueryGraphCandidate) -> str:
    """Read all actual dates in one query; never break ties with LIMIT 1.

    This is an evidence query, not the final answer projection.  The caller
    must use ``resolve_stability_extrema_rows`` before changing its result.
    """
    from .lowering import lower_sparql
    proof = candidate.provenance.get("stability_extrema_scope")
    if not isinstance(proof, Mapping) or proof.get("status") != "compiled":
        raise ValueError("Missing compiled temporal extrema certificate")
    value_var = str(proof.get("date_value_var", ""))
    if not _VAR.fullmatch(value_var) or not candidate.operators:
        raise ValueError("Missing bound temporal value variable")
    operator = candidate.operators[-1]
    if (operator.get("type") != proof.get("extrema_type") or operator.get("input_var") != value_var
            or not operator.get("_order_bound_value")):
        raise ValueError("Compiled temporal extrema operator changed")
    ordinary = deepcopy(candidate)
    ordinary.operators = ordinary.operators[:-1]
    owner = str(proof.get("date_owner_var", ""))
    parent = str(proof.get("record_relation_id", ""))
    attribute = str(proof.get("date_relation_id", ""))
    suffix = [[ordinary.answer_var, parent, owner], [owner, attribute, value_var]]
    if any(ordinary.triples.count(t) != 1 for t in suffix):
        raise ValueError("Compiled temporal suffix is no longer unique")
    for triple in suffix:
        ordinary.triples.remove(triple)
    query = lower_sparql(ordinary, limit=None)
    expected = "SELECT DISTINCT ?" + ordinary.answer_var
    if not query.startswith(expected + "\nWHERE {") or query.count("\n}") != 1:
        raise ValueError("Evidence query requires one unlimited basic source graph")
    query = query.replace(expected, expected + " ?" + value_var + " ?" + owner + " ?stability_owner_type (DATATYPE(?" + value_var
                          + ") AS ?stability_datetime_type)", 1)
    query = query.replace("\n}", "\n  OPTIONAL { ?" + ordinary.answer_var
                          + " <http://rdf.freebase.com/ns/" + parent + "> ?" + owner + " ."
                          + " OPTIONAL { ?" + owner + " <http://rdf.freebase.com/ns/" + attribute + "> ?"
                          + value_var + " . } }\n  OPTIONAL { ?" + ordinary.answer_var
                          + " <http://rdf.freebase.com/ns/type.object.type> ?stability_owner_type . }\n}", 1)
    return query


def _datetime_value(value: str, datatype: str) -> datetime | None:
    datatype = datatype.rsplit("#", 1)[-1]
    if datatype == "date":
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(Z|[+-]\d{2}:\d{2})?", value)
        if match is None:
            return None
        text = match[1] + "T00:00:00" + (match[2] or "")
    elif datatype == "dateTime":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?", value):
            return None
        text = value
    else:
        # In particular no gYear, signed-year, zero-year or unknown cast.
        return None
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if result.utcoffset() is not None and abs(result.utcoffset()) > timedelta(hours=14):
            return None
        return result
    except (ValueError, OverflowError):
        return None


def resolve_stability_extrema_rows(
    candidate: QueryGraphCandidate, rows: Sequence[Mapping[str, Any]], *,
    current_source_ids: Sequence[str], current_source_row_count: int,
    query_result_complete: bool, ontology: Any = None,
) -> tuple[list[str] | None, dict[str, Any]]:
    """Use complete same-query dates; retain every tied answer entity.

    All source entities must appear.  A source with no position record is
    ineligible under the explicit question obligation.  Any existing record
    with no date is unknown, not permission to discard that entity.
    Source and date row caps, invalid dates, mixed
    datatypes/precision or timezone-presence all cause abstention.
    """
    proof = candidate.provenance.get("stability_extrema_scope")
    if not isinstance(proof, Mapping) or proof.get("status") != "compiled":
        return None, _evidence("compiled_extrema_certificate_missing")
    source = list(current_source_ids)
    if (not query_result_complete or current_source_row_count != len(source)
            or len(source) < 2 or len(set(source)) != len(source)
            or not all(re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", str(v)) for v in source)
            or not isinstance(rows, Sequence) or not rows or len(rows) > 4096):
        return None, _evidence("source_or_datetime_query_not_complete")
    values: list[tuple[str, datetime]] = []
    timezone_present: set[bool] = set()
    datatypes: set[str] = set()
    owner_types: dict[str, set[str]] = {}
    actual_source_seen: set[str] = set()
    eligible_source: set[str] = set()
    eligible_records: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return None, _evidence("invalid_datetime_query_row")
        identifier = str(row.get(candidate.answer_var, ""))
        identifier = identifier.removeprefix("http://rdf.freebase.com/ns/")
        value = str(row.get(str(proof.get("date_value_var")), ""))
        datatype = str(row.get("stability_datetime_type", ""))
        if identifier not in source:
            return None, _evidence("actual_source_entity_not_certified")
        actual_source_seen.add(identifier)
        actual_type = str(row.get("stability_owner_type", ""))
        if actual_type.startswith("http://rdf.freebase.com/ns/"):
            actual_type = actual_type.removeprefix("http://rdf.freebase.com/ns/")
        if actual_type:
            owner_types.setdefault(identifier, set()).add(actual_type)
        record = str(row.get(str(proof.get("date_owner_var")), "")).removeprefix("http://rdf.freebase.com/ns/")
        if not record:
            if value or datatype:
                return None, _evidence("date_without_actual_position_record")
            continue
        if not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", record):
            return None, _evidence("position_record_is_not_an_actual_resource")
        eligible_source.add(identifier)
        eligible_records.add((identifier, record))
        if not value or not datatype:
            return None, _evidence("actual_eligible_position_record_has_no_date")
        if not datatype.startswith("http://www.w3.org/2001/XMLSchema#"):
            return None, _evidence("datetime_datatype_not_certified")
        parsed = _datetime_value(value, datatype)
        if parsed is None:
            return None, _evidence("datetime_value_not_exactly_normalizable")
        datatypes.add(datatype)
        timezone_present.add(parsed.tzinfo is not None)
        values.append((identifier, parsed))
    if (actual_source_seen != set(source) or not eligible_source
            or len(datatypes) != 1 or len(timezone_present) != 1
            or {v for v, _ in values} != eligible_source):
        return None, _evidence("date_precision_timezone_or_source_date_coverage_unproved")
    if proof.get("requires_same_query_actual_owner_type_certificate"):
        wanted = str(proof.get("record_attachment_domain", ""))
        if not all(any(actual == wanted or (ontology is not None
                       and wanted in ontology.supertypes(actual))
                       for actual in owner_types.get(identifier, set())) for identifier in eligible_source):
            return None, _evidence("actual_source_answer_record_attachment_type_unproved")
    try:
        extreme = (max if proof["extrema_type"] == "ARGMAX" else min)(v for _, v in values)
        accepted_set = {identifier for identifier, value in values if value == extreme}
    except (ValueError, TypeError, OverflowError):
        return None, _evidence("datetime_values_not_comparable")
    if not accepted_set or not accepted_set < set(source):
        return None, _evidence("no_nonempty_strict_subset_of_complete_source")
    accepted = [v for v in source if v in accepted_set]
    return accepted, {**_evidence("complete_same_query_datetime_extrema_with_all_ties"),
                      "status": "qualified", "source_answer_count": len(source),
                      "date_row_count": len(rows), "extremal_entity_count": len(accepted),
                      "all_tied_entities_retained": True,
                      "all_source_entities_covered": True,
                      "eligible_position_entity_count": len(eligible_source),
                      "eligible_position_record_count": len(eligible_records),
                      "all_eligible_position_records_have_actual_date": True,
                      "source_entities_without_any_position_record": [v for v in source if v not in eligible_source],
                      "actual_datetime_datatype": next(iter(datatypes)),
                      "same_query_record_attachment_type_verified": True,
                      "timezone_presence_uniform": True,
                      "new_repair_denotation_verified": True}
