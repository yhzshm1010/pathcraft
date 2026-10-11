"""Gold-blind protection of already executed, explicitly attached constraints.

No graph is generated, changed or executed here.  A scalar can only be added
to an unchanged answer/context, or a single projection error can be reversed
when question grammar, decomposition co-reference, original Semantic, Compose
and the schema all identify the same owner.  In particular, a smaller answer
set and an ARGMIN returning one answer are not evidence of correctness.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any

from .nested_name_schema_guard import _anchors, _complete, _singular
from .ontology import relation_id_from_label


_COMPARISON = {
    "less than": "LESS_THAN", "fewer than": "LESS_THAN", "lower than": "LESS_THAN",
    "greater than": "GREATER_THAN", "more than": "GREATER_THAN",
    "higher than": "GREATER_THAN", "larger than": "GREATER_THAN",
    "at most": "LESS_OR_EQUAL", "at least": "GREATER_OR_EQUAL",
    "before": "LESS_THAN", "after": "GREATER_THAN", "on": "EQUAL",
}
_COMPARATOR = "|".join(re.escape(x) for x in sorted(_COMPARISON, key=len, reverse=True))
_VALUE = r"(?:\d{4}-\d{2}-\d{2}|[+-]?\d[\d,]*(?:\.\d+)?)"
_EXPLICIT = re.compile(rf"\b(?P<comparison>{_COMPARATOR})\s+(?P<value>{_VALUE})(?!\w|\.\d)", re.I)
_NUMBER = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")
_VARIABLE = re.compile(r"(?:V\d+|P\d+\.V\d+|\?[A-Za-z_]\w*)\Z")
_EXTREMA = re.compile(r"\b(?:smallest|largest|lowest|highest|earliest|latest|first|last)\b", re.I)
_EXTREMA_KIND = {"smallest": "ARGMIN", "lowest": "ARGMIN", "earliest": "ARGMIN",
                 "first": "ARGMIN", "largest": "ARGMAX", "highest": "ARGMAX",
                 "latest": "ARGMAX", "last": "ARGMAX"}
_SCALAR = frozenset({"COUNT", "ARGMIN", "ARGMAX", "TC", "EQUAL", "LESS_THAN",
                     "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"})
_NUMERIC = frozenset({"type.int", "type.float"})


def _words(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", str(value).replace("_", " ").casefold()))


def _leaf(value: Any) -> tuple[str, ...]:
    return _words(str(value).rsplit(".", 1)[-1])


def _contains(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    return bool(phrase) and any(words[i:i + len(phrase)] == phrase
                                for i in range(len(words) - len(phrase) + 1))


def _decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value).replace(",", "").strip())
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _decomposition_items(values: Any) -> set[str]:
    items: set[str] = set()
    if not isinstance(values, (list, tuple)):
        return items
    for value in values:
        if isinstance(value, str):
            items.add(" ".join(value.split()).casefold())
        else:
            nested = value.get("decomposition") if isinstance(value, Mapping) else getattr(value, "decomposition", None)
            if isinstance(nested, list):
                items.update(" ".join(x.split()).casefold() for x in nested if isinstance(x, str))
    return items


def _semantic_views(values: Any):
    if not isinstance(values, (list, tuple)):
        return
    for value in values:
        if not isinstance(value, Mapping) or value.get("valid") is False:
            continue
        graph = value.get("output", value)
        if isinstance(graph, Mapping) and isinstance(graph.get("semantic_paths"), list):
            yield graph


def _obligation(question: str) -> dict[str, Any] | None:
    # Quoted content is not a scalar condition, and may contain comparator
    # words or extremal adjectives.  Unbalanced quotations are ambiguous.
    if str(question).count('"') % 2:
        return None
    text = re.sub(r'"[^\"]*"', " ", str(question))
    if re.search(r"\b(?:not|no|never|or|either|except|nearest|approximately|about)\b", text, re.I):
        return None
    matches = list(_EXPLICIT.finditer(text))
    extrema = list(_EXTREMA.finditer(text))
    if len(matches) == 1 and not extrema:
        match = matches[0]
        raw = match["value"]
        temporal = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw))
        if temporal:
            try:
                date.fromisoformat(raw)
            except ValueError:
                return None
            # No second date/number is permitted, including entity-name years.
            remainder = text[:match.start("value")] + text[match.end("value"):]
            if _NUMBER.search(remainder):
                return None
        elif len(_NUMBER.findall(text)) != 1 or _decimal(raw) is None:
            return None
        if not temporal and match["comparison"].casefold() in {"before", "after", "on"}:
            return None
        return {"kind": _COMPARISON[match["comparison"].casefold()], "value": raw,
                "datatype": "date" if temporal else "number", "comparison": match["comparison"].casefold(),
                "prefix": text[:match.start()], "text": text}
    if len(extrema) == 1 and not matches and not _NUMBER.search(text):
        head = re.match(r"\s*(?:what|which)\s+([A-Za-z]+)\s+(?:that|with|has|is|was|contains)\b", text, re.I)
        # Executed ARGMIN/ARGMAX commonly has LIMIT 1. A plural target would
        # require independent evidence that tied extrema are not suppressed.
        if head is None or _singular(head[1]) != head[1].casefold():
            return None
        match = extrema[0]
        return {"kind": _EXTREMA_KIND[match.group().casefold()], "value": "",
                "datatype": "ordering", "prefix": text, "text": text}
    return None


def _internal_obligation(question: str, spec_provider: Any = None) -> dict[str, Any] | None:
    """Opt-in, post-model fallback; never replace a previously supported form.

    A supplied dictionary is not a semantic witness. Recompute the strict
    word-normalization proof locally and verify its source span/cue. Native
    parser outputs lacking this explicit proof cannot widen this guard.
    """
    native = _obligation(question)
    if native is not None or not callable(spec_provider):
        return native
    from .comparison_number_normalization import normalize_comparison_constraint_spec
    provided = spec_provider(question)
    if not isinstance(provided, Mapping) or provided.get("_normalization_scope") != "internal_post_model_constraint_spec_only":
        return None
    verified = normalize_comparison_constraint_spec(question)
    fields = ("type", "value", "value_type", "_normalization_kind", "_source_literal",
              "_source_comparison_cue", "_source_literal_span", "_comparison_date_precision")
    if verified is None or any(provided.get(k) != verified.get(k) for k in fields):
        return None
    value_type = str(verified["value_type"])
    if value_type == "dateTime" and verified.get("_comparison_date_precision") != "day":
        return None
    literal_start = verified["_source_literal_span"][0]
    cue = re.search(re.escape(verified["_source_comparison_cue"]) + r"\s*$", str(question)[:literal_start], re.I)
    if cue is None:
        return None
    return {"kind": verified["type"], "value": verified["value"],
            "datatype": "date" if value_type == "dateTime" else "number",
            "comparison": verified["_source_comparison_cue"].casefold(),
            "prefix": str(question)[:cue.start()], "text": str(question),
            "internal_normalization_proof": {k: verified[k] for k in (
                "_normalization_kind", "_source_literal", "_source_comparison_cue", "_source_literal_span")}}


def _operators(graph: Any) -> tuple[list[dict], list[dict]] | None:
    values = getattr(graph, "operators", None)
    if not isinstance(values, list) or any(not isinstance(x, dict) for x in values):
        return None
    return ([x for x in values if str(x.get("type", "")).upper() in _SCALAR],
            [x for x in values if str(x.get("type", "")).upper() not in _SCALAR])


def _signatures(values: list[dict]) -> set[str]:
    return {json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for x in values}


def _triples(graph: Any) -> set[tuple[str, str, str]] | None:
    triples = getattr(graph, "triples", None)
    if not isinstance(triples, list) or not triples or any(not isinstance(x, (list, tuple)) or len(x) != 3 for x in triples):
        return None
    return {tuple(map(str, x)) for x in triples}


def _binding(graph: Any, operator: dict, ontology: Any,
             *, prefer_direct_answer_owner: bool = False) -> dict[str, Any] | None:
    """Find one schema-proven owner and primitive attribute, not an entity URI."""
    triples = _triples(graph)
    if triples is None:
        return None
    input_var = str(operator.get("input_var") or "")
    if not _VARIABLE.fullmatch(input_var):
        return None
    attribute = str(operator.get("attribute_relation_id") or "")
    if not attribute:
        incoming = [(s, p) for s, p, o in triples if o == input_var]
        if len(incoming) != 1:
            return None
        owner, attribute = incoming[0]
        if str(ontology.range_for_relation(attribute)) not in _NUMERIC | {"type.datetime"}:
            return None
        return {"owner": owner, "attribute": attribute, "property": attribute,
                "datatype": str(ontology.range_for_relation(attribute)), "branches": {(owner, attribute, input_var)}}
    datatype = str(ontology.range_for_relation(attribute) or "")
    domain = str(ontology.domain_for_relation(attribute) or "")
    if datatype not in _NUMERIC | {"type.datetime"} or not domain:
        return None
    if prefer_direct_answer_owner and input_var == str(graph.answer_var):
        incident = {str(ontology.range_for_relation(p)) for _, p, o in triples if o == input_var}
        incident |= {str(ontology.domain_for_relation(p)) for s, p, _ in triples if s == input_var}
        if domain in incident:
            # A direct answer-country.calling_code operator must not be
            # rebound onto the language that supplied that answer country.
            return {"owner": input_var, "attribute": attribute, "property": attribute,
                    "datatype": datatype, "branches": set()}
    incoming = {(s, p) for s, p, o in triples if o == input_var
                and str(ontology.range_for_relation(p)) == domain}
    if len(incoming) == 1:
        owner, prop = next(iter(incoming))
        branches = {(s, p, o) for s, p, o in triples if s == owner and p == prop
                    and _VARIABLE.fullmatch(o)}
        # Redundant existential copies of the same scalar CVT are permitted
        # only if every copy is a degree-one, otherwise-unused leaf.
        for _, _, leaf in branches:
            if leaf != input_var and (sum(leaf in (s, o) for s, _, o in triples) != 1
                                      or leaf == str(graph.answer_var)
                                      or any(leaf in str(x) for x in graph.operators)):
                return None
        return {"owner": owner, "attribute": attribute, "property": prop,
                "datatype": datatype, "branches": branches}
    if incoming:
        return None
    # A direct primitive attribute on the output entity is supported only
    # when an incident schema explicitly declares that attribute's domain.
    incident = {str(ontology.range_for_relation(p)) for _, p, o in triples if o == input_var}
    incident |= {str(ontology.domain_for_relation(p)) for s, p, _ in triples if s == input_var}
    if domain not in incident:
        return None
    return {"owner": input_var, "attribute": attribute, "property": attribute,
            "datatype": datatype, "branches": set()}


def _matches(operator: dict, binding: dict, obligation: dict) -> bool:
    if str(operator.get("type", "")).upper() != obligation["kind"]:
        return False
    if operator.get("inputs") not in (None, []):
        return False
    if obligation["datatype"] == "number":
        if operator.get("_numeric_comparison_cast") and operator.get("_numeric_cast_kind") != (
                "float" if binding["datatype"] == "type.float" else "integer"):
            return False
        return (binding["datatype"] in _NUMERIC
                and str(operator.get("value_type", "")).casefold() == "number"
                and _decimal(operator.get("value")) == _decimal(obligation["value"])
                and not operator.get("_numeric_cast_compare"))
    if obligation["datatype"] == "date":
        return (binding["datatype"] == "type.datetime"
                and str(operator.get("value_type", "")).casefold() == "date"
                and str(operator.get("value")) == obligation["value"]
                and operator.get("_comparison_date_precision") == "day"
                and (obligation["kind"] == "EQUAL" or operator.get("_calendar_day_boundary") is True)
                and not operator.get("_numeric_comparison_cast"))
    if operator.get("_numeric_order_cast") and operator.get("_numeric_order_cast") != (
            "float" if binding["datatype"] == "type.float" else "integer"):
        return False
    return not str(operator.get("value") or "") and binding["datatype"] in _NUMERIC | {"type.datetime"}


def _property_supported(binding: dict, obligation: dict) -> bool:
    # Every property-leaf word is required, not just 'date', 'number', 'rate'
    # or a convenient ARGMIN. Copyright date does not match first book written.
    if not _contains(_words(obligation["prefix"]), _leaf(binding["property"])):
        return False
    if obligation["datatype"] == "ordering" and binding["property"] != binding["attribute"]:
        return _contains(_words(obligation["text"]), _leaf(binding["attribute"]))
    return True


def _semantic_attachment(graphs: Any, decompositions: set[str], prop: str,
                         obligation: dict, spec_provider: Any = None) -> list[tuple[Mapping, Mapping, Mapping]]:
    matches = []
    seen = set()
    for graph in _semantic_views(graphs):
        for path in graph["semantic_paths"]:
            if not isinstance(path, Mapping):
                continue
            goal = " ".join(str(path.get("goal", "")).split()).casefold()
            if goal not in decompositions:
                continue
            condition = _internal_obligation(goal, spec_provider)
            if condition is None or (condition["kind"], condition["value"], condition["datatype"]) != (
                    obligation["kind"], obligation["value"], obligation["datatype"]):
                continue
            steps = path.get("steps")
            if not isinstance(steps, list):
                continue
            attached = [step for step in steps if isinstance(step, Mapping)
                        and isinstance(step.get("relation_label"), list)
                        and step["relation_label"]
                        and _words(step["relation_label"][-1]) == _leaf(prop)
                        and step.get("direction") == "forward"]
            if len(attached) != 1:
                continue
            signature = json.dumps({"anchors": graph.get("anchors"), "path": path}, sort_keys=True)
            if signature not in seen:
                seen.add(signature)
                matches.append((graph, path, attached[0]))
    return matches


def _compose_owner(graph: Any, owner: str, prop: str) -> bool:
    compose = getattr(graph, "compose_output", None)
    if not isinstance(compose, Mapping) or str(compose.get("answer_var")) != owner:
        return False
    return any(isinstance(t, Mapping) and str(t.get("subject")) == owner
               and isinstance(t.get("relation_label"), list) and t["relation_label"]
               and _words(t["relation_label"][-1]) == _leaf(prop)
               for t in compose.get("triples", []))


def _prefix_owners(question: str, selected: Any, candidate: Any, attachment: list) -> set[str] | None:
    """Map original Semantic's property subject through actual/Compose joins.

    A scalar path starting on a province cannot be rebound onto an answer
    country just because both have a population predicate. Opaque or divergent
    cross-stage variable mappings are not certificates.
    """
    provenance = getattr(selected.graph, "provenance", None)
    compose = getattr(candidate.graph, "compose_output", None)
    if not isinstance(provenance, Mapping) or not isinstance(compose, Mapping):
        return None
    bindings = provenance.get("anchor_bindings")
    if not isinstance(bindings, Mapping):
        return None
    entities = compose.get("entities")
    compose_triples = compose.get("triples")
    if not isinstance(entities, list) or not isinstance(compose_triples, list):
        return None
    actual = _triples(candidate.graph)
    owners: set[str] = set()
    for graph, path, scalar in attachment:
        anchor_id = str(path.get("anchor_ref", ""))
        anchor = [a for a in graph.get("anchors", []) if isinstance(a, Mapping) and str(a.get("id")) == anchor_id]
        linked = bindings.get(anchor_id)
        if len(anchor) != 1 or not isinstance(linked, Mapping):
            return None
        surface = str(anchor[0].get("surface", ""))
        if (not surface or _words(surface) != _words(linked.get("label", ""))
                or not _contains(_words(question), _words(surface))):
            return None
        matches = [e for e in entities if isinstance(e, Mapping) and _words(e.get("surface", "")) == _words(surface)]
        if len(matches) != 1:
            return None
        entity_ref, entity_id = str(matches[0].get("id", "")), str(linked.get("id", ""))
        composed = set()
        for triple in compose_triples:
            if not isinstance(triple, Mapping):
                return None
            subject, object_ = str(triple.get("subject", "")), str(triple.get("object", ""))
            composed.add((entity_id if subject == entity_ref else subject,
                          relation_id_from_label(triple.get("relation_label")),
                          entity_id if object_ == entity_ref else object_))
        current = {entity_id}
        for step in path.get("steps", []):
            if not isinstance(step, Mapping):
                return None
            if step is scalar:
                break
            predicate = relation_id_from_label(step.get("relation_label"))
            if step.get("direction") == "forward":
                current = {o for s, p, o in actual & composed if s in current and p == predicate}
            elif step.get("direction") == "backward":
                current = {s for s, p, o in actual & composed if o in current and p == predicate}
            else:
                return None
            if len(current) != 1:
                return None
        owners.update(current)
    return owners


def _projection_certificate(question: str, selected: Any, candidate: Any,
                            source_binding: dict, binding: dict, attachment: list,
                            obligation: dict) -> dict[str, Any] | None:
    # This is a grammatical owner-qualification shape, not a population or
    # containment rule. A nested 'which countries contain provinces whose ...'
    # has a different question contract and must keep the intermediate owner.
    if obligation["datatype"] != "number":
        return None
    qualifier = re.fullmatch(r"\s*where\s+with\s+(?:a|an|the)\s+(.+?)\s*", obligation["prefix"], re.I)
    if not qualifier:
        return None
    attribute_words = _words(re.sub(r"\s+once\s+of\s*$", "", qualifier[1], flags=re.I))
    if attribute_words != _leaf(binding["property"]):
        return None
    if (source_binding["owner"], source_binding["property"], source_binding["attribute"], source_binding["datatype"]) != (
            binding["owner"], binding["property"], binding["attribute"], binding["datatype"]):
        return None
    owner = binding["owner"]
    if str(candidate.graph.answer_var) != owner or str(selected.graph.answer_var) == owner:
        return None
    if not _compose_owner(candidate.graph, owner, binding["property"]):
        return None
    source_context = _triples(selected.graph) - source_binding["branches"]
    candidate_context = _triples(candidate.graph) - binding["branches"]
    if len(source_context) != 2 or len(candidate_context) != 1 or not candidate_context < source_context:
        return None
    first = next(iter(candidate_context))
    extra = next(iter(source_context - candidate_context))
    if (first[2] != owner or extra[0] != owner or extra[2] != str(selected.graph.answer_var)
            or extra[1] != first[1]):
        return None
    certificates = set()
    for semantic, path, scalar_step in attachment:
        steps = path.get("steps", [])
        if (len(steps) != 2 or scalar_step is not steps[1] or steps[0].get("direction") != "forward"
                or _words(steps[0].get("relation_label", [""])[-1]) != _leaf(first[1])
                or str(steps[1].get("from")) != str(steps[0].get("to"))):
            continue
        anchor_id = str(path.get("anchor_ref"))
        anchors = [a for a in semantic.get("anchors", []) if isinstance(a, Mapping) and str(a.get("id")) == anchor_id]
        if len(anchors) != 1:
            continue
        anchor_surface = str(anchors[0].get("surface", ""))
        provenance = getattr(selected.graph, "provenance", None)
        linked = provenance.get("anchor_bindings", {}).get(anchor_id) if isinstance(provenance, Mapping) else None
        if (not isinstance(linked, Mapping) or str(linked.get("id")) != first[0]
                or _words(linked.get("label", "")) != _words(anchor_surface)
                or not _contains(_words(question), _words(anchor_surface))):
            continue
        # Cross-stage coreference is explicit: the same queried relation owner,
        # not merely a property on an arbitrary intermediate node.
        labels = steps[0].get("relation_label", [])
        if len(labels) < 2:
            continue
        type_words = _words(labels[-2])
        relation_words = _words(labels[-1])
        goal = str(path.get("goal", ""))
        clauses = [x.strip().rstrip(".").strip() for x in goal.split("?") if x.strip()]
        if len(clauses) != 3:
            continue
        expected_first = ("what", "is", *type_words, *_words(anchor_surface), *relation_words)
        expected_second = ("what", "is", "the", *attribute_words, "of", "that", "same", *relation_words, *type_words)
        if _words(clauses[0]) != expected_first or _words(clauses[1]) != expected_second:
            continue
        expected_third = ("the", *attribute_words, "of", "that", *type_words, "is",
                          *obligation["comparison"].split())
        if _words(clauses[2])[:len(expected_third)] != expected_third:
            continue
        certificates.add((first, extra, anchor_surface, str(steps[0].get("to"))))
    if len(certificates) != 1:
        return None
    return {"owner_var": owner, "semantic_owner_certificates": len(certificates),
            "unrequested_projection_edge": list(extra), "retained_context_edge": list(first),
            "same_owner_question_decomposition_semantic_compose": True}


def constraint_execution_challenger(
    question: str, selected: Any, executions: list[Any], ontology: Any,
    *, semantic_graphs: Any = None, decompositions: Any = None,
    prior_selection: Any = None, spec_provider: Any = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Select a unique complete existing execution or conservatively abstain."""
    evidence: dict[str, Any] = {
        "status": "not_applied", "source_graph_id": str(selected.graph.graph_id),
        "additional_model_calls": 0, "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0, "uses_gold_answers": False,
        "uses_existing_executions_only": True, "uses_entity_or_relation_whitelist": False,
    }
    def abstain(reason: str):
        evidence["reason"] = reason
        return None, evidence

    obligation = _internal_obligation(question, spec_provider)
    if obligation is None:
        return abstain("not_one_unambiguous_explicit_scalar_obligation")
    evidence["explicit_obligation"] = {k: obligation[k] for k in ("kind", "value", "datatype")}
    if obligation.get("internal_normalization_proof"):
        evidence["internal_normalization_proof"] = obligation["internal_normalization_proof"]
    if ontology is None or not _complete(selected) or _triples(selected.graph) is None:
        return abstain("source_execution_or_schema_incomplete")
    source_ops = _operators(selected.graph)
    if source_ops is None or len(source_ops[0]) > 1:
        return abstain("source_operator_contract_ambiguous")
    provenance = getattr(selected.graph, "provenance", None)
    if isinstance(provenance, Mapping) and provenance.get("selection_answer_union"):
        return abstain("union_is_not_one_actual_execution")
    if (obligation.get("internal_normalization_proof") and isinstance(provenance, Mapping)
            and provenance.get("selection_answer_intersection")):
        return abstain("intersection_is_not_one_actual_execution")
    items = _decomposition_items(decompositions)
    if not items:
        return abstain("decomposition_attachment_unavailable")
    source_binding = _binding(selected.graph, source_ops[0][0], ontology,
                              prefer_direct_answer_owner=bool(obligation.get("internal_normalization_proof"))) if source_ops[0] else None
    if source_ops[0] and (source_binding is None or not _matches(source_ops[0][0], source_binding, obligation)):
        # Do not correct existing comparison directions/dates globally: such a
        # change can alter a currently perfect answer even when semantically
        # warranted. This module only repairs missing obligations/projection.
        return abstain("existing_scalar_semantics_not_identical_preserved")
    qualified = []
    rejected = []
    for candidate in executions:
        if candidate is selected or not _complete(candidate) or _triples(candidate.graph) is None:
            continue
        ops = _operators(candidate.graph)
        if ops is None or len(ops[0]) != 1 or _signatures(ops[1]) != _signatures(source_ops[1]):
            continue
        binding = _binding(candidate.graph, ops[0][0], ontology,
                           prefer_direct_answer_owner=bool(obligation.get("internal_normalization_proof")))
        if binding is None or not _matches(ops[0][0], binding, obligation) or not _property_supported(binding, obligation):
            continue
        if _anchors(selected.graph) != _anchors(candidate.graph) or not _anchors(selected.graph):
            continue
        attachment = _semantic_attachment(semantic_graphs, items, binding["property"], obligation, spec_provider)
        if not attachment:
            rejected.append({"graph_id": candidate.graph.graph_id, "reason": "original_semantic_attachment_missing"})
            continue
        prefix_owners = _prefix_owners(question, selected, candidate, attachment)
        if prefix_owners != {binding["owner"]}:
            rejected.append({"graph_id": candidate.graph.graph_id, "reason": "semantic_compose_owner_mapping_not_unique"})
            continue
        certificate = None
        if source_ops[0]:
            certificate = _projection_certificate(question, selected, candidate, source_binding, binding, attachment, obligation)
        elif (str(candidate.graph.answer_var) == str(selected.graph.answer_var) == binding["owner"]
              and _triples(candidate.graph) - binding["branches"] == _triples(selected.graph)
              and (_compose_owner(candidate.graph, binding["owner"], binding["property"])
                   or (obligation.get("internal_normalization_proof") and not binding["branches"]
                       and isinstance(getattr(candidate.graph, "compose_output", None), Mapping)
                       and str(candidate.graph.compose_output.get("answer_var")) == binding["owner"]))):
            certificate = {"owner_var": binding["owner"], "retains_all_source_triples_and_operators": True,
                           "answer_var_unchanged": True}
        if certificate is not None:
            qualified.append((candidate, binding, certificate))
    evidence["qualified_graph_ids"] = [x[0].graph.graph_id for x in qualified]
    if rejected:
        evidence["rejected_candidates"] = rejected
    if not qualified:
        return abstain("no_complete_execution_with_unique_explicit_owner_contract")
    denotations = {tuple(sorted(str(x) for x in entry[0].answer_ids)) for entry in qualified}
    bindings = {(entry[1]["owner"], entry[1]["property"], entry[1]["attribute"], entry[1]["datatype"])
                for entry in qualified}
    if len(denotations) != 1 or len(bindings) != 1:
        return abstain("qualified_existing_denotations_or_attributes_ambiguous")
    candidate, binding, certificate = qualified[0]
    if set(candidate.answer_ids) == set(selected.answer_ids):
        return abstain("existing_denotation_unchanged")
    evidence.update(status="applied", reason="unique_existing_explicit_constraint_owner_execution",
                    selected_graph_id=candidate.graph.graph_id, attribute_relation=binding["attribute"],
                    owner_property=binding["property"], owner_certificate=certificate,
                    source_answer_count=len(selected.answer_ids), selected_answer_count=len(candidate.answer_ids))
    return candidate, evidence
