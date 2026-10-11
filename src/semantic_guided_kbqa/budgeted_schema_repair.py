"""Compile already-grounded obligations without changing the model beam.

The old retrieval guard treats unrelated entity roles as disjoint types.
Freebase roles are open-world: two unrelated roles can describe one entity.
Keep that distinction local to a rejected candidate; do not widen Compose or
Operator input beams.  A caller may store qualified candidates in memory and
use a compiled repair *instead of* one existing late execution slot.

This module performs no model, embedding, endpoint, file, or Gold access.  It
does not decide which denotation wins.  The caller owns the fixed execution
budget and must validate the resulting denotation before selecting it.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Callable, Mapping, Sequence

from .contracts import ExecutedGraph, GroundedSemanticCandidate, QueryGraphCandidate
from .ontology import relation_id_from_label, relation_label_from_id


SCALAR_TYPES = frozenset({
    "type.boolean", "type.datetime", "type.enumeration", "type.float",
    "type.int", "type.key", "type.rawstring", "type.text", "type.uri",
})
NUMERIC_TYPES = frozenset({"type.int", "type.float"})
STRING_TYPES = frozenset({"type.key", "type.rawstring", "type.text"})
COMPARISON_TYPES = frozenset({
    "GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "EQUAL",
})
_VAR = re.compile(r"^V\d+$")
_COMPARISON = re.compile(
    r"\b(no more than|no less than|at most|at least|less than|fewer than|"
    r"lower than|smaller than|prior to|more than|greater than|higher than|"
    r"larger than|later than|under|below|before|over|above|after)\b", re.I,
)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)")
_MONTH = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_NAMED_DATE = re.compile(
    rf"\b(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}\b|"
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH})[,]?\s+\d{{4}}\b", re.I,
)


def _evidence(reason: str) -> dict[str, Any]:
    return {"status": "not_applied", "reason": reason,
            "additional_model_calls": 0, "additional_endpoint_queries": 0,
            "additional_embedding_calls": 0, "uses_evaluation_gold": False}


def schema_type_overlap(left: str, right: str, ontology: Any) -> str:
    """Return compatible, unknown, or a provable RDF-kind contradiction.

    No inheritance relation between ordinary entity roles proves nothing.
    Missing CVT/topic inheritance is also not proof of disjointness.
    """
    left, right = str(left).strip(), str(right).strip()
    if not left or not right or ontology is None:
        return "unknown"
    if left == right:
        return "compatible"
    if left in SCALAR_TYPES or right in SCALAR_TYPES:
        if {left, right} <= NUMERIC_TYPES or {left, right} <= STRING_TYPES:
            return "compatible"
        return "hard_conflict"
    if left in {"common.topic", "type.object"} or right in {
        "common.topic", "type.object",
    }:
        return "compatible"
    closures = getattr(ontology, "supertypes", None)
    if callable(closures) and (
        left in set(map(str, closures(right)))
        or right in set(map(str, closures(left)))
    ):
        return "compatible"
    return "unknown"


def grounding_schema_evidence(
    grounded: GroundedSemanticCandidate, ontology: Any,
) -> dict[str, Any]:
    """Classify every adjacent edge of an actual, complete grounding."""
    result = _evidence("missing_grounded_paths_or_schema")
    paths = grounded.compose_input.get("semantic_paths", [])
    hops = grounded.provenance.get("hops", [])
    if (ontology is None or not isinstance(paths, list) or not paths
            or not isinstance(hops, list) or len(paths) != len(hops)
            or grounded.provenance.get("dropped_paths")):
        return result
    unknown: list[dict[str, Any]] = []
    for path_index, (path, path_hops) in enumerate(zip(paths, hops)):
        steps = path.get("steps", []) if isinstance(path, dict) else []
        if (not steps or not isinstance(path_hops, list)
                or len(steps) != len(path_hops)
                or not all(isinstance(hop, dict)
                           and hop.get("ontology_constrained") is True
                           and hop.get("retrieval_mode") == "ontology_expansion"
                           for hop in path_hops)):
            result["reason"] = "not_complete_pure_ontology_grounding"
            return result
        typed: list[tuple[str, str, str]] = []
        for step in steps:
            relation = str(grounded.relation_bindings.get(str(step.get("id", "")), ""))
            direction = str(step.get("direction", ""))
            domain = str(ontology.domain_for_relation(relation)).strip()
            range_id = str(ontology.range_for_relation(relation)).strip()
            if not relation or not domain or not range_id or direction not in {"forward", "backward"}:
                result["reason"] = "incomplete_schema_edge"
                return result
            input_type, output_type = (domain, range_id) if direction == "forward" else (range_id, domain)
            typed.append((relation, input_type, output_type))
        for edge_index, (left, right) in enumerate(zip(typed, typed[1:])):
            overlap = schema_type_overlap(left[2], right[1], ontology)
            record = {"path_index": path_index, "edge_index": edge_index,
                      "left_relation_id": left[0], "right_relation_id": right[0],
                      "left_output_type": left[2], "right_input_type": right[1]}
            if overlap == "hard_conflict":
                result.update({"status": "hard_conflict", "reason": "proven_rdf_kind_conflict", **record})
                return result
            if overlap == "unknown":
                if left[2].startswith("type.") or right[1].startswith("type."):
                    result["reason"] = "unknown_schema_metatype_not_an_entity_role_witness"
                    return result
                unknown.append(record)
    result.update({"status": "qualified" if unknown else "not_applied",
                   "reason": "unproven_entity_role_discontinuity" if unknown else "no_unproven_role_discontinuity",
                   "unknown_role_boundaries": unknown})
    return result


def capture_unproven_schema_rejection(
    grounded: GroundedSemanticCandidate, rejection: Mapping[str, Any], ontology: Any,
) -> tuple[bool, dict[str, Any]]:
    """Qualify memory retention, without retaining it in a model input beam."""
    if rejection.get("reason") != "pure_ontology_expansion_schema_discontinuity":
        return False, _evidence("not_the_open_world_role_guard")
    evidence = grounding_schema_evidence(grounded, ontology)
    return evidence.get("status") == "qualified", evidence


def _number(value: Any) -> Decimal | None:
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _date_literals(text: str) -> list[str]:
    values: list[str] = []
    for match in _ISO_DATE.finditer(text):
        try:
            values.append(date(*map(int, match.groups())).isoformat())
        except ValueError:
            return []
    for match in _NAMED_DATE.finditer(text):
        value = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", match.group(0), flags=re.I)
        value = re.sub(r"\bSept\b", "Sep", value.replace(",", ""), flags=re.I)
        for fmt in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
            try:
                values.append(datetime.strptime(value.title(), fmt).date().isoformat())
                break
            except ValueError:
                continue
    return values


def _explicit_spec(question: str, spec: Mapping[str, Any], labels: Sequence[str]) -> bool:
    text = str(question)
    for label in sorted(set(labels), key=len, reverse=True):
        if label:
            text = re.sub(re.escape(label), " ", text, flags=re.I)
    operator = str(spec.get("type", "")).upper()
    value_type = str(spec.get("value_type", "")).casefold()
    value = str(spec.get("value", "")).strip()
    if operator not in COMPARISON_TYPES or not value:
        return False
    cues = list(_COMPARISON.finditer(text))
    if operator != "EQUAL":
        if len(cues) != 1:
            return False
        cue = cues[0].group(0).casefold()
        if cue in {"at most", "no more than"}:
            expected = "LESS_OR_EQUAL"
        elif cue in {"at least", "no less than"}:
            expected = "GREATER_OR_EQUAL"
        elif cue in {"less than", "fewer than", "lower than", "smaller than", "prior to", "under", "below", "before"}:
            expected = "LESS_THAN"
        else:
            expected = "GREATER_THAN"
        if operator != expected:
            return False
        text = text[cues[0].end():]
    elif cues:
        return False
    dates = _date_literals(text)
    if value_type in {"datetime", "date", "year"}:
        if dates:
            return bool(len(dates) == 1 and value == dates[0]
                        and not re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[Tt]\d", text))
        return bool(re.fullmatch(r"\d{4}", value)
                    and re.findall(r"(?<!\w)\d{4}(?!\w)", text) == [value])
    numbers = re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w.])", text)
    return bool(value_type == "number" and not dates and len(numbers) == 1
                and _number(value) is not None and _number(numbers[0]) == _number(value))


def _compile_grounded(grounded: GroundedSemanticCandidate) -> tuple[list[list[str]], dict[str, str]]:
    names: dict[str, str] = {}
    triples: list[list[str]] = []
    for path in grounded.compose_input.get("semantic_paths", []):
        for step in path.get("steps", []):
            source, target = str(step.get("from", "")), str(step.get("to", ""))
            if source in grounded.anchor_bindings:
                source_ref = grounded.anchor_bindings[source].entity_id
            else:
                source_ref = names.get(source, "")
            target_ref = names.setdefault(target, f"V{len(names)}")
            relation = str(grounded.relation_bindings.get(str(step.get("id", "")), ""))
            if not source_ref or not relation:
                return [], {}
            subject, object_ = (source_ref, target_ref) if step.get("direction") == "forward" else (target_ref, source_ref)
            triples.append([subject, relation, object_])
    return triples, names


def _unique_embedding(source: QueryGraphCandidate, triples: list[list[str]]) -> dict[str, str] | None:
    """Find one rooted, injective variable mapping, with a tiny CPU bound."""
    mappings: list[dict[str, str]] = []
    searches = 0

    def visit(index: int, mapping: dict[str, str], used: set[int]) -> None:
        nonlocal searches
        searches += 1
        if searches > 64 or len(mappings) > 1:
            return
        if index == len(source.triples):
            if mapping not in mappings:
                mappings.append(mapping)
            return
        left, relation, right = map(str, source.triples[index])
        for target_index, (a, predicate, b) in enumerate(triples):
            if target_index in used or relation != predicate:
                continue
            extended = dict(mapping)
            for src, dst in ((left, str(a)), (right, str(b))):
                if _VAR.fullmatch(src):
                    if not _VAR.fullmatch(dst) or (src in extended and extended[src] != dst):
                        break
                    if src not in extended and dst in extended.values():
                        break
                    extended[src] = dst
                elif src != dst:
                    break
            else:
                visit(index + 1, extended, used | {target_index})

    visit(0, {}, set())
    if searches > 64 or len(mappings) != 1 or str(source.answer_var) not in mappings[0]:
        return None
    return mappings[0]


def compile_grounded_constraint_repair(
    grounded: GroundedSemanticCandidate, source_graph: QueryGraphCandidate,
    question: str, spec: Mapping[str, Any], ontology: Any, *, graph_id: str,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Compile a role-guard-rejected path using an existing graph's answer role.

    Qualification is structural, not a question/relation whitelist.  The
    source must be a uniquely embedded existing answer graph.  Its operators
    cannot already enforce a different substantive condition.  The added
    scalar leaf must be unique, fully grounded, and named in the question.
    """
    evidence = grounding_schema_evidence(grounded, ontology)
    if evidence.get("status") != "qualified":
        return None, evidence
    labels = [str(item.label) for item in grounded.anchor_bindings.values()]
    if not _explicit_spec(question, spec, labels):
        return None, _evidence("constraint_not_one_explicit_exact_literal")
    if not source_graph.triples or any(
        not isinstance(op, dict) or str(op.get("type", "")).upper() != "NO_EQUAL"
        for op in source_graph.operators
    ):
        return None, _evidence("source_already_has_substantive_operator")
    triples, names = _compile_grounded(grounded)
    if not triples or len(triples) <= len(source_graph.triples):
        return None, _evidence("no_complete_grounded_extension")
    source_anchors = {str(term) for triple in source_graph.triples for term in (triple[0], triple[2])
                      if not _VAR.fullmatch(str(term))}
    target_anchors = {str(term) for triple in triples for term in (triple[0], triple[2])
                      if not _VAR.fullmatch(str(term))}
    if not source_anchors or source_anchors != target_anchors:
        return None, _evidence("different_actual_entity_anchors")
    mapping = _unique_embedding(source_graph, triples)
    if mapping is None:
        return None, _evidence("source_answer_role_not_uniquely_embedded")
    source_triples = Counter((mapping.get(str(a), str(a)), str(p), mapping.get(str(b), str(b)))
                             for a, p, b in source_graph.triples)
    added = list((Counter(map(tuple, triples)) - source_triples).elements())
    scalar = [(a, p, b) for a, p, b in added
              if str(ontology.range_for_relation(p)) in SCALAR_TYPES]
    if len(scalar) != 1:
        return None, _evidence("not_one_existing_grounded_scalar_leaf")
    owner, relation, scalar_var = scalar[0]
    range_id = str(ontology.range_for_relation(relation))
    value_type = str(spec.get("value_type", "")).casefold()
    if not ((value_type == "number" and range_id in NUMERIC_TYPES)
            or (value_type in {"datetime", "date", "year"} and range_id == "type.datetime")):
        return None, _evidence("explicit_literal_and_scalar_datatype_disagree")
    # The leaf phrase must be real question evidence.  A path generated for
    # "age" cannot silently become a height condition just because both are
    # numeric.  Canonical morphology is generic and applies to every leaf.
    leaf = set(re.findall(r"[A-Za-z]+", relation.rsplit(".", 1)[-1].casefold()))
    words = set(re.findall(r"[A-Za-z]+", str(question).casefold()))
    def normalized(tokens: set[str]) -> set[str]:
        return {token[:-1] if len(token) > 4 and token.endswith("s") else token for token in tokens}
    if not leaf or not normalized(leaf) <= normalized(words):
        return None, _evidence("grounded_scalar_leaf_not_explicitly_named")
    if scalar_var == mapping[source_graph.answer_var] or any(a == scalar_var for a, _, _ in triples):
        return None, _evidence("scalar_leaf_is_answer_or_is_used_as_resource")
    operators = deepcopy(source_graph.operators)
    for op in operators:
        op["input_var"] = mapping.get(str(op.get("input_var", "")), str(op.get("input_var", "")))
        if isinstance(op.get("inputs"), list):
            op["inputs"] = [mapping.get(str(item), str(item)) for item in op["inputs"]]
    operator = {"type": str(spec["type"]).upper(), "inputs": [], "input_var": scalar_var,
                "attribute_relation_id": "", "attribute_relation_label": [],
                "attribute_relation_labels": [], "value": str(spec["value"]),
                "value_type": str(spec["value_type"]), "_attribute_value_already_bound": True}
    if range_id in NUMERIC_TYPES:
        operator["value"] = format(_number(spec["value"]), "f")
        operator.update({"_numeric_comparison_cast": True,
                         "_numeric_cast_kind": "float" if range_id == "type.float" else "integer"})
    elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(spec["value"])):
        if operator["type"] == "EQUAL":
            operator["_comparison_date_precision"] = "day"
        else:
            operator["_calendar_day_boundary"] = True
    elif re.fullmatch(r"\d{4}", str(spec["value"])):
        operator["_comparison_date_precision"] = "year"
    operators.append(operator)
    provenance = deepcopy(source_graph.provenance)
    provenance["budgeted_grounded_schema_repair"] = {
        "source_graph_id": source_graph.graph_id, "source_answer_var": source_graph.answer_var,
        "source_variable_mapping": mapping, "grounded_variable_mapping": names,
        "grounded_semantic": deepcopy(grounded.provenance),
        "unknown_role_boundaries": evidence["unknown_role_boundaries"],
        "constraint_owner_var": owner, "scalar_var": scalar_var, "scalar_relation_id": relation,
        "requires_existing_slot_replacement": True, "additional_model_calls": 0,
        "additional_endpoint_queries": 0, "uses_evaluation_gold": False,
    }
    result = QueryGraphCandidate(graph_id=graph_id, triples=deepcopy(triples),
                                 answer_var=mapping[source_graph.answer_var], operators=operators,
                                 score=float(grounded.score), compose_output=deepcopy(source_graph.compose_output),
                                 provenance=provenance)
    evidence.update({"status": "compiled", "reason": "complete_grounded_extension_with_preserved_answer_role",
                     "graph_id": graph_id, "source_graph_id": source_graph.graph_id,
                     "scalar_relation_id": relation, "constraint_owner_var": owner,
                     "requires_existing_slot_replacement": True})
    return result, evidence


def bind_existing_scalar_value(
    graph: QueryGraphCandidate, ontology: Any,
) -> tuple[QueryGraphCandidate, dict[str, Any]]:
    """Fix only a proven literal->attribute hop, using its exact bound edge.

    Different ordinary owner roles are not evidence.  An incoming edge with
    the very same scalar predicate is: the comparison value already exists.
    The answer projection, graph, threshold, and comparison are preserved.
    """
    evidence = _evidence("no_proven_literal_attribute_hop")
    if ontology is None:
        return graph, evidence
    repairs: list[dict[str, str]] = []
    operators = deepcopy(graph.operators)
    for index, operator in enumerate(operators):
        if not isinstance(operator, dict) or str(operator.get("type", "")).upper() not in COMPARISON_TYPES:
            continue
        relation = str(operator.get("attribute_relation_id", ""))
        variable = str(operator.get("input_var", ""))
        if not relation or not _VAR.fullmatch(variable) or str(ontology.range_for_relation(relation)) not in SCALAR_TYPES:
            continue
        exact = [(str(a), str(p), str(b)) for a, p, b in graph.triples
                 if str(p) == relation and str(b) == variable]
        if len(exact) != 1 or any(str(a) == variable for a, _, _ in graph.triples):
            continue
        operator["attribute_relation_id"] = ""
        operator["attribute_relation_label"] = []
        operator["attribute_relation_labels"] = []
        operator["_attribute_value_already_bound"] = True
        repairs.append({"operator_index": str(index), "scalar_var": variable,
                        "owner_var": exact[0][0], "relation_id": relation})
    if not repairs:
        return graph, evidence
    result = deepcopy(graph)
    result.operators = operators
    result.sparql = ""
    result.provenance["budgeted_existing_scalar_value_binding"] = repairs
    evidence.update({"status": "applied", "reason": "comparison_on_exact_already_bound_scalar",
                     "repairs": repairs, "graph_id": graph.graph_id})
    return result, evidence


def _words(value: Any) -> frozenset[str]:
    return frozenset(token[:-1] if len(token) > 4 and token.endswith("s") else token
                     for token in re.findall(r"[A-Za-z]+", str(value).casefold()))


def _explicit_question_head(question: str) -> frozenset[str]:
    match = re.match(
        r"^\s*(?:what\s+(?:is|are)|what['’]s|which)\s+(?:the\s+)?"
        r"([A-Za-z][A-Za-z\s-]*?)(?=\s*[,?]|\s+(?:of|that|where|whose|with|has|have|does|do|is|are)\b)",
        str(question), re.I,
    )
    return _words(match.group(1)) if match else frozenset()


def explicit_terminal_role_witness(
    grounded: GroundedSemanticCandidate, question: str,
    source_semantic: Mapping[str, Any], ontology: Any,
) -> dict[str, Any]:
    """Prove one explicit terminal property and, if needed, its entity leaf.

    The original Semantic variable/direction and question head must agree;
    grounded schema alone is not a role witness.  An anonymous relation
    record is never the answer merely because its type name contains the
    requested word.  Its unique field must produce a positively known topic
    type matching the complete requested head.
    """
    evidence = grounding_schema_evidence(grounded, ontology)
    if evidence.get("status") != "qualified":
        return evidence
    head = _explicit_question_head(question)
    if not head or _COMPARISON.search(str(question)):
        return _evidence("question_has_no_unambiguous_explicit_terminal_head")
    paths = grounded.compose_input.get("semantic_paths", [])
    original_paths = source_semantic.get("semantic_paths", []) if isinstance(source_semantic, Mapping) else []
    if len(paths) != 1 or len(original_paths) != 1:
        return _evidence("terminal_role_requires_one_original_and_grounded_path")
    path, original = paths[0], original_paths[0]
    steps, original_steps = path.get("steps", []), original.get("steps", [])
    if (not original_steps or len(steps) != len(original_steps)
            or path.get("id") != original.get("id")
            or path.get("anchor_ref") != original.get("anchor_ref")
            or path.get("path_output_var") != original.get("path_output_var")
            or any(tuple(step.get(key) for key in ("id", "from", "to", "direction"))
                   != tuple(prior.get(key) for key in ("id", "from", "to", "direction"))
                   for step, prior in zip(steps, original_steps))):
        return _evidence("original_semantic_variable_or_direction_proof_missing")
    # Explicitly named original anchors cannot disappear during grounding.
    # Extra model-invented anchors absent from the question are not promoted
    # into requirements.  This catches a missing contributor/person anchor.
    labels = {str(item.get("id", "")): str(item.get("surface", ""))
              for item in source_semantic.get("anchors", []) if isinstance(item, dict)}
    grounded_labels = {str(item.get("id", "")): str(item.get("surface", ""))
                       for item in grounded.compose_input.get("anchors", []) if isinstance(item, dict)}
    root_id = str(path.get("anchor_ref", ""))
    if (not labels.get(root_id) or " ".join(labels[root_id].casefold().split())
            != " ".join(grounded_labels.get(root_id, "").casefold().split())):
        return _evidence("original_semantic_anchor_identity_does_not_match_grounding")
    for identifier, label in labels.items():
        if label and label.casefold() in str(question).casefold() and identifier not in grounded.anchor_bindings:
            return _evidence("explicit_original_anchor_was_lost")
    terminal = steps[-1]
    prior_label = original_steps[-1].get("relation_label", [])
    relation = str(grounded.relation_bindings.get(str(terminal.get("id", "")), ""))
    if (not isinstance(prior_label, (list, tuple)) or not prior_label
            or _words(prior_label[-1]) != head or _words(relation.rsplit(".", 1)[-1]) != head
            or terminal.get("direction") != "forward"):
        return _evidence("question_original_semantic_and_grounded_property_do_not_agree")
    endpoint_type = str(ontology.range_for_relation(relation))
    expected_type, leaf = "", ""
    if endpoint_type not in SCALAR_TYPES and "common.topic" in set(ontology.supertypes(endpoint_type)):
        if _words(endpoint_type.rsplit(".", 1)[-1]) != head:
            return _evidence("terminal_entity_type_does_not_prove_question_head")
        expected_type = endpoint_type
    else:
        fields = []
        for predicate in ontology.relations_for_domain(endpoint_type):
            range_id = str(ontology.range_for_relation(predicate))
            if (range_id and range_id not in SCALAR_TYPES
                    and "common.topic" in set(ontology.supertypes(range_id))
                    and _words(range_id.rsplit(".", 1)[-1]) == head):
                fields.append((str(predicate), range_id))
        if len(fields) != 1:
            return _evidence("terminal_record_has_no_unique_entity_role_leaf")
        leaf, expected_type = fields[0]
    evidence.update({"status": "qualified", "reason": "independent_explicit_terminal_entity_role_proof",
                     "question_head_tokens": sorted(head), "original_path_id": str(original["id"]),
                     "original_answer_local_var": str(original["path_output_var"]),
                     "terminal_relation_id": relation, "terminal_range_type": endpoint_type,
                     "entity_leaf_relation_id": leaf, "expected_entity_type": expected_type,
                     "expected_entity_is_topic": True})
    return evidence


def _terminal_core(
    grounded: GroundedSemanticCandidate, witness: Mapping[str, Any],
) -> QueryGraphCandidate | None:
    triples, names = _compile_grounded(grounded)
    terminal = names.get(str(witness.get("original_answer_local_var", "")))
    if not triples or not terminal:
        return None
    return QueryGraphCandidate("grounded_role_core", triples, terminal, [], float(grounded.score), {})


def _redundant_terminal_echo(
    source: QueryGraphCandidate, core: QueryGraphCandidate,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """Prove a fresh existential fan-in is redundant, not a lost condition."""
    mapping = _unique_embedding(core, source.triples)
    if mapping is None or source.operators:
        return None, _evidence("source_does_not_have_an_operator_free_unique_grounded_core")
    core_nodes = set(mapping.values())
    embedded = Counter((mapping.get(str(a), str(a)), str(p), mapping.get(str(b), str(b)))
                       for a, p, b in core.triples)
    extra = list((Counter(map(tuple, source.triples)) - embedded).elements())
    if len(extra) != 1:
        return None, _evidence("source_has_unproven_nonterminal_constraints")
    left, relation, right = map(str, extra[0])
    source_answer = str(source.answer_var)
    if (left != source_answer or source_answer in core_nodes
            or right != mapping[core.answer_var]
            or not _VAR.fullmatch(source_answer)
            or sum(source_answer in (str(a), str(b)) for a, _, b in source.triples) != 1
            or not any(str(p) == relation and str(b) == right for _, p, b in embedded)):
        return None, _evidence("source_answer_is_not_a_proven_redundant_terminal_owner_echo")
    # Same predicate and same object already occur in the core.  With no
    # inequality or other use of the fresh subject it can bind to the existing
    # subject.  Removing this existential triple is a logical equivalence.
    return mapping, {"status": "qualified", "reason": "fresh_terminal_owner_echo_is_logically_redundant",
                     "redundant_triple": [left, relation, right],
                     "source_answer_var": source_answer, "core_variable_mapping": mapping}


def compile_grounded_terminal_role_repair(
    grounded: GroundedSemanticCandidate, source_graph: QueryGraphCandidate,
    question: str, source_semantic: Mapping[str, Any], ontology: Any, *, graph_id: str,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Repair a proven terminal owner echo, never directly project a CVT."""
    witness = explicit_terminal_role_witness(grounded, question, source_semantic, ontology)
    if witness.get("status") != "qualified":
        return None, witness
    core = _terminal_core(grounded, witness)
    if core is None:
        return None, _evidence("cannot_compile_complete_grounded_role_core")
    mapping, echo = _redundant_terminal_echo(source_graph, core)
    if mapping is None:
        return None, echo
    result = deepcopy(core)
    result.graph_id = graph_id
    result.compose_output = deepcopy(source_graph.compose_output)
    leaf = str(witness["entity_leaf_relation_id"])
    if leaf:
        variable = f"V{max(int(value[1:]) for triple in core.triples for value in (triple[0], triple[2]) if _VAR.fullmatch(str(value))) + 1}"
        result.triples.append([core.answer_var, leaf, variable])
        result.answer_var = variable
    result.provenance = deepcopy(source_graph.provenance)
    result.provenance["budgeted_grounded_terminal_role_repair"] = {
        "source_graph_id": source_graph.graph_id, "role_witness": deepcopy(witness),
        "redundant_owner_echo_proof": echo, "grounded_semantic": deepcopy(grounded.provenance),
        "requires_existing_slot_replacement": True, "additional_model_calls": 0,
        "additional_endpoint_queries": 0, "uses_evaluation_gold": False,
    }
    witness.update({"status": "compiled", "reason": "terminal_owner_echo_replaced_by_proven_entity_role",
                    "source_graph_id": source_graph.graph_id, "graph_id": graph_id,
                    "requires_existing_slot_replacement": True})
    return result, witness


def existing_grounded_terminal_role_challenger(
    grounded: GroundedSemanticCandidate, selected: ExecutedGraph,
    executions: Sequence[ExecutedGraph], question: str,
    source_semantic: Mapping[str, Any], ontology: Any,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Use one unique complete existing entity-role denotation, without queries."""
    proposed, evidence = compile_grounded_terminal_role_repair(
        grounded, selected.graph, question, source_semantic, ontology, graph_id="role_witness_only",
    )
    if proposed is None:
        return None, evidence
    core = _terminal_core(grounded, evidence)
    leaf = str(evidence["entity_leaf_relation_id"])
    inverse = set(ontology.reverse_for_relation(leaf)) if leaf else set()
    complete: list[ExecutedGraph] = []
    incomplete: list[str] = []
    for execution in executions:
        candidate = execution.graph
        if candidate is selected.graph or candidate.operators:
            continue
        mapping = _unique_embedding(core, candidate.triples)
        if mapping is None:
            continue
        embedded = Counter((mapping.get(str(a), str(a)), str(p), mapping.get(str(b), str(b)))
                           for a, p, b in core.triples)
        extra = list((Counter(map(tuple, candidate.triples)) - embedded).elements())
        if not leaf:
            matches = not extra and candidate.answer_var == mapping[core.answer_var]
        else:
            matches = (len(extra) == 1 and str(candidate.answer_var) not in mapping.values()
                       and (tuple(map(str, extra[0])) == (mapping[core.answer_var], leaf, str(candidate.answer_var))
                            or (str(extra[0][0]) == str(candidate.answer_var)
                                and str(extra[0][1]) in inverse
                                and str(extra[0][2]) == mapping[core.answer_var])))
        if not matches:
            continue
        if (execution.row_count != len(execution.answer_ids)
                or len(set(execution.answer_ids)) != len(execution.answer_ids)):
            incomplete.append(candidate.graph_id)
        elif execution.answer_ids:
            complete.append(execution)
    denotations = {frozenset(execution.answer_ids) for execution in complete}
    if incomplete or len(denotations) != 1:
        evidence.update({"status": "not_applied", "reason": "existing_entity_role_denotation_incomplete_or_ambiguous",
                         "complete_candidate_ids": [item.graph.graph_id for item in complete],
                         "incomplete_candidate_ids": incomplete})
        return None, evidence
    chosen = min(complete, key=lambda item: (-float(item.graph.score), item.graph.graph_id))
    evidence.update({"status": "applied", "reason": "unique_existing_explicit_terminal_entity_role_denotation",
                     "selected_graph_id": chosen.graph.graph_id, "additional_endpoint_queries": 0,
                     "uses_existing_executions_only": True})
    evidence.pop("requires_existing_slot_replacement", None)
    return chosen, evidence


def _regular_role_stems(value: Any) -> frozenset[str]:
    """Small regular verb/agent-noun morphology, not a relation routing list."""
    words = re.findall(r"[A-Za-z]+", str(value).casefold())
    if len(words) != 1:
        return frozenset()
    word = words[0]
    roots = {word}
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        roots.add(word[:-1])
    for current in tuple(roots):
        if len(current) > 5 and current.endswith("ed"):
            roots.add(current[:-2])
        if len(current) > 5 and current.endswith("er"):
            roots.add(current[:-2])
        if len(current) > 6 and current.endswith("or"):
            roots.add(current[:-2])
    for current in tuple(roots):
        if len(current) > 4 and current.endswith("e"):
            roots.add(current[:-1])
    return frozenset(roots)


def _one_explicit_scalar_spec(
    question: str, labels: Sequence[str], range_id: str,
) -> dict[str, str] | None:
    """Require one full literal across the anchor-masked question."""
    text = str(question)
    for label in sorted(set(labels), key=len, reverse=True):
        if label:
            text = re.sub(re.escape(label), " ", text, flags=re.I)
    comparisons = list(_COMPARISON.finditer(text))
    if len(comparisons) != 1:
        return None
    cue = comparisons[0].group(0).casefold()
    if cue in {"at most", "no more than"}:
        operator = "LESS_OR_EQUAL"
    elif cue in {"at least", "no less than"}:
        operator = "GREATER_OR_EQUAL"
    elif cue in {"less than", "fewer than", "lower than", "smaller than", "prior to", "under", "below", "before"}:
        operator = "LESS_THAN"
    else:
        operator = "GREATER_THAN"
    dates = _date_literals(text)
    if range_id == "type.datetime":
        remaining = _NAMED_DATE.sub(" ", _ISO_DATE.sub(" ", text))
        if len(dates) != 1 or re.search(r"\d", remaining):
            return None
        spec = {"type": operator, "value": dates[0], "value_type": "dateTime"}
    elif range_id in NUMERIC_TYPES:
        numbers = re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w.])", text)
        if dates or len(numbers) != 1 or _number(numbers[0]) is None:
            return None
        spec = {"type": operator, "value": format(_number(numbers[0]), "f"), "value_type": "number"}
    else:
        return None
    return spec if _explicit_spec(question, spec, labels) else None


def _resolved_record_answer_witness(
    graph: QueryGraphCandidate, anchor: str, ontology: Any,
) -> dict[str, Any] | None:
    """Prove WHO projects a literal or an endpoint-resolved mediator.

    Different ordinary entity roles are not considered incompatible here.
    No absence of common.topic inheritance is promoted to a CVT proof.
    """
    edges = [(str(a), str(p), str(b)) for a, p, b in graph.triples
             if str(a) == anchor and str(b) == str(graph.answer_var)]
    if len(edges) != 1:
        return None
    _, relation, variable = edges[0]
    range_id = str(ontology.range_for_relation(relation))
    if range_id in SCALAR_TYPES:
        return {"kind": "literal_answer_to_entity_who", "relation_id": relation,
                "answer_var": variable, "range_type": range_id}
    grounding = graph.provenance.get("grounded_semantic", {})
    hops = grounding.get("hops", []) if isinstance(grounding, dict) else []
    for path in hops if isinstance(hops, list) else []:
        for hop in path if isinstance(path, list) else []:
            if (isinstance(hop, dict) and str(hop.get("relation_id", "")) == relation
                    and hop.get("direction") == "forward"
                    and hop.get("reaches_cvt") is True
                    and hop.get("cvt_detection_status") == "resolved"):
                return {"kind": "endpoint_resolved_record_answer_to_entity_who",
                        "relation_id": relation, "answer_var": variable,
                        "range_type": range_id, "cvt_detection_status": "resolved"}
    return None


def budgeted_repair_source_witness(
    question: str, selected: ExecutedGraph, executions: Sequence[ExecutedGraph],
    grounded: GroundedSemanticCandidate, source_semantic: Mapping[str, Any], ontology: Any,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Qualify an existing primary-role source, not a repaired denotation.

    This permits considering a complete grounded constraint path when the
    incumbent is not a subset of its primary-role source.  It requires a
    direct regular active WHO clause, independently aligned original Semantic
    variables/anchor, a uniquely grounded primary entity relation, and proof
    that the incumbent omits that relation while returning a literal or an
    endpoint-resolved record.  Ordinary role differences alone abstain.

    A successful return authorizes neither a query nor final selection.  The
    caller must have saved count/time credit and execute only within it.  Any
    resulting repair must be a nonempty *strict subset* of this real source
    execution, preserve its entity answer role and enforce the complete
    normalized scalar obligation.  Disjointness from the incumbent alone is
    never positive evidence.
    """
    evidence = grounding_schema_evidence(grounded, ontology)
    if evidence.get("status") != "qualified":
        return None, evidence
    paths = grounded.compose_input.get("semantic_paths", [])
    originals = source_semantic.get("semantic_paths", []) if isinstance(source_semantic, Mapping) else []
    if len(paths) != 1 or len(originals) != 1:
        return None, _evidence("primary_role_requires_one_original_and_grounded_path")
    path, original = paths[0], originals[0]
    steps, prior = path.get("steps", []), original.get("steps", [])
    if (not prior or len(steps) != len(prior) or len(steps) < 2
            or any(path.get(key) != original.get(key) for key in ("id", "anchor_ref", "path_output_var"))
            or any(tuple(step.get(key) for key in ("id", "from", "to", "direction"))
                   != tuple(previous.get(key) for key in ("id", "from", "to", "direction"))
                   for step, previous in zip(steps, prior))):
        return None, _evidence("original_primary_role_variable_or_direction_proof_missing")
    anchor_ref = str(path.get("anchor_ref", ""))
    original_labels = {str(item.get("id", "")): str(item.get("surface", ""))
                       for item in source_semantic.get("anchors", []) if isinstance(item, dict)}
    actual_labels = {str(item.get("id", "")): str(item.get("surface", ""))
                     for item in grounded.compose_input.get("anchors", []) if isinstance(item, dict)}
    label = " ".join(original_labels.get(anchor_ref, "").split())
    if (not label or label.casefold() != " ".join(actual_labels.get(anchor_ref, "").split()).casefold()
            or anchor_ref not in grounded.anchor_bindings):
        return None, _evidence("original_primary_anchor_identity_does_not_match_grounding")
    text = " ".join(str(question).split())
    clause = re.match(r"^Who\s+([A-Za-z]+)\s+(?:the\s+)?[\"“]?" + re.escape(label)
                      + r"[\"”]?(?=\s*[,?]|\s+(?:who|which|that|whose|with|after|before|from|in|on)\b|$)", text, re.I)
    if not clause or not re.fullmatch(r"[A-Za-z]+(?:ed|s)", clause.group(1)):
        return None, _evidence("question_has_no_unambiguous_direct_regular_active_who_clause")
    verb = clause.group(1).casefold()
    if verb in {"is", "was", "does", "has"}:
        return None, _evidence("question_verb_is_an_auxiliary_not_a_primary_role")
    if not verb.endswith("ed") and verb.endswith(("ers", "ors")):
        return None, _evidence("question_primary_token_could_be_a_plural_agent_noun")
    for identifier, surface in original_labels.items():
        if surface and surface.casefold() in text.casefold() and identifier not in grounded.anchor_bindings:
            return None, _evidence("explicit_original_anchor_was_lost")
    first = steps[0]
    relation = str(grounded.relation_bindings.get(str(first.get("id", "")), ""))
    prior_label = prior[0].get("relation_label", [])
    stems = _regular_role_stems(verb)
    if (first.get("direction") != "forward" or first.get("from") != anchor_ref
            or not isinstance(prior_label, (list, tuple)) or not prior_label
            or not stems & _regular_role_stems(prior_label[-1])
            or not stems & _regular_role_stems(relation.rsplit(".", 1)[-1])):
        return None, _evidence("question_verb_original_primary_role_and_grounded_relation_disagree")
    role_type = str(ontology.range_for_relation(relation))
    if role_type in SCALAR_TYPES or "common.topic" not in set(ontology.supertypes(role_type)):
        return None, _evidence("primary_role_has_no_positive_entity_answer_schema_proof")
    anchor = str(grounded.anchor_bindings[anchor_ref].entity_id)
    selected_anchors = {str(term) for triple in selected.graph.triples for term in (triple[0], triple[2])
                        if not _VAR.fullmatch(str(term))}
    if selected_anchors != {anchor}:
        return None, _evidence("incumbent_entity_linking_or_anchor_scope_differs")
    inverse = set(ontology.reverse_for_relation(relation))
    if any((str(a) == anchor and str(p) == relation)
           or (str(b) == anchor and str(p) in inverse)
           for a, p, b in selected.graph.triples):
        return None, _evidence("incumbent_already_preserves_the_explicit_primary_relation")
    incumbent = _resolved_record_answer_witness(selected.graph, anchor, ontology)
    if incumbent is None or stems & _regular_role_stems(incumbent["relation_id"].rsplit(".", 1)[-1]):
        return None, _evidence("incumbent_primary_role_or_entity_answer_contradiction_not_proven")
    scalar_steps = [(step, str(grounded.relation_bindings.get(str(step.get("id", "")), "")))
                    for step in steps if step.get("direction") == "forward"
                    and str(ontology.range_for_relation(str(grounded.relation_bindings.get(str(step.get("id", "")), "")))) in SCALAR_TYPES]
    if len(scalar_steps) != 1:
        return None, _evidence("primary_role_repair_requires_one_complete_grounded_scalar_obligation")
    constraint = _one_explicit_scalar_spec(text, [*original_labels.values(), *actual_labels.values()],
                                         str(ontology.range_for_relation(scalar_steps[0][1])))
    if constraint is None:
        return None, _evidence("scalar_obligation_is_ambiguous_invalid_or_not_full_precision")
    qualified: list[ExecutedGraph] = []
    incomplete: list[str] = []
    for execution in executions:
        graph = execution.graph
        if (len(graph.triples) != 1 or graph.operators
                or [str(part) for part in graph.triples[0]] != [anchor, relation, str(graph.answer_var)]):
            continue
        answer_labels = {str(item.get("id", "")): str(item.get("label", "")).strip()
                         for item in execution.answers if isinstance(item, dict)}
        if (not execution.answer_ids or execution.row_count != len(execution.answer_ids)
                or len(set(execution.answer_ids)) != len(execution.answer_ids)
                or any(not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", str(identifier))
                       or not answer_labels.get(str(identifier)) for identifier in execution.answer_ids)):
            incomplete.append(graph.graph_id)
            continue
        compiled, compilation = compile_grounded_constraint_repair(
            grounded, graph, text, constraint, ontology, graph_id="source_witness_only",
        )
        if compiled is not None:
            qualified.append(execution)
    if incomplete or len({frozenset(execution.answer_ids) for execution in qualified}) != 1:
        evidence.update({"status": "not_applied", "reason": "primary_role_source_execution_incomplete_or_ambiguous",
                         "qualified_source_graph_ids": [item.graph.graph_id for item in qualified],
                         "incomplete_source_graph_ids": incomplete})
        return None, evidence
    chosen = min(qualified, key=lambda item: (-float(item.graph.score), item.graph.graph_id))
    evidence.update({"status": "qualified", "reason": "explicit_primary_entity_role_source_with_proven_incumbent_obligation_loss",
                     "source_graph_id": chosen.graph.graph_id, "source_answer_count": len(chosen.answer_ids),
                     "source_answer_var": chosen.graph.answer_var, "question_active_verb": verb,
                     "primary_relation_id": relation, "primary_entity_type": role_type,
                     "original_primary_local_var": str(first.get("to", "")),
                     "incumbent_answer_kind_witness": incumbent,
                     "incumbent_omits_required_primary_relation_and_its_inverse": True,
                     "normalized_scalar_spec": constraint,
                     "requires_saved_execution_count_and_time_credit": True,
                     "required_repair_denotation": "nonempty_strict_subset_of_actual_source_answer_ids",
                     "incumbent_disjointness_used_as_positive_evidence": False,
                     "query_executed": False, "new_repair_denotation_verified": False})
    return chosen, evidence


def _same_schema_property(label: Any, relation: str, ontology: Any) -> bool:
    """Recognize canonical morphology and a positively proved owner subtype."""
    if not isinstance(label, (list, tuple)) or len(label) < 3:
        return False
    original = relation_id_from_label(label)
    if original == relation:
        return True
    if _words(label[-1]) != _words(relation.rsplit(".", 1)[-1]):
        return False
    requested_domain = relation_id_from_label(label[:-1])
    actual_domain = str(ontology.domain_for_relation(relation))
    return bool(requested_domain and actual_domain
                and schema_type_overlap(requested_domain, actual_domain, ontology) == "compatible")


def _native_scalar_obligation(
    question: str, labels: Sequence[str], range_id: str,
) -> dict[str, str] | None:
    comparison = _one_explicit_scalar_spec(question, labels, range_id)
    if comparison is not None:
        return comparison
    text = str(question)
    for label in sorted(set(labels), key=len, reverse=True):
        if label:
            text = re.sub(re.escape(label), " ", text, flags=re.I)
    if _COMPARISON.search(text):
        return None
    if re.search(r"\b(?:less|fewer|lower|smaller|greater|more|higher|larger|later|earlier)\b", text, re.I):
        return None
    dates = _date_literals(text)
    if range_id == "type.datetime":
        remaining = _NAMED_DATE.sub(" ", _ISO_DATE.sub(" ", text))
        if len(dates) != 1 or re.search(r"\d", remaining):
            return None
        spec = {"type": "EQUAL", "value": dates[0], "value_type": "dateTime"}
        # Explicit "on" is equality, not merely an arbitrary number/date
        # mentioned elsewhere in a question.  Comparison cues were excluded.
        if not re.search(r"\bon\s+(?:[A-Za-z]+\s+)?\d", text, re.I):
            return None
    elif range_id in NUMERIC_TYPES:
        values = re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w.])", text)
        if dates or len(values) != 1 or _number(values[0]) is None:
            return None
        spec = {"type": "EQUAL", "value": format(_number(values[0]), "f"), "value_type": "number"}
    else:
        return None
    return spec if _explicit_spec(question, spec, labels) else None


def _validated_provided_scalar_spec(
    question: str, labels: Sequence[str],
    provider: Callable[[str], Mapping[str, Any] | None],
) -> dict[str, Any] | None:
    text = str(question)
    for label in sorted(set(labels), key=len, reverse=True):
        if label:
            text = re.sub(re.escape(label), " ", text, flags=re.I)
    proposed = provider(text)
    if not isinstance(proposed, Mapping):
        return None
    if (not _COMPARISON.search(text)
            and re.search(r"\b(?:less|fewer|lower|smaller|greater|more|higher|larger|later|earlier)\b", text, re.I)):
        return None
    if _explicit_spec(question, proposed, labels):
        return dict(proposed)
    if proposed.get("_normalization_scope") != "internal_post_model_constraint_spec_only":
        return None
    # Independently recheck the strict local word parser's literal and cue;
    # a caller-supplied number is not accepted merely because it has a tag.
    from .comparison_number_normalization import normalize_comparison_constraint_spec

    verified = normalize_comparison_constraint_spec(text)
    if (not isinstance(verified, Mapping)
            or any(proposed.get(key) != verified.get(key) for key in ("type", "value", "value_type", "_source_literal", "_normalization_kind"))):
        return None
    return dict(verified)


def _normal_anchor_scope(
    grounded: GroundedSemanticCandidate, source_semantic: Mapping[str, Any],
    question: str, linked_entity_context: Sequence[Mapping[str, Any]] | None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Separate path roots, unreferenced literals, schema kinds and topics."""
    originals = source_semantic.get("semantic_paths", [])
    referenced = {str(p.get("anchor_ref", "")) for p in originals if isinstance(p, dict)}
    original_labels = {str(a.get("id", "")): str(a.get("surface", ""))
                       for a in source_semantic.get("anchors", []) if isinstance(a, dict)}
    actual_labels = {str(a.get("id", "")): str(a.get("surface", ""))
                     for a in grounded.compose_input.get("anchors", []) if isinstance(a, dict)}
    masked_labels: list[str] = []
    ignored: list[dict[str, str]] = []
    for identifier, surface in original_labels.items():
        literal = bool(_date_literals(surface) or _number(surface) is not None)
        if identifier in referenced:
            if (literal or identifier not in grounded.anchor_bindings
                    or surface.casefold() != actual_labels.get(identifier, "").casefold()):
                return False, [], {"reason": "referenced_original_topic_anchor_missing_or_literal", "anchor_id": identifier}
            masked_labels.append(surface)
            continue
        if not surface or not re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", str(question), re.I):
            continue
        if literal:
            ignored.append({"anchor_id": identifier, "reason": "unreferenced_literal_is_question_constraint_not_topic"})
            continue
        matches = [item for item in linked_entity_context or [] if isinstance(item, Mapping)
                   and str(item.get("name", item.get("label", ""))).casefold() == surface.casefold()]
        if not matches:
            return False, [], {"reason": "unreferenced_question_anchor_topic_kind_unknown", "anchor_id": identifier}
        meta = {"type", "property", "domain", "namespace", "type type", "type property", "type domain", "type namespace"}
        statuses = []
        for item in matches:
            tags = {" ".join(re.findall(r"[A-Za-z]+", str(tag).casefold())) for tag in item.get("types", [])}
            identifier_value = str(item.get("id", ""))
            statuses.append("schema_kind" if tags & meta else "named_topic" if tags and re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", identifier_value) else "unknown")
        if set(statuses) == {"schema_kind"}:
            ignored.append({"anchor_id": identifier, "reason": "existing_linker_proves_unreferenced_schema_kind_not_named_topic"})
        else:
            return False, [], {"reason": "unreferenced_explicit_topic_or_ambiguous_kind_is_unproved_scope", "anchor_id": identifier}
    return True, masked_labels, {"reason": "path_roots_and_existing_literal_or_schema_kind_evidence_agree", "ignored_anchors": ignored}


def aligned_scalar_grounding_witness(
    grounded: GroundedSemanticCandidate, source_semantic: Mapping[str, Any],
    question: str, ontology: Any, *,
    spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
    linked_entity_context: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Qualify complete normal groundings without changing any model beam.

    Unlike the open-world rejected-role channel, this channel requires every
    adjacent owner to have positive schema compatibility.  All original
    variable/direction references and actual grounded properties must agree.
    No missing attribute, unknown owner or dropped Semantic branch is filled
    by guessing.  The optional pure provider is for already-normalized local
    surface specifications, not an extra model/embedding lookup.
    """
    evidence = _evidence("normal_scalar_alignment_unavailable")
    paths = grounded.compose_input.get("semantic_paths", [])
    originals = source_semantic.get("semantic_paths", []) if isinstance(source_semantic, Mapping) else []
    if (ontology is None or len(paths) != 1 or len(originals) != 1
            or grounded.provenance.get("dropped_paths")):
        return evidence
    path, original = paths[0], originals[0]
    steps, prior = path.get("steps", []), original.get("steps", [])
    if (len(steps) < 2 or len(steps) != len(prior)
            or any(path.get(key) != original.get(key) for key in ("id", "anchor_ref", "path_output_var"))
            or any(tuple(step.get(key) for key in ("id", "from", "to", "direction"))
                   != tuple(previous.get(key) for key in ("id", "from", "to", "direction"))
                   for step, previous in zip(steps, prior))):
        return _evidence("normal_scalar_original_variables_or_directions_do_not_align")
    root = str(path.get("anchor_ref", ""))
    scope_ok, labels, scope = _normal_anchor_scope(grounded, source_semantic, question, linked_entity_context)
    if not scope_ok:
        return {**_evidence("normal_scalar_original_anchor_identity_or_coverage_missing"), "anchor_scope_evidence": scope}
    types: list[tuple[str, str]] = []
    relations: list[str] = []
    for step, previous in zip(steps, prior):
        relation = str(grounded.relation_bindings.get(str(step.get("id", "")), ""))
        domain, range_id = str(ontology.domain_for_relation(relation)), str(ontology.range_for_relation(relation))
        if not relation or not domain or not range_id or not _same_schema_property(previous.get("relation_label"), relation, ontology):
            return _evidence("normal_scalar_actual_property_not_proved_by_original_semantic")
        if step.get("direction") == "forward":
            types.append((domain, range_id))
        elif step.get("direction") == "backward":
            types.append((range_id, domain))
        else:
            return _evidence("normal_scalar_invalid_grounded_direction")
        relations.append(relation)
    if any(schema_type_overlap(left[1], right[0], ontology) != "compatible"
           for left, right in zip(types, types[1:])):
        return _evidence("normal_scalar_owner_schema_not_positively_compatible")
    scalar = [index for index, (_, out) in enumerate(types) if out in SCALAR_TYPES]
    if scalar != [len(steps) - 1] or steps[-1].get("direction") != "forward":
        return _evidence("normal_scalar_requires_one_actual_terminal_scalar_edge")
    constraint = _native_scalar_obligation(question, labels, types[-1][1])
    if constraint is None and spec_provider is not None:
        constraint = _validated_provided_scalar_spec(question, labels, spec_provider)
    if constraint is None:
        return _evidence("normal_scalar_question_obligation_ambiguous_or_unsupported")
    value_type = str(constraint.get("value_type", "")).casefold()
    if not ((types[-1][1] in NUMERIC_TYPES and value_type == "number")
            or (types[-1][1] == "type.datetime" and value_type in {"date", "datetime", "year"})):
        return _evidence("normal_scalar_normalized_literal_and_actual_range_disagree")
    # The actual field was independently proposed by the original Semantic
    # path and has a positively typed owner.  A full calendar-day "on" request
    # on its sole temporal leaf does not require a fuzzy lexical rewrite of
    # the already-proposed field (e.g. the question may spell adopted adapted).
    evidence.update({"status": "qualified", "reason": "complete_original_aligned_scalar_grounding",
                     "path_id": str(path["id"]), "root_anchor_id": root,
                     "actual_relations": relations,
                     "scalar_relation_id": relations[-1], "scalar_range_type": types[-1][1],
                     "scalar_owner_local_var": str(steps[-1]["from"]),
                     "scalar_value_local_var": str(steps[-1]["to"]),
                     "normalized_scalar_spec": constraint,
                     "owner_schema_all_positively_compatible": True,
                     "anchor_scope_evidence": scope,
                     "requires_internal_memory_only_not_model_beam_insertion": True})
    return evidence


def capture_complete_scalar_grounding(
    grounded: GroundedSemanticCandidate, source_semantic: Mapping[str, Any],
    question: str, ontology: Any, *,
    spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
    linked_entity_context: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[bool, dict[str, Any]]:
    evidence = aligned_scalar_grounding_witness(
        grounded, source_semantic, question, ontology, spec_provider=spec_provider,
        linked_entity_context=linked_entity_context,
    )
    return evidence.get("status") == "qualified", evidence


def compile_aligned_scalar_grounding_repair(
    grounded: GroundedSemanticCandidate, source_graph: QueryGraphCandidate,
    question: str, source_semantic: Mapping[str, Any], ontology: Any, *, graph_id: str,
    spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
    linked_entity_context: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Compile a full normal grounding, preserving an existing entity prefix."""
    evidence = aligned_scalar_grounding_witness(
        grounded, source_semantic, question, ontology, spec_provider=spec_provider,
        linked_entity_context=linked_entity_context,
    )
    if evidence.get("status") != "qualified":
        return None, evidence
    if (not source_graph.triples or any(not isinstance(op, dict) or str(op.get("type", "")).upper() != "NO_EQUAL"
                                       for op in source_graph.operators)):
        return None, _evidence("normal_scalar_source_already_has_substantive_conditions")
    triples, names = _compile_grounded(grounded)
    mapping = _unique_embedding(source_graph, triples)
    if mapping is None:
        return None, _evidence("normal_scalar_source_answer_role_not_uniquely_embedded")
    source = Counter((mapping.get(str(a), str(a)), str(p), mapping.get(str(b), str(b)))
                     for a, p, b in source_graph.triples)
    actual = Counter(map(tuple, triples))
    if not source < actual:
        return None, _evidence("normal_scalar_not_a_strict_complete_grounded_extension")
    actual_anchors = {str(value) for t in triples for value in (t[0], t[2]) if not _VAR.fullmatch(str(value))}
    source_anchors = {str(value) for t in source_graph.triples for value in (t[0], t[2]) if not _VAR.fullmatch(str(value))}
    if source_anchors != actual_anchors:
        return None, _evidence("normal_scalar_source_anchors_differ")
    answer = mapping[source_graph.answer_var]
    scalar_var = names[evidence["scalar_value_local_var"]]
    if answer == scalar_var:
        return None, _evidence("normal_scalar_source_projects_the_scalar_not_the_entity_role")
    constraint = evidence["normalized_scalar_spec"]
    operator = {"type": constraint["type"], "inputs": [], "input_var": scalar_var,
                "attribute_relation_id": "", "attribute_relation_label": [], "attribute_relation_labels": [],
                "value": str(constraint["value"]), "value_type": str(constraint["value_type"]),
                "_attribute_value_already_bound": True}
    if evidence["scalar_range_type"] in NUMERIC_TYPES:
        operator["_numeric_cast_kind"] = "float" if evidence["scalar_range_type"] == "type.float" else "integer"
        if operator["type"] == "EQUAL":
            operator["_numeric_cast_compare"] = True
        else:
            operator["_numeric_comparison_cast"] = True
    elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", operator["value"]):
        if operator["type"] == "EQUAL":
            operator["_comparison_date_precision"] = "day"
        else:
            operator["_calendar_day_boundary"] = True
    operators = deepcopy(source_graph.operators)
    for op in operators:
        op["input_var"] = mapping.get(str(op.get("input_var", "")), str(op.get("input_var", "")))
        if isinstance(op.get("inputs"), list):
            op["inputs"] = [mapping.get(str(value), str(value)) for value in op["inputs"]]
    operators.append(operator)
    result = QueryGraphCandidate(graph_id, triples, answer, operators, float(grounded.score), deepcopy(source_graph.compose_output),
                                 deepcopy(source_graph.provenance))
    result.provenance["budgeted_aligned_scalar_grounding_repair"] = {
        "source_graph_id": source_graph.graph_id, "source_variable_mapping": mapping,
        "grounded_variable_mapping": names, "original_aligned_scalar_witness": deepcopy(evidence),
        "requires_saved_execution_count_and_time_credit": True,
        "additional_model_calls": 0, "additional_endpoint_queries": 0,
        "uses_evaluation_gold": False,
    }
    evidence.update({"status": "compiled", "reason": "complete_normal_grounded_scalar_extension",
                     "source_graph_id": source_graph.graph_id, "graph_id": graph_id,
                     "requires_saved_execution_count_and_time_credit": True,
                     "new_repair_denotation_verified": False})
    return result, evidence


def _scalar_property_surface_supported(question: str, relation: str) -> bool:
    leaf = relation.rsplit(".", 1)[-1].casefold()
    words = _words(leaf)
    if words and words <= _words(question):
        return True
    if leaf in {"from", "from_date", "start", "start_date", "begin", "begin_date"}:
        return bool(re.search(r"\b(?:began|begin|begun|starts?|started|starting|since|from)\b", question, re.I)
                    and not re.search(r"\b(?:ended|ending|finished|out\s+of\s+office)\b", question, re.I))
    if leaf in {"to", "to_date", "end", "end_date", "finish", "finish_date"}:
        return bool(re.search(r"\b(?:ends?|ended|ending|finish|finished|until|out\s+of\s+office)\b", question, re.I)
                    and not re.search(r"\b(?:began|begin|begun|starts?|started|starting|since)\b", question, re.I))
    generic = {"date", "time", "initial", "value", "number", "total", "maximum", "minimum"}
    roots = set().union(*(_regular_role_stems(word) for word in words - generic)) if words - generic else set()
    question_roots = set().union(*(_regular_role_stems(word) for word in _words(question)))
    return bool(roots & question_roots)


def compile_saved_compose_scalar_extension(
    grounded: GroundedSemanticCandidate, compose_output: Mapping[str, Any],
    source_graph: QueryGraphCandidate, question: str, ontology: Any, *, graph_id: str,
    spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Reuse a real Compose scalar suffix that lowering could not transfer.

    All grounded and source edges must occur exactly in the already-produced
    Compose graph.  Extra edges must be a single short forward scalar chain
    with positively compatible schema owners, attached to an existing node.
    The returned entity role is the source role, or an explicit official
    inverse owner of the same relation record; in the latter case the caller
    still has to prove the new IDs are a strict subset of the actual source.
    No relation is discovered or guessed and no new model input is produced.
    """
    evidence = _evidence("saved_compose_scalar_extension_unqualified")
    if (ontology is None or grounded.provenance.get("dropped_paths")
            or len(grounded.compose_input.get("semantic_paths", [])) != 1
            or not source_graph.triples
            or any(not isinstance(op, dict) or str(op.get("type", "")).upper() != "NO_EQUAL"
                   for op in source_graph.operators)):
        return None, evidence
    entities = compose_output.get("entities", []) if isinstance(compose_output, Mapping) else []
    raw_triples = compose_output.get("triples", []) if isinstance(compose_output, Mapping) else []
    if not isinstance(entities, list) or not isinstance(raw_triples, list) or not raw_triples:
        return None, evidence
    binding_by_surface = {str(value.label).casefold(): str(value.entity_id)
                          for value in grounded.anchor_bindings.values()}
    for anchor in grounded.compose_input.get("anchors", []):
        if str(anchor.get("id", "")) in grounded.anchor_bindings:
            binding_by_surface[str(anchor.get("surface", "")).casefold()] = grounded.anchor_bindings[str(anchor["id"])].entity_id
    bindings: dict[str, str] = {}
    for entity in entities:
        if not isinstance(entity, dict) or str(entity.get("surface", "")).casefold() not in binding_by_surface:
            return None, _evidence("saved_compose_has_unproved_new_entity_or_literal_anchor")
        bindings[str(entity.get("id", ""))] = binding_by_surface[str(entity["surface"]).casefold()]
    triples: list[list[str]] = []
    for raw in raw_triples:
        if not isinstance(raw, dict):
            return None, evidence
        a, b = str(raw.get("subject", "")), str(raw.get("object", ""))
        if any(node not in bindings and not _VAR.fullmatch(node) for node in (a, b)):
            return None, _evidence("saved_compose_references_unproved_node")
        relation = relation_id_from_label(raw.get("relation_label", []))
        if not relation or not ontology.domain_for_relation(relation) or not ontology.range_for_relation(relation):
            return None, _evidence("saved_compose_property_not_in_actual_schema")
        triples.append([bindings.get(a, a), relation, bindings.get(b, b)])
    node_types: dict[str, set[str]] = {}
    for a, p, b in triples:
        node_types.setdefault(a, set()).add(str(ontology.domain_for_relation(p)))
        node_types.setdefault(b, set()).add(str(ontology.range_for_relation(p)))
    if any(types & SCALAR_TYPES and types - SCALAR_TYPES for types in node_types.values()):
        return None, _evidence("saved_compose_has_proven_literal_resource_schema_conflict")
    answer = str(compose_output.get("answer_var", ""))
    if not _VAR.fullmatch(answer):
        return None, _evidence("saved_compose_has_no_entity_answer_variable")
    core_triples, core_names = _compile_grounded(grounded)
    if not core_triples:
        return None, _evidence("saved_compose_grounded_core_incomplete")
    core = QueryGraphCandidate("actual_grounded_core", core_triples,
                               core_names[str(grounded.compose_input["semantic_paths"][0]["path_output_var"])], [], grounded.score, {})
    core_mapping = _unique_embedding(core, triples)
    source_mapping = _unique_embedding(source_graph, triples)
    if core_mapping is None or source_mapping is None:
        return None, _evidence("saved_compose_does_not_uniquely_preserve_all_actual_grounded_and_source_edges")
    source_anchors = {str(v) for t in source_graph.triples for v in (t[0], t[2]) if not _VAR.fullmatch(str(v))}
    target_anchors = {str(v) for t in triples for v in (t[0], t[2]) if not _VAR.fullmatch(str(v))}
    if not source_anchors or source_anchors != target_anchors:
        return None, _evidence("saved_compose_anchor_scope_differs_from_actual_source")
    embedded_core = Counter((core_mapping.get(str(a), str(a)), str(p), core_mapping.get(str(b), str(b)))
                            for a, p, b in core.triples)
    extra = list((Counter(map(tuple, triples)) - embedded_core).elements())
    scalar = [t for t in extra if str(ontology.range_for_relation(str(t[1]))) in SCALAR_TYPES]
    if len(scalar) != 1 or len(extra) not in {1, 2}:
        return None, _evidence("saved_compose_extras_are_not_one_short_scalar_chain")
    owner, attribute, scalar_var = map(str, scalar[0])
    if scalar_var == answer or any(str(a) == scalar_var for a, _, _ in triples):
        return None, _evidence("saved_compose_scalar_is_answer_or_resource")
    core_nodes = {str(v) for t in embedded_core for v in (t[0], t[2])}
    current = owner
    if len(extra) == 2:
        parents = [(str(a), str(p), str(b)) for a, p, b in extra if str(b) == owner and str(p) != attribute]
        if len(parents) != 1 or parents[0][0] not in core_nodes:
            return None, _evidence("saved_compose_scalar_chain_has_no_unique_existing_attachment")
        independent_context = str(question) + " " + " ".join(str(p.get("goal", "")) for p in grounded.compose_input.get("semantic_paths", []))
        if not _scalar_property_surface_supported(independent_context, parents[0][1]):
            return None, _evidence("saved_compose_added_record_relation_not_supported_by_question_or_original_goal")
        current = parents[0][0]
        parent_type = str(ontology.range_for_relation(parents[0][1]))
        if parent_type != str(ontology.domain_for_relation(attribute)):
            return None, _evidence("saved_compose_scalar_record_owner_not_exactly_typed")
        leading = str(ontology.domain_for_relation(parents[0][1]))
    elif owner not in core_nodes:
        return None, _evidence("saved_compose_scalar_owner_is_not_an_existing_grounded_node")
    else:
        leading = str(ontology.domain_for_relation(attribute))
    existing_types = {str(ontology.domain_for_relation(str(p))) for a, p, _ in embedded_core if str(a) == current}
    existing_types.update(str(ontology.range_for_relation(str(p))) for _, p, b in embedded_core if str(b) == current)
    if not existing_types or not any(schema_type_overlap(t, leading, ontology) == "compatible" for t in existing_types):
        return None, _evidence("saved_compose_new_attribute_owner_not_positively_compatible")
    source_answer = source_mapping[source_graph.answer_var]
    role_bridge: dict[str, Any] | None = None
    if answer != source_answer:
        bridges = [(str(a), str(p), str(b), str(q))
                   for a, p, b in embedded_core for x, q, y in embedded_core
                   if str(a) == source_answer and str(b) == str(x) and str(y) == answer
                   and str(q) in set(ontology.reverse_for_relation(str(p)))]
        if len(bridges) != 1 or owner != bridges[0][2]:
            return None, _evidence("saved_compose_answer_role_not_preserved_or_proved_by_exact_inverse_owner")
        role_bridge = {"source_entity_var": source_answer, "relation_record_var": bridges[0][2],
                       "returned_entity_var": answer, "relation_id": bridges[0][1], "inverse_relation_id": bridges[0][3]}
    answer_types = {str(ontology.range_for_relation(str(p))) for _, p, b in triples if str(b) == answer}
    answer_types.update(str(ontology.domain_for_relation(str(p))) for a, p, _ in triples if str(a) == answer)
    if not any(t not in SCALAR_TYPES and "common.topic" in set(ontology.supertypes(t)) for t in answer_types):
        return None, _evidence("saved_compose_returned_role_is_not_a_positively_known_topic_entity")
    if not _scalar_property_surface_supported(question, attribute):
        return None, _evidence("saved_compose_scalar_property_or_temporal_phase_not_supported_by_question")
    if attribute.rsplit(".", 1)[-1].casefold() in {
        "from", "from_date", "start", "start_date", "begin", "begin_date",
        "to", "to_date", "end", "end_date", "finish", "finish_date",
    }:
        # A correct end cue is not enough to switch from an office tenure to
        # a party membership or another unrelated dated record.  The actual
        # record owner must also be independently named by the question or
        # the already produced Semantic goal, not only by Compose itself.
        generic_owner = {"government", "held", "relationship", "tenure", "record", "membership", "term", "dated"}
        owner_words = _words(str(ontology.domain_for_relation(attribute)).rsplit(".", 1)[-1]) - generic_owner
        context = str(question) + " " + " ".join(str(p.get("goal", "")) for p in grounded.compose_input.get("semantic_paths", []))
        if not owner_words or not owner_words & _words(context):
            return None, _evidence("saved_compose_generic_date_phase_has_no_independent_record_owner_kind_proof")
    labels = [*binding_by_surface]
    range_id = str(ontology.range_for_relation(attribute))
    numeric_string = range_id in (STRING_TYPES | {"type.enumeration"}) and bool(_COMPARISON.search(question))
    constraint = _native_scalar_obligation(question, labels, "type.float" if numeric_string else range_id)
    if constraint is None and spec_provider is not None:
        constraint = _validated_provided_scalar_spec(question, labels, spec_provider)
    if constraint is None:
        return None, _evidence("saved_compose_single_explicit_scalar_obligation_not_normalizable")
    value_type = str(constraint.get("value_type", "")).casefold()
    if not (((range_id in NUMERIC_TYPES or numeric_string) and value_type == "number")
            or (range_id == "type.datetime" and value_type in {"date", "datetime", "year"})):
        return None, _evidence("saved_compose_normalized_literal_and_actual_attribute_range_disagree")
    operator = {"type": constraint["type"], "inputs": [], "input_var": scalar_var,
                "attribute_relation_id": "", "attribute_relation_label": [], "attribute_relation_labels": [],
                "value": str(constraint["value"]), "value_type": str(constraint["value_type"]),
                "_attribute_value_already_bound": True}
    if range_id in NUMERIC_TYPES or numeric_string:
        operator["_numeric_cast_kind"] = "float" if numeric_string or range_id == "type.float" else "integer"
        operator["_numeric_cast_compare" if operator["type"] == "EQUAL" else "_numeric_comparison_cast"] = True
        if numeric_string:
            operator["_budgeted_numeric_string_value_guard"] = True
    elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", operator["value"]):
        operator["_comparison_date_precision" if operator["type"] == "EQUAL" else "_calendar_day_boundary"] = "day" if operator["type"] == "EQUAL" else True
    elif range_id == "type.datetime" and re.fullmatch(r"\d{4}", operator["value"]):
        operator["_comparison_date_precision"] = "year"
    operators = deepcopy(source_graph.operators)
    for op in operators:
        op["input_var"] = source_mapping.get(str(op.get("input_var", "")), str(op.get("input_var", "")))
        if isinstance(op.get("inputs"), list):
            op["inputs"] = [source_mapping.get(str(v), str(v)) for v in op["inputs"]]
    operators.append(operator)
    result = QueryGraphCandidate(graph_id, triples, answer, operators, float(grounded.score), deepcopy(dict(compose_output)),
                                 deepcopy(source_graph.provenance))
    result.provenance["budgeted_saved_compose_scalar_extension"] = {
        "source_graph_id": source_graph.graph_id, "grounded_core_variable_mapping": core_mapping,
        "source_variable_mapping": source_mapping, "inverse_owner_role_bridge": role_bridge,
        "actual_compose_output_reused": True, "scalar_attribute": attribute,
        "requires_nonempty_strict_subset_of_actual_source": True,
        "requires_saved_execution_count_and_time_credit": True,
        "additional_model_calls": 0, "additional_endpoint_queries": 0, "uses_evaluation_gold": False,
    }
    evidence.update({"status": "compiled", "reason": "actual_saved_compose_scalar_suffix_with_exact_owner",
                     "source_graph_id": source_graph.graph_id, "graph_id": graph_id,
                     "scalar_relation_id": attribute, "scalar_owner_var": owner, "scalar_value_var": scalar_var,
                     "normalized_scalar_spec": constraint, "inverse_owner_role_bridge": role_bridge,
                     "requires_nonempty_strict_subset_of_actual_source": True,
                     "requires_saved_execution_count_and_time_credit": True,
                     "new_repair_denotation_verified": False})
    return result, evidence


def lower_budgeted_scalar_query(graph: QueryGraphCandidate, *, limit: int | None = 256) -> str:
    """Render a compiled repair; reject nonnumeric enum/text values explicitly.

    This is only for the new post-model repair channel.  Ordinary query
    lowering is untouched.  Virtuoso numeric casts can turn alphabetic text
    into zero, so a numeric string repair must not rely on the cast alone.
    """
    from .lowering import lower_sparql

    query = lower_sparql(graph, limit=limit)
    guards = []
    for operator in graph.operators:
        if not isinstance(operator, dict) or not operator.get("_budgeted_numeric_string_value_guard"):
            continue
        variable = str(operator.get("input_var", ""))
        if (not _VAR.fullmatch(variable) or operator.get("attribute_relation_id")
                or not operator.get("_attribute_value_already_bound")):
            raise ValueError("Budgeted numeric string guard requires one already-bound scalar variable")
        guards.append(f'  FILTER(REGEX(STR(?{variable}), "^[+-]?[0-9]+([.][0-9]+)?$"))')
    if guards:
        before, after = query.rsplit("\n}", 1)
        query = before + "\n" + "\n".join(guards) + "\n}" + after
    return query


def _existing_exact_scalar_intent(
    graph: QueryGraphCandidate, attribute: str, constraint: Mapping[str, Any], ontology: Any,
) -> bool:
    """Preserve a program already expressing the exact required field/value."""
    for op in graph.operators:
        if not isinstance(op, dict) or str(op.get("type", "")).upper() != constraint.get("type"):
            continue
        expected, actual = str(constraint.get("value", "")), str(op.get("value", ""))
        if str(constraint.get("value_type", "")).casefold() == "number":
            if _number(expected) is None or _number(actual) != _number(expected):
                continue
        elif expected != actual:
            continue
        direct = str(op.get("attribute_relation_id", "")) or relation_id_from_label(op.get("attribute_relation_label", []))
        owner = str(op.get("input_var", ""))
        if direct == attribute:
            types = {str(ontology.range_for_relation(str(p))) for _, p, b in graph.triples if str(b) == owner}
            types.update(str(ontology.domain_for_relation(str(p))) for a, p, _ in graph.triples if str(a) == owner)
            if any(t not in SCALAR_TYPES and schema_type_overlap(t, str(ontology.domain_for_relation(attribute)), ontology) == "compatible" for t in types):
                return True
        if not direct and any(str(p) == attribute and str(b) == owner for _, p, b in graph.triples):
            return True
    return False


def budgeted_saved_compose_source_witness(
    question: str, selected: ExecutedGraph, executions: Sequence[ExecutedGraph],
    grounded: GroundedSemanticCandidate, compose_output: Mapping[str, Any], ontology: Any, *,
    source_semantic: Mapping[str, Any] | None = None,
    linked_entity_context: Sequence[Mapping[str, Any]] | None = None,
    spec_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Qualify complete real sources for a missing saved Compose obligation.

    This is a post-model-only gate.  It neither executes nor selects the
    proposed query.  Incomplete source rows/labels remain unknown; an existing
    exact field/comparison/literal is preserved.  Extra original named topic
    scope and non-scalar ordering/count requirements cannot be discarded.
    """
    if source_semantic is None:
        return None, _evidence("saved_compose_originating_semantic_scope_missing")
    scope_ok, _, scope = _normal_anchor_scope(grounded, source_semantic, question, linked_entity_context)
    if not scope_ok:
        return None, {**_evidence("saved_compose_original_question_anchor_scope_unproved"), "anchor_scope_evidence": scope}
    if re.search(r"\b(?:earliest|latest|first|last|smallest|largest|highest|lowest|maximum|minimum|most|least)\b|\bhow\s+many\b", question, re.I):
        return None, _evidence("saved_compose_question_has_additional_extrema_or_count_obligation")
    qualified: list[tuple[ExecutedGraph, dict[str, Any]]] = []
    incomplete: list[str] = []
    for execution in executions:
        proposed, evidence = compile_saved_compose_scalar_extension(
            grounded, compose_output, execution.graph, question, ontology,
            graph_id="saved_compose_source_witness_only", spec_provider=spec_provider,
        )
        if proposed is None:
            continue
        if _existing_exact_scalar_intent(selected.graph, evidence["scalar_relation_id"], evidence["normalized_scalar_spec"], ontology):
            return None, _evidence("incumbent_already_has_exact_saved_compose_scalar_intent")
        names = {str(a.get("id", "")): str(a.get("label", "")).strip()
                 for a in execution.answers if isinstance(a, dict)}
        if (not execution.answer_ids or execution.row_count != len(execution.answer_ids)
                or len(set(execution.answer_ids)) != len(execution.answer_ids)
                or any(not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", str(identifier)) or not names.get(str(identifier))
                       for identifier in execution.answer_ids)):
            incomplete.append(execution.graph.graph_id)
        else:
            qualified.append((execution, evidence))
    if incomplete or len({frozenset(item.answer_ids) for item, _ in qualified}) != 1:
        return None, {**_evidence("saved_compose_primary_execution_incomplete_or_ambiguous"),
                      "complete_source_graph_ids": [item.graph.graph_id for item, _ in qualified],
                      "incomplete_source_graph_ids": incomplete}
    chosen, evidence = min(qualified, key=lambda item: (-float(item[0].graph.score), item[0].graph.graph_id))
    evidence.update({"status": "qualified", "reason": "existing_named_primary_source_and_missing_explicit_saved_compose_obligation",
                     "source_graph_id": chosen.graph.graph_id, "source_answer_count": len(chosen.answer_ids),
                     "anchor_scope_evidence": scope, "incumbent_exact_obligation_present": False,
                     "requires_nonempty_strict_subset_of_actual_source": True,
                     "requires_saved_execution_count_and_time_credit": True,
                     "new_repair_denotation_verified": False})
    return chosen, evidence
