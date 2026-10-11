"""Pure veto of a late challenger that only changes a requested answer role.

Cached schema metadata is role evidence, not proof that sibling Freebase types
are disjoint. In particular no agent/non-agent root or movie/film synonym is
used. An opaque question head needs a uniquely quoted obligation whose original
Semantic and decomposition explicitly corroborate the source terminal role.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import re
from types import SimpleNamespace
from typing import Any

from .evaluation import canonicalize_answer_id
from .nested_name_schema_guard import (
    _anchors, _complete, _concrete_type, _entity_answers, _schema_status, _singular,
    _substantive_operators, _type_index,
)
from .ontology import relation_id_from_label
from .source_role_execution_safety import _decomposition_lines, _semantic_graphs, _tokens


_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_HEAD = re.compile(r"^\s*(?:what|which)\s+(?:(?:is|was|are|were)\s+)?"
                   r"(?:the\s+)?([A-Za-z]+)\b", re.I)
_WH_ROLE = re.compile(r"\b(?:what|which)\s+(?:is|are|was|were)\s+(?:the\s+)?"
                      r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)", re.I)
_QUOTED = re.compile(r'"([^"\n]+)"')


def _phrase(value: str) -> str:
    return " ".join(_tokens(str(value).replace("_", " ")))


def _type_role(type_id: str) -> str:
    """Remove only a repeated namespace prefix, not arbitrary type modifiers."""
    namespace, _, leaf = str(type_id).rpartition(".")
    words, prefix = _tokens(leaf.replace("_", " ")), _tokens(namespace.rsplit(".", 1)[-1])
    if prefix and words[:len(prefix)] == prefix and len(words) > len(prefix):
        words = words[len(prefix):]
    return " ".join(words)


def _last_role(line: str) -> tuple[str, str] | None:
    values = _WH_ROLE.findall(str(line))
    return (_phrase(values[-1][0]), _phrase(values[-1][1])) if values else None


def _projection(graph: Any) -> tuple[str, str, str] | None:
    triples = getattr(graph, "triples", None)
    if (not isinstance(triples, list) or not triples
            or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in triples)):
        return None
    incident = [tuple(map(str, row)) for row in triples
                if str(row[0]) == str(graph.answer_var) or str(row[2]) == str(graph.answer_var)]
    return incident[0] if len(incident) == 1 and incident[0][2] == str(graph.answer_var) else None


def _question_head(question: str) -> str:
    match = _HEAD.match(str(question))
    if not match:
        return ""
    head = _singular(match.group(1))
    # Do not treat an apostrophe, hyphen or digit as a head-word boundary.
    tail = str(question)[match.end():]
    if tail and not tail[0].isspace() and tail[0] not in "?,.;:!":
        return ""
    return head


def _quoted_obligation(question: str, semantic: Mapping[str, Any], path: Mapping[str, Any],
                       goal: str, ontology: Any) -> tuple[str, str] | None:
    if str(question).count('"') != 2:
        return None
    quoted = _QUOTED.findall(str(question))
    if len(quoted) != 1 or quoted[0] not in str(goal):
        return None
    anchors = {str(anchor.get("id")): str(anchor.get("surface", ""))
               for anchor in semantic["anchors"] if isinstance(anchor, Mapping)}
    if anchors.get(str(path.get("anchor_ref"))) != quoted[0]:
        return None
    steps = path.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[0], Mapping):
        return None
    first = steps[0]
    relation = relation_id_from_label(first.get("relation_label"))
    field = _phrase(relation.rsplit(".", 1)[-1])
    # The phrase must be the requested field, not an unrelated entity title.
    # Missing original field metadata is retained only as textual role context;
    # the separate complete-literal lane requires its actual known text schema.
    question_without_quote = _phrase(_QUOTED.sub(" ", str(question)))
    datatype = str(ontology.range_for_relation(relation) or "")
    if (first.get("direction") != "backward"
            or str(first.get("from")) != str(path.get("anchor_ref"))
            or datatype and datatype not in {"type.text", "type.rawstring"}
            or not field or not re.search(r"(?<!\w)" + re.escape(field) + r"(?!\w)", question_without_quote)):
        return None
    return quoted[0], relation


def _terminal_role_evidence(question: str, source_edge: tuple[str, str, str], ontology: Any,
                            semantic_context: Any, decomposition_context: Any):
    source_type = str(ontology.range_for_relation(source_edge[1]) or "")
    if not _concrete_type(source_type):
        return None, {"reason": "source_terminal_schema_unknown_or_generic"}
    source_role = _type_role(source_type)
    head = _question_head(question)
    if not head:
        return None, {"reason": "question_not_single_entity_wh_request"}
    known_head_types = set(_type_index(ontology).get(head, ()))
    direct_head = source_role == head
    source_graph = type("RoleGraph", (), {"triples": [source_edge], "answer_var": source_edge[2]})()
    source_status, source_schema = _schema_status(source_graph, known_head_types, ontology)
    if known_head_types and source_status != "compatible":
        return None, {"reason": "known_question_head_does_not_support_source_role",
                      "head": head, "source_schema": source_schema}
    decomposition_roles = {_last_role(line) for line in _decomposition_lines(decomposition_context)} - {None}
    witnesses = []
    for semantic in _semantic_graphs(semantic_context):
        for path in semantic["semantic_paths"]:
            if not isinstance(path, Mapping):
                continue
            steps, goal = path.get("steps"), str(path.get("goal", ""))
            role = _last_role(goal)
            if not isinstance(steps, list) or not steps or role not in decomposition_roles:
                continue
            last = steps[-1]
            if not isinstance(last, Mapping) or last.get("direction") != "forward":
                continue
            requested_relation = relation_id_from_label(last.get("relation_label"))
            parts = requested_relation.split(".")
            if len(parts) < 3:
                continue
            expected_role = _phrase(parts[-1])
            expected_owner = _phrase(parts[-2])
            if (role != (expected_role, expected_owner) or expected_role != source_role
                    or requested_relation.rsplit(".", 1)[0] != source_edge[1].rsplit(".", 1)[0]):
                continue
            # An exact schema head is direct evidence. An unknown head is not
            # translated: it instead needs the literal obligation chain above.
            obligation = _quoted_obligation(question, semantic, path, goal, ontology)
            if not direct_head and obligation is None:
                continue
            witnesses.append({"requested_role": expected_role, "requested_owner": expected_owner,
                              "requested_semantic_relation_id": requested_relation,
                              "quoted_obligation": obligation})
    signatures = {(row["requested_role"], row["requested_owner"], row["quoted_obligation"])
                  for row in witnesses}
    if len(signatures) != 1:
        return None, {"reason": "question_semantic_decomposition_role_missing_or_ambiguous",
                      "head": head, "source_terminal_role": source_role,
                      "role_witness_count": len(signatures)}
    return source_role, {"head": head, "source_terminal_type": source_type,
                         "source_terminal_role": source_role,
                         "question_head_ontology_known": bool(known_head_types),
                         "no_question_head_synonym_mapping": True,
                         "quoted_field_schema_proven": bool(witnesses[0]["quoted_obligation"]
                             and ontology.range_for_relation(witnesses[0]["quoted_obligation"][1])),
                         "does_not_assert_literal_constraint_satisfied": True,
                         "role_witness": witnesses[0]}


def zero_graph_role_regression(
    question: str, source: Any, challenger_graph: Any, challenger_answer_ids: Any,
    ontology: Any, semantic_context: Any, decomposition_context: Any,
) -> tuple[bool, dict[str, Any]]:
    """Veto a pure requested-terminal switch, without choosing/executing a graph.

    The source is the actual complete selected execution before the late gate.
    Semantic and decomposition must be existing original stage context. A true
    result protects its declared role only; it does not assert that all missing
    question constraints were answered or that schema sibling types are disjoint.
    """
    evidence = {"status": "not_applied", "source_graph_id": str(source.graph.graph_id),
                "challenger_graph_id": str(challenger_graph.graph_id),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "protects_declared_role_only": True, "schema_disjointness_claimed": False}
    try:
        if not _complete(source) or _entity_answers(source) is None:
            evidence["reason"] = "source_complete_entity_execution_not_attested"
            return False, evidence
        if (isinstance(getattr(source.graph, "provenance", None), Mapping)
                and source.graph.provenance.get("selection_answer_union")):
            evidence["reason"] = "source_union_binding_scope_unproven"
            return False, evidence
        if not isinstance(challenger_answer_ids, list) or not challenger_answer_ids:
            evidence["reason"] = "challenger_entity_answers_missing"
            return False, evidence
        challenger_ids = {canonicalize_answer_id(value) for value in challenger_answer_ids}
        if len(challenger_ids) != len(challenger_answer_ids) or not all(_ENTITY.fullmatch(value) for value in challenger_ids):
            evidence["reason"] = "challenger_answers_not_unique_entities"
            return False, evidence
        before, after = _projection(source.graph), _projection(challenger_graph)
        if (before is None or after is None or before[1] == after[1]
                or before[0] != after[0] or before[2] != after[2]
                or str(source.graph.answer_var) != str(challenger_graph.answer_var)):
            evidence["reason"] = "not_single_same_owner_terminal_projection_change"
            return False, evidence
        source_context = Counter(tuple(map(str, row)) for row in source.graph.triples if tuple(map(str, row)) != before)
        challenger_context = Counter(tuple(map(str, row)) for row in challenger_graph.triples if tuple(map(str, row)) != after)
        source_ops, challenger_ops = _substantive_operators(source.graph), _substantive_operators(challenger_graph)
        if (not source_context or source_context != challenger_context
                or _anchors(source.graph) != _anchors(challenger_graph)
                or source_ops is None or source_ops != challenger_ops):
            evidence["reason"] = "challenger_changes_context_anchor_or_operator_obligations"
            return False, evidence
        source_owner = str(ontology.domain_for_relation(before[1]) or "")
        challenger_owner = str(ontology.domain_for_relation(after[1]) or "")
        if not _concrete_type(source_owner) or source_owner != challenger_owner:
            evidence["reason"] = "terminal_owner_schema_unknown_or_changed"
            return False, evidence
        role, role_evidence = _terminal_role_evidence(
            question, before, ontology, semantic_context, decomposition_context)
        evidence.update(role_evidence)
        if role is None:
            return False, evidence
        alternative_type = str(ontology.range_for_relation(after[1]) or "")
        if not _concrete_type(alternative_type):
            evidence["reason"] = "challenger_terminal_schema_unknown_or_generic"
            return False, evidence
        alternative_role = _type_role(alternative_type)
        source_type = role_evidence["source_terminal_type"]
        if (alternative_role == role or source_type in ontology.supertypes(alternative_type)
                or alternative_type in ontology.supertypes(source_type)):
            evidence["reason"] = "terminal_role_same_or_type_hierarchy_compatible"
            return False, evidence
        if challenger_ids <= _entity_answers(source):
            evidence["reason"] = "challenger_denotation_does_not_add_different_role_answers"
            return False, evidence
        evidence.update(status="applied", reason="late_challenger_only_changes_corroborated_terminal_role",
                        challenger_terminal_type=alternative_type, challenger_terminal_role=alternative_role,
                        preserved_context_edge_count=sum(source_context.values()),
                        preserved_operator_count=len(source_ops),
                        preserved_terminal_owner_type=source_owner,
                        newly_satisfied_requested_constraint_count=0,
                        source_answer_count=len(source.answer_ids), challenger_answer_count=len(challenger_ids))
        return True, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="role_context_or_schema_error",
                        error=f"{type(exc).__name__}: {exc}")
        return False, evidence


_RDF_TERM = r'(?:\?[A-Za-z_]\w*|[A-Za-z_]\w*:[A-Za-z0-9_.]+|<[^<>\s]+>|"(?:[^"\\]|\\.)*"(?:@[A-Za-z-]+|\^\^(?:<[^<>\s]+>|[A-Za-z_]\w*:[A-Za-z0-9_.]+))?)'
_BGP_TRIPLE = re.compile(r"\s*(?P<s>" + _RDF_TERM + r")\s+(?P<p>" + _RDF_TERM
                         + r")\s+(?P<o>" + _RDF_TERM + r")\s*\.\s*")
_VAR = re.compile(r"\?[A-Za-z_]\w*\Z")


def _compiled_template_bgp(execution: Any) -> tuple[Any | None, str]:
    """Parse a deliberately small, fully covered SELECT-DISTINCT BGP subset.

    Unknown constructs fail closed. A LIMIT is qualified only below its cap;
    existing row count plus a truncated capped result is not completeness proof.
    The parser never executes/lowers a query or imports a SPARQL/network package.
    """
    text = str(getattr(execution.graph, "sparql", "") or "")
    prefixes = {}
    while True:
        match = re.match(r"\s*PREFIX\s+([A-Za-z_]\w*):\s*<([^>]+)>\s*", text, re.I)
        if not match:
            break
        prefixes[match.group(1)] = match.group(2)
        text = text[match.end():]
    query = re.fullmatch(r"\s*SELECT\s+DISTINCT\s+(\?[A-Za-z_]\w*)\s+WHERE\s*\{([\s\S]*)\}"
                         r"\s*(?:LIMIT\s+(\d+))?\s*", text, re.I)
    if not query:
        return None, "compiled_template_not_supported_select_distinct_bgp"
    answer, body, limit = query.groups()
    if answer.lstrip("?") != str(execution.graph.answer_var).lstrip("?"):
        return None, "compiled_template_answer_variable_conflict"
    if limit and execution.row_count >= int(limit):
        return None, "compiled_template_limit_completeness_unproven"

    def term(value):
        if _VAR.fullmatch(value):
            return value
        if value.startswith('"'):
            return value
        if value.startswith("<"):
            uri = value[1:-1]
        elif ":" in value:
            prefix, suffix = value.split(":", 1)
            if prefix not in prefixes:
                raise ValueError("Unknown template prefix")
            uri = prefixes[prefix] + suffix
        else:
            raise ValueError("Unknown template term")
        if not uri.startswith("http://rdf.freebase.com/ns/"):
            raise ValueError("Non-Freebase template term")
        return canonicalize_answer_id(uri)

    # These two common template filters are proven true for the existing pure
    # entity denotation. Arbitrary filters, OR/UNION branches and ordering fail.
    position = 0
    triples = []
    while position < len(body):
        whitespace = re.match(r"\s*", body[position:])
        position += whitespace.end()
        if position == len(body):
            break
        if re.match(r"FILTER\b", body[position:], re.I):
            filter_match = re.match(r"FILTER\s*\(", body[position:], re.I)
            if not filter_match:
                return None, "compiled_template_filter_unsupported"
            begin = position + filter_match.end()
            cursor, depth, quote, escaped = begin, 1, "", False
            while cursor < len(body) and depth:
                char = body[cursor]
                if quote:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = ""
                elif char in "\"'":
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                cursor += 1
            if depth or quote:
                return None, "compiled_template_filter_unbalanced"
            expression = body[begin:cursor-1].strip()
            inequality = re.fullmatch(re.escape(answer) + r"\s*!=\s*(" + _RDF_TERM + r")", expression)
            language = re.fullmatch(r"!\s*isLiteral\s*\(\s*" + re.escape(answer)
                + r"\s*\)\s*(?:OR|\|\|)\s*lang\s*\(\s*" + re.escape(answer)
                + r"\s*\)\s*=\s*''\s*(?:OR|\|\|)\s*langMatches\s*\(\s*lang\s*\(\s*"
                + re.escape(answer) + r"\s*\)\s*,\s*'[A-Za-z-]+'\s*\)", expression, re.I)
            if inequality:
                excluded = term(inequality.group(1))
                if not _ENTITY.fullmatch(excluded) or excluded in _entity_answers(execution):
                    return None, "compiled_template_exclusion_not_attested"
            elif not language:
                return None, "compiled_template_filter_unsupported"
            position = cursor
            continue
        triple = _BGP_TRIPLE.match(body, position)
        if not triple:
            return None, "compiled_template_body_not_fully_covered_bgp"
        subject, predicate, object_ = map(term, triple.groups())
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", predicate):
            return None, "compiled_template_predicate_not_canonical"
        triples.append((subject, predicate, object_))
        position = triple.end()
    if not triples or len(set(triples)) != len(triples):
        return None, "compiled_template_empty_or_duplicate_bgp"
    return SimpleNamespace(triples=triples, answer_var=answer, operators=[]), "compiled_template_bgp_qualified"


def _literal_text(value: str) -> str | None:
    match = re.fullmatch(r'("(?:[^"\\]|\\.)*")(?:@[A-Za-z-]+|\^\^\S+)?', value)
    if not match:
        return None
    try:
        return str(json.loads(match.group(1)))
    except (TypeError, ValueError):
        return None


def _source_embedding(source_graph: Any, candidate_graph: Any):
    """Injective, answer-bound alpha mapping, preserving all source triples."""
    source = [tuple(map(str, row)) for row in source_graph.triples]
    candidate = [tuple(map(str, row)) for row in candidate_graph.triples]
    constants = _anchors(source_graph)
    source_vars = {node for row in source for node in (row[0], row[2]) if node not in constants}
    states = [{str(source_graph.answer_var): str(candidate_graph.answer_var)}]
    for subject, relation, object_ in source:
        next_states = []
        for state in states:
            for c_subject, c_relation, c_object in candidate:
                if relation != c_relation:
                    continue
                updated = dict(state)
                valid = True
                for left, right in ((subject, c_subject), (object_, c_object)):
                    if left not in source_vars:
                        valid &= left == right
                    elif not _VAR.fullmatch(right) or left in updated and updated[left] != right:
                        valid = False
                    elif left not in updated:
                        if right in updated.values():
                            valid = False
                        else:
                            updated[left] = right
                if valid:
                    next_states.append(updated)
        states = next_states
        if not states or len(states) > 64:
            return []
    distinct = {json.dumps(value, sort_keys=True): value for value in states}
    return list(distinct.values())


def quoted_constraint_execution_challenger(
    question: str, source: Any, executions: list[Any], ontology: Any,
    semantic_context: Any, decomposition_context: Any,
) -> tuple[Any | None, dict[str, Any]]:
    """A complete already-executed template may refine even an entity source.

    Require a strict answer subset, exact typed quoted literal, source-core
    preservation under a unique alpha mapping, and the original literal owner
    path to the declared answer. No rank, singleton preference, Gold, or query
    execution is involved. A different description owner is an abstention.
    """
    evidence = {"status": "not_applied", "source_graph_id": str(source.graph.graph_id),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "uses_existing_executions_only": True}
    try:
        if (not _complete(source) or _entity_answers(source) is None
                or source.graph.operators or _projection(source.graph) is None):
            evidence["reason"] = "source_denotation_operator_or_projection_unqualified"
            return None, evidence
        if (isinstance(getattr(source.graph, "provenance", None), Mapping)
                and source.graph.provenance.get("selection_answer_union")):
            evidence["reason"] = "source_union_scope_unproven"
            return None, evidence
        edge = _projection(source.graph)
        role, proof = _terminal_role_evidence(question, edge, ontology, semantic_context, decomposition_context)
        evidence["role_evidence"] = proof
        if role is None or not proof.get("quoted_field_schema_proven"):
            evidence["reason"] = "original_typed_quoted_obligation_and_owner_unproven"
            return None, evidence
        quoted, field = proof["role_witness"]["quoted_obligation"]
        field_type = str(ontology.range_for_relation(field) or "")
        if field_type not in {"type.text", "type.rawstring"}:
            evidence["reason"] = "original_quoted_field_not_known_text"
            return None, evidence
        source_ids = _entity_answers(source)
        eligible, excluded = [], []
        for candidate in executions:
            ids = _entity_answers(candidate)
            if candidate is source or ids is None or not _complete(candidate) or not ids < source_ids:
                continue
            parsed, reason = _compiled_template_bgp(candidate)
            if parsed is None:
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": reason})
                continue
            literal_edges = [row for row in parsed.triples if row[1] == field and _literal_text(row[2]) == quoted]
            if len(literal_edges) != 1:
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "exact_original_literal_field_missing_or_ambiguous"})
                continue
            bindings = _source_embedding(source.graph, parsed)
            if len(bindings) != 1:
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "source_core_binding_missing_or_ambiguous"})
                continue
            # The original Semantic literal path must reach the exact source
            # projection owner. A literal on an unrelated/disconnected entity
            # cannot supply the requested obligation.
            mapping = bindings[0]
            if literal_edges[0][0] != mapping.get(edge[0]):
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "quoted_constraint_attached_to_different_owner"})
                continue
            mapped_source = {(mapping.get(str(row[0]), str(row[0])), str(row[1]),
                              mapping.get(str(row[2]), str(row[2]))) for row in source.graph.triples}
            if set(parsed.triples) != mapped_source | {literal_edges[0]}:
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "template_adds_other_unproven_obligations"})
                continue
            # This lane is intentionally limited to a direct owner-to-answer
            # literal obligation. Longer paths need their own variable proof.
            literal_owner = str(ontology.domain_for_relation(field) or "")
            if not _concrete_type(literal_owner) or literal_owner != str(ontology.domain_for_relation(edge[1]) or ""):
                excluded.append({"graph_id": str(candidate.graph.graph_id), "reason": "original_literal_and_answer_owner_types_differ"})
                continue
            eligible.append(candidate)
        evidence.update(eligible_graph_ids=[str(candidate.graph.graph_id) for candidate in eligible],
                        excluded_candidates=excluded)
        denotations = {tuple(sorted(_entity_answers(candidate))) for candidate in eligible}
        if len(denotations) != 1:
            evidence["reason"] = "no_unique_complete_quoted_constraint_denotation"
            return None, evidence
        chosen = eligible[0]
        evidence.update(status="applied", reason="existing_strict_subset_fulfils_original_typed_literal_same_owner",
                        selected_graph_id=str(chosen.graph.graph_id), selected_answer_count=len(chosen.answer_ids),
                        exact_quoted_field=field, exact_quoted_value=quoted, original_owner_preserved=True)
        return chosen, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="compiled_template_or_context_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence
