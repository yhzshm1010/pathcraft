#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAEDE_SRC = REPO_ROOT / "vendor" / "kaede_vqg" / "src"
FREEBASE_NS = "http://rdf.freebase.com/ns/"
MID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
SPARQL_MID_RE = re.compile(r"\bns:(?P<mid>[mg]\.[A-Za-z0-9_]+)\b")
SELECT_VARIABLE_RE = re.compile(
    r"\bSELECT\s+(?:DISTINCT\s+)?(?P<variable>\?[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
ANSWER_EXCLUSION_RE = re.compile(
    r"FILTER\s*\(\s*(?P<variable>\?[A-Za-z_][A-Za-z0-9_]*)\s*!=\s*"
    r"(?:ns:(?P<compact>[mg]\.[A-Za-z0-9_]+)|"
    r"<http://rdf\.freebase\.com/ns/(?P<uri>[mg]\.[A-Za-z0-9_]+)>)\s*\)",
    re.IGNORECASE,
)
VARIABLE_ANSWER_EXCLUSION_RE = re.compile(
    r"FILTER\s*\(\s*(?P<answer>\?[A-Za-z_][A-Za-z0-9_]*)\s*!=\s*"
    r"(?P<excluded>\?[A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.IGNORECASE,
)
SPARQL_RELATION_TRIPLE_RE = re.compile(
    r"(?P<subject>\?[A-Za-z_][A-Za-z0-9_]*|ns:[A-Za-z0-9_\.]+)\s+"
    r"ns:(?P<relation>[A-Za-z0-9_\.]+)\s+"
    r"(?P<object>\?[A-Za-z_][A-Za-z0-9_]*|ns:[A-Za-z0-9_\.]+)\s*\.",
    re.IGNORECASE,
)
NUMERIC_COMPARISON_FILTER_RE = re.compile(
    r"FILTER\(\s*(?P<input>\?[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<operator><=|>=|<|>)\s*"
    r'"(?P<value>[-+]?\d+(?:\.\d+)?)"\^\^xsd:'
    r"(?P<datatype>integer|double)\s*\)",
    re.IGNORECASE,
)


def main() -> int:
    try:
        payload = _read_payload()
        result = validate(payload)
    except Exception as exc:  # noqa: BLE001 - validator must report JSON to the caller.
        result = {"valid": False, "errors": [f"Freebase execution validator failed: {exc}"]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    stage = "source validation"
    try:
        source_key = "cwq_source" if isinstance(payload.get("cwq_source"), dict) else "webqsp_source"
        source = payload.get(source_key)
        if not isinstance(source, dict):
            return _reject("missing webqsp_source or cwq_source")
        gold_sparql = str(source.get("sparql", "")).strip()
        if not gold_sparql:
            return _reject(f"{source_key}.sparql is empty; cannot execution-validate this row")

        _ensure_kaede_import()
        from kaede_vqg.freebase import (  # noqa: PLC0415
            FreebaseConfig,
            FreebaseGroundingBackend,
            FreebaseQueryBackend,
            FreebaseSparqlClient,
            GoldenEntityLinker,
        )

        config = FreebaseConfig.from_env()
        client = FreebaseSparqlClient(config)
        stage = "gold SPARQL"
        gold_payload = client.select(gold_sparql)
        gold_answers = _answer_set_from_sparql_payload(gold_payload)
        if not gold_answers:
            return _reject("source SPARQL executed but returned no answers")
        excluded_answer_ids = _gold_answer_exclusions(gold_sparql)

        compose_program = payload.get("compose_program")
        if not isinstance(compose_program, dict):
            return _reject("compose_program must be an object")
        if source_key == "cwq_source":
            compose_program = _prepare_cwq_variable_exclusions(compose_program, source)

        stage = "gold entity label resolution"
        context = {
            "golden_entities": _golden_entities(source, client=client),
            source_key: source,
        }
        grounding = FreebaseGroundingBackend(
            client,
            entity_linker=GoldenEntityLinker(
                minimum_surface_score=float(os.environ.get("WEBQSP_GOLDEN_ENTITY_MIN_SCORE", "0.65"))
            ),
        )
        stage = "candidate grounding"
        graphs = grounding.ground_logical_program(
            question=str(payload.get("question", "")),
            logical_program=compose_program,
            top_k=int(os.environ.get("WEBQSP_VALIDATOR_TOP_K", "3")),
            relation_top_k=int(os.environ.get("WEBQSP_VALIDATOR_RELATION_TOP_K", "1")),
            beam_size=int(os.environ.get("WEBQSP_VALIDATOR_BEAM_SIZE", "5")),
            context=context,
        )
        if not graphs:
            return _reject("candidate compose_program could not be grounded to Freebase")

        query_backend = FreebaseQueryBackend(
            client,
            answer_limit=max(
                int(os.environ.get("WEBQSP_VALIDATOR_ANSWER_LIMIT", str(config.answer_limit))),
                len(gold_answers) + 1,
            ),
            relation_substitution=False,
        )
        executions: list[dict[str, Any]] = []
        for graph in graphs:
            execution_graph = _prepare_candidate_graph(
                graph,
                gold_sparql=gold_sparql,
            )
            execution_graph = _restore_gold_predicate_casing(
                execution_graph,
                gold_sparql,
            )
            stage = "candidate query execution"
            execution = _execute_candidate_graph(query_backend, execution_graph)
            raw_candidate_answers = {
                _normalize_answer(value) for value in execution.get("answer_ids", []) or []
            }
            raw_candidate_answers.discard("")
            candidate_answers = raw_candidate_answers - excluded_answer_ids
            executions.append(
            {
                "graph_id": graph.get("graph_id"),
                "candidate_answers": sorted(candidate_answers),
                "excluded_candidate_answers": sorted(
                    raw_candidate_answers & excluded_answer_ids
                ),
                "sparql": execution.get("sparql"),
                "error": execution.get("error"),
                "grounded_entities": graph.get("grounded_entities"),
                "grounded_relations": execution_graph.get("grounded_relations"),
            }
            )
            if candidate_answers == gold_answers:
                return {
                "valid": True,
                "errors": [],
                "gold_answer_count": len(gold_answers),
                "gold_answers": sorted(gold_answers),
                "candidate_sparql": execution.get("sparql", ""),
                "candidate_graph_id": graph.get("graph_id", ""),
                }

        return {
        "valid": False,
        "errors": [
            "candidate execution answers do not match source SPARQL answers",
            f"gold_answers={sorted(gold_answers)[:20]}",
            f"candidate_answers={executions[0].get('candidate_answers', [])[:20] if executions else []}",
        ],
        "gold_answer_count": len(gold_answers),
        "gold_answers": sorted(gold_answers)[:50],
        "executions": executions[:3],
        }
    except Exception as exc:  # noqa: BLE001 - validator must report structured diagnostics.
        return _reject(f"Freebase execution validator failed during {stage}: {exc}")


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("stdin payload is empty")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("stdin payload must be a JSON object")
    return payload


def _ensure_kaede_import() -> None:
    src = Path(os.environ.get("KAEDE_VQG_SRC") or DEFAULT_KAEDE_SRC).resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _answer_set_from_sparql_payload(payload: dict[str, Any]) -> set[str]:
    output: set[str] = set()
    results = payload.get("results", {})
    bindings = results.get("bindings", []) if isinstance(results, dict) else []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        term = binding.get("answer") or binding.get("x") or next(iter(binding.values()), {})
        if isinstance(term, dict):
            value = _normalize_answer(term.get("value", ""))
            if value:
                output.add(value)
    return output


def _gold_answer_exclusions(gold_sparql: str) -> set[str]:
    select_match = SELECT_VARIABLE_RE.search(gold_sparql)
    if select_match is None:
        return set()
    answer_variable = select_match.group("variable").casefold()
    output: set[str] = set()
    for match in ANSWER_EXCLUSION_RE.finditer(gold_sparql):
        if match.group("variable").casefold() != answer_variable:
            continue
        value = match.group("compact") or match.group("uri") or ""
        if value:
            output.add(value)
    return output


def _prepare_candidate_graph(
    graph: dict[str, Any],
    *,
    gold_sparql: str = "",
) -> dict[str, Any]:
    prepared = deepcopy(graph)
    gold_uses_integer_comparison = bool(
        re.search(r"\bxsd:integer\s*\(", gold_sparql, re.IGNORECASE)
    )
    operator_groups: list[Any] = [prepared.get("operators")]
    composition = prepared.get("composition")
    if isinstance(composition, dict):
        operator_groups.append(composition.get("operators"))
    for operators in operator_groups:
        if not isinstance(operators, list):
            continue
        for operator in operators:
            if not isinstance(operator, dict):
                continue
            operator_type = str(operator.get("type", "")).upper()
            if operator_type in {
                "GREATER_THAN",
                "LESS_THAN",
                "GREATER_OR_EQUAL",
                "LESS_OR_EQUAL",
            }:
                # ``number`` is the public correction-contract type, but it is
                # not an XML Schema datatype.  Passing it through makes KaeDe
                # emit ``xsd:number``, which Virtuoso may compare
                # lexicographically or ignore.  Keep the persisted correction
                # unchanged and select a real XSD numeric datatype only for
                # execution.
                value = str(operator.get("value", "")).strip()
                value_type = str(operator.get("value_type", "")).casefold()
                if value_type == "number":
                    operator["value_type"] = (
                        "integer" if re.fullmatch(r"[-+]?\d+", value) else "double"
                    )
                elif gold_uses_integer_comparison and re.fullmatch(r"[-+]?\d+", value):
                    # A few CWQ rows mislabel numeric counts as ``year`` in the
                    # public source contract. The read-only gold query is the
                    # authority for execution semantics and explicitly casts
                    # the value with xsd:integer.
                    operator["value_type"] = "integer"
            if operator_type != "TC":
                continue
            relation_ids = operator.get("attribute_relation_ids")
            if isinstance(relation_ids, list) and relation_ids and all(
                isinstance(value, str) for value in relation_ids
            ):
                # KaeDe treats a flat string list as one semantic relation label.
                # Keep materialized TC interval bounds as distinct relation IDs.
                operator["attribute_relation_ids"] = [[value] for value in relation_ids]
    return prepared


def _restore_gold_predicate_casing(
    graph: dict[str, Any],
    gold_sparql: str,
) -> dict[str, Any]:
    """Restore case-sensitive legacy Freebase predicates from gold evidence.

    KAeDe case-folds human-readable relation labels.  Most Freebase predicates
    are lowercase, but legacy predicates such as ``Kind_of_symbol`` are not.
    Only the casing is restored: the grounded predicate must otherwise match a
    predicate in the read-only gold query case-insensitively.
    """
    canonical = {
        match.group("relation").casefold(): match.group("relation")
        for match in SPARQL_RELATION_TRIPLE_RE.finditer(gold_sparql)
    }
    if not canonical:
        return graph

    prepared = deepcopy(graph)
    triples = prepared.get("triples", [])
    if isinstance(triples, list):
        for triple in triples:
            if not isinstance(triple, list) or len(triple) != 3:
                continue
            relation_id = str(triple[1])
            triple[1] = canonical.get(relation_id.casefold(), relation_id)

    grounded_relations = prepared.get("grounded_relations", [])
    if isinstance(grounded_relations, list):
        for relation in grounded_relations:
            if not isinstance(relation, dict):
                continue
            relation_id = str(relation.get("relation_id", ""))
            relation["relation_id"] = canonical.get(
                relation_id.casefold(), relation_id
            )
    return prepared


def _execute_candidate_graph(query_backend: Any, graph: dict[str, Any]) -> dict[str, Any]:
    """Execute with numeric casts matching CWQ's comparison semantics.

    KaeDe serializes a public ``number`` operator as a typed threshold but does
    not cast the Freebase value variable.  CWQ gold explicitly uses
    ``xsd:integer(?value)``; some Freebase numeric properties are stored as
    untyped strings, so the uncast candidate either returns every row or none.
    Only replace the exact comparison form emitted from the execution-only
    normalization above. Other queries continue through KaeDe unchanged.
    """
    from kaede_vqg.freebase import query_graph_to_sparql  # noqa: PLC0415

    sparql = query_graph_to_sparql(graph, limit=query_backend.answer_limit)
    casted = _cast_numeric_comparison_inputs(sparql)
    if casted == sparql:
        return query_backend.execute(graph)
    try:
        payload = query_backend.client.select(casted)
    except Exception as exc:  # noqa: BLE001 - preserve backend result contract.
        return {"answer_ids": [], "can_answer": False, "error": str(exc), "sparql": casted}
    answer_ids = sorted(_answer_set_from_sparql_payload(payload))
    return {
        "answer_ids": answer_ids,
        "can_answer": True,
        "row_count": len(answer_ids),
        "sparql": casted,
        "executed_graph": deepcopy(graph),
        "substitution_attempted": False,
    }


def _cast_numeric_comparison_inputs(sparql: str) -> str:
    def replace(match: re.Match[str]) -> str:
        datatype = match.group("datatype").casefold()
        cast = "integer" if datatype == "integer" else "double"
        return (
            f"FILTER(xsd:{cast}({match.group('input')}) "
            f"{match.group('operator')} {match.group('value')})"
        )

    return NUMERIC_COMPARISON_FILTER_RE.sub(replace, sparql)


def _prepare_cwq_variable_exclusions(
    compose_program: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Lower CWQ answer exclusions to candidate variables for execution.

    CWQ gold often expresses ``FILTER(?answer != ?intermediate)`` while the
    public correction contract carries the human-readable composition answer
    as a NO_EQUAL entity value. That entity is not necessarily an independent
    Freebase node and must not be grounded as an extra placeholder.
    """
    repaired = json.loads(json.dumps(compose_program, ensure_ascii=False))
    sparql = str(source.get("sparql", ""))
    composition_answer = str(source.get("composition_answer", "")).strip()
    answer_match = SELECT_VARIABLE_RE.search(sparql)
    variable_exclusions = [
        (match.group("answer"), match.group("excluded"))
        for match in VARIABLE_ANSWER_EXCLUSION_RE.finditer(sparql)
        if answer_match is not None
        and match.group("answer").casefold() == answer_match.group("variable").casefold()
    ]
    if not variable_exclusions:
        return repaired

    answer_var = str(repaired.get("answer_var", "")).strip()
    excluded_var = _candidate_exclusion_variable(
        repaired,
        sparql,
        variable_exclusions[0][1],
        answer_var,
    )
    if not excluded_var:
        return repaired

    operators = repaired.get("operators", [])
    if not isinstance(operators, list):
        operators = []
        repaired["operators"] = operators
    normalized = False
    if composition_answer:
        for operator in operators:
            if not isinstance(operator, dict):
                continue
            if str(operator.get("type", "")).upper() != "NO_EQUAL":
                continue
            if (
                str(operator.get("value", "")).strip().casefold()
                != composition_answer.casefold()
            ):
                continue
            operator["input_var"] = answer_var
            operator["value"] = excluded_var
            operator["value_type"] = "variable"
            normalized = True
    if not normalized:
        operators.append(
            {
                "type": "NO_EQUAL",
                "inputs": [],
                "input_var": answer_var,
                "attribute_relation_label": [],
                "attribute_relation_labels": [],
                "value": excluded_var,
                "value_type": "variable",
            }
        )

    if composition_answer:
        used_entity_ids = {
            str(term)
            for triple in repaired.get("triples", []) or []
            if isinstance(triple, dict)
            for term in (triple.get("subject", ""), triple.get("object", ""))
            if re.fullmatch(r"E\d+", str(term))
        }
        entities = repaired.get("entities", [])
        if isinstance(entities, list):
            repaired["entities"] = [
                entity
                for entity in entities
                if not (
                    isinstance(entity, dict)
                    and str(entity.get("surface", "")).strip().casefold()
                    == composition_answer.casefold()
                    and str(entity.get("id", "")) not in used_entity_ids
                )
            ]
    return repaired


def _candidate_exclusion_variable(
    compose_program: dict[str, Any],
    sparql: str,
    excluded_sparql_var: str,
    answer_var: str,
) -> str:
    excluded_name = excluded_sparql_var.casefold()
    source_relation_positions: dict[str, set[str]] = {}
    for match in SPARQL_RELATION_TRIPLE_RE.finditer(sparql):
        positions: set[str] = set()
        if match.group("subject").casefold() == excluded_name:
            positions.add("subject")
        if match.group("object").casefold() == excluded_name:
            positions.add("object")
        if positions:
            source_relation_positions.setdefault(
                match.group("relation").casefold(), set()
            ).update(positions)
    candidates: list[str] = []
    for triple in compose_program.get("triples", []) or []:
        if not isinstance(triple, dict):
            continue
        relation = ".".join(
            str(value).strip().replace(" ", "_")
            for value in triple.get("relation_label", []) or []
        ).casefold()
        if source_relation_positions and relation not in source_relation_positions:
            continue
        subject = str(triple.get("subject", ""))
        obj = str(triple.get("object", ""))
        allowed_positions = source_relation_positions.get(
            relation, {"subject", "object"}
        )
        if (
            "subject" in allowed_positions
            and re.fullmatch(r"V\d+", subject)
            and subject != answer_var
        ):
            candidates.append(subject)
        if (
            "object" in allowed_positions
            and re.fullmatch(r"V\d+", obj)
            and obj != answer_var
        ):
            candidates.append(obj)
    if candidates:
        return sorted(set(candidates), key=lambda value: (int(value[1:]), value))[0]

    fallback = {
        str(term)
        for triple in compose_program.get("triples", []) or []
        if isinstance(triple, dict)
        for term in (triple.get("subject", ""), triple.get("object", ""))
        if re.fullmatch(r"V\d+", str(term)) and str(term) != answer_var
    }
    return sorted(fallback, key=lambda value: (int(value[1:]), value))[0] if fallback else ""


def _golden_entities(
    source: dict[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, str]:
    """Return labelled gold entities for grounding, including CWQ SPARQL MIDs.

    CWQ source contexts intentionally have a compact schema and do not expose
    WebQSP-style ``topic_entities`` / ``constraints`` metadata.  Without this
    fallback the GoldenEntityLinker receives an empty inventory and triggers a
    costly full-Freebase surface search for every candidate anchor.
    """
    entities: dict[str, str] = {}
    for item in source.get("topic_entities", []) or []:
        if isinstance(item, dict):
            entity_id = _normalize_answer(item.get("id", ""))
            label = str(item.get("name") or entity_id).strip()
            if entity_id:
                entities[entity_id] = label or entity_id
    for item in source.get("constraints", []) or []:
        if not isinstance(item, dict):
            continue
        entity_id = _normalize_answer(item.get("Argument", ""))
        if entity_id and MID_RE.fullmatch(entity_id):
            label = str(item.get("EntityName") or entity_id).strip()
            entities[entity_id] = label or entity_id
    if entities or client is None:
        return entities

    entity_ids = _sparql_constant_entity_ids(str(source.get("sparql", "")))
    if not entity_ids:
        return entities
    values = " ".join(f"ns:{entity_id}" for entity_id in entity_ids)
    query = (
        "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
        "SELECT ?entity ?label WHERE {\n"
        f"  VALUES ?entity {{ {values} }}\n"
        "  ?entity ns:type.object.name ?label .\n"
        '  FILTER(lang(?label) = "en")\n'
        "}"
    )
    try:
        payload = client.select(query)
    except Exception:  # noqa: BLE001 - retain normal grounding fallback.
        return entities
    bindings = payload.get("results", {}).get("bindings", []) if isinstance(payload, dict) else []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        entity_id = _normalize_answer((binding.get("entity") or {}).get("value", ""))
        label = str((binding.get("label") or {}).get("value", "")).strip()
        if entity_id and label:
            entities[entity_id] = label
    return entities


def _sparql_constant_entity_ids(sparql: str) -> list[str]:
    """Return namespace constants used as triple subjects or objects.

    CWQ constraints sometimes use schema/type constants such as
    ``ns:location.location`` rather than only ``m.*``/``g.*`` entities.  These
    constants still need to enter the golden linker inventory so a readable
    anchor surface grounds to the exact source node instead of an unrelated MID
    with the same label.
    """
    output: set[str] = set()
    for match in SPARQL_RELATION_TRIPLE_RE.finditer(sparql):
        for group in ("subject", "object"):
            term = match.group(group)
            if term.casefold().startswith("ns:"):
                output.add(term[3:])
    return sorted(output)


def _normalize_answer(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    if text.startswith(FREEBASE_NS):
        text = text[len(FREEBASE_NS) :]
    if text.startswith("ns:"):
        text = text[3:]
    if text.endswith("-08:00"):
        text = text[:-6]
    return text


def _reject(*errors: str) -> dict[str, Any]:
    return {"valid": False, "errors": [error for error in errors if error]}


if __name__ == "__main__":
    raise SystemExit(main())
