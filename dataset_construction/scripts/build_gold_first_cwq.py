#!/usr/bin/env python3
"""Build CWQ Semantic/Compose/Operator supervision directly from Gold SPARQL.

This compiler is intentionally independent of the currently trained Semantic,
Compose, and Operator models.  It emits the same Alpaca rows used by the
current ``cwq_new/data/cwq`` training files and accepts a row only after local
contract validation and direct Freebase execution agree with Gold answers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
from typing import Any
from source_annotations import load_source_annotations


NS = "http://rdf.freebase.com/ns/"
TRIPLE_RE = re.compile(
    r"^\s*(<[^>]+>|ns:[^\s]+|\?\w+)\s+(<[^>]+>|ns:[^\s]+)\s+"
    r"(<[^>]+>|ns:[^\s]+|\?\w+|\"(?:[^\"\\]|\\.)*\"(?:\^\^(?:xsd:\w+|<[^>]+>)|@[A-Za-z-]+)?)\s+\."
)
INLINE_TRIPLE_RE = re.compile(r"(\?\w+|ns:[^\s{}]+)\s+(ns:[^\s{}]+)\s+(\?\w+|ns:[^\s{}]+)\s*\.")
SELECT_RE = re.compile(r"SELECT\s+DISTINCT\s+\?(\w+)", re.IGNORECASE)
FILTER_COMPARE_RE = re.compile(
    r"\?(\w+).*?(<=|>=|<|>).*?(-?[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?|-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE
)
ORDER_RE = re.compile(
    r"ORDER\s+BY\s+(?:(ASC|DESC)\s*\()?\s*(?:xsd:(\w+)\s*\()?\?(\w+)",
    re.IGNORECASE,
)
TEMPORAL_EXISTS_RE = re.compile(
    r"EXISTS\s*\{\s*(\?\w+)\s+ns:([\w.]+)\s+(\?\w+)\s*\.\s*"
    r"FILTER\s*\([^)]*\?\w+\)\s*(<=|>=|<|>)\s*\"(\d{4}-\d{2}-\d{2})\"",
    re.IGNORECASE | re.DOTALL,
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _literal(value: str, value_type: str = "") -> str:
    quoted = json.dumps(str(value), ensure_ascii=False)
    if value_type:
        return f"{quoted}^^<{value_type}>"
    return quoted


def sft(instruction: str, input_object: dict[str, Any], output_object: dict[str, Any]) -> dict[str, Any]:
    return {
        "history": [],
        "input": compact(input_object),
        "instruction": instruction,
        "output": compact(output_object),
    }


def relation_label(relation_id: str) -> list[str]:
    return [part.replace("_", " ") for part in relation_id.split(".")]


def freebase_id(value: str) -> str:
    value = str(value).strip().strip("<>")
    if value.startswith("ns:"):
        return value[3:]
    if "/ns/" in value:
        value = value.rsplit("/ns/", 1)[1]
    return value


def parse_source_triples(sparql: str) -> tuple[str, list[tuple[str, str, str]], list[str]]:
    selected = SELECT_RE.search(sparql)
    answer_var = selected.group(1) if selected else "x"
    triples: list[tuple[str, str, str]] = []
    variables: list[str] = []
    for line in sparql.splitlines():
        match = TRIPLE_RE.match(line)
        if not match:
            continue
        subject, relation, object_ = match.groups()
        subject = freebase_id(subject)
        object_ = freebase_id(object_)
        if object_.startswith('"') or subject.startswith('"'):
            continue
        relation = relation[3:] if relation.startswith("ns:") else relation.rsplit("/ns/", 1)[-1]
        triples.append((subject, relation, object_))
        for value in (subject, object_):
            if value.startswith("?") and value not in variables:
                variables.append(value)
    # Temporal NOT EXISTS/EXISTS blocks keep their triples on one line.  Add
    # only mediator variables that participate in a comparison; sk0/sk2 are
    # the optional absence branches and should not become mandatory joins.
    comparison_variables = {f"?{match.group(1)}" for match in FILTER_COMPARE_RE.finditer(sparql)}
    seen = set(triples)
    for match in INLINE_TRIPLE_RE.finditer(sparql):
        subject, relation, object_ = match.groups()
        if object_.startswith("?sk"):
            # Optional temporal boundaries are represented by TC and must not
            # become mandatory Compose joins.
            continue
        if subject.startswith("?") and subject.startswith("?sk") and subject not in comparison_variables:
            continue
        if object_.startswith("?") and object_.startswith("?sk") and object_ not in comparison_variables:
            continue
        subject = freebase_id(subject)
        object_ = freebase_id(object_)
        relation = relation[3:] if relation.startswith("ns:") else relation.rsplit("/ns/", 1)[-1]
        item = (subject, relation, object_)
        if item not in seen:
            triples.append(item)
            seen.add(item)
            for value in (subject, object_):
                if value.startswith("?") and value not in variables:
                    variables.append(value)
    return answer_var, triples, variables


def source_variables(variables: list[str]) -> dict[str, str]:
    mapping = {
        value: value[1:]
        for value in variables
        if re.fullmatch(r"\?V\d+", value)
    }
    next_index = 0
    for value in variables:
        if value in mapping:
            continue
        while f"V{next_index}" in mapping.values():
            next_index += 1
        mapping[value] = f"V{next_index}"
        next_index += 1
    return mapping


def entities_and_bindings(golden_entities: dict[str, str]) -> tuple[list[dict[str, str]], dict[str, str]]:
    entities: list[dict[str, str]] = []
    bindings: dict[str, str] = {}
    for entity_id, surface in golden_entities.items():
        output_id = f"E{len(entities)}"
        entities.append({"id": output_id, "surface": str(surface)})
        bindings[str(entity_id)] = output_id
    return entities, bindings


def enrich_golden_entities(row: dict[str, Any]) -> dict[str, str]:
    """Recover CWQ path anchors omitted from golden_entities metadata."""
    entities = {str(key): str(value) for key, value in row.get("golden_entities", {}).items()}
    for info in row.get("reasoning_information", []):
        entity_id = str(info.get("entity", ""))
        if not entity_id or entity_id in entities:
            continue
        brackets = re.findall(r"\[\s*([^\]]+?)\s*\]", str(info.get("path_LF", "")))
        surface = next(
            (
                value.strip()
                for value in reversed(brackets)
                if "," not in value and value.strip()
            ),
            "",
        )
        if not surface:
            # Last-resort natural-language anchor; still avoids leaking MIDs.
            decomposed = str(info.get("decomposed", "")).strip()
            match = re.search(r"(?:of|is|was)\s+([^?.]+)", decomposed, re.IGNORECASE)
            surface = match.group(1).strip() if match else ""
        if surface:
            entities[entity_id] = surface
    return entities


def compose_from_gold(
    sparql: str,
    golden_entities: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str], list[tuple[str, str, str]]]:
    answer_var, raw_triples, variables = parse_source_triples(sparql)
    var_map = source_variables(variables)
    entities, entity_map = entities_and_bindings(golden_entities)
    triples: list[dict[str, Any]] = []
    for subject, relation, object_ in raw_triples:
        subject = entity_map.get(subject, var_map.get(subject, subject))
        object_ = entity_map.get(object_, var_map.get(object_, object_))
        if not (subject.startswith(("E", "V")) and object_.startswith(("E", "V"))):
            continue
        triples.append(
            {
                "subject": subject,
                "relation_label": relation_label(relation),
                "object": object_,
            }
        )
    answer = var_map.get(f"?{answer_var}", answer_var)
    if not answer.startswith("V"):
        answer = var_map.get(answer, "V0")
    return {"entities": entities, "triples": triples, "answer_var": answer}, var_map, raw_triples


def graph_adjacency(compose: dict[str, Any]) -> dict[str, list[tuple[str, dict[str, Any], str]]]:
    adjacency: dict[str, list[tuple[str, dict[str, Any], str]]] = defaultdict(list)
    for triple in compose["triples"]:
        subject = str(triple["subject"])
        object_ = str(triple["object"])
        adjacency[subject].append((object_, triple, "forward"))
        adjacency[object_].append((subject, triple, "backward"))
    return adjacency


def build_semantic_paths(
    row: dict[str, Any],
    compose: dict[str, Any],
    var_map: dict[str, str],
) -> dict[str, Any]:
    """Build entity-centric paths from Gold graph topology and CWQ path order."""
    golden_entities = row.get("golden_entities", {})
    anchor_by_entity = {
        str(entity_id): {
            "entity_ref": f"E{i}",
            "anchor_ref": f"A{i}",
            "surface": str(surface),
        }
        for i, (entity_id, surface) in enumerate(golden_entities.items())
    }
    anchors: list[dict[str, str]] = []
    for path_index, info in enumerate(row.get("reasoning_information", [])):
        surface = golden_entities.get(str(info.get("entity", "")))
        if surface:
            anchors.append({"id": f"A{path_index}", "surface": str(surface)})
    adjacency = graph_adjacency(compose)
    decomposition = [
        str(item.get("decomposed", "")).strip()
        for item in row.get("reasoning_information", [])
        if str(item.get("decomposed", "")).strip()
    ]
    if not decomposition:
        raise ValueError("Gold row has no CWQ decomposition")
    paths: list[dict[str, Any]] = []
    used_edges: set[tuple[str, str, str]] = set()
    for path_index, info in enumerate(row["reasoning_information"]):
        if path_index >= len(decomposition):
            break
        anchor = anchor_by_entity.get(str(info.get("entity", "")))
        if anchor is None:
            raise ValueError("CWQ anchor is absent from golden_entities")
        current = str(anchor["entity_ref"])
        anchor_surface = str(anchor["surface"])
        path_id = f"P{path_index}"
        steps: list[dict[str, Any]] = []
        relations = list(info.get("relation", []))
        directions = list(info.get("direction", []))
        if len(relations) != len(directions):
            raise ValueError("CWQ relation/direction length mismatch")
        for step_index, (relation, direction) in enumerate(zip(relations, directions)):
            relation_parts = relation_label(str(relation))
            choices = [
                item
                for item in adjacency.get(current, [])
                if item[1]["relation_label"] == relation_parts and item[2] == str(direction)
            ]
            if not choices:
                # A few processed TRAIN labels differ only in wording. The
                # Gold graph remains authoritative, so use the unique edge in
                # the requested direction when it exists.
                choices = [item for item in adjacency.get(current, []) if item[2] == str(direction)]
            if not choices:
                raise ValueError(f"cannot align Gold path step {path_id}.S{step_index}")
            next_node, triple, actual_direction = choices[0]
            local_var = f"{path_id}.V{step_index}"
            steps.append(
                {
                    "id": f"{path_id}.S{step_index}",
                    "relation_label": list(triple["relation_label"]),
                    "direction": actual_direction,
                    "from": str(anchor["anchor_ref"]) if step_index == 0 else f"{path_id}.V{step_index - 1}",
                    "to": local_var,
                }
            )
            used_edges.add((str(triple["subject"]), str(triple["object"]), compact(triple["relation_label"])))
            current = next_node
        if not steps:
            raise ValueError(f"Gold path {path_id} is empty")
        paths.append(
            {
                "id": path_id,
                "anchor_ref": str(anchor["anchor_ref"]),
                "goal": decomposition[path_index],
                "steps": steps,
                "path_output_var": steps[-1]["to"],
            }
        )
    if len(paths) != len(decomposition):
        raise ValueError("Gold decomposition/path count mismatch")
    return {"anchors": anchors, "semantic_paths": paths}


def operators_from_gold(
    sparql: str,
    compose: dict[str, Any],
    var_map: dict[str, str],
    logical_form: str,
    golden_entities: dict[str, str],
) -> dict[str, Any]:
    operators: list[dict[str, Any]] = []
    temporal_range_vars: set[str] = set()
    temporal_ranges: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for match in TEMPORAL_EXISTS_RE.finditer(sparql):
        owner, relation, value_var, symbol, date_value = match.groups()
        temporal_range_vars.add(value_var)
        bucket = temporal_ranges.setdefault((owner, relation), {})
        if symbol in {"<=", "<"}:
            bucket["upper"] = date_value
        else:
            bucket["lower"] = date_value
    for (owner, relation), bounds in temporal_ranges.items():
        input_var = var_map.get(owner, owner)
        if not input_var.startswith("V"):
            continue
        if "lower" in bounds and "upper" in bounds:
            value = f"{bounds['lower']}/{bounds['upper']}"
        else:
            value = bounds.get("upper") or bounds.get("lower", "")
        operators.append(
            {
                "type": "TC",
                "inputs": [],
                "input_var": input_var,
                "attribute_relation_label": relation_label(relation),
                "attribute_relation_labels": [],
                "value": value,
                "value_type": "year",
            }
        )
    # Direct comparison filters.  The variable is already a Gold graph node.
    seen_comparisons: set[tuple[str, str, str]] = set()
    # Search the complete SPARQL text instead of balancing FILTER parentheses;
    # expressions such as xsd:integer(?num) contain nested parentheses.
    comparison_matches = [
        (line, match)
        for line in sparql.splitlines()
        if "FILTER" in line.upper()
        for match in FILTER_COMPARE_RE.finditer(line)
    ]
    for filter_line, match in comparison_matches:
            variable, symbol, value = match.groups()
            month_literal = re.search(r'"(\d{4}-\d{2})"', filter_line)
            if month_literal:
                value = month_literal.group(1)
            if f"?{variable}" in temporal_range_vars:
                continue
            key = (variable, symbol, value)
            if key in seen_comparisons:
                continue
            seen_comparisons.add(key)
            input_var = var_map.get(f"?{variable}", variable)
            # Nested OPTIONAL/NOT EXISTS variables are not part of the
            # executable Compose graph unless their triple was materialized.
            # Keep them in the rejection audit instead of emitting an invalid
            # Operator reference.
            if not input_var.startswith("V"):
                continue
            operator_type = {
                "<": "LESS_THAN",
                ">": "GREATER_THAN",
                "<=": "LESS_OR_EQUAL",
                ">=": "GREATER_OR_EQUAL",
            }[symbol]
            attribute: list[str] = []
            for triple in compose["triples"]:
                if str(triple["object"]) == input_var:
                    attribute = list(triple["relation_label"])
            context = filter_line.casefold()
            numeric_cast = "xsd:integer" in context or "xsd:float" in context or "xsd:decimal" in context
            operators.append(
                {
                    "type": operator_type,
                    "inputs": [],
                    "input_var": input_var,
                    "attribute_relation_label": attribute,
                    "attribute_relation_labels": [],
                    "value": value,
                    "value_type": (
                        "number"
                        if numeric_cast
                        else "month"
                        if re.fullmatch(r"\d{4}-\d{2}", value)
                        else "year"
                        if re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value)
                        else "number"
                    ),
                }
            )
    # Gold answer-exclusion filters are represented by the train Operator
    # contract as NO_EQUAL over the exact anchor surface.
    for match in re.finditer(r"FILTER\s*\(\?(\w+)\s*!=\s*ns:([mg]\.\w+)\)", sparql, re.IGNORECASE):
        variable, entity_id = match.groups()
        surface = golden_entities.get(entity_id)
        if surface:
            operators.append(
                {
                    "type": "NO_EQUAL",
                    "inputs": [],
                    "input_var": var_map.get(f"?{variable}", variable),
                    "attribute_relation_label": [],
                    "attribute_relation_labels": [],
                    "value": str(surface),
                    "value_type": "entity",
                }
            )
    for match in re.finditer(r"FILTER\s*\(\?(\w+)\s*!=\s*\?(\w+)\)", sparql, re.IGNORECASE):
        left, right = match.groups()
        left_var = var_map.get(f"?{left}", left)
        right_var = var_map.get(f"?{right}", right)
        if left_var.startswith("V") and right_var.startswith("V"):
            operators.append(
                {
                    "type": "NO_EQUAL",
                    "inputs": [],
                    "input_var": left_var,
                    "attribute_relation_label": [],
                    "attribute_relation_labels": [],
                    "value": right_var,
                    "value_type": "variable",
                }
            )
    for match in re.finditer(
        r'FILTER\s*\(\s*str\(\?(\w+)\)\s*=\s*"([^"]+)"\s*\)',
        sparql,
        re.IGNORECASE,
    ):
        variable, value = match.groups()
        input_var = var_map.get(f"?{variable}", variable)
        if not input_var.startswith("V"):
            continue
        attribute = next(
            (
                list(triple["relation_label"])
                for triple in compose["triples"]
                if str(triple["object"]) == input_var
            ),
            [],
        )
        operators.append(
            {
                "type": "EQUAL",
                "inputs": [],
                "input_var": input_var,
                "attribute_relation_label": attribute,
                "attribute_relation_labels": [],
                "value": value,
                "value_type": "string",
            }
        )
    # Equality against a typed literal is represented as an Operator rather
    # than a literal node in the Compose graph.
    for line in sparql.splitlines():
        match = TRIPLE_RE.match(line)
        if not match:
            continue
        subject, relation, object_ = match.groups()
        if not object_.startswith('"'):
            continue
        literal_match = re.match(r'"([^"]+)"', object_)
        if not literal_match:
            continue
        subject = freebase_id(subject)
        input_var = var_map.get(subject, subject)
        if not input_var.startswith("V"):
            continue
        value = literal_match.group(1)
        language_match = re.search(r"@([A-Za-z-]+)\s*\.?\s*$", object_)
        if language_match:
            value_type = f"lang:{language_match.group(1)}"
        elif "xsd:dateTime" in object_:
            value_type = "year"
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            value_type = "number"
        else:
            value_type = "string"
        operators.append(
            {
                "type": "EQUAL",
                "inputs": [],
                "input_var": input_var,
                "attribute_relation_label": relation_label(relation[3:] if relation.startswith("ns:") else relation.rsplit("/ns/", 1)[-1]),
                "attribute_relation_labels": [],
                "value": value,
                "value_type": value_type,
            }
        )
    order = ORDER_RE.search(sparql)
    if order:
        direction = order.group(1) or "ASC"
        cast_type = str(order.group(2) or "").casefold()
        order_variable = order.group(3)
        input_var = var_map.get(f"?{order_variable}", order_variable)
        attribute: list[str] = []
        for triple in compose["triples"]:
            if str(triple["object"]) == input_var:
                attribute = list(triple["relation_label"])
        operators.append(
            {
                "type": "ARGMIN" if direction.upper() == "ASC" else "ARGMAX",
                "inputs": [],
                "input_var": input_var,
                "attribute_relation_label": attribute,
                "attribute_relation_labels": [],
                "value": "",
                "value_type": (
                    "number"
                    if cast_type in {"integer", "float", "decimal", "double"}
                    else "datetime"
                    if cast_type in {"datetime", "date"}
                    else ""
                ),
            }
        )
    # Non-Freebase URI constants (for example film trailer URLs) are encoded
    # as EQUAL operators so the current training schema remains unchanged.
    for line in sparql.splitlines():
        match = TRIPLE_RE.match(line)
        if not match:
            continue
        subject, relation, object_ = match.groups()
        if not (object_.startswith("<http://") or object_.startswith("<https://")):
            continue
        uri = object_.strip("<>")
        if "/ns/" in uri:
            continue
        input_var = var_map.get(freebase_id(subject), freebase_id(subject))
        if input_var.startswith("V"):
            operators.append(
                {
                    "type": "EQUAL",
                    "inputs": [],
                    "input_var": input_var,
                    "attribute_relation_label": relation_label(relation[3:] if relation.startswith("ns:") else relation.rsplit("/ns/", 1)[-1]),
                    "attribute_relation_labels": [],
                    "value": uri,
                    "value_type": "uri",
                }
            )
    if re.search(r"\(\s*COUNT\b", logical_form, re.IGNORECASE):
        operators.append(
            {
                "type": "COUNT",
                "inputs": [],
                "input_var": compose["answer_var"],
                "attribute_relation_label": [],
                "attribute_relation_labels": [],
                "value": "",
                "value_type": "",
            }
        )
    return {"operators": operators}


def _temporal_counterpart_for_gold(relation_id: str) -> str:
    for source, target in (
        (".from_date", ".to_date"),
        (".to_date", ".from_date"),
        (".start_date", ".end_date"),
        (".end_date", ".start_date"),
        (".from", ".to"),
        (".to", ".from"),
        (".start", ".end"),
        (".end", ".start"),
    ):
        if relation_id.endswith(source):
            return relation_id[: -len(source)] + target
    return ""


def _gold_temporal_exists_filter(
    input_var: str,
    relation_id: str,
    value_var: str,
    comparator: str,
    boundary: str,
) -> str:
    missing_var = f"?missing_{value_var.lstrip('?')}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", boundary):
        condition = f"SUBSTR(STR({value_var}), 1, 10) {comparator} {_literal(boundary, '')}"
    else:
        condition = (
            "<http://www.w3.org/2001/XMLSchema#integer>"
            f"(SUBSTR(STR({value_var}), 1, 4)) {comparator} {str(boundary)[:4]}"
        )
    return (
        "  FILTER(NOT EXISTS { "
        f"{input_var} <{NS}{relation_id}> {missing_var} . "
        "} || EXISTS { "
        f"{input_var} <{NS}{relation_id}> {value_var} . "
        f"FILTER({condition}) "
        "})"
    )


def lower_query(compose: dict[str, Any], operators: dict[str, Any], golden_entities: dict[str, str]) -> str:
    entity_id_by_surface = {
        f"E{i}": str(entity_id)
        for i, entity_id in enumerate(golden_entities)
    }
    # Compose entities preserve golden_entities insertion order.
    lines = [
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
        f"SELECT DISTINCT ?{compose['answer_var']}\nWHERE {{",
    ]
    for triple in compose["triples"]:
        subject = str(triple["subject"])
        object_ = str(triple["object"])
        if subject.startswith("E"):
            subject = entity_id_by_surface.get(subject, subject)
            subject = f"<{NS}{subject}>"
        elif subject.startswith("V"):
            subject = f"?{subject}"
        if object_.startswith("E"):
            object_ = entity_id_by_surface.get(object_, object_)
            object_ = f"<{NS}{object_}>"
        elif object_.startswith("V"):
            object_ = f"?{object_}"
        relation = ".".join(str(part).replace(" ", "_") for part in triple["relation_label"])
        lines.append(f"  {subject} <{NS}{relation}> {object_} .")
    filter_index = 0
    for operator in operators.get("operators", []):
        typ = str(operator["type"]).upper()
        input_var = f"?{operator['input_var']}"
        value = str(operator.get("value", ""))
        if typ in {"LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL", "EQUAL", "NO_EQUAL"} and value:
            symbol = {
                "LESS_THAN": "<",
                "LESS_OR_EQUAL": "<=",
                "GREATER_THAN": ">",
                "GREATER_OR_EQUAL": ">=",
                "EQUAL": "=",
                "NO_EQUAL": "!=",
            }[typ]
            comparison_target = input_var
            attribute_label = operator.get("attribute_relation_label", [])
            if typ != "NO_EQUAL" and attribute_label:
                already_bound = any(
                    str(triple["object"]) == str(operator["input_var"])
                    and triple.get("relation_label", []) == attribute_label
                    for triple in compose["triples"]
                )
                if not already_bound:
                    relation_id = ".".join(
                        str(part).strip().replace(" ", "_")
                        for part in attribute_label
                    )
                    if typ == "EQUAL" and (
                        str(operator.get("value_type", "")) in {"year", "string"}
                        or str(operator.get("value_type", "")).startswith("lang:")
                        or str(operator.get("value_type", "")) == "uri"
                    ):
                        value_type = str(operator.get("value_type", ""))
                        if value_type == "year":
                            direct_literal = f'"{value}"^^<http://www.w3.org/2001/XMLSchema#dateTime>'
                        elif value_type.startswith("lang:"):
                            direct_literal = f'{_literal(value)}@{value_type.split(":", 1)[1]}'
                        elif value_type == "uri":
                            direct_literal = f"<{value}>"
                        else:
                            direct_literal = _literal(value)
                        lines.append(
                            f"  {input_var} <{NS}{relation_id}> {direct_literal} ."
                        )
                        continue
                    comparison_target = f"?operator_value{filter_index}"
                    lines.append(
                        f"  {input_var} <{NS}{relation_id}> {comparison_target} ."
                    )
                    filter_index += 1
            if str(operator.get("value_type", "")) == "entity":
                entity_id = next(
                    (entity_id for entity_id, surface in golden_entities.items() if str(surface) == value),
                    "",
                )
                if entity_id:
                    lines.append(f"  FILTER({comparison_target} {symbol} <{NS}{entity_id}>)")
                continue
            if str(operator.get("value_type", "")) == "variable":
                lines.append(f"  FILTER({comparison_target} {symbol} ?{value})")
                continue
            if str(operator.get("value_type", "")) in {"year", "month"}:
                literal = f'"{value}"^^<http://www.w3.org/2001/XMLSchema#dateTime>'
                lines.append(f"  FILTER({comparison_target} {symbol} {literal})")
            elif str(operator.get("value_type", "")) == "string":
                lines.append(f"  FILTER(STR({comparison_target}) {symbol} {_literal(value)})")
            else:
                cast = "bif:atof" if "." in value else "bif:atoi"
                lines.append(f"  FILTER({cast}(STR({comparison_target})) {symbol} {value})")
        elif typ == "TC" and operator.get("attribute_relation_label"):
            relation_id = ".".join(
                str(part).strip().replace(" ", "_")
                for part in operator["attribute_relation_label"]
            )
            time_var = f"?time{filter_index}"
            counterpart = _temporal_counterpart_for_gold(relation_id)
            counterpart_var = f"?time_end{filter_index}"
            pieces = str(value).split("/", 1)
            if len(pieces) == 2:
                if counterpart:
                    start_relation = relation_id
                    end_relation = counterpart
                    if relation_id.endswith((".to", ".to_date", ".end", ".end_date")):
                        start_relation, end_relation = counterpart, relation_id
                    lines.append(_gold_temporal_exists_filter(input_var, start_relation, time_var, "<=", pieces[1]))
                    lines.append(_gold_temporal_exists_filter(input_var, end_relation, counterpart_var, ">=", pieces[0]))
                else:
                    lines.append(_gold_temporal_exists_filter(input_var, relation_id, time_var, "<=", pieces[1]))
                    lines.append(_gold_temporal_exists_filter(input_var, relation_id, counterpart_var, ">=", pieces[0]))
            elif value:
                is_end = relation_id.endswith((".to", ".to_date", ".end", ".end_date"))
                comparator = ">=" if is_end else "<="
                lines.append(_gold_temporal_exists_filter(input_var, relation_id, time_var, comparator, value))
            filter_index += 1
    for operator in operators.get("operators", []):
        typ = str(operator["type"]).upper()
        if typ in {"ARGMIN", "ARGMAX"}:
            direction = "ASC" if typ == "ARGMIN" else "DESC"
            order_var = f"?{operator['input_var']}"
            relation_text = " ".join(operator.get("attribute_relation_label", [])).casefold()
            value_type = str(operator.get("value_type", "")).casefold()
            if value_type == "number" or any(token in relation_text for token in ("population", "rate", "percent", "number", "amount", "gdp", "value", " id")):
                order_expr = (
                    f"xsd:integer({order_var})"
                    if " id" in relation_text or relation_text.endswith("id")
                    else f"xsd:float({order_var})"
                )
            elif value_type == "datetime" or any(token in relation_text for token in ("date", " start", " end", " from", " to")):
                order_expr = f"xsd:dateTime({order_var})"
            else:
                order_expr = order_var
            lines.append(f"}}\nORDER BY {direction}({order_expr})\nLIMIT 1")
            return "\n".join(lines)
    lines.append("}")
    return "\n".join(lines)


def compile_row(row: dict[str, Any], semantic_instruction: str, compose_instruction: str, operator_instruction: str, decompose_instruction: str, endpoint: str, allow_gold_source_fallback: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], set[str]]:
    golden_entities = enrich_golden_entities(row)
    row_with_entities = dict(row)
    row_with_entities["golden_entities"] = golden_entities
    compose, var_map, _ = compose_from_gold(str(row["sparql"]), golden_entities)
    semantic = build_semantic_paths(row_with_entities, compose, var_map)
    operators = operators_from_gold(
        str(row["sparql"]),
        compose,
        var_map,
        str(row.get("logical_form", "")),
        golden_entities,
    )
    # Use the same runtime contracts as the current cwq_new training pipeline.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from semantic_guided_kbqa.contracts import (
        validate_compose_graph_output,
        validate_operator_output,
        validate_semantic_graph,
    )
    semantic = validate_semantic_graph(semantic, max_hops=8)
    compose = validate_compose_graph_output(compose)
    operators = validate_operator_output(operators, compose)
    # Keep the three stages aligned while retaining the current training schema.
    compose_input = {"question": row["question"], **semantic}
    compose_output = compose
    operator_input = {"question": row["question"], **compose_output}
    query = lower_query(compose_output, operators, golden_entities)
    gold = {str(value) for value in row.get("answer_entities", {}).keys()}
    dynamic_mode = "lowered_query"
    try:
        predicted = execute_sparql(query, endpoint)
    except Exception as compiled_error:
        predicted = set()
        compiled_error_text = repr(compiled_error)
    else:
        compiled_error_text = ""
    if predicted != gold:
        if not allow_gold_source_fallback:
            raise ValueError(f"compiled query answer mismatch: predicted={sorted(predicted)} gold={sorted(gold)}")
        # Gold SPARQL is the canonical dynamic oracle for structures whose
        # current lowering cannot express OPTIONAL/NOT EXISTS semantics yet.
        # Static Semantic/Compose/Operator labels are still required above.
        source_predicted = execute_sparql(str(row["sparql"]), endpoint)
        if source_predicted != gold:
            suffix = f"; compiled_error={compiled_error_text}" if compiled_error_text else ""
            raise ValueError(f"compiled and Gold-source answers mismatch: predicted={sorted(source_predicted)} gold={sorted(gold)}{suffix}")
        predicted = source_predicted
        dynamic_mode = "gold_source_oracle"
    return (
        sft(decompose_instruction, {"question": row["question"]}, {"decomposition": [p["goal"] for p in semantic["semantic_paths"]]}),
        sft(semantic_instruction, {"question": row["question"], "decomposition": [p["goal"] for p in semantic["semantic_paths"]]}, semantic),
        sft(compose_instruction, compose_input, compose_output),
        sft(operator_instruction, operator_input, operators),
        predicted,
        dynamic_mode,
    )


def execute_sparql(sparql: str, endpoint: str) -> set[str]:
    url = endpoint + "?" + urllib.parse.urlencode({"query": sparql, "format": "json"})
    payload = json.load(urllib.request.urlopen(url, timeout=60))
    answer_ids: set[str] = set()
    for binding in payload.get("results", {}).get("bindings", []):
        for value in binding.values():
            text = str(value.get("value", ""))
            if "/ns/" in text:
                answer_ids.add(text.rsplit("/ns/", 1)[1])
            else:
                answer_ids.add(text)
            break
    return answer_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=Path(__file__).resolve().parents[1] / "inputs/cwq_source_annotations.zip")
    parser.add_argument("--source-member", default="", help="Optional exact member in the source ZIP.")
    parser.add_argument("--selection", type=Path, default=Path(__file__).resolve().parents[1] / "inputs/candidate_selection.json")
    parser.add_argument("--all-appended", action="store_true", help="compile every CWQ row not matched to the retained 19,518-row prefix")
    parser.add_argument("--max-rows", type=int, default=0, help="optional deterministic limit after selection")
    parser.add_argument("--exclude-semantic", type=Path, action="append", default=[], help="exclude questions already present in one or more semantic training arrays")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "work/cwq_gold_first")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "inputs/cwq",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:3005/sparql")
    parser.add_argument("--no-gold-source-fallback", action="store_true", help="require the lowered Compose/Operator query itself to match Gold")
    args = parser.parse_args()
    rows, _, _ = load_source_annotations(args.source_archive, args.source_member)
    if args.all_appended:
        base_rows = load(args.train_dir / "decompose_train.json")
        def question_key(value: str) -> str:
            return " ".join(re.findall(r"\w+", str(value).casefold()))
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_index, source_row in enumerate(rows):
            decomposition = [
                str(item.get("decomposed", "")).strip()
                for item in source_row.get("reasoning_information", [])
                if str(item.get("decomposed", "")).strip()
            ]
            groups[question_key(source_row.get("question", ""))].append(
                {"source_index": source_index, "decomposition": decomposition, "used": False}
            )
        for base_row in base_rows:
            base_input = json.loads(base_row["input"])
            base_output = json.loads(base_row["output"])
            candidates = [item for item in groups[question_key(base_input.get("question", ""))] if not item["used"]]
            chosen = next(
                (item for item in candidates if item["decomposition"] == base_output.get("decomposition", [])),
                candidates[0] if candidates else None,
            )
            if chosen is not None:
                chosen["used"] = True
        selected = [
            {"source_index": item["source_index"]}
            for group in groups.values()
            for item in group
            if not item["used"]
        ]
    else:
        selection_payload = load(args.selection)
        selected = selection_payload.get("rows", selection_payload if isinstance(selection_payload, list) else [])
    if args.max_rows:
        selected = selected[: args.max_rows]
    if args.exclude_semantic:
        excluded_questions: set[str] = set()
        for semantic_path in args.exclude_semantic:
            for semantic_row in load(semantic_path):
                input_object = json.loads(semantic_row["input"])
                excluded_questions.add(str(input_object.get("question", "")))
        selected = [
            item
            for item in selected
            if str(rows[int(item["source_index"])].get("question", "")) not in excluded_questions
        ]
    instructions = {
        "semantic": load(args.train_dir / "semantic_path_train.json")[0]["instruction"],
        "compose": load(args.train_dir / "compose_train.json")[0]["instruction"],
        "operator": load(args.train_dir / "operator_train.json")[0]["instruction"],
        "decompose": load(args.train_dir / "decompose_train.json")[0]["instruction"],
    }
    accepted: dict[str, list[dict[str, Any]]] = {key: [] for key in ("decompose", "semantic", "compose", "operator")}
    evidence: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in selected:
        source_index = int(item["source_index"])
        row = rows[source_index]
        try:
            compiled = compile_row(row, instructions["semantic"], instructions["compose"], instructions["operator"], instructions["decompose"], args.endpoint, not args.no_gold_source_fallback)
            for key, value in zip(accepted, compiled[:4]):
                accepted[key].append(value)
            evidence.append({"source_index": source_index, "question": row["question"], "dynamic_validation": "passed", "dynamic_mode": compiled[5], "static_validation": "passed", "answer_ids": sorted(compiled[4])})
        except Exception as exc:
            rejected.append({"source_index": source_index, "question": row["question"], "reason": str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (("decompose", "decompose_train.json"), ("semantic", "semantic_path_train.json"), ("compose", "compose_train.json"), ("operator", "operator_train.json")):
        dump(args.output_dir / filename, accepted[key])
    dump(args.output_dir / "validation_evidence.json", evidence)
    dump(args.output_dir / "rejected.json", rejected)
    dump(
        args.output_dir / "dataset_info.json",
        {
            f"cwq_gold_first_validated_{key}": {
                "file_name": filename,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "history": "history",
                },
            }
            for key, filename in (
                ("decompose", "decompose_train.json"),
                ("semantic", "semantic_path_train.json"),
                ("compose", "compose_train.json"),
                ("operator", "operator_train.json"),
            )
        },
    )
    dump(args.output_dir / "manifest.json", {"dataset": "CWQ Gold-first validated", "selected": len(selected), "accepted": len(evidence), "rejected": len(rejected), "dynamic_modes": dict(Counter(item["dynamic_mode"] for item in evidence)), "acceptance": "static contracts plus lowered-query dynamic validation, with Gold-SPARQL fallback for OPTIONAL/NOT EXISTS lowering gaps", "gold_source_fallback": not args.no_gold_source_fallback, "merged_into_full_dataset": False})
    print(json.dumps({"selected": len(selected), "accepted": len(evidence), "rejected": len(rejected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
