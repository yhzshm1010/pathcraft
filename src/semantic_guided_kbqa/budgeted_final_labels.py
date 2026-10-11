"""Fill only missing final names, paid from this question's saved KG budget.

This presentation-only step never changes IDs, graphs, operators, model
requests or genuine existing labels. It cannot use Gold, old traces or aliases
to pick a name. Only a unique actual English type.object.name is accepted.
"""
from collections import defaultdict
import time

from .final_label_evidence import _entity_id, _genuine_label
from .question_query_budget import current_query_budget, execute_repair_from_saved_budget, charge_repair_preparation


def complete_budgeted_final_labels(result, kg, *, linked_context=None):
    state = current_query_budget()
    if state is None:
        return result
    ids = result.get("answer_ids")
    answers = result.get("answers")
    if not isinstance(ids, list) or not ids or not isinstance(answers, list):
        return result
    ordered = list(dict.fromkeys(_entity_id(x) for x in ids if _entity_id(x)))
    genuine = {_entity_id(a.get("id")) for a in answers if isinstance(a, dict)
               and (_genuine_label(a.get("label")) or
                    (_entity_id(a.get("label")) and
                     _entity_id(a.get("label")) != _entity_id(a.get("id"))))}
    missing = [x for x in ordered if x not in genuine]
    if not missing or len(missing) > 128:
        return result
    preparation_started = time.monotonic()
    names = defaultdict(dict)
    for identifier in missing:
        names[identifier].update(state.kg_name_evidence.get(identifier, {}))
    for item in linked_context or []:
        if not isinstance(item, dict):
            continue
        identifier, label = _entity_id(item.get("id")), _genuine_label(item.get("name"))
        if identifier in missing and label:
            names[identifier].setdefault(" ".join(label.casefold().split()), label)
    to_query = [x for x in missing if not names[x]]
    # Individual known entity IDs are validated by _entity_id, not guessed
    # from question text, model-generated names or evaluation fields.
    values = " ".join("<http://rdf.freebase.com/ns/" + x + ">" for x in to_query)
    query = ("SELECT DISTINCT ?entity ?label WHERE {\n"
             "  VALUES ?entity { " + values + " }\n"
             "  ?entity <http://rdf.freebase.com/ns/type.object.name> ?name .\n"
             '  FILTER(LANGMATCHES(LANG(?name), "EN"))\n'
             "  BIND(STR(?name) AS ?label)\n}")
    rows = None
    execution = {"status": "not_executed", "reason": "existing_name_evidence_or_no_saved_budget"}
    if (to_query and kg is not None and state.request_credits > 0
            and state.seconds_saved - state.repair_seconds >= 0.02):
        charge_repair_preparation(preparation_started)
        rows, execution = execute_repair_from_saved_budget(kg, query)
        preparation_started = time.monotonic()
    if rows is not None:
        for row in rows:
            identifier = _entity_id(row.get("entity"))
            label = _genuine_label(row.get("label"))
            if identifier in missing and label:
                normalized = " ".join(label.casefold().split())
                names[identifier].setdefault(normalized, label)
    unique = {identifier: next(iter(labels.values())) for identifier, labels in names.items()
              if len(labels) == 1}
    filled = set()
    output = []
    present = set()
    for item in answers:
        if not isinstance(item, dict):
            output.append(item)
            continue
        identifier = _entity_id(item.get("id"))
        present.add(identifier)
        if identifier in unique and not _genuine_label(item.get("label")):
            output.append({**item, "label": unique[identifier]})
            filled.add(identifier)
        else:
            output.append(item)
    for raw in ids:
        identifier = _entity_id(raw)
        if identifier not in present and identifier in unique:
            output.append({"id": raw, "label": unique[identifier]})
            present.add(identifier)
            filled.add(identifier)
    if not filled and execution["status"] == "not_executed":
        charge_repair_preparation(preparation_started)
        return result
    evidence = {"status": "filled_missing_final_names" if filled else "original_names_preserved",
                "execution": execution,
                "queried_missing_entity_count": len(to_query) if execution["status"] != "not_executed" else 0,
                "reused_current_question_kg_name_evidence": bool(filled and len(to_query) < len(missing)),
                "filled_entity_ids": sorted(filled),
                "conflicting_name_entity_ids": sorted(x for x, n in names.items() if len(n) > 1),
                "answer_ids_changed": False, "graph_changed": False,
                "genuine_names_overwritten": False, "additional_model_calls": 0,
                "uses_evaluation_gold": False, "uses_historical_trace": False,
                "request_and_time_paid_from_exact_duplicate_queries": True}
    completed = {**result, "answers": output,
                 "traces": [*result.get("traces", []),
                            {"stage": "budgeted_final_entity_names", "input": {}, "output": evidence}]}
    charge_repair_preparation(preparation_started)
    return completed
