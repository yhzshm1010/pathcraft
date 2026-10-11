"""Consolidated labels; original rules and budgets retained."""
from __future__ import annotations


# ---- final_label_evidence ----
"""Complete final entity labels from this invocation's existing name evidence.

This is deliberately a presentation-only finalization step. It never retrieves
labels, reads historical results, sees evaluation Gold, or changes the answer
set. The explicit trace paths below contain supplied entity context or names
attached to executed KG answers; model-generated graph/repair text is excluded.
"""


import re
from typing import Any, Iterable

from .evaluation import canonicalize_answer_id


_ENTITY_ID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")


def _final_label_evidence_entity_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    canonical = canonicalize_answer_id(value)
    return canonical if _ENTITY_ID.fullmatch(canonical) else ""


def _final_label_evidence_genuine_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    label = value.strip()
    # A serialized entity URI or MID is a fallback, not a name. Do not
    # reinterpret literals/dates/numbers, or use another entity ID as a name.
    return label if label and not _final_label_evidence_entity_id(label) else ""


def _executed_answers(execution: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(execution, dict):
        return
    answers = execution.get("answers")
    ids = execution.get("answer_ids")
    if not isinstance(answers, list) or not isinstance(ids, list):
        return
    allowed_ids = {_final_label_evidence_entity_id(value) for value in ids}
    for answer in answers:
        if isinstance(answer, dict) and _final_label_evidence_entity_id(answer.get("id")) in allowed_ids:
            yield answer


def complete_final_answer_labels(result: dict[str, Any]) -> dict[str, Any]:
    """Fill missing/fallback labels without changing final answers or choices.

    Conflicting genuine names abstain rather than choosing an alias by score,
    frequency, or source. Already-present genuine names are never overwritten.
    Results without a missing entity label are returned untouched, avoiding
    trace scanning for the common path. The caller must invoke this only after
    all inference, selection, restoration, and failure recovery are finished.
    """
    answer_ids = result.get("answer_ids")
    answers = result.get("answers")
    if not isinstance(answer_ids, list) or not answer_ids:
        return result
    if not isinstance(answers, list):
        answers = []
    final_ids = {_final_label_evidence_entity_id(value) for value in answer_ids} - {""}
    if not final_ids:
        return result
    present = {
        _final_label_evidence_entity_id(answer.get("id"))
        for answer in answers
        if isinstance(answer, dict)
    }
    missing = final_ids - present
    missing.update(
        _final_label_evidence_entity_id(answer.get("id"))
        for answer in answers
        if isinstance(answer, dict)
        and _final_label_evidence_entity_id(answer.get("id")) in final_ids
        and not _final_label_evidence_genuine_label(answer.get("label"))
    )
    if not missing:
        return result

    # Keep only evidence relevant to final missing IDs, not whole KG results.
    evidence: dict[str, dict[str, dict[str, Any]]] = {}

    def remember(answer: dict[str, Any], source: str, field: str = "label") -> None:
        entity_id = _final_label_evidence_entity_id(answer.get("id"))
        label = _final_label_evidence_genuine_label(answer.get(field))
        if entity_id not in missing or not label:
            return
        # Mere capitalization/whitespace variants are not conflicting aliases.
        normalized = " ".join(label.casefold().split())
        record = evidence.setdefault(entity_id, {}).setdefault(
            normalized, {"label": label, "sources": set()}
        )
        record["sources"].add(source)

    for answer in answers:
        if isinstance(answer, dict):
            remember(answer, "final_kg_answer")
    traces = result.get("traces")
    if not isinstance(traces, list):
        traces = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        stage, output = trace.get("stage"), trace.get("output")
        if stage == "linked_entity_context" and isinstance(output, list):
            for entity in output:
                if isinstance(entity, dict):
                    remember(entity, "supplied_linked_entity_context", "name")
        elif stage == "freebase_execution" and isinstance(output, list):
            for execution in output:
                for answer in _executed_answers(execution):
                    remember(answer, "freebase_execution")
        elif stage in {"glm_final_graph_ranking", "executed_graph_selection"}:
            if not isinstance(output, dict):
                continue
            candidates = output.get("candidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    for answer in _executed_answers(candidate):
                        remember(answer, "executed_selector_candidate")

    resolved = {
        entity_id: next(iter(names.values()))
        for entity_id, names in evidence.items()
        if len(names) == 1
    }
    conflicts = sorted(
        entity_id for entity_id, names in evidence.items() if len(names) > 1
    )
    if not resolved and not conflicts:
        return result

    completed = []
    filled_ids: set[str] = set()
    for answer in answers:
        if not isinstance(answer, dict):
            completed.append(answer)
            continue
        entity_id = _final_label_evidence_entity_id(answer.get("id"))
        if entity_id in resolved and not _final_label_evidence_genuine_label(answer.get("label")):
            completed.append({**answer, "label": resolved[entity_id]["label"]})
            filled_ids.add(entity_id)
        else:
            completed.append(answer)
    # Preserve IDs and ordering as supplied by inference, including URI form.
    for raw_id in answer_ids:
        entity_id = _final_label_evidence_entity_id(raw_id)
        if entity_id not in present and entity_id in resolved:
            completed.append({"id": raw_id, "label": resolved[entity_id]["label"]})
            present.add(entity_id)
            filled_ids.add(entity_id)

    diagnostics = {
        "status": "completed" if filled_ids else "conflicting_evidence_abstained",
        "missing_entity_label_count": len(missing),
        "filled_count": len(filled_ids),
        "filled_entity_ids": sorted(filled_ids),
        "conflicting_entity_ids": conflicts,
        "unresolved_count": len(missing - filled_ids),
        "evidence_sources": {
            entity_id: sorted(resolved[entity_id]["sources"])
            for entity_id in sorted(filled_ids)
        },
        "answer_ids_changed": False,
        "graph_changed": False,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "uses_gold_answers": False,
        "uses_historical_trace": False,
    }
    return {
        **result,
        "answers": completed,
        "traces": [
            *traces,
            {"stage": "final_answer_label_evidence", "input": {}, "output": diagnostics},
        ],
    }



# ---- budgeted_final_labels ----
"""Fill only missing final names, paid from this question's saved KG budget.

This presentation-only step never changes IDs, graphs, operators, model
requests or genuine existing labels. It cannot use Gold, old traces or aliases
to pick a name. Only a unique actual English type.object.name is accepted.
"""
from collections import defaultdict
import time

_budgeted_final_labels_entity_id = _final_label_evidence_entity_id
_budgeted_final_labels_genuine_label = _final_label_evidence_genuine_label
from .runtime_context import current_query_budget, execute_repair_from_saved_budget, charge_repair_preparation


def complete_budgeted_final_labels(result, kg, *, linked_context=None):
    state = current_query_budget()
    if state is None:
        return result
    ids = result.get("answer_ids")
    answers = result.get("answers")
    if not isinstance(ids, list) or not ids or not isinstance(answers, list):
        return result
    ordered = list(dict.fromkeys(_budgeted_final_labels_entity_id(x) for x in ids if _budgeted_final_labels_entity_id(x)))
    genuine = {_budgeted_final_labels_entity_id(a.get("id")) for a in answers if isinstance(a, dict)
               and (_budgeted_final_labels_genuine_label(a.get("label")) or
                    (_budgeted_final_labels_entity_id(a.get("label")) and
                     _budgeted_final_labels_entity_id(a.get("label")) != _budgeted_final_labels_entity_id(a.get("id"))))}
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
        identifier, label = _budgeted_final_labels_entity_id(item.get("id")), _budgeted_final_labels_genuine_label(item.get("name"))
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
            identifier = _budgeted_final_labels_entity_id(row.get("entity"))
            label = _budgeted_final_labels_genuine_label(row.get("label"))
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
        identifier = _budgeted_final_labels_entity_id(item.get("id"))
        present.add(identifier)
        if identifier in unique and not _budgeted_final_labels_genuine_label(item.get("label")):
            output.append({**item, "label": unique[identifier]})
            filled.add(identifier)
        else:
            output.append(item)
    for raw in ids:
        identifier = _budgeted_final_labels_entity_id(raw)
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



__all__ = ['complete_budgeted_final_labels', 'complete_final_answer_labels']
