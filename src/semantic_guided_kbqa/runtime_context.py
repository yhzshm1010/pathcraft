"""Consolidated runtime context; original rules and budgets retained."""
from __future__ import annotations


# ---- question_context ----
"""Read-only KG context for already-linked entities; never reads Gold answers."""
import re
from .evaluation import canonicalize_answer_id as _question_context_canonicalize_answer_id


def linked_entity_context(kg, linked_entities):
    ids = [x for x in linked_entities if re.fullmatch(r'[mg]\.[A-Za-z0-9_]+', x)]
    if not ids:
        return []
    values = ' '.join('ns:' + x for x in ids)
    query = '''PREFIX ns: <http://rdf.freebase.com/ns/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?type ?label ?start ?end WHERE {
VALUES ?entity { ''' + values + ''' }
OPTIONAL { ?entity ns:type.object.type ?type .
FILTER(!STRSTARTS(STR(?type), "http://rdf.freebase.com/ns/base.") && !STRSTARTS(STR(?type), "http://rdf.freebase.com/ns/user.") && ?type != ns:common.topic)
OPTIONAL { ?type rdfs:label ?label . FILTER(LANG(?label) = "en") } }
OPTIONAL { ?entity ns:time.event.start_date ?start . }
OPTIONAL { ?entity ns:time.event.end_date ?end . }
} LIMIT 256'''
    records = {x: {'id': x, 'name': linked_entities[x], 'types': []} for x in ids}
    for row in kg.execute(query):
        entity = _question_context_canonicalize_answer_id(row.get('entity', ''))
        if entity not in records:
            continue
        record = records[entity]
        label = row.get('label') or _question_context_canonicalize_answer_id(row.get('type', '')).replace('.', ' ')
        if label and label not in record['types']:
            record['types'].append(label)
        for key in ('start', 'end'):
            if row.get(key):
                record[key + '_date'] = _question_context_canonicalize_answer_id(row[key])
    # The Gold ID/name itself is useful anchor evidence even when the optional
    # type/date lookup returns no row. Never silently drop a configured entity.
    return list(records.values())



# ---- question_query_budget ----
"""Per-question exact read-only query reuse and paid-for bounded repairs.

Only successful, deterministic SELECT results are reused, with exact query
bytes and the same client identity. No failure, cross-question, variable-name
or graph-ID shortcut is cached. Optional new repair queries require both a
saved request credit and a measured saved-time credit, and never call a model.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field, replace
import re
import time
from .evaluation import canonicalize_answer_id as _question_query_budget_canonicalize_answer_id


@dataclass
class QuestionQueryBudget:
    question: str
    cache: dict = field(default_factory=dict)
    actual_requests: int = 0
    cache_hits: int = 0
    request_credits: int = 0
    seconds_saved: float = 0.0
    repair_requests: int = 0
    repair_seconds: float = 0.0
    repair_preparation_seconds: float = 0.0
    rejected_groundings: list = field(default_factory=list)
    semantic_contexts: list = field(default_factory=list)
    decomposition_contexts: list = field(default_factory=list)
    answer_depth: int = 0
    kg_name_evidence: dict = field(default_factory=dict)
    proposed_scalar_compositions: list = field(default_factory=list)

    def receipt(self):
        return {"actual_requests": self.actual_requests, "exact_query_cache_hits": self.cache_hits,
                "same_flow_requests_without_cache_or_new_repairs":
                    self.actual_requests + self.cache_hits - self.repair_requests,
                "net_request_delta_from_cache_and_repairs": self.repair_requests - self.cache_hits,
                "unused_request_credits": self.request_credits,
                "measured_seconds_saved": self.seconds_saved,
                "repair_requests": self.repair_requests, "repair_seconds": self.repair_seconds,
                "repair_preparation_seconds": self.repair_preparation_seconds,
                "remaining_saved_seconds": max(0.0, self.seconds_saved - self.repair_seconds),
                "repair_time_budget_overrun_seconds": max(0.0, self.repair_seconds - self.seconds_saved),
                "additional_model_calls": 0, "cross_question_reuse": False}


_BUDGET = ContextVar("cwq_question_query_budget", default=None)
_REPAIR_DEADLINE = ContextVar("cwq_budgeted_repair_deadline", default=None)
_NONDETERMINISTIC = re.compile(r"\b(?:NOW|RAND|UUID|STRUUID|BNODE)\s*\(|\bSERVICE\b", re.I)


@contextmanager
def question_query_scope(question):
    current = _BUDGET.get()
    if current is not None and current.question == question:
        yield current
        return
    token = _BUDGET.set(QuestionQueryBudget(str(question)))
    try:
        yield _BUDGET.get()
    finally:
        _BUDGET.reset(token)


def current_query_budget():
    return _BUDGET.get()


def has_saved_repair_budget():
    state = _BUDGET.get()
    return bool(state is not None and state.request_credits > 0
                and state.seconds_saved - state.repair_seconds >= 0.02)


def charge_repair_preparation(started):
    state = _BUDGET.get()
    if state is None:
        return
    elapsed = max(0.0, time.monotonic() - started)
    state.repair_seconds += elapsed
    state.repair_preparation_seconds += elapsed


def remember_kg_label_rows(rows):
    """Keep real label-query results only, never model-produced names."""
    state = _BUDGET.get()
    if state is None:
        return
    for row in rows:
        identifier = _question_query_budget_canonicalize_answer_id(row.get("entity", ""))
        label = str(row.get("label", "")).strip()
        if (not re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", identifier) or not label
                or re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", _question_query_budget_canonicalize_answer_id(label))):
            continue
        if identifier not in state.kg_name_evidence and len(state.kg_name_evidence) >= 2048:
            continue
        normalized = " ".join(label.casefold().split())
        state.kg_name_evidence.setdefault(identifier, {}).setdefault(normalized, label)


def bounded_request_timeout(timeout):
    deadline = _REPAIR_DEADLINE.get()
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Saved-query time budget exhausted")
    return min(float(timeout), remaining)


def current_repair_deadline():
    return _REPAIR_DEADLINE.get()


def check_repair_deadline():
    deadline = _REPAIR_DEADLINE.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Saved-query absolute time budget exhausted")


def _clone(rows):
    return [dict(row) for row in rows]


def execute_with_question_budget(client, query, execute_raw):
    state = _BUDGET.get()
    cacheable = (state is not None and re.search(r"\bSELECT\b", query, re.I)
                 and not _NONDETERMINISTIC.search(query))
    key = (id(client), query)
    if cacheable and key in state.cache:
        rows, elapsed = state.cache[key]
        started = time.monotonic()
        result = _clone(rows)
        overhead = time.monotonic() - started
        state.cache_hits += 1
        state.request_credits += 1
        state.seconds_saved += max(0.0, elapsed - overhead)
        return result
    if state is not None:
        state.actual_requests += 1
    started = time.monotonic()
    rows = execute_raw()
    elapsed = time.monotonic() - started
    # Bound memory and avoid caching unusually expensive result copies.
    if (cacheable and len(state.cache) < 128 and isinstance(rows, list)
            and len(rows) <= 4096 and all(isinstance(row, dict) for row in rows)):
        state.cache[key] = (_clone(rows), elapsed)
    return rows


def execute_repair_from_saved_budget(client, query, *, maximum_seconds=1.0):
    state = _BUDGET.get()
    evidence = {"status": "not_executed", "additional_model_calls": 0,
                "paid_from_saved_request_and_time": True}
    if state is None or state.request_credits <= 0:
        return None, {**evidence, "reason": "no_saved_query_credit"}
    remaining = state.seconds_saved - state.repair_seconds
    timeout = min(float(maximum_seconds), remaining)
    if timeout < 0.02:
        return None, {**evidence, "reason": "insufficient_saved_time_credit"}
    config = getattr(client, "config", None)
    if config is None or not callable(getattr(client, "_select", None)):
        return None, {**evidence, "reason": "client_cannot_enforce_bounded_request"}
    try:
        bounded = copy(client)
        bounded.config = replace(config, timeout=timeout)
    except (TypeError, AttributeError):
        return None, {**evidence, "reason": "client_config_cannot_be_copied"}
    state.request_credits -= 1
    state.actual_requests += 1
    state.repair_requests += 1
    started = time.monotonic()
    deadline_token = _REPAIR_DEADLINE.set(started + timeout)
    try:
        # Bypass execute() to avoid creating credits from the repair itself.
        payload = bounded._select(query)
        check_repair_deadline()
        bindings = payload.get("results", {}).get("bindings", [])
        rows = [{key: str(value.get("value", "")) for key, value in row.items()
                 if isinstance(value, dict)} for row in bindings if isinstance(row, dict)]
        check_repair_deadline()
        evidence.update(status="executed", answer_row_count=len(rows), timeout_seconds=timeout)
        return rows, evidence
    except Exception as exc:
        return None, {**evidence, "status": "bounded_repair_failed", "reason": str(exc),
                      "timeout_seconds": timeout}
    finally:
        state.repair_seconds += time.monotonic() - started
        _REPAIR_DEADLINE.reset(deadline_token)



__all__ = ['QuestionQueryBudget', 'bounded_request_timeout', 'charge_repair_preparation', 'check_repair_deadline', 'current_query_budget', 'current_repair_deadline', 'execute_repair_from_saved_budget', 'execute_with_question_budget', 'has_saved_repair_budget', 'linked_entity_context', 'question_query_scope', 'remember_kg_label_rows']
