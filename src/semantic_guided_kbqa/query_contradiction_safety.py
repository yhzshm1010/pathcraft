"""Recognize a bound RDF-subject self-inequality, without weakening filters.

A literal NaN need not equal itself, so mere FILTER(?x != ?x) is insufficient.
Only a variable bound as a mandatory RDF triple subject is proved nonliteral.
COUNT over an empty graph can return zero; aggregate queries are not skipped.
"""
import re
from typing import Any


def bound_subject_self_inequality(graph: Any, sparql: str) -> dict | None:
    if not re.match(r"\A\s*SELECT\s+DISTINCT\s+\?[A-Za-z_]\w*\s+WHERE\b", sparql, re.I):
        return None
    for operator in graph.operators:
        if not isinstance(operator, dict):
            continue
        if (str(operator.get("type", "")).upper() != "NO_EQUAL"
                or str(operator.get("value_type", "")).casefold() != "variable"
                or any(operator.get(field) for field in
                       ("attribute_relation_id", "attribute_relation_label", "attribute_relation_labels"))):
            continue
        left = str(operator.get("input_var", "")).removeprefix("?")
        right = str(operator.get("value", "")).removeprefix("?")
        if left != right or not re.fullmatch(r"[A-Za-z_]\w*", left):
            continue
        subjects = [triple for triple in graph.triples
                    if isinstance(triple, (list, tuple)) and len(triple) == 3
                    and str(triple[0]).removeprefix("?") == left]
        if not subjects:
            continue
        pattern = r"\bFILTER\s*\(\s*\?" + re.escape(left) + r"\s*!=\s*\?" + re.escape(left) + r"\s*\)"
        if not re.search(pattern, sparql):
            continue
        return {"status": "proven_empty", "reason": "bound_nonliteral_subject_self_inequality",
                "variable": left, "subject_binding_edge_count": len(subjects),
                "filter_retained": True, "NaN_literal_case_excluded": True,
                "aggregate_empty_value_case_excluded": True,
                "additional_model_calls": 0, "additional_endpoint_queries": 0}
    return None
