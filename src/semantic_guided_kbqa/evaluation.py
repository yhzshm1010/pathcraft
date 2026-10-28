"""Primary scoring with separate answer-set Hit, first-answer hit, and exact match.

``hit`` checks whether any normalized predicted answer belongs to Gold.
The reported metric is answer-set Hit (``summary.hit``): exact canonical
Freebase ID matching for entity-only answers, otherwise normalized
literal/answer-text exact matching. It is not label-substring matching.
``hits_at_1`` instead checks the first normalized, deduplicated prediction in
supplied order. Neither requires full recall, and the evaluator does not rank
answers. ``exact_match`` requires equality of the predicted and Gold sets.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable


EVALUATION_VERSION = "answer-id-primary-v5-hit-failures-zero"
# Evaluate every WebQSP item from its configured Gold record; no question-level
# perfect-score exceptions are injected by the runtime.
_EMPTY_GOLD_PERFECT_QUESTIONS = frozenset()
_FREEBASE_NS_PREFIXES = (
    "http://rdf.freebase.com/ns/",
    "https://rdf.freebase.com/ns/",
)
_SLASH_MID_RE = re.compile(r"^/?([mg])/([A-Za-z0-9_]+)$")
_TYPED_LITERAL_RE = re.compile(
    r'^(?P<quoted>"(?:[^"\\]|\\.)*")(?:@[A-Za-z0-9-]+|\^\^\S+)?$'
)
_GYEAR_WITH_TIMEZONE_RE = re.compile(r"^(?P<year>[12][0-9]{3})[-+][0-9]{2}:[0-9]{2}$")
_DATE_WITH_TIMEZONE_RE = re.compile(
    r"^(?P<date>[12][0-9]{3}-[0-9]{2}-[0-9]{2})[-+][0-9]{2}:[0-9]{2}$"
)
_WEBQSP_PUNCT_RE = re.compile(r"[\s\-_.,;:!?/\\()（）【】\[\]{}<>《》、，。；：！？\"']+")


def canonicalize_answer_id(value: Any) -> str:
    """Normalize equivalent Freebase IDs and serialized literal forms."""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    for prefix in _FREEBASE_NS_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    if text.startswith("ns:"):
        return text[3:]
    slash_mid = _SLASH_MID_RE.fullmatch(text)
    if slash_mid:
        return f"{slash_mid.group(1)}.{slash_mid.group(2)}"
    typed_literal = _TYPED_LITERAL_RE.fullmatch(text)
    if typed_literal:
        try:
            text = str(json.loads(typed_literal.group("quoted"))).strip()
        except json.JSONDecodeError:
            pass
    date = _DATE_WITH_TIMEZONE_RE.fullmatch(text)
    if date:
        return date.group("date")
    gyear = _GYEAR_WITH_TIMEZONE_RE.fullmatch(text)
    if gyear:
        return gyear.group("year")
    return text


def canonicalize_answer_ids(values: Iterable[Any]) -> set[str]:
    return {
        canonical
        for value in values
        if (canonical := canonicalize_answer_id(value))
    }


def canonicalize_answer_id_list(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        canonical = canonicalize_answer_id(value)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        output.append(canonical)
    return output


def normalize_answer_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", canonicalize_answer_id(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def normalize_webqsp_text(value: Any) -> str:
    """Normalize WebQSP answer text after RDF literal canonicalization."""
    text = canonicalize_answer_id(value).strip()
    for prefix in _FREEBASE_NS_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return _WEBQSP_PUNCT_RE.sub("", text).strip().lower()


def webqsp_unique_answers(values: Iterable[Any]) -> list[str]:
    """Deduplicate answer strings with WebQSP's punctuation normalization."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = normalize_webqsp_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def dedupe_answer_labels(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = unicodedata.normalize("NFKC", str(value)).strip()
        normalized = normalize_answer_label(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(text)
    return output


def gold_answer_ids(record: dict[str, Any]) -> set[str]:
    answers = record.get("answer_entities", {})
    if isinstance(answers, dict):
        values = (
            value if str(key).casefold() == "value" else key
            for key, value in answers.items()
        )
        return canonicalize_answer_ids(values)
    if isinstance(answers, list):
        return canonicalize_answer_ids(answers)
    return set()


def gold_answer_labels(record: dict[str, Any]) -> list[str]:
    answers = record.get("answer_entities", {})
    if isinstance(answers, dict):
        return webqsp_unique_answers(answers.values())
    return []


def score_answer_ids(gold: Iterable[Any], predicted: Iterable[Any]) -> dict[str, Any]:
    """Score canonical ID sets; ``hit`` is 1 whenever their overlap is nonempty."""
    gold_set = canonicalize_answer_ids(gold)
    predicted_set = canonicalize_answer_ids(predicted)
    if not gold_set:
        return {"exact": None, "precision": None, "recall": None, "f1": None, "hit": None}
    if not predicted_set:
        return {"exact": False, "precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": 0.0}
    overlap = gold_set & predicted_set
    precision = len(overlap) / len(predicted_set)
    recall = len(overlap) / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "exact": predicted_set == gold_set,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": float(bool(overlap)),
    }


def score_webqsp_answers(
    gold_answers: Iterable[Any],
    predicted_answers: Iterable[Any],
) -> dict[str, Any]:
    """Score normalized text sets with both any-answer and first-answer hit.

    ``hit`` is 1 if any normalized predicted answer matches a Gold answer.
    ``hits_at_1`` preserves normalized, deduplicated input order; a correct
    answer later in the list is insufficient. ``exact_match`` instead requires
    set equality, and ``exact`` is its boolean compatibility alias.
    """
    gold_values = webqsp_unique_answers(gold_answers)
    predicted_values = webqsp_unique_answers(predicted_answers)
    gold_keys = {normalize_webqsp_text(value) for value in gold_values}
    predicted_keys = [normalize_webqsp_text(value) for value in predicted_values]
    predicted_key_set = set(predicted_keys)
    if not gold_keys:
        return {
            "labeled": False,
            "precision": None,
            "recall": None,
            "f1": None,
            "hit": None,
            "hits_at_1": None,
            "exact_match": None,
            "exact": None,
        }
    true_positive = len(gold_keys & predicted_key_set)
    precision = true_positive / len(predicted_key_set) if predicted_key_set else 0.0
    recall = true_positive / len(gold_keys)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact_match = 1.0 if predicted_key_set == gold_keys else 0.0
    first = predicted_keys[0] if predicted_keys else ""
    return {
        "labeled": True,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": float(true_positive > 0),
        "hits_at_1": 1.0 if first and first in gold_keys else 0.0,
        "exact_match": exact_match,
        # Preserve the historical field as a boolean compatibility alias.
        "exact": bool(exact_match),
    }


def predicted_answer_labels(row: dict[str, Any]) -> list[str]:
    labels_by_id = _predicted_answer_label_map(row)
    return dedupe_answer_labels(
        labels_by_id.get(canonicalize_answer_id(answer_id), answer_id)
        for answer_id in row.get("prediction", []) or []
    )


def primary_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop auxiliary scores, add missing primary Hit, and zero labeled failures.

    For stored primary scores from the supported earlier runtime versions,
    standard set F1 is positive iff there is an answer overlap. This permits
    adding the missing ``hit`` field without changing existing metrics or the
    original row. Auxiliary label-matching scores are not used for this step.
    """
    scored = dict(row)
    for key in tuple(scored):
        if key.endswith("_label_score") or key in {"evaluation_prediction", "raw_text_score"}:
            scored.pop(key, None)
    if isinstance(scored.get("delta"), dict):
        scored["delta"] = {
            key: value for key, value in scored["delta"].items()
            if not key.endswith("_label")
        }
    score = scored.get("score", {})
    if is_error_result_row(scored) and (
        (isinstance(score, dict) and score.get("labeled") is True)
        or bool(canonicalize_answer_ids(scored.get("gold_answers", []) or []))
        or bool(webqsp_unique_answers(scored.get("gold_answer_labels", []) or []))
    ):
        # A partial prediction is diagnostic evidence, not a successful answer.
        # Missing Gold is never silently turned into a labeled zero-score item.
        scored["score"] = {
            "labeled": True,
            "exact": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hit": 0.0,
            "hits_at_1": 0.0,
            "exact_match": 0.0,
        }
    elif isinstance(score, dict) and "hit" not in score:
        if score.get("labeled") is False:
            scored["score"] = {**score, "hit": None}
        elif score.get("labeled") is True and score.get("f1") is not None:
            f1 = float(score["f1"])
            if not 0.0 <= f1 <= 1.0:
                raise ValueError("Cannot recover Hit from an invalid primary F1 score")
            scored["score"] = {**score, "hit": float(f1 > 0.0)}
    return scored


def score_result_row(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Attach primary ID/text scores without changing prediction order.

    ``hit`` tests whether any predicted answer belongs to Gold, independently
    of answer order. It does not require full recall or exact set equality.
    ``hits_at_1`` tests the first normalized, deduplicated prediction, not
    any-answer overlap or full recall. ``exact_match`` requires set equality.
    No confidence ranking is performed here. Labeled failures receive zeros;
    auxiliary label diagnostics are omitted.
    """
    scored = primary_result_row(row)
    gold_ids = sorted(gold_answer_ids(record))
    raw_gold_labels = gold_answer_labels(record)
    gold_labels = webqsp_unique_answers(raw_gold_labels)
    predicted_ids = row.get("prediction", row.get("answer_ids", [])) or []
    scored["prediction"] = canonicalize_answer_id_list(predicted_ids)
    scored["answer_ids"] = list(scored["prediction"])
    raw_predicted_labels = predicted_answer_labels(scored)
    predicted_labels = webqsp_unique_answers(raw_predicted_labels)
    scored["gold_answers"] = gold_ids
    scored["gold_answer_labels"] = gold_labels
    scored["predicted_answers"] = predicted_labels
    text_score = score_webqsp_answers(gold_labels, predicted_labels)
    # Entity identity is independent of label availability, aliases and preview
    # limits. Keep the original text protocol for literal/text predictions.
    entity_ids = bool(gold_ids) and all(_is_freebase_id(value) for value in gold_ids)
    entity_predictions = all(_is_freebase_id(value) for value in scored["prediction"])
    if entity_ids and entity_predictions:
        id_score = score_answer_ids(gold_ids, scored["prediction"])
        # Keep first-answer hit separate from the answer-set Hit in id_score.
        scored["score"] = {
            "labeled": True,
            **id_score,
            "hits_at_1": float(bool(scored["prediction"]) and scored["prediction"][0] in gold_ids),
            "exact_match": float(id_score["exact"]),
        }
        scored["score_basis"] = "answer_ids"
    else:
        scored["score"] = text_score
        scored["score_basis"] = "answer_text"
    return primary_result_row(scored)


def summarize_result_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average all labeled rows, including zero-scored failures."""
    scored = []
    for row in rows:
        normalized = primary_result_row(row)
        if is_metric_result_row(normalized):
            score = normalized["score"]
            if not isinstance(score.get("hit"), (int, float)) or score["hit"] not in (0, 1):
                raise ValueError("Labeled primary scores must contain a binary Hit value")
            scored.append(score)
    if not scored:
        return {"total": len(rows), "labeled": 0, "unlabeled": len(rows),
                "count": 0, "scored": 0, "hit": None}
    precision = sum(float(item["precision"] or 0.0) for item in scored) / len(scored)
    recall = sum(float(item["recall"] or 0.0) for item in scored) / len(scored)
    f1 = sum(float(item["f1"] or 0.0) for item in scored) / len(scored)
    hit = sum(float(item["hit"]) for item in scored) / len(scored)
    exact_match = sum(float(item["exact_match"] or 0.0) for item in scored) / len(scored)
    hits_at_1 = sum(float(item["hits_at_1"] or 0.0) for item in scored) / len(scored)
    summary: dict[str, Any] = {
        "total": len(rows),
        "labeled": len(scored),
        "unlabeled": len(rows) - len(scored),
        "count": len(scored),
        "scored": len(scored),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "hit": hit,
        "hits_at_1": hits_at_1,
        "exact_match": exact_match,
        # Keep the previous summary names as numerical aliases.
        "exact": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return summary


def is_error_result_row(row: dict[str, Any]) -> bool:
    """Return whether pipeline execution failed for a result row."""
    return bool(row.get("failure"))


def is_metric_result_row(row: dict[str, Any]) -> bool:
    """Include every labeled item, including failures scored as zero."""
    score = row.get("score", {})
    return (
        isinstance(score, dict)
        and score.get("labeled") is True
        and score.get("f1") is not None
    )


def row_metrics_are_all_one(row: dict[str, Any]) -> bool:
    score = row.get("score")
    return isinstance(score, dict) and _metric_value_is_one(score.get("f1")) and _metric_value_is_one(
        score.get("recall")
    )


def row_id_f1_recall_are_all_one(row: dict[str, Any]) -> bool:
    """Return whether primary F1 and recall are both perfect."""
    score = row.get("score")
    return (
        isinstance(score, dict)
        and _metric_value_is_one(score.get("f1"))
        and _metric_value_is_one(score.get("recall"))
    )


def evaluation_metadata(source: str) -> dict[str, Any]:
    """Describe the reported Hit protocol without changing numeric scores."""
    return {
        "version": EVALUATION_VERSION,
        "source": source,
        "matching": "canonical ID sets when Gold and predictions are Freebase entities; otherwise WebQSP-compatible normalized answer text sets",
        "normalization": "strip Freebase namespace, remove WebQSP punctuation/separators, lowercase, deduplicate",
        "gold_source": "answer_entities IDs/labels from the configured Gold record",
        "protocol_change": "entity identity takes precedence over labels; this primary metric differs from the historical text-only protocol",
        "hit_definition": "1 if any normalized predicted answer belongs to Gold, otherwise 0; independent of prediction order and not a full-recall requirement; labeled failures are zero",
        "hits_at_1_definition": "1 if the first normalized, deduplicated prediction belongs to Gold, otherwise 0; not any-answer overlap or full recall; labeled failures are zero",
        "prediction_order": "preserve supplied order after normalization and deduplication; the evaluator does not rank answers by confidence",
        "exact_match_definition": "1 only when the predicted and Gold answer sets are equal, otherwise 0; labeled failures are zero; per-row exact is the boolean alias",
        "hit_output": "per-question score.hit and aggregate summary.hit in [0, 1]; multiply by 100 for a percentage; v5 adds this field without changing existing metric formulas",
        "reported_hit_metric": {
            "name": "Answer-set Hit (canonical Freebase ID / normalized-literal exact matching)",
            "name_zh": "基于规范化 Freebase 实体 ID／字面量精确匹配的答案集合命中率",
            "per_question_field": "score.hit",
            "summary_field": "summary.hit",
            "entity_matching": "exact canonical Freebase ID equality when all Gold and predicted answers are Freebase entity IDs",
            "literal_or_text_fallback": "exact normalized literal/answer-text equality for the existing non-entity-only scoring branch",
            "per_question_matching_basis_field": "score_basis (answer_ids or answer_text)",
            "substring_matching": False,
            "first_answer_only": False,
            "aggregation": "mean of per-question binary Hit over all labeled items, including zero-scored failures; multiply by 100 for a percentage",
        },
        "historical_cwq_hit": "hit uses the same any-answer-overlap criterion as the checked archived CWQ ID summary; answer matching, failure policy and denominator must still be checked separately for historical comparisons",
        "aggregation": "macro precision/recall/F1, hit, hits_at_1 and exact_match over all labeled rows, including pipeline failures scored as zero",
        "failure_policy": "labeled failures receive zero precision/recall/F1/hit/hits_at_1/exact_match and remain in the denominator; error rows are also stored separately",
        "missing_gold_policy": "rows without evaluable Gold remain unlabeled, are reported separately, and are not silently assigned zero",
    }


def _bridge_predictions_to_gold_ids(
    record: dict[str, Any],
    row: dict[str, Any],
) -> list[str]:
    """Bridge literal answers to gold MIDs only through unique exact labels."""
    answers = record.get("answer_entities", {})
    predicted_ids = row.get("prediction", []) or []
    if not isinstance(answers, dict):
        return list(predicted_ids)

    label_to_ids: dict[str, set[str]] = {}
    for raw_id, raw_label in answers.items():
        if str(raw_id).casefold() == "value":
            continue
        gold_id = canonicalize_answer_id(raw_id)
        label = normalize_answer_label(raw_label)
        if gold_id and label:
            label_to_ids.setdefault(label, set()).add(gold_id)

    bridged: list[str] = []
    seen: set[str] = set()
    gold_ids = gold_answer_ids(record)
    labels_by_id = _predicted_answer_label_map(row)
    for predicted_id in predicted_ids:
        label = labels_by_id.get(predicted_id, predicted_id)
        matching_ids = label_to_ids.get(normalize_answer_label(label), set())
        evaluation_id = (
            next(iter(matching_ids))
            if not _is_freebase_id(predicted_id)
            and predicted_id not in gold_ids
            and len(matching_ids) == 1
            else predicted_id
        )
        if evaluation_id and evaluation_id not in seen:
            seen.add(evaluation_id)
            bridged.append(evaluation_id)
    return bridged


def _predicted_answer_label_map(row: dict[str, Any]) -> dict[str, str]:
    return {
        canonicalize_answer_id(item.get("id", "")): str(
            item.get("label", item.get("id", ""))
        )
        for item in row.get("answers", []) or []
        if isinstance(item, dict) and canonicalize_answer_id(item.get("id", ""))
    }


def _is_freebase_id(value: str) -> bool:
    return re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", value) is not None


def _metric_value_is_one(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return abs(float(value) - 1.0) <= 1e-12
    return False


def _metric_block_is_all_one(block: Any) -> bool:
    if not isinstance(block, dict) or not block:
        return False
    return all(_metric_value_is_one(value) for value in block.values())
