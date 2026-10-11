"""Consolidated constraint repairs; original rules and budgets retained."""
from __future__ import annotations


# ---- constraint_property_ranking ----
"""Conservative evidence for ranking a late scalar constraint property.

This module is intentionally independent from :mod:`pipeline`.  It contains
only question/schema scoring and answer-set safety checks, so an offline
replay can evaluate a new ranker without changing production behaviour.

No entity, relation, question, or sample whitelist is used.  The only lexical
constants are language stop words and generic scalar words; dates and numbers
remain handled by the caller's existing normalization.
"""



from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping as _constraint_property_ranking_Mapping, Sequence as _constraint_property_ranking_Sequence


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "did", "do", "does", "for", "had", "has", "have", "he",
    "her", "his", "how", "i", "in", "is", "it", "its", "name",
    "of", "on", "or", "our", "she", "that", "the", "their", "them", "to",
    "there", "these", "they", "this", "those", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "with",
}

_GENERIC_SCALAR_WORDS = {
    "amount", "date", "datetime", "float", "id", "identifier", "integer",
    "number", "rate", "time", "unit", "value", "year",
}

_EXTREMA_WORDS = {
    "earliest", "first", "highest", "largest", "last", "latest", "least",
    "lowest", "maximum", "minimum", "most", "recent", "smallest",
}

_COMPARISON_WORDS = {
    "above", "after", "before", "below", "fewer", "greater", "higher",
    "larger", "later", "less", "lower", "more", "over", "prior",
    "smaller", "under",
}

_TEMPORAL_WORDS = {
    "begin", "birth", "date", "death", "die", "end", "found", "from",
    "open", "release", "start", "time", "to", "year",
}

_NUMERIC_WORDS = {
    "amount", "area", "army", "casualty", "code", "count", "height",
    "id", "length", "number", "population", "postgraduate", "rate",
    "soldier", "total", "undergraduate",
}

_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}


def _word_key(value: str) -> str:
    """Apply the same deliberately small morphology used by late repair."""
    token = str(value).casefold()
    if token in {"died", "dead"}:
        return "death"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("sed") and len(token) > 4:
        return token[:-1]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def content_tokens(value: str, *, drop_scalar_words: bool = False) -> set[str]:
    """Return meaningful normalized English tokens from surface/schema text."""
    tokens = {
        _word_key(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(value))
    }
    tokens = {token for token in tokens if len(token) > 1}
    tokens.difference_update(_STOP_WORDS)
    if drop_scalar_words:
        tokens.difference_update(_GENERIC_SCALAR_WORDS)
    return tokens


def relation_tail_text(relation: str) -> str:
    """Convert a relation id to its terminal human-readable property name."""
    return " ".join(
        re.findall(
            r"[A-Za-z0-9]+",
            str(relation).rsplit(".", 1)[-1].replace("_", " "),
        )
    )


def constraint_focus_phrase(focus: str, *, window: int = 7) -> str:
    """Keep the local attribute phrase instead of unrelated question nouns.

    Full-question similarity overweights words describing the source graph
    (for example ``country`` in a question whose requested attribute is
    ``army``).  A bounded window around the explicit comparison/extremum is a
    more faithful embedding query and does not require a domain lexicon.
    """
    tokens = re.findall(r"[A-Za-z0-9]+(?:[.,/-][A-Za-z0-9]+)*", str(focus))
    if not tokens:
        return str(focus)
    normalized = [_word_key(token) for token in tokens]
    cue_indexes = [
        index
        for index, token in enumerate(normalized)
        if token in _EXTREMA_WORDS or token in _COMPARISON_WORDS
    ]
    numeric_indexes = [
        index for index, token in enumerate(tokens) if re.search(r"\d", token)
    ]
    if cue_indexes:
        pivot = cue_indexes[-1]
        # In an extrema phrase the attribute normally follows the cue
        # (``smallest army``, ``earliest released film``).  Starting at the
        # cue prevents the answer target immediately before it (``country``)
        # from looking like property evidence.  If no meaningful word follows
        # the cue, fall back to the left context for forms such as
        # ``number of undergraduates the largest`` or ``position from latest``.
        if normalized[pivot] in _EXTREMA_WORDS:
            right_end = min(len(tokens), pivot + max(3, int(window)))
            right = tokens[pivot:right_end]
            right_content = content_tokens(" ".join(tokens[pivot + 1:right_end]))
            if right_content:
                return " ".join(right)
            return " ".join(tokens[max(0, pivot - 5): pivot + 1])
    elif numeric_indexes:
        pivot = numeric_indexes[-1]
    else:
        return " ".join(tokens[-max(3, int(window)):])
    # Two words before a comparison normally cover its attribute phrase
    # (``calling codes over`` / ``released after``) without pulling in the
    # answer target (``countries`` / ``movies``).
    start = max(0, pivot - 2)
    # Equality phrases often insert light verbs immediately before the value
    # (``population was once 10,004,081``). Walk a few tokens farther left to
    # retain the nearest substantive property word without using a domain
    # vocabulary.
    light_tokens = {
        "a", "an", "at", "been", "being", "had", "has", "have", "is",
        "of", "once", "the", "to", "was", "were", "with",
    }
    if all(
        token in light_tokens
        for token in normalized[max(0, pivot - 2):pivot]
    ):
        for index in range(pivot - 3, max(-1, pivot - 7), -1):
            if index < 0:
                break
            if normalized[index] not in light_tokens:
                start = index
                break
    # A written calendar date occupies two surface tokens (``December
    # 25,2003``), so the usual two-token look-behind stops at the preposition
    # and loses the temporal predicate.  For this time-normalization form only,
    # retain the nearest local temporal verb (released/opened/founded/etc.).
    # Numeric constraints keep the original tight window.
    if any(
        token in _MONTH_WORDS
        for token in normalized[max(0, pivot - 2): pivot + 1]
    ):
        temporal_indexes = [
            index
            for index in range(max(0, pivot - 5), pivot)
            if normalized[index] in _TEMPORAL_WORDS
        ]
        if temporal_indexes:
            start = temporal_indexes[-1]
    end = min(len(tokens), pivot + max(3, int(window)))
    return " ".join(tokens[start:end])


def requested_family(spec: _constraint_property_ranking_Mapping[str, Any]) -> str:
    """Return ``temporal``, ``numeric`` or an empty unknown family."""
    value_type = str(spec.get("value_type", "")).casefold()
    if value_type == "datetime":
        return "temporal"
    if value_type == "number":
        return "numeric"
    tokens = content_tokens(str(spec.get("focus", "")))
    if tokens & _TEMPORAL_WORDS:
        return "temporal"
    if tokens & _NUMERIC_WORDS:
        return "numeric"
    return ""


def property_family(
    first_relation: str,
    second_relation: str,
    terminal_range: str,
) -> str:
    """Infer a scalar property's family from ontology range and leaf words."""
    terminal = second_relation or first_relation
    tokens = content_tokens(relation_tail_text(terminal))
    if str(terminal_range) == "type.datetime" or tokens & _TEMPORAL_WORDS:
        return "temporal"
    if str(terminal_range) in {"type.float", "type.int", "type.enumeration"}:
        return "numeric"
    if tokens & _NUMERIC_WORDS:
        return "numeric"
    return ""


@dataclass(frozen=True, slots=True)
class PropertyRankEvidence:
    """Inspectable, sortable evidence for one schema property path."""

    score: float
    family_score: float
    lexical_recall: float
    lexical_precision: float
    semantic_similarity: float
    directness: float
    matched_tokens: tuple[str, ...]
    first_matched_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    property_tokens: tuple[str, ...]
    terminal_relation: str

    @property
    def rank_key(self) -> tuple[float, float, float, float, float]:
        return (
            self.score,
            self.family_score,
            self.lexical_recall,
            self.semantic_similarity,
            self.lexical_precision,
        )


def property_rank_evidence(
    *,
    first_relation: str,
    second_relation: str,
    spec: _constraint_property_ranking_Mapping[str, Any],
    terminal_range: str,
    semantic_similarity: float,
    graph_score: float = 0.0,
) -> PropertyRankEvidence:
    """Score one property without variables, answer ids, or answer values.

    A relation token is counted at most once across a two-hop path.  This is
    the critical difference from adding independent first/second-hop overlap:
    ``release_dates -> release_date`` must not receive two votes for the same
    word and outrank the canonical direct ``initial_release_date`` property.
    """
    local_focus = constraint_focus_phrase(str(spec.get("focus", "")))
    query_tokens = content_tokens(local_focus, drop_scalar_words=True)
    first_tokens = content_tokens(
        relation_tail_text(first_relation), drop_scalar_words=True
    )
    second_tokens = content_tokens(
        relation_tail_text(second_relation), drop_scalar_words=True
    )
    path_tokens = first_tokens | second_tokens
    matched = query_tokens & path_tokens
    lexical_recall = len(matched) / max(1, len(query_tokens))
    lexical_precision = len(matched) / max(1, len(path_tokens))

    wanted_family = requested_family(spec)
    actual_family = property_family(
        first_relation,
        second_relation,
        terminal_range,
    )
    family_score = (
        1.0
        if wanted_family and actual_family == wanted_family
        else -1.0
        if wanted_family and actual_family and actual_family != wanted_family
        else 0.0
    )

    if not second_relation:
        directness = 0.10
    elif query_tokens & first_tokens:
        directness = 0.05
    elif float(semantic_similarity) >= 0.62:
        # A strong local-focus paraphrase is sufficient for a normal Freebase
        # measurement CVT (army -> size_of_armed_forces -> number).
        directness = 0.025
    else:
        directness = -0.10

    # Scores are deliberately continuous: BGE can bridge a genuine lexical
    # paraphrase (army -> armed forces), while lexical precision keeps an
    # unrelated long relation from winning on one accidental word.
    score = (
        1.40 * lexical_recall
        # Precision is a modest tie-breaker. A canonical property can carry a
        # useful modifier not repeated verbatim in the question (``initial``
        # release date), so unmatched schema words must not dominate recall
        # and local semantic similarity.
        + 0.20 * lexical_precision
        + 1.25 * float(semantic_similarity)
        + 0.55 * family_score
        + directness
        + 0.02 * float(graph_score)
    )
    return PropertyRankEvidence(
        score=score,
        family_score=family_score,
        lexical_recall=lexical_recall,
        lexical_precision=lexical_precision,
        semantic_similarity=float(semantic_similarity),
        directness=directness,
        matched_tokens=tuple(sorted(matched)),
        first_matched_tokens=tuple(sorted(query_tokens & first_tokens)),
        query_tokens=tuple(sorted(query_tokens)),
        property_tokens=tuple(sorted(path_tokens)),
        terminal_relation=str(second_relation or first_relation),
    )


def semantic_only_is_confident(
    best: PropertyRankEvidence,
    alternatives: Iterable[PropertyRankEvidence],
    *,
    minimum_similarity: float = 0.62,
    minimum_margin: float = 0.05,
) -> bool:
    """Gate a property supported by BGE but by no literal surface token.

    Lexical matches are already auditable evidence.  A pure paraphrase is
    accepted only when its local-focus similarity is both strong and clearly
    separated from other property paths.  This admits ``army`` -> ``size of
    armed forces`` while abstaining on an underspecified phrase such as
    ``smallest on the program``.
    """
    if any(len(token) >= 5 for token in best.matched_tokens):
        return True
    distinct = sorted(
        (
            item.semantic_similarity
            for item in alternatives
            if item.terminal_relation != best.terminal_relation
            and item.family_score == best.family_score
        ),
        reverse=True,
    )
    runner_up = distinct[0] if distinct else -1.0
    return bool(
        best.semantic_similarity >= float(minimum_similarity)
        and best.semantic_similarity - runner_up >= float(minimum_margin)
    )


def answer_set_safety_reason(
    source_ids: _constraint_property_ranking_Sequence[str],
    repaired_ids: _constraint_property_ranking_Sequence[str],
    *,
    operator_type: str,
    source_answers: _constraint_property_ranking_Sequence[_constraint_property_ranking_Mapping[str, Any]] = (),
    repaired_answers: _constraint_property_ranking_Sequence[_constraint_property_ranking_Mapping[str, Any]] = (),
) -> str:
    """Return an answer-evidence-independent reason to abstain, or ``""``.

    The checks use only cardinality and labels already fetched for execution.
    They never inspect gold.  They target two observed failure modes: applying
    an extremum to a very broad/incomplete source graph, and choosing a nested
    child label when a discarded source answer is its named parent.
    """
    source = set(map(str, source_ids))
    repaired = set(map(str, repaired_ids))
    if not repaired or not repaired < source:
        return "not_strict_subset"
    if (
        str(operator_type).upper() in {"ARGMIN", "ARGMAX"}
        and len(source) > 32
        and len(repaired) == 1
        and len(repaired) * 20 < len(source)
    ):
        return "broad_source_extrema_collapse"

    if (
        str(operator_type).upper()
        in {"GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL"}
        and len(repaired) == 1
        and source_answers
        and repaired_answers
    ):
        if nested_label_parent_ids(
            source_ids,
            repaired_ids,
            source_answers=source_answers,
            repaired_answers=repaired_answers,
        ):
            return "nested_child_label_collapse"
    return ""


def nested_label_parent_ids(
    source_ids: _constraint_property_ranking_Sequence[str],
    repaired_ids: _constraint_property_ranking_Sequence[str],
    *,
    source_answers: _constraint_property_ranking_Sequence[_constraint_property_ranking_Mapping[str, Any]] = (),
    repaired_answers: _constraint_property_ranking_Sequence[_constraint_property_ranking_Mapping[str, Any]] = (),
) -> list[str]:
    """Return source answers that are lexical parents of one repaired child.

    This is deliberately label-structural: a parent must lose at least two
    content tokens relative to the child, and no entity or relation identity
    is consulted.  Source order is preserved so a unique parent can be kept
    alongside a numeric-boundary child without restoring unrelated answers.
    """

    repaired = set(map(str, repaired_ids))
    if len(repaired) != 1:
        return []
    repaired_id = next(iter(repaired))
    source_labels = {
        str(item.get("id", "")): content_tokens(str(item.get("label", "")))
        for item in source_answers
        if isinstance(item, _constraint_property_ranking_Mapping)
    }
    repaired_labels = {
        str(item.get("id", "")): content_tokens(str(item.get("label", "")))
        for item in repaired_answers
        if isinstance(item, _constraint_property_ranking_Mapping)
    }
    child_tokens = repaired_labels.get(repaired_id) or source_labels.get(
        repaired_id, set()
    )
    if not child_tokens:
        return []
    return [
        answer_id
        for answer_id in map(str, source_ids)
        if answer_id not in repaired
        and len(source_labels.get(answer_id, set())) >= 2
        and source_labels.get(answer_id, set()) < child_tokens
        and len(child_tokens - source_labels.get(answer_id, set())) >= 2
    ]



# ---- compose_scalar_scope_safety ----
"""Prove every originating Semantic path survived before a local repair.

Binding a second named topic is not evidence that its path was retained. This
gate compares all original path/variable/direction contracts, not just anchor
names. Extra auto-expanded paths require independent evidence and currently
abstain; no scoring, queries, models or evaluation fields are consulted.
"""
from collections.abc import Mapping as _compose_scalar_scope_safety_Mapping


def _compose_scalar_scope_safety_original_semantic_scope_certificate(grounded, source_semantic, *, ontology=None):
    evidence = {"status": "not_authorized", "additional_model_calls": 0,
                "additional_endpoint_queries": 0, "uses_evaluation_gold": False,
                "all_original_paths_preserved": False,
                "original_predicates_independently_checked": ontology is not None}
    actual = getattr(grounded, "compose_input", None)
    if not isinstance(actual, _compose_scalar_scope_safety_Mapping) or not isinstance(source_semantic, _compose_scalar_scope_safety_Mapping):
        return False, {**evidence, "reason": "missing_actual_original_semantic_contract"}
    originals = source_semantic.get("semantic_paths")
    paths = actual.get("semantic_paths")
    if (not isinstance(originals, list) or not originals or not isinstance(paths, list)
            or len(originals) != len(paths)):
        return False, {**evidence, "reason": "original_and_grounded_path_count_differs"}
    by_id = {}
    for path in paths:
        if (not isinstance(path, _compose_scalar_scope_safety_Mapping) or not isinstance(path.get("id"), str)
                or not path["id"] or path["id"] in by_id):
            return False, {**evidence, "reason": "actual_path_identity_unknown_or_duplicated"}
        by_id[path["id"]] = path
    seen = set()
    for original in originals:
        if (not isinstance(original, _compose_scalar_scope_safety_Mapping) or not isinstance(original.get("id"), str)
                or not original["id"] or original["id"] in seen):
            return False, {**evidence, "reason": "original_path_identity_unknown_or_duplicated"}
        seen.add(original["id"])
        current = by_id.get(original["id"])
        if current is None:
            return False, {**evidence, "reason": "original_path_not_represented", "path_id": original["id"]}
        for key in ("anchor_ref", "path_output_var"):
            if not isinstance(original.get(key), str) or not original[key] or current.get(key) != original[key]:
                return False, {**evidence, "reason": "original_root_or_output_variable_changed", "path_id": original["id"]}
        before, after = original.get("steps"), current.get("steps")
        if not isinstance(before, list) or not before or not isinstance(after, list) or len(before) != len(after):
            return False, {**evidence, "reason": "original_steps_missing_or_unproved_auto_expansion", "path_id": original["id"]}
        for first, second in zip(before, after):
            if (not isinstance(first, _compose_scalar_scope_safety_Mapping) or not isinstance(second, _compose_scalar_scope_safety_Mapping)
                    or first.get("direction") not in {"forward", "backward"}
                    or any(not isinstance(first.get(key), str) or not first[key]
                           or second.get(key) != first[key] for key in ("id", "from", "to", "direction"))):
                return False, {**evidence, "reason": "original_step_variable_identity_or_direction_changed", "path_id": original["id"]}
            if ontology is not None:
                from .budgeted_schema_repair import _same_schema_property
                bindings = getattr(grounded, "relation_bindings", {})
                actual_relation = bindings.get(first["id"]) if isinstance(bindings, _compose_scalar_scope_safety_Mapping) else None
                if not actual_relation or not _same_schema_property(first.get("relation_label"), actual_relation, ontology):
                    return False, {**evidence, "reason": "original_predicate_replaced_by_unproved_schema_role",
                                   "path_id": original["id"], "step_id": first["id"]}
    return True, {**evidence, "status": "authorized", "reason": "every_original_path_and_variable_contract_preserved",
                  "all_original_paths_preserved": True, "original_path_ids": sorted(seen)}



# ---- constraint_execution_safety ----
"""Gold-blind protection of already executed, explicitly attached constraints.

No graph is generated, changed or executed here.  A scalar can only be added
to an unchanged answer/context, or a single projection error can be reversed
when question grammar, decomposition co-reference, original Semantic, Compose
and the schema all identify the same owner.  In particular, a smaller answer
set and an ARGMIN returning one answer are not evidence of correctness.
"""


from collections.abc import Mapping as _constraint_execution_safety_Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any

from .safety import _nested_name_schema_guard_anchors as _constraint_execution_safety_anchors, _nested_name_schema_guard_complete as _constraint_execution_safety_complete, _nested_name_schema_guard_singular as _singular
from .ontology import relation_id_from_label as _constraint_execution_safety_relation_id_from_label


_constraint_execution_safety_COMPARISON = {
    "less than": "LESS_THAN", "fewer than": "LESS_THAN", "lower than": "LESS_THAN",
    "greater than": "GREATER_THAN", "more than": "GREATER_THAN",
    "higher than": "GREATER_THAN", "larger than": "GREATER_THAN",
    "at most": "LESS_OR_EQUAL", "at least": "GREATER_OR_EQUAL",
    "before": "LESS_THAN", "after": "GREATER_THAN", "on": "EQUAL",
}
_COMPARATOR = "|".join(re.escape(x) for x in sorted(_constraint_execution_safety_COMPARISON, key=len, reverse=True))
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


def _constraint_execution_safety_words(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", str(value).replace("_", " ").casefold()))


def _leaf(value: Any) -> tuple[str, ...]:
    return _constraint_execution_safety_words(str(value).rsplit(".", 1)[-1])


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
            nested = value.get("decomposition") if isinstance(value, _constraint_execution_safety_Mapping) else getattr(value, "decomposition", None)
            if isinstance(nested, list):
                items.update(" ".join(x.split()).casefold() for x in nested if isinstance(x, str))
    return items


def _semantic_views(values: Any):
    if not isinstance(values, (list, tuple)):
        return
    for value in values:
        if not isinstance(value, _constraint_execution_safety_Mapping) or value.get("valid") is False:
            continue
        graph = value.get("output", value)
        if isinstance(graph, _constraint_execution_safety_Mapping) and isinstance(graph.get("semantic_paths"), list):
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
        return {"kind": _constraint_execution_safety_COMPARISON[match["comparison"].casefold()], "value": raw,
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
    from .normalization import normalize_comparison_constraint_spec
    provided = spec_provider(question)
    if not isinstance(provided, _constraint_execution_safety_Mapping) or provided.get("_normalization_scope") != "internal_post_model_constraint_spec_only":
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
    if not _contains(_constraint_execution_safety_words(obligation["prefix"]), _leaf(binding["property"])):
        return False
    if obligation["datatype"] == "ordering" and binding["property"] != binding["attribute"]:
        return _contains(_constraint_execution_safety_words(obligation["text"]), _leaf(binding["attribute"]))
    return True


def _semantic_attachment(graphs: Any, decompositions: set[str], prop: str,
                         obligation: dict, spec_provider: Any = None) -> list[tuple[_constraint_execution_safety_Mapping, _constraint_execution_safety_Mapping, _constraint_execution_safety_Mapping]]:
    matches = []
    seen = set()
    for graph in _semantic_views(graphs):
        for path in graph["semantic_paths"]:
            if not isinstance(path, _constraint_execution_safety_Mapping):
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
            attached = [step for step in steps if isinstance(step, _constraint_execution_safety_Mapping)
                        and isinstance(step.get("relation_label"), list)
                        and step["relation_label"]
                        and _constraint_execution_safety_words(step["relation_label"][-1]) == _leaf(prop)
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
    if not isinstance(compose, _constraint_execution_safety_Mapping) or str(compose.get("answer_var")) != owner:
        return False
    return any(isinstance(t, _constraint_execution_safety_Mapping) and str(t.get("subject")) == owner
               and isinstance(t.get("relation_label"), list) and t["relation_label"]
               and _constraint_execution_safety_words(t["relation_label"][-1]) == _leaf(prop)
               for t in compose.get("triples", []))


def _prefix_owners(question: str, selected: Any, candidate: Any, attachment: list) -> set[str] | None:
    """Map original Semantic's property subject through actual/Compose joins.

    A scalar path starting on a province cannot be rebound onto an answer
    country just because both have a population predicate. Opaque or divergent
    cross-stage variable mappings are not certificates.
    """
    provenance = getattr(selected.graph, "provenance", None)
    compose = getattr(candidate.graph, "compose_output", None)
    if not isinstance(provenance, _constraint_execution_safety_Mapping) or not isinstance(compose, _constraint_execution_safety_Mapping):
        return None
    bindings = provenance.get("anchor_bindings")
    if not isinstance(bindings, _constraint_execution_safety_Mapping):
        return None
    entities = compose.get("entities")
    compose_triples = compose.get("triples")
    if not isinstance(entities, list) or not isinstance(compose_triples, list):
        return None
    actual = _triples(candidate.graph)
    owners: set[str] = set()
    for graph, path, scalar in attachment:
        anchor_id = str(path.get("anchor_ref", ""))
        anchor = [a for a in graph.get("anchors", []) if isinstance(a, _constraint_execution_safety_Mapping) and str(a.get("id")) == anchor_id]
        linked = bindings.get(anchor_id)
        if len(anchor) != 1 or not isinstance(linked, _constraint_execution_safety_Mapping):
            return None
        surface = str(anchor[0].get("surface", ""))
        if (not surface or _constraint_execution_safety_words(surface) != _constraint_execution_safety_words(linked.get("label", ""))
                or not _contains(_constraint_execution_safety_words(question), _constraint_execution_safety_words(surface))):
            return None
        matches = [e for e in entities if isinstance(e, _constraint_execution_safety_Mapping) and _constraint_execution_safety_words(e.get("surface", "")) == _constraint_execution_safety_words(surface)]
        if len(matches) != 1:
            return None
        entity_ref, entity_id = str(matches[0].get("id", "")), str(linked.get("id", ""))
        composed = set()
        for triple in compose_triples:
            if not isinstance(triple, _constraint_execution_safety_Mapping):
                return None
            subject, object_ = str(triple.get("subject", "")), str(triple.get("object", ""))
            composed.add((entity_id if subject == entity_ref else subject,
                          _constraint_execution_safety_relation_id_from_label(triple.get("relation_label")),
                          entity_id if object_ == entity_ref else object_))
        current = {entity_id}
        for step in path.get("steps", []):
            if not isinstance(step, _constraint_execution_safety_Mapping):
                return None
            if step is scalar:
                break
            predicate = _constraint_execution_safety_relation_id_from_label(step.get("relation_label"))
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
    attribute_words = _constraint_execution_safety_words(re.sub(r"\s+once\s+of\s*$", "", qualifier[1], flags=re.I))
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
                or _constraint_execution_safety_words(steps[0].get("relation_label", [""])[-1]) != _leaf(first[1])
                or str(steps[1].get("from")) != str(steps[0].get("to"))):
            continue
        anchor_id = str(path.get("anchor_ref"))
        anchors = [a for a in semantic.get("anchors", []) if isinstance(a, _constraint_execution_safety_Mapping) and str(a.get("id")) == anchor_id]
        if len(anchors) != 1:
            continue
        anchor_surface = str(anchors[0].get("surface", ""))
        provenance = getattr(selected.graph, "provenance", None)
        linked = provenance.get("anchor_bindings", {}).get(anchor_id) if isinstance(provenance, _constraint_execution_safety_Mapping) else None
        if (not isinstance(linked, _constraint_execution_safety_Mapping) or str(linked.get("id")) != first[0]
                or _constraint_execution_safety_words(linked.get("label", "")) != _constraint_execution_safety_words(anchor_surface)
                or not _contains(_constraint_execution_safety_words(question), _constraint_execution_safety_words(anchor_surface))):
            continue
        # Cross-stage coreference is explicit: the same queried relation owner,
        # not merely a property on an arbitrary intermediate node.
        labels = steps[0].get("relation_label", [])
        if len(labels) < 2:
            continue
        type_words = _constraint_execution_safety_words(labels[-2])
        relation_words = _constraint_execution_safety_words(labels[-1])
        goal = str(path.get("goal", ""))
        clauses = [x.strip().rstrip(".").strip() for x in goal.split("?") if x.strip()]
        if len(clauses) != 3:
            continue
        expected_first = ("what", "is", *type_words, *_constraint_execution_safety_words(anchor_surface), *relation_words)
        expected_second = ("what", "is", "the", *attribute_words, "of", "that", "same", *relation_words, *type_words)
        if _constraint_execution_safety_words(clauses[0]) != expected_first or _constraint_execution_safety_words(clauses[1]) != expected_second:
            continue
        expected_third = ("the", *attribute_words, "of", "that", *type_words, "is",
                          *obligation["comparison"].split())
        if _constraint_execution_safety_words(clauses[2])[:len(expected_third)] != expected_third:
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
    if ontology is None or not _constraint_execution_safety_complete(selected) or _triples(selected.graph) is None:
        return abstain("source_execution_or_schema_incomplete")
    source_ops = _operators(selected.graph)
    if source_ops is None or len(source_ops[0]) > 1:
        return abstain("source_operator_contract_ambiguous")
    provenance = getattr(selected.graph, "provenance", None)
    if isinstance(provenance, _constraint_execution_safety_Mapping) and provenance.get("selection_answer_union"):
        return abstain("union_is_not_one_actual_execution")
    if (obligation.get("internal_normalization_proof") and isinstance(provenance, _constraint_execution_safety_Mapping)
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
        if candidate is selected or not _constraint_execution_safety_complete(candidate) or _triples(candidate.graph) is None:
            continue
        ops = _operators(candidate.graph)
        if ops is None or len(ops[0]) != 1 or _signatures(ops[1]) != _signatures(source_ops[1]):
            continue
        binding = _binding(candidate.graph, ops[0][0], ontology,
                           prefer_direct_answer_owner=bool(obligation.get("internal_normalization_proof")))
        if binding is None or not _matches(ops[0][0], binding, obligation) or not _property_supported(binding, obligation):
            continue
        if _constraint_execution_safety_anchors(selected.graph) != _constraint_execution_safety_anchors(candidate.graph) or not _constraint_execution_safety_anchors(selected.graph):
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
                       and isinstance(getattr(candidate.graph, "compose_output", None), _constraint_execution_safety_Mapping)
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



# ---- budgeted_primary_projection_repair ----
"""Pure compilation of a closed-question primary-node projection repair.

No candidate query is executed, and no answer is selected here. The caller may
spend only an already saved same-question request and measured time credit,
after all model stages. Every original triple/operator stays unchanged. An
ancestor-compatible answer type is not a trigger: exact closed question grammar,
a typed anchored main relation, and original Semantic variable evidence are.
"""


from collections.abc import Mapping as _budgeted_primary_projection_repair_Mapping
from copy import deepcopy
import re
from typing import Any

from .evaluation import canonicalize_answer_id as _budgeted_primary_projection_repair_canonicalize_answer_id
from .safety import _nested_name_schema_guard_anchors as _budgeted_primary_projection_repair_anchors, _nested_name_schema_guard_complete as _budgeted_primary_projection_repair_complete, _nested_name_schema_guard_concrete_type as _budgeted_primary_projection_repair_concrete_type, _nested_name_schema_guard_entity_answers as _entity_answers, _nested_name_schema_guard_type_index as _type_index
from .ontology import relation_id_from_label as _budgeted_primary_projection_repair_relation_id_from_label
from .safety import _source_role_execution_safety_semantic_graphs as _budgeted_primary_projection_repair_semantic_graphs, _semantic_path_nodes as _budgeted_primary_projection_repair_semantic_path_nodes, _source_role_execution_safety_tokens as _budgeted_primary_projection_repair_tokens, _word_forms as _budgeted_primary_projection_repair_word_forms


_budgeted_primary_projection_repair_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_WORD_PHRASE = r"[A-Za-z]+(?:\s+[A-Za-z]+){0,3}"
_ACTIVE = re.compile(r"^(?:what|which)\s+(" + _WORD_PHRASE + r")\s+([A-Za-z]+)(?:\s+the)?\s*$", re.I)
_RELATIVE = re.compile(r"^(?:what|which)\s+(?:is|was)\s+(?:(?:a|an|the)\s+)?("
                       + _WORD_PHRASE + r")\s+(?:where|for\s+which)\s*$", re.I)
_ROLE_SUFFIX = re.compile(r"^\s*(?:is|was)\s+(?:(?:a|an|the)\s+)?(" + _WORD_PHRASE + r")\s*[?.!]*\s*$", re.I)
_GOAL_HEAD = re.compile(r"\b(?:what|which)\s+(?:is|was|are|were)\s+(?:(?:a|an|the)\s+)?("
                        + _WORD_PHRASE + r")\s*(?=[?.!]|$)", re.I)


def _budgeted_primary_projection_repair_phrase(value: str) -> str:
    return " ".join(_budgeted_primary_projection_repair_tokens(value.replace("_", " ")))


def _budgeted_primary_projection_repair_finite_forms(word: str) -> set[str]:
    value = word.casefold()
    forms = {value}
    if len(value) > 4 and value.endswith("ed"):
        forms.update((value[:-2], value[:-1]))
    elif len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        forms.add(value[:-1])
    return forms


def _closed_question(question: str, label: str):
    text = str(question).strip()
    # Full topic span is allowed to contain a year/date; any number or
    # additional wording outside it must be accounted for by the closed grammar.
    words = str(label).strip().split()
    if not words:
        return None, "current_topic_label_missing"
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in words) + r"(?!\w)"
    matches = list(re.finditer(pattern, text, re.I))
    if len(matches) != 1:
        return None, "full_current_topic_label_not_unique_in_question"
    match = matches[0]
    before, after = text[:match.start()].strip(), text[match.end():]
    relative, suffix = _RELATIVE.fullmatch(before), _ROLE_SUFFIX.fullmatch(after)
    if relative and suffix:
        return {"grammar": "closed_relative_primary_role", "head": _budgeted_primary_projection_repair_phrase(relative[1]),
                "relation_cue": _budgeted_primary_projection_repair_phrase(suffix[1]), "topic_span": [match.start(), match.end()]}, "closed_question"
    active = _ACTIVE.fullmatch(re.sub(r"\s+the\s*$", "", before, flags=re.I))
    if active and re.fullmatch(r"\s*[?.!]*\s*", after):
        return {"grammar": "closed_active_primary_role", "head": _budgeted_primary_projection_repair_phrase(active[1]),
                "relation_cue": active[2].casefold(), "topic_span": [match.start(), match.end()]}, "closed_question"
    return None, "question_has_unsupported_extra_clause_modifier_or_scalar"


def _relation_cue_supported(spec: _budgeted_primary_projection_repair_Mapping[str, Any], relation: str) -> bool:
    words = re.findall(r"[A-Za-z]+", relation.rsplit(".", 1)[-1].replace("_", " "))
    cue = str(spec["relation_cue"])
    if spec["grammar"] == "closed_active_primary_role":
        # Do not turn a nominal 'founding figures' field into an active verb
        # 'founded': source/subject voice would be reversed. Only finite lexical
        # forms of an actual predicate word support the active construction.
        return any(not word.casefold().endswith("ing") and _budgeted_primary_projection_repair_finite_forms(cue) & _budgeted_primary_projection_repair_finite_forms(word)
                   for word in words)
    cue_words = _budgeted_primary_projection_repair_tokens(cue)
    schema_words = _budgeted_primary_projection_repair_tokens(" ".join(words))
    return bool(cue_words) and all(any(_budgeted_primary_projection_repair_word_forms(word) & _budgeted_primary_projection_repair_word_forms(schema_word)
                                      for schema_word in schema_words) for word in cue_words)


def _operators_supported(graph: Any, anchor: str) -> bool:
    variables = {str(node) for row in graph.triples for node in (row[0], row[2])
                 if _budgeted_primary_projection_repair_canonicalize_answer_id(node) != anchor}
    for operator in graph.operators:
        if (not isinstance(operator, _budgeted_primary_projection_repair_Mapping) or str(operator.get("type", "")).upper() != "NO_EQUAL"
                or str(operator.get("input_var", "")) not in variables
                or operator.get("inputs") or operator.get("attribute_relation_label")
                or operator.get("attribute_relation_labels") or operator.get("attribute_relation_id")):
            return False
        if operator.get("value_type") == "variable":
            if str(operator.get("value", "")) not in variables:
                return False
        elif operator.get("value_type") in {"entity", "mid"}:
            value = operator.get("_value_entity_id") or operator.get("value")
            if _budgeted_primary_projection_repair_canonicalize_answer_id(value) != anchor:
                return False
        else:
            return False
    return True


def _semantic_primary_witness(source: Any, target: str, relation: str, anchor: str,
                              names: set[str], endpoint_type: str, ontology: Any, semantics: Any,
                              context: list[_budgeted_primary_projection_repair_Mapping[str, Any]]):
    index = _type_index(ontology)
    witnesses = []
    for semantic in _budgeted_primary_projection_repair_semantic_graphs(semantics):
        references = {str(item.get("id")) for item in semantic["anchors"]
                      if isinstance(item, _budgeted_primary_projection_repair_Mapping) and str(item.get("surface", "")).strip().casefold() in names}
        for path in semantic["semantic_paths"]:
            if not isinstance(path, _budgeted_primary_projection_repair_Mapping) or str(path.get("anchor_ref")) not in references:
                continue
            steps = path.get("steps")
            if not isinstance(steps, list) or not steps or not isinstance(steps[0], _budgeted_primary_projection_repair_Mapping):
                continue
            first = steps[0]
            if (str(first.get("from")) != str(path.get("anchor_ref"))
                    or _budgeted_primary_projection_repair_relation_id_from_label(first.get("relation_label")) != relation):
                continue
            prefix = {**path, "steps": [first], "path_output_var": str(first.get("to", ""))}
            if _budgeted_primary_projection_repair_semantic_path_nodes(source, semantic, prefix, context) != {target}:
                continue
            heads = [_budgeted_primary_projection_repair_phrase(match[1]) for match in _GOAL_HEAD.finditer(str(path.get("goal", "")))]
            matching = [head for head in heads if endpoint_type in index.get(head, ())]
            if not matching:
                continue
            witnesses.append({"semantic_anchor_ref": str(path.get("anchor_ref")),
                              "semantic_primary_variable": str(first.get("to", "")),
                              "mapped_actual_primary_variable": target,
                              "semantic_primary_direction": str(first.get("direction", "")),
                              "original_wh_primary_heads": matching})
    return witnesses


def compile_primary_projection_repair(
    question: str, source: Any, ontology: Any, semantic_graphs: Any,
    *, linked_entity_context: Any = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Compile only a different answer_var, never execute or choose answers.

    Protect singleton, incomplete, union, multi-anchor and scalar/extrema cases.
    A nominal adjective not represented by an exact ontology head (e.g. 'major
    religion') is unknown and abstains; there is no descriptor exception list.
    """
    graph = source.graph
    evidence = {"status": "not_compiled", "source_graph_id": str(graph.graph_id),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "requires_saved_request_and_remaining_measured_time": True,
                "requires_post_model_execution": True, "compiler_selects_answers": False,
                "preserves_all_source_triples_and_operators": True,
                "ancestor_schema_compatibility_is_not_trigger": True}

    def abstain(reason):
        evidence["reason"] = reason
        return None, evidence

    try:
        ids = _entity_answers(source)
        if ids is None or not _budgeted_primary_projection_repair_complete(source) or len(ids) <= 1:
            return abstain("source_singleton_incomplete_or_not_pure_entities")
        provenance = getattr(graph, "provenance", None)
        if not isinstance(provenance, _budgeted_primary_projection_repair_Mapping) or provenance.get("selection_answer_union"):
            return abstain("source_current_bindings_missing_or_union_scope_unproven")
        original_question = provenance.get("original_question")
        if original_question is not None and str(original_question).strip() != str(question).strip():
            return abstain("current_source_question_provenance_conflict")
        anchors = _budgeted_primary_projection_repair_anchors(graph)
        if len(anchors) != 1 or not all(_budgeted_primary_projection_repair_ENTITY.fullmatch(value) for value in anchors):
            return abstain("not_single_entity_anchor_current_graph")
        anchor = next(iter(anchors))
        bindings = provenance.get("anchor_bindings")
        if not isinstance(bindings, _budgeted_primary_projection_repair_Mapping):
            return abstain("source_current_anchor_bindings_missing")
        labels = {str(value.get("label", "")).strip() for value in bindings.values()
                  if isinstance(value, _budgeted_primary_projection_repair_Mapping) and _budgeted_primary_projection_repair_canonicalize_answer_id(value.get("id", "")) == anchor}
        labels.discard("")
        if len(labels) != 1:
            return abstain("source_current_anchor_label_missing_or_ambiguous")
        label = next(iter(labels))
        spec, reason = _closed_question(question, label)
        if spec is None:
            return abstain(reason)
        evidence["closed_question"] = spec
        expected = set(_type_index(ontology).get(spec["head"], ()))
        if not expected:
            evidence["unsupported_head_phrase"] = spec["head"]
            evidence["unknown_modifier_or_head"] = True
            return abstain("primary_head_or_modifier_not_exact_ontology_vocabulary")
        if not _operators_supported(graph, anchor):
            return abstain("source_has_unsupported_scalar_extrema_or_external_operator")
        if not isinstance(linked_entity_context, (list, tuple)):
            return abstain("current_entity_context_missing")
        names = {str(item.get("name") or item.get("label") or "").strip().casefold()
                 for item in linked_entity_context if isinstance(item, _budgeted_primary_projection_repair_Mapping)
                 and _budgeted_primary_projection_repair_canonicalize_answer_id(item.get("id", "")) == anchor}
        names.discard("")
        if names != {label.casefold()}:
            return abstain("current_context_and_bound_topic_label_conflict")
        # The whole unchanged BGP must be connected, so another disconnected
        # branch cannot become an unreported obligation of the new projection.
        adjacency = {}
        for subject, relation, object_ in graph.triples:
            subject, object_ = str(subject), str(object_)
            adjacency.setdefault(subject, set()).add(object_)
            adjacency.setdefault(object_, set()).add(subject)
        reached, pending = {anchor}, [anchor]
        while pending:
            for neighbor in adjacency[pending.pop()] - reached:
                reached.add(neighbor)
                pending.append(neighbor)
        if reached != set(adjacency):
            return abstain("source_bgp_disconnected")
        candidates, unknown = [], []
        for subject, relation, object_ in graph.triples:
            subject, relation, object_ = map(str, (subject, relation, object_))
            if anchor not in (subject, object_) or not _relation_cue_supported(spec, relation):
                continue
            target = object_ if subject == anchor else subject
            side = "range_for_relation" if subject == anchor else "domain_for_relation"
            endpoint_type = str(getattr(ontology, side)(relation) or "")
            if not _budgeted_primary_projection_repair_concrete_type(endpoint_type):
                unknown.append(relation)
                continue
            if endpoint_type not in expected:
                continue
            witnesses = _semantic_primary_witness(source, target, relation, anchor, names,
                endpoint_type, ontology, semantic_graphs, list(linked_entity_context))
            if not witnesses:
                continue
            candidates.append((target, relation, endpoint_type, witnesses))
        targets = {target for target, _, _, _ in candidates}
        evidence.update(primary_candidate_variables=sorted(targets), unknown_primary_relation_schema=sorted(set(unknown)))
        if unknown or len(targets) != 1:
            return abstain("primary_variable_schema_or_original_semantic_not_unique")
        target = next(iter(targets))
        if target == str(graph.answer_var):
            return abstain("closed_question_primary_variable_already_projected")
        target, relation, endpoint_type, witnesses = next(value for value in candidates if value[0] == target)
        repaired = deepcopy(graph)
        repaired.graph_id = str(graph.graph_id) + "__primary_projection"
        repaired.answer_var = target
        repaired.sparql = ""
        repaired.provenance = deepcopy(dict(provenance))
        evidence.update(status="compiled", reason="closed_question_typed_primary_variable_projection",
                        target_answer_var=target, original_answer_var=str(graph.answer_var),
                        primary_relation_id=relation, primary_endpoint_type=endpoint_type,
                        original_semantic_variable_witnesses=witnesses)
        repaired.provenance["budgeted_primary_projection_repair"] = deepcopy(evidence)
        return repaired, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="source_question_or_schema_evidence_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence



# ---- semantic_scalar_path_restore ----
"""Pure compilation from one complete original Semantic scalar path.

No models, endpoint calls, Gold access or sample routing inside this compiler.
An ordinary changed predicate is not a contradiction.  Only an official
inverse used without reversing direction, or an extra same-predicate hop
that replaces the explicitly identified scalar owner's answer role, can
qualify.  Unproved original topic scope is always rejected.
"""
from copy import deepcopy
import re
from typing import Any, Callable, Mapping as _semantic_scalar_path_restore_Mapping, Sequence as _semantic_scalar_path_restore_Sequence

from .budgeted_schema_repair import NUMERIC_TYPES as _semantic_scalar_path_restore_NUMERIC_TYPES, SCALAR_TYPES as _semantic_scalar_path_restore_SCALAR_TYPES, STRING_TYPES as _semantic_scalar_path_restore_STRING_TYPES, _VAR as _semantic_scalar_path_restore_VAR, _normal_anchor_scope, _native_scalar_obligation, _number as _semantic_scalar_path_restore_number, _validated_provided_scalar_spec, schema_type_overlap
from .contracts import EntityCandidate, ExecutedGraph as _semantic_scalar_path_restore_ExecutedGraph, GroundedSemanticCandidate, QueryGraphCandidate as _semantic_scalar_path_restore_QueryGraphCandidate
from .ontology import relation_id_from_label as _semantic_scalar_path_restore_relation_id_from_label


def _semantic_scalar_path_restore_evidence(reason):
    return {"status": "not_qualified", "reason": reason, "pure_compiler_only": True,
            "model_calls": 0, "endpoint_queries": 0, "embedding_calls": 0,
            "uses_gold": False, "new_denotation_verified": False}


def _semantic_scalar_path_restore_tokens(value):
    tokens = set()
    for word in re.findall(r"[A-Za-z]+", str(value).casefold()):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.add(word)
    return tokens


def _question_head(question):
    match = re.match(r"^\s*(?:(?:in|at|on)\s+)?(?:which|what)\s+(?:the\s+)?([A-Za-z]+)", question, re.I)
    return _semantic_scalar_path_restore_tokens(match.group(1)) if match else set()


def compile_original_scalar_path_restore(
    question: str, selected: _semantic_scalar_path_restore_ExecutedGraph, original_semantic: _semantic_scalar_path_restore_Mapping[str, Any],
    topic_bindings: _semantic_scalar_path_restore_Mapping[str, EntityCandidate], ontology: Any, *, graph_id: str,
    linked_entity_context: _semantic_scalar_path_restore_Sequence[_semantic_scalar_path_restore_Mapping[str, Any]] | None = None,
    spec_provider: Callable[[str], _semantic_scalar_path_restore_Mapping[str, Any] | None] | None = None,
) -> tuple[_semantic_scalar_path_restore_QueryGraphCandidate | None, dict[str, Any]]:
    evidence = _semantic_scalar_path_restore_evidence("original_scalar_path_not_completely_proved")
    paths = original_semantic.get("semantic_paths", [])
    if ontology is None or len(paths) != 1:
        return None, _semantic_scalar_path_restore_evidence("requires_one_original_path_no_dropped_original_branches")
    path = paths[0]
    steps = path.get("steps", [])
    if len(steps) != 2 or not all(isinstance(s, dict) for s in steps):
        return None, _semantic_scalar_path_restore_evidence("requires_one_direct_primary_edge_and_one_scalar_attribute")
    first, last = steps
    root = str(path.get("anchor_ref", ""))
    owner_local, scalar_local = str(first.get("to", "")), str(last.get("to", ""))
    if (str(first.get("from", "")) != root or str(last.get("from", "")) != owner_local
            or str(path.get("path_output_var", "")) != scalar_local
            or last.get("direction") != "forward" or first.get("direction") not in {"forward", "backward"}
            or root not in topic_bindings):
        return None, _semantic_scalar_path_restore_evidence("original_variable_owner_or_direction_is_ambiguous")
    surfaces = [str(a.get("surface", "")) for a in original_semantic.get("anchors", [])
                if isinstance(a, _semantic_scalar_path_restore_Mapping) and str(a.get("id", "")) == root]
    actual_label = str(topic_bindings[root].label)
    if (len(surfaces) != 1 or not surfaces[0]
            or " ".join(surfaces[0].casefold().split()) != " ".join(actual_label.casefold().split())
            or not re.search(r"(?<!\w)" + re.escape(surfaces[0]) + r"(?!\w)", question, re.I)):
        return None, _semantic_scalar_path_restore_evidence("actual_bound_topic_label_does_not_match_original_question_semantic_root")
    owned_question = getattr(selected.graph, "provenance", {}).get("original_question")
    if owned_question is not None and str(owned_question).strip() != str(question).strip():
        return None, _semantic_scalar_path_restore_evidence("current_selected_graph_question_contract_conflict")
    # This adapter is solely an anchor-scope representation of ORIGINAL
    # Semantic plus actual existing topic bindings, not fabricated grounding.
    adapter = GroundedSemanticCandidate(dict(original_semantic), dict(topic_bindings), {}, 0.0,
                                         {"source": "original_semantic_scope_only"})
    scope_ok, labels, scope = _normal_anchor_scope(adapter, original_semantic, question, linked_entity_context)
    if not scope_ok:
        return None, {**_semantic_scalar_path_restore_evidence("original_extra_topic_or_schema_kind_scope_unknown"), "scope": scope}
    anchor = str(topic_bindings[root].entity_id)
    if not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", anchor):
        return None, _semantic_scalar_path_restore_evidence("actual_topic_binding_is_not_a_resolved_freebase_entity")
    for context in linked_entity_context or []:
        name = str(context.get("name", context.get("label", "")))
        tags = _semantic_scalar_path_restore_tokens(" ".join(map(str, context.get("types", []))))
        if (name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, re.I)
                and not tags & {"type", "property", "domain", "namespace"}
                and str(context.get("id", "")) not in {str(b.entity_id) for b in topic_bindings.values()}):
            return None, _semantic_scalar_path_restore_evidence("question_has_an_existing_linked_topic_not_in_original_scope")
    primary = _semantic_scalar_path_restore_relation_id_from_label(first.get("relation_label", []))
    attribute = _semantic_scalar_path_restore_relation_id_from_label(last.get("relation_label", []))
    p_domain, p_range = str(ontology.domain_for_relation(primary)), str(ontology.range_for_relation(primary))
    a_domain, a_range = str(ontology.domain_for_relation(attribute)), str(ontology.range_for_relation(attribute))
    if not all((p_domain, p_range, a_domain, a_range)):
        return None, _semantic_scalar_path_restore_evidence("original_exact_predicate_not_completely_in_ontology")
    output_type = p_range if first["direction"] == "forward" else p_domain
    numeric_string = a_range in (_semantic_scalar_path_restore_STRING_TYPES | {"type.enumeration"})
    if (a_range not in _semantic_scalar_path_restore_NUMERIC_TYPES and not numeric_string
            or schema_type_overlap(output_type, a_domain, ontology) != "compatible"
            or "common.topic" not in set(ontology.supertypes(a_domain))):
        return None, _semantic_scalar_path_restore_evidence("original_scalar_owner_not_a_positively_compatible_entity")
    head = _question_head(question)
    if not head or head != _semantic_scalar_path_restore_tokens(a_domain.rsplit(".", 1)[-1]):
        return None, _semantic_scalar_path_restore_evidence("question_head_does_not_uniquely_prove_the_original_attribute_owner_role")
    if not _semantic_scalar_path_restore_tokens(attribute.rsplit(".", 1)[-1]) <= _semantic_scalar_path_restore_tokens(question):
        return None, _semantic_scalar_path_restore_evidence("exact_original_scalar_property_not_explicit_in_question")
    spec = _native_scalar_obligation(question, labels, "type.float" if numeric_string else a_range)
    if spec is None and spec_provider is not None:
        spec = _validated_provided_scalar_spec(question, labels, spec_provider)
    if not spec or spec.get("value_type") != "number" or spec.get("type") not in {
        "GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "EQUAL",
    }:
        return None, _semantic_scalar_path_restore_evidence("single_exact_numeric_obligation_not_proved")
    current = selected.graph
    constants = {str(v) for t in current.triples for v in (t[0], t[2]) if not _semantic_scalar_path_restore_VAR.fullmatch(str(v))}
    if constants != {anchor}:
        return None, _semantic_scalar_path_restore_evidence("incumbent_topic_anchor_scope_differs")
    roots = [(str(p), str(b), "forward") for a, p, b in current.triples if str(a) == anchor and _semantic_scalar_path_restore_VAR.fullmatch(str(b))]
    roots += [(str(p), str(a), "backward") for a, p, b in current.triples if str(b) == anchor and _semantic_scalar_path_restore_VAR.fullmatch(str(a))]
    if len(roots) != 1:
        return None, _semantic_scalar_path_restore_evidence("incumbent_primary_edge_not_unique")
    actual_relation, first_owner, actual_direction = roots[0]
    inverse = set(ontology.reverse_for_relation(primary))
    desired_direction = str(first["direction"])
    contradiction = None
    if actual_relation in inverse and actual_direction == desired_direction:
        # Preserve all actual source conditions; arbitrary extra clauses are
        # not authorized to disappear during this narrowly proved inversion.
        if len(current.triples) == 1 and not current.operators and current.answer_var == first_owner:
            contradiction = {"kind": "official_inverse_substituted_without_direction_flip", "actual_relation": actual_relation,
                             "required_relation": primary, "unchanged_direction": desired_direction}
    elif actual_relation == primary and actual_direction == desired_direction and current.answer_var != first_owner:
        if len(current.triples) == 2 and len(current.operators) == 1:
            extra = next((tuple(map(str, t)) for t in current.triples if anchor not in (str(t[0]), str(t[2]))), None)
            expected = (first_owner, primary, str(current.answer_var)) if desired_direction == "forward" else (str(current.answer_var), primary, first_owner)
            op = current.operators[0]
            actual_attribute = str(op.get("attribute_relation_id", "")) or _semantic_scalar_path_restore_relation_id_from_label(op.get("attribute_relation_label", []))
            if (extra == expected and op.get("input_var") == current.answer_var and actual_attribute == attribute
                    and op.get("type") == spec["type"] and _semantic_scalar_path_restore_number(op.get("value")) == _semantic_scalar_path_restore_number(spec["value"])):
                contradiction = {"kind": "extra_same_predicate_hop_overrode_original_unique_entity_owner", "original_owner_var": first_owner,
                                 "actual_owner_and_answer_var": current.answer_var, "extra_triple": list(extra)}
    if contradiction is None:
        return None, _semantic_scalar_path_restore_evidence("no_proved_inverse_error_or_unique_owner_projection_override")
    first_triple = [anchor, primary, "V0"] if desired_direction == "forward" else ["V0", primary, anchor]
    op = {"type": spec["type"], "input_var": "V1", "inputs": [], "attribute_relation_id": "",
          "attribute_relation_label": [], "attribute_relation_labels": [], "value": str(spec["value"]),
          "value_type": "number", "_attribute_value_already_bound": True,
          "_numeric_cast_kind": "float" if numeric_string or a_range == "type.float" else "integer"}
    op["_numeric_cast_compare" if spec["type"] == "EQUAL" else "_numeric_comparison_cast"] = True
    if numeric_string:
        op["_budgeted_numeric_string_value_guard"] = True
    proposed = _semantic_scalar_path_restore_QueryGraphCandidate(graph_id, [first_triple, ["V0", attribute, "V1"]], "V0", [op], float(current.score), {},
        {"original_scalar_path_restore_shadow": {"original_path_id": str(path.get("id", "")), "original_owner_local_var": owner_local,
         "original_value_local_var": scalar_local, "actual_topic_binding": anchor, "contradiction_witness": contradiction,
         "question_owner_schema_type": a_domain, "requires_saved_query_count_and_time_credit": True,
         "uses_gold": False, "caller_owns_execution_and_budget": True}})
    evidence.update({"status": "compiled_not_executed", "reason": "complete_original_path_with_proved_incumbent_direction_or_owner_contradiction",
                     "contradiction": contradiction, "primary_relation": primary, "attribute_relation": attribute,
                     "normalized_spec": dict(spec), "required_owner_type": a_domain, "original_scope": scope,
                     "incumbent_subset_used_as_positive_evidence": False,
                     "remaining_requirements": ["actual paid execution and complete entity rows", "no query truncation",
                                                "fresh and old nonreg influence audit", "safe final selection and labels"]})
    return proposed, evidence



# ---- unrecognized_intersection_scalar ----
"""Pure shadow compiler for an intersection that missed a scalar obligation.

Requires actual prior applied-intersection metadata with an explicitly null
scalar spec, and a full reviewed single path from the same current flow.
Existing source BGP/operators are preserved; one schema-exact scalar edge and
the independently normalized comparator are appended. No models, execution,
file/Gold access, answer selection, sample routing or global guard change.
"""
from collections.abc import Mapping as _unrecognized_intersection_scalar_Mapping, Sequence as _unrecognized_intersection_scalar_Sequence
from copy import deepcopy
import math
import re
from typing import Any

from .budgeted_schema_repair import NUMERIC_TYPES as _unrecognized_intersection_scalar_NUMERIC_TYPES, STRING_TYPES as _unrecognized_intersection_scalar_STRING_TYPES, _existing_exact_scalar_intent, _number as _unrecognized_intersection_scalar_number, _one_explicit_scalar_spec as _unrecognized_intersection_scalar_one_explicit_scalar_spec, lower_budgeted_scalar_query
from .normalization import normalize_comparison_constraint_spec, parse_english_number
from .contracts import ExecutedGraph as _unrecognized_intersection_scalar_ExecutedGraph, QueryGraphCandidate as _unrecognized_intersection_scalar_QueryGraphCandidate
from .ontology import relation_id_from_label as _unrecognized_intersection_scalar_relation_id_from_label


_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_unrecognized_intersection_scalar_VAR = re.compile(r"V\d+\Z")


def _unrecognized_intersection_scalar_words(value):
    words = set()
    for token in re.findall(r"[a-z]+", str(value).replace("_", " ").casefold()):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        words.add(token)
    return words


def _head(question):
    match = re.match(r"\s*(?:(?:in|of|among)\s+)?(?:what|which)\s+([A-Za-z]+)\b", question, re.I)
    return _unrecognized_intersection_scalar_words(match[1]) if match else set()


def _entity(value):
    if isinstance(value, _unrecognized_intersection_scalar_Mapping):
        return str(value.get("id", value.get("entity_id", ""))), str(value.get("label", value.get("name", "")))
    return str(getattr(value, "entity_id", "")), str(getattr(value, "label", ""))


def _unrecognized_intersection_scalar_evidence(reason):
    return {"status": "not_compiled", "reason": reason, "shadow_only": True,
            "model_calls": 0, "endpoint_queries": 0, "embedding_calls": 0,
            "uses_gold_answers_or_sparql": False, "uses_entity_relation_or_sample_whitelist": False,
            "production_activation_authorized": False, "compiler_selects_answers": False,
            "requires_actual_full_source_execution_and_saved_query_time_budget": True}


def intersection_scalar_question_necessary(question):
    """Question-only necessary shape; never supplies missing prior authority."""
    if re.search(r'["“”]|\b(?:not|never|or|either|except|approximately|about|between|most|least|first|last|latest|earliest)\b', str(question), re.I):
        return _unrecognized_intersection_scalar_evidence("question_not_one_unquoted_numeric_comparison")
    spec = _unrecognized_intersection_scalar_one_explicit_scalar_spec(str(question), [], "type.float")
    if spec is None:
        spec = normalize_comparison_constraint_spec(str(question))
    if not spec or spec.get("value_type") != "number" or spec.get("type") not in {"LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"}:
        return _unrecognized_intersection_scalar_evidence("prior_missing_spec_is_not_now_one_proven_numeric_comparison")
    if len(re.findall(r"\band\b", str(question), re.I)) != 1:
        return _unrecognized_intersection_scalar_evidence("question_has_other_conjuncts_or_no_same_owner_and_clause")
    return {**_unrecognized_intersection_scalar_evidence("necessary_question_shape_only_actual_prior_and_full_scope_still_required"), "necessary_gate": True,
            "normalized_spec": deepcopy(spec)}


def intersection_scalar_obligation_necessary(question, prior):
    if (not isinstance(prior, _unrecognized_intersection_scalar_Mapping) or prior.get("status") != "applied"
            or "explicit_constraint_obligation" not in prior or prior["explicit_constraint_obligation"] is not None
            or prior.get("grounded_anchor_obligation_count") != 1):
        return _unrecognized_intersection_scalar_evidence("not_proven_applied_intersection_with_explicitly_unrecognized_scalar")
    proof = intersection_scalar_question_necessary(question)
    if proof.get("necessary_gate"):
        proof["reason"] = "necessary_prior_and_number_only_full_scope_still_required"
    return proof


def _program_core(graph):
    read = graph.get if isinstance(graph, _unrecognized_intersection_scalar_Mapping) else lambda key, default=None: getattr(graph, key, default)
    triples, operators, answer = read("triples"), read("operators"), read("answer_var")
    if (not isinstance(triples, list) or not triples or not isinstance(operators, list)
            or any(not isinstance(t, (list, tuple)) or len(t) != 3
                   or any(not isinstance(v, str) or not v for v in t) for t in triples)
            or any(not isinstance(op, _unrecognized_intersection_scalar_Mapping) for op in operators)
            or not isinstance(answer, str) or not _unrecognized_intersection_scalar_VAR.fullmatch(answer)):
        return None
    return tuple(tuple(t) for t in triples), answer, deepcopy(operators)


def _complete_entity_execution(execution):
    values, count = getattr(execution, "answer_ids", None), getattr(execution, "row_count", None)
    return (isinstance(values, list) and bool(values) and isinstance(count, int) and not isinstance(count, bool)
            and count == len(values) and all(isinstance(v, str) and _MID.fullmatch(v) for v in values)
            and len(set(values)) == len(values))


def _finite_score(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def resolve_unrecognized_intersection_source(
    question: str, selected: _unrecognized_intersection_scalar_ExecutedGraph, executions: _unrecognized_intersection_scalar_Sequence[_unrecognized_intersection_scalar_ExecutedGraph],
    prior_selection: _unrecognized_intersection_scalar_Mapping[str, Any] | None, stages: _unrecognized_intersection_scalar_Sequence[_unrecognized_intersection_scalar_Mapping[str, Any]],
) -> tuple[_unrecognized_intersection_scalar_ExecutedGraph | None, dict[str, Any]]:
    """Attest the actual full source before any paid query; never completes IDs.

    Runtime operands, the complete passed selector record, its latest actual
    stage, captured previews/programs, and endpoint counts must all agree.
    Uniqueness is tested over *all* matching executions, before completeness.
    No artifacts, ontology, models, endpoint calls, or answer synthesis.
    """
    proof = {**_unrecognized_intersection_scalar_evidence("current_source_not_attested"), "pure_current_execution_attestation": True,
             "answer_ids_synthesized": False, "additional_endpoint_queries": 0,
             "additional_model_calls": 0, "additional_embedding_calls": 0}

    def abstain(reason):
        return None, {**proof, "reason": reason}

    if (not isinstance(question, str) or not question or not isinstance(prior_selection, _unrecognized_intersection_scalar_Mapping)
            or not isinstance(stages, _unrecognized_intersection_scalar_Sequence) or isinstance(stages, (str, bytes))
            or not isinstance(executions, _unrecognized_intersection_scalar_Sequence) or isinstance(executions, (str, bytes))):
        return abstain("actual_passed_prior_or_current_runtime_contract_missing")
    rankings = [s for s in stages if isinstance(s, _unrecognized_intersection_scalar_Mapping) and s.get("stage") == "glm_final_graph_ranking"]
    if (not rankings or not isinstance(rankings[-1].get("output"), _unrecognized_intersection_scalar_Mapping)
            or rankings[-1]["output"] != prior_selection):
        return abstain("passed_prior_not_identical_to_latest_actual_final_ranking_record")
    ranking_input = rankings[-1].get("input", {})
    if (not isinstance(ranking_input, _unrecognized_intersection_scalar_Mapping)
            or "question" in ranking_input and ranking_input["question"] != question):
        return abstain("latest_final_ranking_not_owned_by_current_question")
    graph = getattr(selected, "graph", None)
    provenance = getattr(graph, "provenance", None)
    marker = provenance.get("selection_answer_intersection") if isinstance(provenance, _unrecognized_intersection_scalar_Mapping) else None
    gate = prior_selection.get("hard_conjunctive_answer_intersection_gate")
    if (not isinstance(marker, _unrecognized_intersection_scalar_Mapping) or not isinstance(gate, _unrecognized_intersection_scalar_Mapping)
            or not intersection_scalar_obligation_necessary(question, gate).get("necessary_gate")
            or provenance.get("original_question") != question or "selection_answer_union" in provenance):
        return abstain("current_question_owned_intersection_and_explicit_null_applied_prior_missing")
    left_id, right_id = marker.get("left_graph_id"), marker.get("right_graph_id")
    if (not isinstance(left_id, str) or not left_id or not isinstance(right_id, str) or not right_id or left_id == right_id
            or getattr(graph, "graph_id", None) != left_id or prior_selection.get("selected_graph_id") != left_id
            or prior_selection.get("hard_conjunctive_answer_intersection_gate_source_graph_id") != left_id
            or gate.get("source_graph_id") != left_id or gate.get("selected_graph_id") != left_id
            or gate.get("left_graph_id") != left_id or gate.get("right_graph_id") != right_id
            or not _finite_score(marker.get("score_margin")) or not _finite_score(gate.get("score_margin"))
            or marker["score_margin"] != gate["score_margin"]):
        return abstain("marker_gate_and_current_source_graph_identities_or_margin_conflict")
    actual = {}
    for graph_id in (left_id, right_id):
        matches = [e for e in executions if getattr(getattr(e, "graph", None), "graph_id", None) == graph_id]
        if len(matches) != 1 or not _complete_entity_execution(matches[0]) or _program_core(matches[0].graph) is None:
            return abstain("unique_complete_actual_operand_execution_missing")
        execution = matches[0]
        owned = getattr(execution.graph, "provenance", None)
        if (not isinstance(owned, _unrecognized_intersection_scalar_Mapping) or owned.get("original_question") != question
                or "selection_answer_intersection" in owned or "selection_answer_union" in owned):
            return abstain("actual_operand_not_question_owned_or_is_postprocessed")
        actual[graph_id] = execution
    left, right = actual[left_id], actual[right_id]
    if (not _complete_entity_execution(selected) or _program_core(graph) != _program_core(left.graph)
            or getattr(graph, "score", None) != left.graph.score
            or provenance.get("anchor_bindings") != left.graph.provenance.get("anchor_bindings")):
        return abstain("current_complete_intersection_core_score_or_root_binding_drifted")
    intersected = set(left.answer_ids) & set(right.answer_ids)
    if (not intersected or set(selected.answer_ids) != intersected or not intersected < set(left.answer_ids)
            or gate.get("source_answer_count") != len(left.answer_ids)
            or gate.get("selected_answer_count") != len(selected.answer_ids)
            or any(not isinstance(gate.get(k), int) or isinstance(gate.get(k), bool)
                   for k in ("source_answer_count", "selected_answer_count", "operator_count"))
            or gate.get("operator_count") != len(left.graph.operators)
            or gate.get("operator_count") != len(right.graph.operators) or isinstance(gate.get("operator_count"), bool)):
        return abstain("current_ids_or_gate_counts_not_the_exact_complete_actual_intersection")
    snapshots = prior_selection.get("candidates")
    if not isinstance(snapshots, list):
        return abstain("actual_captured_selector_operand_contracts_missing")
    for graph_id, execution in actual.items():
        records = [c for c in snapshots if isinstance(c, _unrecognized_intersection_scalar_Mapping) and c.get("graph_id") == graph_id]
        if len(records) != 1 or _program_core(records[0]) != _program_core(execution.graph):
            return abstain("captured_selector_operand_core_missing_duplicate_or_conflicting")
        record = records[0]
        values, count, truncated = record.get("answer_ids"), record.get("answer_count"), record.get("answers_truncated")
        if (not isinstance(values, list) or not values or any(not isinstance(v, str) or not _MID.fullmatch(v) for v in values)
                or len(set(values)) != len(values) or not isinstance(count, int) or isinstance(count, bool)
                or count != len(execution.answer_ids) or not set(values) <= set(execution.answer_ids)
                or not (truncated is True and len(values) < count
                        or truncated is False and len(values) == count and set(values) == set(execution.answer_ids))):
            return abstain("captured_selector_preview_count_or_truncation_conflicts_with_actual_full_ids")
        answers = record.get("answers")
        if answers is not None and (not isinstance(answers, list) or any(not isinstance(a, _unrecognized_intersection_scalar_Mapping) for a in answers)
                                    or [a.get("id") for a in answers] != values):
            return abstain("captured_selector_answer_objects_conflict_with_preview_ids")
        score_key = "left_rule_score" if graph_id == left_id else "right_rule_score"
        if (not _finite_score(execution.graph.score) or not _finite_score(record.get("rule_score"))
                or not _finite_score(gate.get(score_key))
                or record["rule_score"] != execution.graph.score or gate[score_key] != execution.graph.score):
            return abstain("captured_selector_actual_operand_and_gate_scores_conflict")
    execution_rows = {left_id: [], right_id: []}
    for stage in stages:
        if not isinstance(stage, _unrecognized_intersection_scalar_Mapping) or stage.get("stage") != "deterministic_sparql_lowering_and_execution":
            continue
        stage_input, output = stage.get("input", {}), stage.get("output")
        if (not isinstance(stage_input, _unrecognized_intersection_scalar_Mapping) or "question" in stage_input and stage_input["question"] != question
                or not isinstance(output, list)):
            return abstain("same_flow_actual_execution_trace_malformed_or_other_question")
        for row in output:
            if isinstance(row, _unrecognized_intersection_scalar_Mapping) and row.get("graph_id") in execution_rows:
                execution_rows[row["graph_id"]].append(row)
    for graph_id, execution in actual.items():
        rows = execution_rows[graph_id]
        if len(rows) != 1:
            return abstain("same_flow_actual_operand_execution_trace_missing_or_duplicate")
        row = rows[0]
        if (any(not isinstance(row.get(k), int) or isinstance(row.get(k), bool) for k in ("row_count", "answer_count"))
                or row["row_count"] != execution.row_count or row["answer_count"] != len(execution.answer_ids)):
            return abstain("same_flow_actual_operand_row_and_answer_counts_conflict")
    proof.update(status="resolved_actual_complete_source", reason="all_current_operand_selector_and_execution_identities_attested",
                 source_graph_id=left_id, secondary_graph_id=right_id, source_answer_count=len(left.answer_ids),
                 current_intersection_answer_count=len(selected.answer_ids), prior_and_latest_actual_ranking_full_record_identical=True,
                 all_matching_executions_counted_before_completeness=True, current_ids_exact_actual_operand_intersection=True,
                 captured_programs_previews_counts_scores_and_actual_execution_counts_agree=True,
                 operands_question_owned_and_not_postprocessed=True, actual_source_returned_without_synthesis=True)
    return left, proof


def compile_unrecognized_intersection_scalar(
    question: str, source_graph: Any, prior: _unrecognized_intersection_scalar_Mapping[str, Any], reviewed_semantic: _unrecognized_intersection_scalar_Mapping[str, Any],
    original_anchor_candidates: _unrecognized_intersection_scalar_Mapping[str, _unrecognized_intersection_scalar_Sequence[_unrecognized_intersection_scalar_Mapping[str, Any]]], topic_contract: _unrecognized_intersection_scalar_Mapping[str, Any],
    ontology: Any, *, graph_id: str = "unrecognized_intersection_scalar_shadow",
) -> tuple[_unrecognized_intersection_scalar_QueryGraphCandidate | None, dict[str, Any]]:
    """Topic authority is explicit, not an invented canonical-label alias.

    ``topic_contract`` must be supplied from the complete current question's
    already-provided/linker MID inputs. A selected graph's anchor subset does
    not create that contract. No existing name-sensitive guard is relaxed.
    """
    proof = intersection_scalar_obligation_necessary(question, prior)
    if not proof.get("necessary_gate"):
        return None, proof

    def abstain(reason):
        proof.update(status="not_compiled", reason=reason)
        return None, proof

    paths = reviewed_semantic.get("semantic_paths", [])
    if ontology is None or len(paths) != 1 or not isinstance(paths[0], _unrecognized_intersection_scalar_Mapping):
        return abstain("reviewed_original_has_other_paths_or_unknown_scope")
    path = paths[0]
    steps = path.get("steps", [])
    if len(steps) != 2 or any(not isinstance(s, _unrecognized_intersection_scalar_Mapping) for s in steps):
        return abstain("reviewed_path_not_exact_entity_prefix_plus_scalar_field")
    first, scalar_step = steps
    root = str(path.get("anchor_ref", ""))
    if (first.get("from") != root or first.get("to") != scalar_step.get("from")
            or scalar_step.get("to") != path.get("path_output_var") or scalar_step.get("direction") != "forward"
            or first.get("direction") not in {"forward", "backward"}):
        return abstain("reviewed_original_variable_owner_or_direction_not_connected")
    primary = _unrecognized_intersection_scalar_relation_id_from_label(first.get("relation_label"))
    field = _unrecognized_intersection_scalar_relation_id_from_label(scalar_step.get("relation_label"))
    owner_type = str(ontology.range_for_relation(primary) if first["direction"] == "forward" else ontology.domain_for_relation(primary))
    field_domain, field_range = str(ontology.domain_for_relation(field)), str(ontology.range_for_relation(field))
    numeric_string = field_range in _unrecognized_intersection_scalar_STRING_TYPES | {"type.enumeration"}
    if (not owner_type or owner_type != field_domain or "common.topic" not in set(ontology.supertypes(owner_type))
            or field_range not in _unrecognized_intersection_scalar_NUMERIC_TYPES and not numeric_string):
        return abstain("source_entity_owner_and_exact_numeric_field_not_same_declared_type")
    if _head(question) != _unrecognized_intersection_scalar_words(owner_type.rsplit(".", 1)[-1]):
        return abstain("question_answer_kind_does_not_match_exact_declared_scalar_owner")
    if (not isinstance(topic_contract, _unrecognized_intersection_scalar_Mapping) or topic_contract.get("question") != question
            or topic_contract.get("binding_source") != "current_question_topic_mid_contract"
            or not isinstance(topic_contract.get("topics"), list) or len(topic_contract["topics"]) != 1):
        return abstain("complete_current_question_topic_mid_contract_missing_or_more_than_one_topic")
    anchor, label = _entity(topic_contract["topics"][0])
    original_roots = [a for a in reviewed_semantic.get("anchors", []) if isinstance(a, _unrecognized_intersection_scalar_Mapping) and str(a.get("id")) == root]
    provenance = getattr(source_graph, "provenance", {})
    actual_topics = provenance.get("anchor_bindings", {}) if isinstance(provenance, _unrecognized_intersection_scalar_Mapping) else {}
    actual_id, actual_label = _entity(actual_topics.get(root)) if isinstance(actual_topics, _unrecognized_intersection_scalar_Mapping) else ("", "")
    root_candidates = original_anchor_candidates.get(root)
    if (not _MID.fullmatch(anchor) or not label or actual_id != anchor or actual_label.casefold() != label.casefold()
            or len(original_roots) != 1 or str(original_roots[0].get("surface", "")).casefold() != label.casefold()
            or not isinstance(root_candidates, (list, tuple)) or len(root_candidates) != 1
            or _entity(root_candidates[0]) != (anchor, label)):
        return abstain("actual_source_reviewed_linker_and_input_topic_identity_do_not_align")
    if provenance.get("original_question") != question:
        return abstain("source_graph_not_owned_by_this_exact_current_question")
    owner_words = _unrecognized_intersection_scalar_words(owner_type.rsplit(".", 1)[-1])
    field_words = _unrecognized_intersection_scalar_words(field.rsplit(".", 1)[-1])
    clause = re.search(r"\band\s+(?:the\s+)?(?P<body>[A-Za-z ]+?)\s+(?:is|has|have)\s+(?:greater|more|higher|larger|less|fewer|lower|smaller|at\s+most|at\s+least|no\s+more|no\s+less)\s+(?:than\s+)?", question, re.I)
    if not clause or _unrecognized_intersection_scalar_words(clause["body"]) != owner_words | field_words:
        return abstain("question_does_not_explicitly_conjoin_same_owner_and_exact_field")
    # An extra capitalized topic in the entity prefix cannot be hidden by
    # the authoritative contract. This does not rewrite names or infer MIDs.
    prefix = question[:clause.start()].strip()
    caps = list(re.finditer(r"\b[A-Z][A-Za-z]*\b", prefix))
    for token in caps:
        if token.start() == 0:
            continue
        if not _unrecognized_intersection_scalar_words(token[0]) <= _unrecognized_intersection_scalar_words(label):
            return abstain("question_prefix_has_unrepresented_named_topic_token")
    spec = proof["normalized_spec"]
    goal = str(path.get("goal", ""))
    goal_spec = _unrecognized_intersection_scalar_one_explicit_scalar_spec(goal, [label], "type.float")
    if goal_spec is None:
        goal_spec = normalize_comparison_constraint_spec(goal)
    if (not goal_spec or goal_spec.get("type") != spec["type"] or _unrecognized_intersection_scalar_number(goal_spec.get("value")) != _unrecognized_intersection_scalar_number(spec["value"])
            or not owner_words | field_words <= _unrecognized_intersection_scalar_words(goal)):
        return abstain("reviewed_goal_does_not_attest_same_owner_field_numeric_comparator_literal")
    scope = []
    referenced = {str(path.get("anchor_ref"))} | {str(s.get(k)) for s in steps for k in ("from", "to")}
    for item in reviewed_semantic.get("anchors", []):
        if not isinstance(item, _unrecognized_intersection_scalar_Mapping):
            return abstain("original_anchor_contract_invalid")
        key, surface = str(item.get("id", "")), str(item.get("surface", ""))
        if key == root:
            continue
        if key in referenced or original_anchor_candidates.get(key) != []:
            return abstain("extra_original_anchor_is_path_referenced_bound_or_unknown")
        literal = _unrecognized_intersection_scalar_number(surface)
        if literal is None:
            literal = _unrecognized_intersection_scalar_number(parse_english_number(surface))
        if literal is not None and literal == _unrecognized_intersection_scalar_number(spec["value"]):
            scope.append({"anchor_id": key, "surface": surface, "kind": "exact_normalized_scalar_threshold"})
        elif _unrecognized_intersection_scalar_words(surface) == owner_words and _head(question) == owner_words:
            scope.append({"anchor_id": key, "surface": surface, "kind": "unbound_unused_kind_proved_by_qhead_and_same_declared_owner"})
        else:
            return abstain("unused_original_topic_kind_or_literal_not_positively_accounted_for")
    if (len([x for x in scope if x["kind"] == "exact_normalized_scalar_threshold"]) > 1
            or len([x for x in scope if x["kind"].startswith("unbound_unused_kind")]) > 1):
        return abstain("duplicate_kind_or_literal_scope_ambiguous")
    if _existing_exact_scalar_intent(source_graph, field, spec, ontology):
        return abstain("source_already_expresses_same_owner_field_comparator_literal")
    triples, operators = getattr(source_graph, "triples", None), getattr(source_graph, "operators", None)
    if (not isinstance(triples, list) or len(triples) != 1 or not isinstance(operators, list)
            or any(op.get("type") != "NO_EQUAL" for op in operators)
            or prior.get("operator_count") != len(operators)):
        return abstain("source_has_other_bgp_or_substantive_operators_not_this_prefix")
    answer = str(source_graph.answer_var)
    expected = [anchor, primary, answer] if first["direction"] == "forward" else [answer, primary, anchor]
    if not _unrecognized_intersection_scalar_VAR.fullmatch(answer) or triples[0] != expected or source_graph.graph_id != prior.get("source_graph_id"):
        return abstain("actual_source_primary_direction_id_or_entity_answer_does_not_match_reviewed_prefix")
    intersection = provenance.get("selection_answer_intersection")
    if (not isinstance(intersection, _unrecognized_intersection_scalar_Mapping) or intersection.get("left_graph_id") != prior.get("left_graph_id")
            or intersection.get("right_graph_id") != prior.get("right_graph_id")
            or source_graph.graph_id not in {prior.get("left_graph_id"), prior.get("right_graph_id")}):
        return abstain("prior_applied_intersection_not_the_current_graph_postprocessing")
    result = deepcopy(source_graph)
    result.graph_id, result.sparql = graph_id, ""
    # This is a new query denotation, not the old answer-ID intersection.
    # Keep the actual prior marker as evidence, without claiming that its
    # postprocessing applies to answers returned by the new scalar query.
    archived_intersection = result.provenance.pop("selection_answer_intersection")
    value_var = "V" + str(max(int(v[1:]) for t in triples for v in (str(t[0]), str(t[2])) if _unrecognized_intersection_scalar_VAR.fullmatch(v)) + 1)
    result.triples.append([answer, field, value_var])
    op = {"type": spec["type"], "inputs": [], "input_var": value_var, "attribute_relation_id": "",
          "attribute_relation_label": [], "attribute_relation_labels": [], "value": spec["value"], "value_type": "number",
          "_attribute_value_already_bound": True, "_numeric_comparison_cast": True,
          "_numeric_cast_kind": "float" if numeric_string or field_range == "type.float" else "integer"}
    if numeric_string:
        op["_budgeted_numeric_string_value_guard"] = True
    result.operators.append(op)
    proof.update(status="compiled_shadow_not_executed", reason="actual_intersection_missed_same_owner_reviewed_numeric_obligation",
                 source_graph_id=source_graph.graph_id, reviewed_path_id=path["id"], scalar_relation_id=field,
                 primary_relation_id=primary, owner_type=owner_type, owner_var=answer, scalar_var=value_var,
                 prior_full_source_answer_count=prior.get("source_answer_count"),
                 prior_selection_answer_intersection_archived=deepcopy(archived_intersection),
                 old_selection_answer_intersection_marker_removed=True,
                 original_unused_anchor_scope=scope, topic_mid_contract_source=topic_contract["binding_source"],
                 actual_topic_id=anchor, canonical_topic_label=label,
                 canonical_label_fully_present_in_question=bool(re.search(re.escape(label), question, re.I)),
                 canonical_label_alias_inferred_or_name_changed=False,
                 provided_current_question_mid_authority_explicitly_used=True,
                 no_global_name_sensitive_guard_or_other_repair_trigger_relaxed=True,
                 source_bgp_and_operators_exactly_preserved=True, source_entity_answer_projection_unchanged=True,
                 restored_scalar_obligation_from_single_original_reviewed_path=True,
                 require_actual_uncapped_nonempty_strict_subset_of_full_source_before_replacement=True,
                 strict_subset_of_postprocessed_incumbent_not_required_or_safety_proof=True)
    result.provenance["unrecognized_intersection_scalar_shadow"] = deepcopy(proof)
    return result, proof


def compile_unrecognized_intersection_from_current_stages(
    question: str, source_graph: Any, stages: _unrecognized_intersection_scalar_Sequence[_unrecognized_intersection_scalar_Mapping[str, Any]], ontology: Any,
    *, graph_id: str = "unrecognized_intersection_scalar_shadow",
    linked_entity_context: _unrecognized_intersection_scalar_Sequence[_unrecognized_intersection_scalar_Mapping[str, Any]] | None = None,
    linked_entity_context_question: str | None = None,
) -> tuple[_unrecognized_intersection_scalar_QueryGraphCandidate | None, dict[str, Any]]:
    """Current stages or explicitly question-owned existing context cache.

    A cache is not a fabricated stage and cannot override actual stage scope.
    Its owner must be the exact question used for the caller's cache lookup.
    Never reads historical artifacts or fetches missing context.
    """
    prior = None
    context = None
    reviewed_decompositions = set()
    pending = None
    source_program_verified = False
    proposals = []
    corpus_failure = None
    for stage in stages:
        if not isinstance(stage, _unrecognized_intersection_scalar_Mapping):
            continue
        if stage.get("stage") == "linked_entity_context":
            next_context = stage.get("output")
            if context is not None and next_context != context:
                corpus_failure = "same_flow_linked_context_contracts_conflict"
            context = next_context
        elif stage.get("stage") == "decomposition_review_and_rewrite":
            output = stage.get("output", {})
            review_input = stage.get("input", {})
            if (not isinstance(review_input, _unrecognized_intersection_scalar_Mapping) or review_input.get("question") != question
                    or not isinstance(output, _unrecognized_intersection_scalar_Mapping) or not isinstance(output.get("final_candidates"), list)):
                corpus_failure = "same_flow_current_question_review_contract_missing_or_conflicting"
                continue
            if any(isinstance(c, _unrecognized_intersection_scalar_Mapping) and c.get("source") == "review_rewrite"
                   and (not isinstance(c.get("decomposition"), list) or not c["decomposition"]
                        or any(not isinstance(s, str) or not s for s in c["decomposition"]))
                   for c in output["final_candidates"]):
                corpus_failure = "reviewed_rewrite_proposal_contract_malformed"
            reviewed_decompositions.update(tuple(c["decomposition"]) for c in output["final_candidates"]
                                           if isinstance(c, _unrecognized_intersection_scalar_Mapping) and c.get("source") == "review_rewrite"
                                           and isinstance(c.get("decomposition"), list)
                                           and all(isinstance(s, str) for s in c["decomposition"]))
        elif stage.get("stage") == "semantic_path_generation":
            output = stage.get("output", [])
            if pending is not None:
                corpus_failure = "earlier_reviewed_semantic_corpus_has_no_complete_grounding_contract"
            semantic_input = stage.get("input", {})
            if (not isinstance(semantic_input, _unrecognized_intersection_scalar_Mapping) or semantic_input.get("question") != question
                    or not isinstance(output, list)):
                corpus_failure = "semantic_stage_not_this_complete_current_question_flow"
                pending = None
            else:
                # The actual pipeline grounds the valid semantic entries,
                # including original fallbacks, and graph_index refers to
                # that valid-only list. Do not reindex after review filtering.
                # Every validated reviewed proposal enters the closed corpus,
                # including competing scope that cannot compile or ground.
                valid_entries = [entry for entry in output if isinstance(entry, _unrecognized_intersection_scalar_Mapping) and entry.get("valid") is True]
                pending = [(index, entry) for index, entry in enumerate(valid_entries)
                           if isinstance(entry.get("decomposition"), list)
                           and all(isinstance(s, str) for s in entry["decomposition"])
                           and tuple(entry["decomposition"]) in reviewed_decompositions]
        elif stage.get("stage") == "semantic_subgraph_grounding_and_ranking" and pending is not None:
            output = stage.get("output", {})
            graphs = output.get("semantic_graphs") if isinstance(output, _unrecognized_intersection_scalar_Mapping) else None
            if not isinstance(graphs, list):
                corpus_failure = "reviewed_semantic_corpus_grounding_contract_missing"
            else:
                if any(not isinstance(g, _unrecognized_intersection_scalar_Mapping) or not isinstance(g.get("graph_index"), int)
                       or isinstance(g.get("graph_index"), bool) for g in graphs):
                    corpus_failure = "reviewed_semantic_corpus_grounding_index_contract_malformed"
                for index, entry in pending:
                    matches = [g for g in graphs if isinstance(g, _unrecognized_intersection_scalar_Mapping) and g.get("graph_index") == index
                               and not isinstance(g.get("graph_index"), bool)]
                    if (len(matches) != 1 or not isinstance(entry.get("output"), _unrecognized_intersection_scalar_Mapping)
                            or matches[0].get("dropped_paths") != [] or not isinstance(matches[0].get("anchors"), list)):
                        corpus_failure = "validated_reviewed_scope_missing_duplicate_or_dropped_grounding_contract"
                        continue
                    graph, semantic = matches[0], entry["output"]
                    anchors = graph["anchors"]
                    if (any(not isinstance(a, _unrecognized_intersection_scalar_Mapping) or not isinstance(a.get("anchor_id"), str)
                            or not isinstance(a.get("candidates"), list) for a in anchors)
                            or len({a["anchor_id"] for a in anchors}) != len(anchors)):
                        corpus_failure = "validated_reviewed_scope_anchor_grounding_contract_incomplete"
                        continue
                    paths = semantic.get("semantic_paths")
                    if (not isinstance(paths, list) or any(not isinstance(p, _unrecognized_intersection_scalar_Mapping)
                                                          or p.get("goal") not in entry["decomposition"] for p in paths)):
                        corpus_failure = "validated_reviewed_scope_goal_not_owned_by_reviewed_decomposition"
                        continue
                    candidates = {a["anchor_id"]: a["candidates"] for a in anchors}
                    semantic_anchors = semantic.get("anchors")
                    if (not isinstance(semantic_anchors, list)
                            or any(not isinstance(a, _unrecognized_intersection_scalar_Mapping) or not isinstance(a.get("id"), str) for a in semantic_anchors)
                            or len({a["id"] for a in semantic_anchors}) != len(semantic_anchors)
                            or set(candidates) != {a["id"] for a in semantic_anchors}):
                        corpus_failure = "reviewed_semantic_and_complete_grounding_anchor_scopes_conflict"
                        continue
                    proposals.append((semantic, candidates))
            pending = None
        elif stage.get("stage") == "glm_final_graph_ranking":
            output = stage.get("output", {})
            prior = output.get("hard_conjunctive_answer_intersection_gate") if isinstance(output, _unrecognized_intersection_scalar_Mapping) else None
            actual = [g for g in output.get("candidates", []) if isinstance(g, _unrecognized_intersection_scalar_Mapping)
                      and g.get("graph_id") == source_graph.graph_id] if isinstance(output, _unrecognized_intersection_scalar_Mapping) else []
            source_program_verified = bool(len(actual) == 1 and actual[0].get("triples") == source_graph.triples
                                           and actual[0].get("operators") == source_graph.operators
                                           and actual[0].get("answer_var") == source_graph.answer_var)
    if pending is not None:
        corpus_failure = "reviewed_semantic_corpus_has_no_complete_current_grounding_contract"
    if corpus_failure is not None:
        return None, _unrecognized_intersection_scalar_evidence(corpus_failure)
    context_origin = "actual_stage"
    if linked_entity_context is not None:
        if (linked_entity_context_question != question or not isinstance(linked_entity_context, list)
                or any(not isinstance(c, _unrecognized_intersection_scalar_Mapping) for c in linked_entity_context)):
            return None, _unrecognized_intersection_scalar_evidence("supplied_current_question_cache_context_owner_or_contract_conflicts")
        if context is not None and context != linked_entity_context:
            return None, _unrecognized_intersection_scalar_evidence("actual_stage_and_current_question_cache_context_conflict")
        if context is None:
            context = linked_entity_context
            context_origin = "current_question_cache"
    if not isinstance(context, list) or any(not isinstance(c, _unrecognized_intersection_scalar_Mapping) for c in context):
        return None, _unrecognized_intersection_scalar_evidence("complete_current_question_topic_contract_context_missing")
    if not source_program_verified:
        return None, _unrecognized_intersection_scalar_evidence("actual_final_selector_source_query_program_not_verified")
    topic_contract = {"question": question, "binding_source": "current_question_topic_mid_contract", "topics": context}
    candidates, rejections = [], []
    for semantic, anchors in proposals:
        candidate, proof = compile_unrecognized_intersection_scalar(question, source_graph, prior, semantic, anchors,
                                                                    topic_contract, ontology, graph_id=graph_id)
        (candidates if candidate is not None else rejections).append((candidate, proof))
    if not candidates:
        return rejections[0] if rejections else (None, _unrecognized_intersection_scalar_evidence("same_current_flow_reviewed_full_scalar_semantic_missing"))
    if rejections:
        return None, {**_unrecognized_intersection_scalar_evidence("competing_validated_reviewed_scope_not_fully_proven"),
                      "reviewed_proposal_count": len(proposals), "rejected_reviewed_scope_proofs": [p for _, p in rejections]}
    keys = {(tuple(tuple(t) for t in c.triples), c.answer_var, str(c.operators)) for c, _ in candidates}
    if len(keys) != 1:
        return None, _unrecognized_intersection_scalar_evidence("multiple_reviewed_scalar_restoration_programs_are_ambiguous")
    candidate, proof = candidates[0]
    proof["same_current_flow_reviewed_semantic_linker_and_actual_prior_verified"] = True
    proof["actual_final_selector_source_query_program_verified"] = True
    proof["entire_validated_reviewed_same_flow_proposal_corpus_checked"] = True
    proof["validated_reviewed_proposal_count"] = len(proposals)
    proof["context_origin"] = context_origin
    proof["current_question_cache_owner_explicitly_verified"] = context_origin == "current_question_cache"
    candidate.provenance["unrecognized_intersection_scalar_shadow"] = deepcopy(proof)
    return candidate, proof


def lower_unrecognized_intersection_scalar(candidate: _unrecognized_intersection_scalar_QueryGraphCandidate, *, limit: int = 256) -> str:
    return lower_budgeted_scalar_query(candidate, limit=limit)



# ---- current_primary_grounded_constraint ----
"""Independent compile-only witness for a current primary ENTITY source.

This pure helper does not itself authorize a
request or select an answer: the caller owns paid count/time credit and must
prove a complete nonempty strict subset of the real source execution.
"""


from collections.abc import Mapping as _current_primary_grounded_constraint_Mapping, Sequence as _current_primary_grounded_constraint_Sequence
import re

from .budgeted_schema_repair import SCALAR_TYPES as _current_primary_grounded_constraint_SCALAR_TYPES, _COMPARISON as _current_primary_grounded_constraint_COMPARISON, _ISO_DATE, _NAMED_DATE, _VAR as _current_primary_grounded_constraint_VAR, _evidence as _current_primary_grounded_constraint_evidence, _one_explicit_scalar_spec as _current_primary_grounded_constraint_one_explicit_scalar_spec, _regular_role_stems, compile_grounded_constraint_repair, grounding_schema_evidence
_current_primary_grounded_constraint_original_semantic_scope_certificate = _compose_scalar_scope_safety_original_semantic_scope_certificate
from .lowering import lower_sparql


def _current_primary_grounded_constraint_tokens(value):
    return tuple(word[:-1] if len(word) > 4 and word.endswith("s") else word
                 for word in re.findall(r"[A-Za-z]+", str(value).casefold()))


def _label(value):
    return " ".join(str(value).split()).casefold()


def _current_primary_grounded_constraint_word_forms(value):
    forms = set(_regular_role_stems(value))
    # An adjective formed from a canonical owner noun is regular morphology,
    # not an occupation/property synonym list (government -> governmental).
    for word in tuple(forms):
        if len(word) > 6 and word.endswith("al"):
            forms.add(word[:-2])
    return forms


def _question_relative_owner_tail(tail, owner_words, leaf_words, range_id):
    """Consume the whole relative condition, never just a scalar prefix."""
    comparisons = list(_current_primary_grounded_constraint_COMPARISON.finditer(tail))
    if len(comparisons) != 1:
        return False
    literal = tail[comparisons[0].end():].strip().rstrip("?!. ").strip()
    if not ((_ISO_DATE.fullmatch(literal) or _NAMED_DATE.fullmatch(literal))
            if range_id == "type.datetime"
            else re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?", literal)):
        return False
    body = tail[:comparisons[0].start()].strip().lstrip(",").strip()
    if re.search(r"\b(?:and|or|where)\b", body, re.I):
        return False
    words = list(_current_primary_grounded_constraint_tokens(body))
    if not words or words.pop(0) not in {"who", "which", "that"}:
        return False
    if words and words[-1] == ("date" if range_id == "type.datetime" else "value"):
        words.pop()
    if not leaf_words or tuple(words[-len(leaf_words):]) != tuple(leaf_words):
        return False
    del words[-len(leaf_words):]
    possessives = [index for index, word in enumerate(words) if word in {"his", "her", "their", "its", "whose"}]
    if len(possessives) != 1 or possessives[0] == len(words) - 1:
        return False
    index = possessives[0]
    auxiliaries = {"is", "are", "was", "were", "has", "have", "had", "does", "do", "did"}
    pre_owner = [word for word in words[:index] if word not in auxiliaries]
    actual_owner = [*pre_owner, *words[index + 1:]]
    if len(actual_owner) != len(owner_words):
        return False
    # Require all owner tokens, without extras or duplicates, in any regular
    # active arrangement. E.g. held + possessive + governmental position is
    # exactly the same canonical owner as government positions held.
    matches = []

    def visit(position, used):
        if position == len(actual_owner):
            matches.append(tuple(sorted(used)))
            return
        for target, word in enumerate(owner_words):
            if target not in used and _current_primary_grounded_constraint_word_forms(actual_owner[position]) & _current_primary_grounded_constraint_word_forms(word):
                visit(position + 1, used | {target})

    if len(actual_owner) > 6:
        return False
    visit(0, set())
    return bool(matches)


def current_primary_grounded_constraint_witness(
    question, source, grounded, source_semantic, ontology, *, decomposition,
    source_query, graph_id="current_primary_constraint",
):
    """Compile the existing f4 extension only after independent source proof.

    Eligibility is narrow: one rooted, complete, purely ontology-grounded
    original path, a direct regular active WHO entity role, an unmodified
    one-edge actual SELECT, and the same full scalar obligation in question,
    original goal and its independently supplied decomposition. An original
    three-clause goal explicitly returns the primary owner, not the scalar.
    """
    evidence = _current_primary_grounded_constraint_evidence("missing_independent_current_primary_source_proof")

    def abstain(reason, **detail):
        return None, {**evidence, "reason": reason, **detail}

    if (not isinstance(decomposition, _current_primary_grounded_constraint_Sequence) or isinstance(decomposition, str)
            or not decomposition or not all(isinstance(value, str) for value in decomposition)):
        return abstain("original_goal_not_independently_attached_to_same_decomposition")
    graph = getattr(source, "graph", None)
    if (graph is None or not isinstance(getattr(graph, "provenance", None), _current_primary_grounded_constraint_Mapping)
            or not isinstance(getattr(graph, "triples", None), list)
            or not isinstance(getattr(graph, "operators", None), list)):
        return abstain("actual_source_graph_contract_unknown")
    if (not isinstance(graph.provenance.get("original_question"), str)
            or graph.provenance["original_question"] != question):
        return abstain("actual_source_original_question_not_identically_owned")
    ok, scope = _current_primary_grounded_constraint_original_semantic_scope_certificate(
        grounded, source_semantic, ontology=ontology,
    )
    if not ok:
        return abstain("original_semantic_scope_not_preserved", scope_certificate=scope)
    schema = grounding_schema_evidence(grounded, ontology)
    if schema.get("status") != "qualified":
        return abstain("not_complete_rejected_entity_role_grounding", schema_evidence=schema)
    paths = source_semantic["semantic_paths"]
    if len(paths) != 1:
        return abstain("current_primary_requires_one_original_path")
    path = paths[0]
    steps = path["steps"]
    if (len(steps) < 2 or steps[0]["direction"] != "forward"
            or steps[0]["from"] != path["anchor_ref"]
            or path["path_output_var"] != steps[-1]["to"]
            or any(step["direction"] != "forward" for step in steps)
            or any(left["to"] != right["from"] for left, right in zip(steps, steps[1:]))):
        return abstain("original_path_is_not_one_unambiguous_forward_owner_chain")
    root = str(path["anchor_ref"])
    anchors = source_semantic.get("anchors")
    actual_anchors = grounded.compose_input.get("anchors")
    if (not isinstance(anchors, list) or not isinstance(actual_anchors, list)
            or any(not isinstance(item, _current_primary_grounded_constraint_Mapping) or not item.get("id")
                   or not str(item.get("surface", "")).strip() for item in anchors)
            or len({item["id"] for item in anchors}) != len(anchors)):
        return abstain("original_anchor_contract_unknown_or_ambiguous")
    labels = {str(item["id"]): str(item["surface"]).strip() for item in anchors}
    actual_labels = {str(item.get("id")): str(item.get("surface", "")).strip()
                     for item in actual_anchors if isinstance(item, _current_primary_grounded_constraint_Mapping)}
    binding = grounded.anchor_bindings.get(root)
    label = labels.get(root, "")
    if (binding is None or not label or len(actual_labels) != len(actual_anchors)
            or _label(label) != _label(actual_labels.get(root, ""))
            or _label(label) != _label(binding.label)
            or set(grounded.anchor_bindings) != {root} or set(actual_labels) != {root}):
        return abstain("actual_root_label_or_grounded_anchor_scope_not_identical")
    source_anchors = graph.provenance.get("anchor_bindings")
    if (not isinstance(source_anchors, _current_primary_grounded_constraint_Mapping) or not source_anchors
            or set(source_anchors) != {root}):
        return abstain("actual_source_anchor_scope_not_exactly_owned_root")
    source_anchor = source_anchors[root]
    if (not isinstance(source_anchor, _current_primary_grounded_constraint_Mapping)
            or not isinstance(source_anchor.get("id"), str) or not source_anchor["id"]
            or source_anchor["id"] != binding.entity_id
            or not isinstance(source_anchor.get("label"), str) or not source_anchor["label"].strip()
            or _label(source_anchor["label"]) != _label(label)
            or _label(source_anchor["label"]) != _label(binding.label)):
        return abstain("actual_source_root_binding_not_identical_to_original_and_grounded_anchor")
    topic_accounting = []
    for identifier, surface in labels.items():
        if identifier == root:
            continue
        used = any(identifier in (step["from"], step["to"]) for step in steps)
        mentioned = any(re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)",
                                  str(value), re.I) for value in (question, path.get("goal", ""), *decomposition))
        if used or mentioned:
            return abstain("extra_original_topic_is_required_or_unaccounted")
        topic_accounting.append({"id": identifier, "surface": surface,
                                 "disposition": "unused_and_absent_from_question"})
    text = " ".join(str(question).split())
    clause = re.match(r"^Who\s+([A-Za-z]+)\s+(?:the\s+)?[\"“]?" + re.escape(label)
                      + r"[\"”]?(?=\s*[,?]|\s+(?:who|which|that|whose|with|after|before|from|in|on)\b|$)",
                      text, re.I)
    if not clause or not re.fullmatch(r"[A-Za-z]+(?:ed|s)", clause.group(1)):
        return abstain("question_has_no_direct_regular_active_who_role")
    verb = clause.group(1).casefold()
    if verb in {"is", "was", "does", "has"} or (not verb.endswith("ed") and verb.endswith(("ers", "ors"))):
        return abstain("question_primary_token_not_an_unambiguous_regular_active_verb")
    first = steps[0]
    relation = str(grounded.relation_bindings.get(first["id"], ""))
    stems = _regular_role_stems(verb)
    if (not stems & _regular_role_stems(first["relation_label"][-1])
            or not stems & _regular_role_stems(relation.rsplit(".", 1)[-1])):
        return abstain("question_original_primary_and_actual_relation_disagree")
    role_type = str(ontology.range_for_relation(relation))
    if role_type in _current_primary_grounded_constraint_SCALAR_TYPES or "common.topic" not in set(ontology.supertypes(role_type)):
        return abstain("primary_relation_lacks_positive_entity_schema_proof")
    provenance = graph.provenance
    if any(key in provenance for key in ("selection_answer_union", "selection_answer_intersection")):
        return abstain("postprocessed_source_is_not_one_actual_execution")
    if graph.operators:
        return abstain("source_already_has_scalar_or_other_operator")
    if (len(graph.triples) != 1 or not _current_primary_grounded_constraint_VAR.fullmatch(str(graph.answer_var))
            or list(map(str, graph.triples[0])) != [str(binding.entity_id), relation, str(graph.answer_var)]):
        return abstain("actual_source_is_not_exact_direct_primary_entity_role")
    expected_query = lower_sparql(graph)
    if (not isinstance(source_query, str) or not source_query.strip()
            or source_query.strip() != expected_query.strip()
            or str(graph.sparql).strip() != source_query.strip()):
        return abstain("actual_source_owned_query_not_exact_unlimited_lowering")
    ids = source.answer_ids
    if (not isinstance(ids, list) or not ids or not all(isinstance(value, str) for value in ids)
            or not isinstance(source.row_count, int) or source.row_count != len(ids)
            or isinstance(source.row_count, bool) or len(set(ids)) != len(ids)
            or any(not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", str(value)) for value in ids)):
        return abstain("actual_source_entity_denotation_not_complete")
    scalar = [(step, str(grounded.relation_bindings.get(step["id"], ""))) for step in steps
              if str(ontology.range_for_relation(str(grounded.relation_bindings.get(step["id"], "")))) in _current_primary_grounded_constraint_SCALAR_TYPES]
    if len(scalar) != 1 or scalar[0][0] is not steps[-1]:
        return abstain("not_one_terminal_original_grounded_scalar_obligation")
    terminal, scalar_relation = scalar[0]
    range_id = str(ontology.range_for_relation(scalar_relation))
    spec = _current_primary_grounded_constraint_one_explicit_scalar_spec(text, list(labels.values()), range_id)
    if spec is None:
        return abstain("question_scalar_literal_or_comparison_is_ambiguous")
    goal = str(path.get("goal", "")).strip()
    if (not isinstance(decomposition, _current_primary_grounded_constraint_Sequence) or isinstance(decomposition, str)
            or list(decomposition) != [goal]):
        return abstain("original_goal_not_independently_attached_to_same_decomposition")
    if _current_primary_grounded_constraint_one_explicit_scalar_spec(goal, list(labels.values()), range_id) != spec:
        return abstain("question_and_original_goal_scalar_obligations_disagree")
    clauses = [part.strip().rstrip(".").strip() for part in goal.split("?") if part.strip()]
    primary = _current_primary_grounded_constraint_tokens(first["relation_label"][-1])
    owner = _current_primary_grounded_constraint_tokens(steps[-2]["relation_label"][-1])
    leaf = _current_primary_grounded_constraint_tokens(terminal["relation_label"][-1])
    expected_first = ("who", "is", "the", *primary, "of", *_current_primary_grounded_constraint_tokens(first["relation_label"][-2]), *_current_primary_grounded_constraint_tokens(label))
    expected_second = ("what", "is", "the", *owner, "of", "the", *primary)
    expected_last = ("return", "the", *primary, "whose", *owner, "has", "a", *leaf,
                     "date" if range_id == "type.datetime" else "value")
    if (len(steps) != 3 or len(clauses) != 3 or _current_primary_grounded_constraint_tokens(clauses[0]) != expected_first
            or _current_primary_grounded_constraint_tokens(clauses[1]) != expected_second
            or _current_primary_grounded_constraint_tokens(clauses[2])[:len(expected_last)] != expected_last
            or _current_primary_grounded_constraint_one_explicit_scalar_spec(clauses[2], [], range_id) != spec):
        return abstain("original_goal_does_not_explicitly_return_and_constrain_same_primary_owner")
    comparison = list(_current_primary_grounded_constraint_COMPARISON.finditer(clauses[2]))
    if len(comparison) != 1 or _current_primary_grounded_constraint_tokens(clauses[2][:comparison[0].start()]) != expected_last:
        return abstain("original_goal_has_unaccounted_condition_words")
    literal_text = clauses[2][comparison[0].end():].strip()
    if not ((_ISO_DATE.fullmatch(literal_text) or _NAMED_DATE.fullmatch(literal_text))
            if range_id == "type.datetime"
            else re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?", literal_text)):
        return abstain("original_goal_has_unaccounted_condition_words")
    # Consume the entire relative condition. An added topic/condition outside
    # the original path is not silently ignored even when its date agrees.
    tail = text[clause.end():]
    if not _question_relative_owner_tail(tail, owner, leaf, range_id):
        return abstain("question_relative_owner_or_extra_scope_not_fully_accounted")
    candidate, compilation = compile_grounded_constraint_repair(
        grounded, graph, question, spec, ontology, graph_id=graph_id,
    )
    if candidate is None:
        return abstain("existing_f4_grounded_extension_did_not_compile", compilation=compilation)
    evidence.update(status="compiled_not_executed", reason="independent_current_primary_entity_missing_scalar_witness",
                    source_graph_id=graph.graph_id, source_answer_count=len(ids),
                    primary_relation_id=relation, primary_entity_type=role_type,
                    question_active_verb=verb, normalized_scalar_spec=spec,
                    scope_certificate=scope, extra_original_topics=topic_accounting,
                    original_goal_decomposition_same_primary_owner=True,
                    actual_source_original_question_and_root_binding_verified=True,
                    actual_source_owned_query_verified=True,
                    required_repair_denotation="complete_nonempty_strict_subset_of_actual_source_ids",
                    requires_actual_paid_saved_count_and_time_credit=True,
                    query_executed=False, full_flow_recovery_proven=False,
                    compilation=compilation)
    return candidate, evidence



# ---- budgeted_parent_character_voice_repair ----
"""Pure typed repair of a parent-name query accidentally returning a date.

Only current original Semantic, already resolved source anchors/context, and
cached official schema/inverses are used. All source BGP and operators remain.
The caller alone may execute one query against saved request AND time credit.
Animated/other unsupported entity-head modifiers are not silently ignored.
"""


from collections.abc import Mapping as _budgeted_parent_character_voice_repair_Mapping
from copy import deepcopy
from datetime import date
import re
from typing import Any

_budgeted_parent_character_voice_repair_finite_forms = _budgeted_primary_projection_repair_finite_forms
from .evaluation import canonicalize_answer_id as _budgeted_parent_character_voice_repair_canonicalize_answer_id
from .safety import _nested_name_schema_guard_anchors as _budgeted_parent_character_voice_repair_anchors, _nested_name_schema_guard_complete as _budgeted_parent_character_voice_repair_complete, _nested_name_schema_guard_concrete_type as _budgeted_parent_character_voice_repair_concrete_type
from .ontology import relation_id_from_label as _budgeted_parent_character_voice_repair_relation_id_from_label
from .safety import _source_role_execution_safety_decomposition_lines as _decomposition_lines, _source_role_execution_safety_semantic_graphs as _budgeted_parent_character_voice_repair_semantic_graphs, _semantic_path_nodes as _budgeted_parent_character_voice_repair_semantic_path_nodes, _source_role_execution_safety_tokens as _budgeted_parent_character_voice_repair_tokens


_QUESTION = re.compile(r"^\s*what\s+is\s+(?:the\s+)?name\s+of\s+(.+?)['’]s\s+"
                       r"([A-Za-z]+)\s+that\s+(.+?)\s+([A-Za-z]+)\s*[?.!]\s*$", re.I)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_budgeted_parent_character_voice_repair_ENTITY = re.compile(r"[mg]\.[A-Za-z0-9_]+\Z")
_budgeted_parent_character_voice_repair_VAR = re.compile(r"V\d+\Z")
_WH_FIELD = re.compile(r"\b(?:what|which)\s+(?:is|was|are|were)\s+(?:the\s+)?"
                       r"([A-Za-z ]+?)\s+of\s+(?:the\s+)?([A-Za-z ]+?)(?=[?.!]|$)", re.I)


def _budgeted_parent_character_voice_repair_phrase(value: str) -> str:
    return " ".join(_budgeted_parent_character_voice_repair_tokens(str(value).replace("_", " ")))


def _context_has_type(context: _budgeted_parent_character_voice_repair_Mapping[str, Any], type_id: str) -> bool:
    values = context.get("types")
    return (isinstance(values, list) and _budgeted_parent_character_voice_repair_concrete_type(type_id)
            and _budgeted_parent_character_voice_repair_phrase(type_id.rsplit(".", 1)[-1]) in {_budgeted_parent_character_voice_repair_phrase(str(value)) for value in values})


def compile_parent_character_voice_repair(
    question: str, source: Any, ontology: Any, semantic_graphs: Any, *,
    linked_entity_context: Any = None, decompositions: Any = None,
    question_anchor_contract: _budgeted_parent_character_voice_repair_Mapping[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Compile a current parent/character/typed performance condition, or abstain.

    Scalar answer contradiction is necessary, so ordinary entity answers are
    untouched. Original resolved IDs are reused, not relinked; the actor name
    must agree with this question and context. The child remains the same
    already resolved original Semantic/source anchor, including supplied mention
    spelling variants; no fuzzy lookup or new entity ID is generated.
    """
    graph = source.graph
    evidence = {"status": "not_compiled", "source_graph_id": str(graph.graph_id),
                "additional_model_calls": 0, "additional_endpoint_queries": 0,
                "additional_embedding_calls": 0, "uses_gold_answers": False,
                "entity_relation_head_or_sample_whitelist": False,
                "requires_saved_request_and_remaining_measured_time": True,
                "requires_post_model_execution": True, "compiler_selects_answers": False,
                "preserves_all_source_bgp_and_operators": True,
                "does_not_relink_or_generate_entity_ids": True}

    def abstain(reason):
        evidence["reason"] = reason
        return None, evidence

    try:
        match = _QUESTION.fullmatch(str(question))
        if not match:
            evidence["unsupported_question_head_or_modifier"] = True
            return abstain("not_closed_parent_name_and_performance_question")
        child_mention, kinship, actor_mention, performance_cue = match.groups()
        if re.search(r"\d", child_mention + actor_mention):
            return abstain("unsupported_numeric_named_clause")
        if not _budgeted_parent_character_voice_repair_complete(source) or not source.answer_ids:
            return abstain("actual_source_complete_denotation_missing")
        normalized = [_budgeted_parent_character_voice_repair_canonicalize_answer_id(value) for value in source.answer_ids]
        if any(_budgeted_parent_character_voice_repair_ENTITY.fullmatch(value) or not _DATE.fullmatch(value) for value in normalized):
            return abstain("source_not_pure_calendar_date_role_contradiction")
        for value in normalized:
            date.fromisoformat(value)
        incoming = [str(relation) for _, relation, object_ in graph.triples if str(object_) == str(graph.answer_var)]
        if (not incoming or any(str(ontology.range_for_relation(relation) or "") != "type.datetime" for relation in incoming)
                or any(str(subject) == str(graph.answer_var) for subject, _, _ in graph.triples)):
            return abstain("source_answer_schema_not_known_date_only")
        provenance = getattr(graph, "provenance", None)
        if (not isinstance(provenance, _budgeted_parent_character_voice_repair_Mapping) or provenance.get("selection_answer_union")
                or str(provenance.get("original_question", "")).strip() != str(question).strip()):
            return abstain("source_question_owned_graph_or_union_scope_unproven")
        anchors = _budgeted_parent_character_voice_repair_anchors(graph)
        if len(anchors) != 2 or not all(_budgeted_parent_character_voice_repair_ENTITY.fullmatch(value) for value in anchors):
            return abstain("source_not_exact_actor_and_child_entity_anchors")
        bindings = provenance.get("anchor_bindings")
        if not isinstance(bindings, _budgeted_parent_character_voice_repair_Mapping) or not isinstance(linked_entity_context, (list, tuple)):
            return abstain("current_anchor_bindings_or_typed_context_missing")
        contexts = {}
        for item in linked_entity_context:
            if not isinstance(item, _budgeted_parent_character_voice_repair_Mapping):
                continue
            entity_id = _budgeted_parent_character_voice_repair_canonicalize_answer_id(item.get("id", ""))
            contexts.setdefault(entity_id, []).append(item)
        actor_ids = {entity_id for entity_id in anchors for item in contexts.get(entity_id, [])
                     if " ".join(str(item.get("name", "")).casefold().split()) == " ".join(actor_mention.casefold().split())}
        if len(actor_ids) != 1:
            return abstain("question_actor_name_not_uniquely_current_resolved_anchor")
        actor_id = next(iter(actor_ids))
        child_id = next(iter(anchors - {actor_id}))
        if len(contexts.get(actor_id, [])) != 1 or len(contexts.get(child_id, [])) != 1:
            return abstain("current_actor_or_child_context_ambiguous")
        child_name = str(contexts[child_id][0].get("name", "")).strip()
        child_mention_exact = " ".join(child_mention.casefold().split()) == " ".join(child_name.casefold().split())
        if not child_mention_exact:
            contract = question_anchor_contract
            if contract is None:
                contract = getattr(source, "question_anchor_contract", None)
            if contract is None:
                contract = provenance.get("question_anchor_contract")
            provided = contract.get("provided_topic_entities") if isinstance(contract, _budgeted_parent_character_voice_repair_Mapping) else None
            if (not isinstance(contract, _budgeted_parent_character_voice_repair_Mapping) or str(contract.get("question", "")) != str(question)
                    or not isinstance(provided, _budgeted_parent_character_voice_repair_Mapping)
                    or str(provided.get(child_id, "")).strip().casefold() != child_name.casefold()
                    or str(provided.get(actor_id, "")).strip().casefold() != str(contexts[actor_id][0].get("name", "")).strip().casefold()):
                return abstain("nonliteral_child_mention_requires_current_question_provided_linking_contract")
            evidence["uses_current_question_provided_topic_contract"] = True
        else:
            evidence["uses_current_question_provided_topic_contract"] = False
        for operator in graph.operators:
            if (not isinstance(operator, _budgeted_parent_character_voice_repair_Mapping) or str(operator.get("type", "")).upper() != "NO_EQUAL"
                    or str(operator.get("input_var", "")) != str(graph.answer_var)
                    or operator.get("inputs") or operator.get("attribute_relation_label")
                    or operator.get("attribute_relation_labels") or operator.get("attribute_relation_id")
                    or operator.get("value_type") not in {"mid", "entity"}
                    or _budgeted_parent_character_voice_repair_canonicalize_answer_id(operator.get("_value_entity_id") or operator.get("value")) not in anchors):
                return abstain("source_other_operator_obligation_not_supported_by_closed_question")
        semantics = list(_budgeted_parent_character_voice_repair_semantic_graphs(semantic_graphs))
        actor_witnesses, parent_witnesses = [], []
        for semantic in semantics:
            for path in semantic["semantic_paths"]:
                if not isinstance(path, _budgeted_parent_character_voice_repair_Mapping):
                    continue
                steps = path.get("steps")
                if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], _budgeted_parent_character_voice_repair_Mapping):
                    continue
                step = steps[0]
                if step.get("direction") != "backward" or str(step.get("from")) != str(path.get("anchor_ref")):
                    continue
                binding = bindings.get(str(path.get("anchor_ref")))
                if not isinstance(binding, _budgeted_parent_character_voice_repair_Mapping):
                    continue
                bound = _budgeted_parent_character_voice_repair_canonicalize_answer_id(binding.get("id", ""))
                relation = _budgeted_parent_character_voice_repair_relation_id_from_label(step.get("relation_label"))
                domain, range_ = str(ontology.domain_for_relation(relation) or ""), str(ontology.range_for_relation(relation) or "")
                if bound == actor_id and _context_has_type(contexts[actor_id][0], range_) and _budgeted_parent_character_voice_repair_concrete_type(domain):
                    nodes = _budgeted_parent_character_voice_repair_semantic_path_nodes(source, semantic, path, linked_entity_context)
                    if nodes is not None and len(nodes) == 1:
                        node = next(iter(nodes))
                        if [node, relation, actor_id] in [list(map(str, row)) for row in graph.triples]:
                            actor_witnesses.append((node, domain, relation))
                if bound != child_id or not _context_has_type(contexts[child_id][0], range_) or not _budgeted_parent_character_voice_repair_concrete_type(domain):
                    continue
                surfaces = {str(item.get("surface", "")).strip().casefold() for item in semantic["anchors"]
                            if isinstance(item, _budgeted_parent_character_voice_repair_Mapping) and str(item.get("id")) == str(path.get("anchor_ref"))}
                if surfaces != {child_name.casefold()} or str(binding.get("label", "")).strip().casefold() != child_name.casefold():
                    continue
                for inverse in getattr(ontology, "reverse_relation_ids", {}).get(relation, ()):
                    if (relation not in getattr(ontology, "reverse_relation_ids", {}).get(inverse, ())
                            or _budgeted_parent_character_voice_repair_phrase(str(inverse).rsplit(".", 1)[-1]) != _budgeted_parent_character_voice_repair_phrase(kinship)
                            or ontology.domain_for_relation(inverse) != range_
                            or ontology.range_for_relation(inverse) != domain):
                        continue
                    parent_witnesses.append((str(inverse), domain, range_, relation))
        actor_unique, parent_unique = set(actor_witnesses), set(parent_witnesses)
        evidence.update(actor_semantic_witness_count=len(actor_unique), official_parent_inverse_witness_count=len(parent_unique))
        if len(actor_unique) != 1 or len(parent_unique) != 1:
            return abstain("original_actor_variable_or_official_parent_inverse_not_unique")
        actor_node, actor_cvt_type, actor_relation = next(iter(actor_unique))
        parent_relation, parent_type, child_type, inverse_relation = next(iter(parent_unique))
        declared = {(_budgeted_parent_character_voice_repair_phrase(property_), _budgeted_parent_character_voice_repair_phrase(owner_)) for line in _decomposition_lines(
            decompositions if decompositions is not None else provenance.get("decomposition"))
            for property_, owner_ in _WH_FIELD.findall(line)}
        character_fields = []
        for relation in ontology.relations_for_domain(actor_cvt_type):
            range_ = str(ontology.range_for_relation(relation) or "")
            if (ontology.domain_for_relation(relation) == actor_cvt_type and _budgeted_parent_character_voice_repair_concrete_type(range_)
                    and parent_type in ontology.supertypes(range_)
                    and (_budgeted_parent_character_voice_repair_phrase(str(relation).rsplit(".", 1)[-1]), _budgeted_parent_character_voice_repair_phrase(actor_cvt_type.rsplit(".", 1)[-1])) in declared):
                character_fields.append(str(relation))
        if len(character_fields) != 1:
            return abstain("declared_same_actor_cvt_parent_character_field_not_unique")
        character_field = character_fields[0]
        voice_candidates = []
        for entity_id, items in contexts.items():
            if entity_id in anchors or len(items) != 1 or not _budgeted_parent_character_voice_repair_ENTITY.fullmatch(entity_id):
                continue
            item = items[0]
            name_words = re.findall(r"[A-Za-z]+", str(item.get("name", "")))
            if len(name_words) != 1 or not (_budgeted_parent_character_voice_repair_finite_forms(name_words[0]) & _budgeted_parent_character_voice_repair_finite_forms(performance_cue)):
                continue
            for relation in ontology.relations_for_domain(actor_cvt_type):
                if (ontology.domain_for_relation(relation) == actor_cvt_type
                        and _context_has_type(item, str(ontology.range_for_relation(relation) or ""))):
                    voice_candidates.append((str(relation), entity_id))
        voice_unique = set(voice_candidates)
        if len(voice_unique) != 1:
            return abstain("current_performance_value_lemma_and_typed_field_not_unique")
        voice_field, voice_id = next(iter(voice_unique))
        if any(str(subject) == actor_node and str(relation) in {character_field, voice_field}
               for subject, relation, _ in graph.triples):
            return abstain("existing_actor_cvt_character_or_performance_binding_requires_separate_mapping")
        variables = {str(node) for row in graph.triples for node in (row[0], row[2]) if str(node) not in anchors}
        if not all(_budgeted_parent_character_voice_repair_VAR.fullmatch(value) for value in variables):
            return abstain("source_global_variable_contract_unproven")
        new_var = "V" + str(max(int(value[1:]) for value in variables) + 1)
        additions = [[actor_node, character_field, new_var], [child_id, parent_relation, new_var],
                     [actor_node, voice_field, voice_id]]
        repaired = deepcopy(graph)
        repaired.graph_id = str(graph.graph_id) + "__typed_parent_voice"
        repaired.answer_var = new_var
        repaired.triples = [*deepcopy(graph.triples), *additions]
        repaired.sparql = ""
        repaired.provenance = deepcopy(dict(provenance))
        evidence.update(status="compiled", reason="date_contradiction_with_original_parent_actor_and_typed_performance_obligations",
                        actor_entity_id=actor_id, child_entity_id=child_id, original_child_mention=child_mention,
                        reuses_original_resolved_child_without_fuzzy_lookup=True,
                        source_date_answer_var=str(graph.answer_var), target_answer_var=new_var,
                        actual_actor_cvt_variable=actor_node, actual_actor_cvt_type=actor_cvt_type,
                        original_actor_relation=actor_relation, character_relation=character_field,
                        parent_relation=parent_relation, original_inverse_relation=inverse_relation,
                        performance_relation=voice_field, current_performance_value_id=voice_id,
                        added_triples=additions, retained_source_triple_count=len(graph.triples),
                        retained_source_operator_count=len(graph.operators))
        repaired.provenance["budgeted_parent_character_voice_repair"] = deepcopy(evidence)
        return repaired, evidence
    except Exception as exc:
        evidence.update(status="unqualified", reason="current_parent_voice_schema_or_context_error",
                        error=f"{type(exc).__name__}: {exc}")
        return None, evidence



__all__ = ['PropertyRankEvidence', '_compose_scalar_scope_safety_original_semantic_scope_certificate', 'answer_set_safety_reason', 'compile_original_scalar_path_restore', 'compile_parent_character_voice_repair', 'compile_primary_projection_repair', 'compile_unrecognized_intersection_from_current_stages', 'compile_unrecognized_intersection_scalar', 'constraint_execution_challenger', 'constraint_focus_phrase', 'content_tokens', 'current_primary_grounded_constraint_witness', 'intersection_scalar_obligation_necessary', 'intersection_scalar_question_necessary', 'lower_unrecognized_intersection_scalar', 'nested_label_parent_ids', 'property_family', 'property_rank_evidence', 'relation_tail_text', 'requested_family', 'resolve_unrecognized_intersection_source', 'semantic_only_is_confident']
