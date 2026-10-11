"""Consolidated normalization; original rules and budgets retained."""
from __future__ import annotations


# ---- comparison_number_normalization ----
"""Strict English comparison literals for internal, budgeted local guards.

This module does not rewrite questions/prompts, mutate operators, create query
graphs or select answers.  It only returns a literal spec.  Callers still must
prove the attribute datatype, owner, comparator, complete executed denotation
and all original joins.  A spelling guess is never a normalized literal.

Native specs can be preserved by ``resolve_internal_constraint_spec``.  In
particular this is not a replacement for the pre-model/global parser.
"""


import calendar
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
import re
from typing import Any


_SMALL = dict(zip(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
     "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"),
    range(20)))
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9, "trillion": 10 ** 12}
_ORDINAL = dict(zip(
    ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
     "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth"),
    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
     "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")))
_ORDINAL.update({"twentieth": "twenty", "thirtieth": "thirty", "fortieth": "forty", "fiftieth": "fifty",
                 "sixtieth": "sixty", "seventieth": "seventy", "eightieth": "eighty", "ninetieth": "ninety",
                 "hundredth": "hundred", "thousandth": "thousand", "millionth": "million",
                 "billionth": "billion", "trillionth": "trillion"})
_NUMBER_WORDS = frozenset(_SMALL) | frozenset(_TENS) | frozenset(_SCALES) | {
    "hundred", "and", "minus", "negative", "plus", "point", "a", "an"}
_BOUNDARY = frozenset({"who", "whose", "which", "that", "where", "with", "without", "from", "in", "at",
                       "for", "is", "are", "was", "were", "has", "have", "had", "did", "does", "do", "having"})
_COMPARISON = {
    "no more than": "LESS_OR_EQUAL", "no less than": "GREATER_OR_EQUAL",
    "at most": "LESS_OR_EQUAL", "at least": "GREATER_OR_EQUAL",
    "less than": "LESS_THAN", "fewer than": "LESS_THAN", "lower than": "LESS_THAN",
    "smaller than": "LESS_THAN", "greater than": "GREATER_THAN", "more than": "GREATER_THAN",
    "higher than": "GREATER_THAN", "larger than": "GREATER_THAN",
    "equal to": "EQUAL", "exactly": "EQUAL", "below": "LESS_THAN", "under": "LESS_THAN",
    "above": "GREATER_THAN", "over": "GREATER_THAN", "before": "LESS_THAN", "after": "GREATER_THAN",
    "prior to": "LESS_THAN", "earlier than": "LESS_THAN", "later than": "GREATER_THAN", "on": "EQUAL",
}
_TEMPORAL_CUES = frozenset({"before", "after", "prior to", "earlier than", "later than", "on"})
_CUE = re.compile(r"\b(" + "|".join(re.escape(x) for x in sorted(_COMPARISON, key=len, reverse=True)) + r")\b", re.I)
_WORD = re.compile(r"[A-Za-z]+(?:[-\u2010\u2011][A-Za-z]+)*")
_DIGIT = re.compile(r"(?<!\w)[+-]?\d+(?:[.,/-]\d+)*(?!\w)")
_MONTHS = {name.casefold(): month for month in range(1, 13)
           for name in (calendar.month_name[month], calendar.month_abbr[month])}
_MONTHS["sept"] = 9
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))


def _tokens(text: str) -> list[str] | None:
    value = str(text).strip().casefold()
    if not re.fullmatch(r"[a-z]+(?:[-\u2010\u2011\s][a-z]+)*", value):
        return None
    return re.split(r"[-\u2010\u2011\s]+", value)


def _under_hundred(tokens: list[str]) -> int | None:
    if len(tokens) == 1:
        return _SMALL.get(tokens[0], _TENS.get(tokens[0]))
    if len(tokens) == 2 and tokens[0] in _TENS and 1 <= _SMALL.get(tokens[1], 0) <= 9:
        return _TENS[tokens[0]] + _SMALL[tokens[1]]
    return None


def _under_thousand(tokens: list[str]) -> int | None:
    if "hundred" not in tokens:
        return _under_hundred(tokens)
    if (len(tokens) < 2 or tokens[1] != "hundred" or tokens.count("hundred") != 1
            or not 1 <= _SMALL.get(tokens[0], 0) <= 9):
        return None
    remainder = tokens[2:]
    if remainder[:1] == ["and"]:
        remainder = remainder[1:]
        if not remainder:
            return None
    tail = _under_hundred(remainder) if remainder else 0
    return None if tail is None or (remainder and tail == 0) else _SMALL[tokens[0]] * 100 + tail


def _cardinal(tokens: list[str]) -> int | None:
    if not tokens:
        return None
    if tokens[0] in {"a", "an"}:
        if len(tokens) < 2 or tokens[1] not in {"hundred", *_SCALES}:
            return None
        tokens = ["one", *tokens[1:]]
    total, previous_scale, start = 0, 10 ** 15, 0
    for index, token in enumerate(tokens):
        if token not in _SCALES:
            continue
        scale = _SCALES[token]
        chunk = tokens[start:index]
        if chunk[:1] == ["and"] and total:
            chunk = chunk[1:]
        coefficient = _under_thousand(chunk)
        if scale >= previous_scale or coefficient is None or coefficient <= 0:
            return None
        total += coefficient * scale
        previous_scale, start = scale, index + 1
    remainder = tokens[start:]
    if remainder[:1] == ["and"] and total:
        remainder = remainder[1:]
        if not remainder:
            return None
    tail = _under_thousand(remainder) if remainder else 0
    if tail is None or (remainder and tail == 0 and total):
        return None
    return total + tail


def parse_english_number(text: str, *, allow_ordinal: bool = False) -> str | None:
    """Parse a complete, grammatical cardinal; ordinals require opt-in.

    No substring recovery, typo correction, arithmetic/range interpretation,
    mixed digit/word value, vague quantifier or entity-name lookup is allowed.
    """
    tokens = _tokens(text)
    if not tokens:
        return None
    if any(token in _ORDINAL for token in tokens):
        if not allow_ordinal or tokens[-1] not in _ORDINAL or any(x in _ORDINAL for x in tokens[:-1]):
            return None
        tokens = [*tokens[:-1], _ORDINAL[tokens[-1]]]
    negative = tokens[0] in {"minus", "negative"}
    if tokens[0] in {"minus", "negative", "plus"}:
        tokens = tokens[1:]
    if not tokens or any(x not in _NUMBER_WORDS for x in tokens):
        return None
    if tokens.count("point") > 1:
        return None
    if "point" in tokens:
        point = tokens.index("point")
        integer = _cardinal(tokens[:point]) if point else 0
        digits = tokens[point + 1:]
        if integer is None or not digits or any(x not in _SMALL or _SMALL[x] > 9 for x in digits):
            return None
        value = str(integer) + "." + "".join(str(_SMALL[x]) for x in digits)
    else:
        integer = _cardinal(tokens)
        if integer is None:
            return None
        value = str(integer)
    return ("-" if negative and Decimal(value) != 0 else "") + value


def _day(text: str) -> int | None:
    text = re.sub(r"^the\s+", "", text.strip(), flags=re.I)
    match = re.fullmatch(r"(\d{1,2})(st|nd|rd|th)?", text, re.I)
    if match:
        day = int(match[1])
        suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        if match[2] and match[2].casefold() != suffix:
            return None
    else:
        value = parse_english_number(text, allow_ordinal=True)
        if value is None or not value.isdigit():
            return None
        day = int(value)
    return day if 1 <= day <= 31 else None


def _year(text: str) -> int | None:
    text = text.strip()
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
    else:
        # Standard century/year reading is distinct from cardinal grammar:
        # eighteen seventy-five -> 1875, not 18 + 75.
        words = _tokens(text)
        if not words:
            return None
        century = _under_hundred(words[:1])
        tail = _under_hundred(words[1:])
        if century is not None and 10 <= century <= 99 and tail is not None and 10 <= tail <= 99:
            year = century * 100 + tail
        elif len(words) == 2 and words[1] == "hundred" and century is not None and 10 <= century <= 99:
            year = century * 100
        else:
            value = parse_english_number(text)
            if value is None or not value.isdigit():
                return None
            year = int(value)
    return year if 1000 <= year <= 9999 else None


def parse_english_calendar_date(text: str) -> str | None:
    """Parse one explicit month/day/year without guessing slash-date order."""
    value = str(text).strip()
    day_first = re.fullmatch(rf"(?P<day>.+?)\s+(?:of\s+)?(?P<month>{_MONTH_RE})\s*,?\s+(?P<year>.+)", value, re.I)
    month_first = re.fullmatch(rf"(?P<month>{_MONTH_RE})\s+(?P<day>.+?)\s*,\s*(?P<year>.+)", value, re.I)
    match = day_first or month_first
    if match is None:
        # Without a comma, numeric/ordinal day has an unambiguous boundary.
        match = re.fullmatch(rf"(?P<month>{_MONTH_RE})\s+(?P<day>\d{{1,2}}(?:st|nd|rd|th)?)\s+(?P<year>.+)", value, re.I)
    if match is None:
        match = re.fullmatch(rf"(?P<month>{_MONTH_RE})\s+(?P<day>.+?)\s+(?P<year>\d{{4}})", value, re.I)
    if match is None:
        return None
    day, year = _day(match["day"]), _year(match["year"])
    if day is None or year is None:
        return None
    try:
        return date(year, _MONTHS[match["month"].casefold()], day).isoformat()
    except ValueError:
        return None


def _unquoted(question: str) -> str | None:
    # Use spaces to keep source offsets exact; paired Unicode quotes are not
    # interpreted as apostrophes in contractions or personal names.
    if question.count('"') % 2 or question.count("“") != question.count("”"):
        return None
    text = re.sub(r'"[^\"]*"|“[^”]*”', lambda m: " " * len(m[0]), question)
    return text


def _literal_segment(tail: str) -> tuple[str, int] | None:
    value = tail.lstrip()
    offset = len(tail) - len(value)
    # Calendar commas are retained; terminal punctuation alone ends a value.
    terminator = re.search(r"[?!;:]|\.(?:\s|$)", value)
    if terminator:
        value = value[:terminator.start()]
    tokens = list(_WORD.finditer(value))
    if not tokens:
        return None
    for match in tokens[1:]:
        if match[0].casefold() in _BOUNDARY:
            value = value[:match.start()]
            break
    value = value.strip()
    if not value:
        return None
    return value, offset + len(value)


def normalize_comparison_constraint_spec(question: str) -> dict[str, Any] | None:
    """Return one internal word-normalized comparison spec, otherwise None.

    Numeric digits alone deliberately abstain: the existing parser remains
    authoritative for its previously supported forms. The supplied question
    itself is never rewritten and no caller is silently redirected to here.
    """
    question = str(question)
    text = _unquoted(question)
    if text is None:
        return None
    cues = list(_CUE.finditer(text))
    if len(cues) != 1:
        return None
    cue = cues[0]
    remainder_without_cue = text[:cue.start()] + text[cue.end():]
    if re.search(r"\b(?:not|never|or|either|except|approximately|about|nearly|roughly|between)\b", remainder_without_cue, re.I):
        return None
    segment = _literal_segment(text[cue.end():])
    if segment is None:
        return None
    literal, consumed = segment
    leading = len(text[cue.end():]) - len(text[cue.end():].lstrip())
    literal_start = cue.end() + leading
    literal_end = cue.end() + consumed
    outside = text[:literal_start] + " " * len(literal) + text[literal_end:]
    if _DIGIT.search(outside) or any(part.casefold() in _SMALL | _TENS | _SCALES | _ORDINAL
                                   for match in _WORD.finditer(outside)
                                   for part in re.split(r"[-\u2010\u2011]", match[0])):
        return None
    cue_text = " ".join(cue[0].casefold().split())
    # A known calendar month is the only cue-independent date signal used.
    temporal = cue_text in _TEMPORAL_CUES or any(w.casefold() in _MONTHS for w in re.findall(r"[A-Za-z]+", literal))
    if temporal:
        value = parse_english_calendar_date(literal)
        precision = "day"
        if value is None and cue_text != "on":
            year = _year(literal)
            value = str(year) if year is not None else None
            precision = "year"
        if value is None or not re.search(r"[A-Za-z]", literal):
            return None
        datatype, kind = "dateTime", "english_calendar_words"
    else:
        value = parse_english_number(literal)
        if value is None:
            return None
        datatype, kind, precision = "number", "english_cardinal_words", ""
    result: dict[str, Any] = {
        "type": _COMPARISON[cue_text], "value": value, "value_type": datatype,
        "focus": question[max(0, cue.start() - 80): min(len(question), literal_end + 80)],
        "_normalization_kind": kind, "_source_literal": question[literal_start:literal_end],
        "_source_comparison_cue": question[cue.start():cue.end()],
        "_source_literal_span": [literal_start, literal_end],
        "_normalization_scope": "internal_post_model_constraint_spec_only",
        "additional_model_calls": 0, "additional_endpoint_queries": 0, "additional_embedding_calls": 0,
        "uses_gold_answers": False, "uses_entity_or_question_whitelist": False,
    }
    if precision:
        result["_comparison_date_precision"] = precision
    return result


def resolve_internal_constraint_spec(
    question: str, native_spec: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Preserve any existing native spec; normalize only an unsupported form."""
    return native_spec if native_spec is not None else normalize_comparison_constraint_spec(question)



# ---- numeric_scalar_safety ----
"""Resolve numerical comparisons onto scalar leaves using local schema only.

This guard is used by model-free challenger generation, not by the ordinary
query lowering path.  It does not inspect answers, questions, or dataset IDs,
and does not issue endpoint/model requests.  Missing schema is left unchanged;
a proven entity-valued numerical target is never cast as though it were a
number.  Anonymous value records can be repaired only when their schema has
exactly one numerical leaf.
"""



from copy import deepcopy
from typing import Any

from .ontology import relation_id_from_label


_NUMERIC_TYPES = frozenset({"type.int", "type.float"})


def _attribute_relation(operator: dict[str, Any]) -> str:
    direct = str(operator.get("attribute_relation_id", ""))
    if direct:
        return direct
    label = operator.get("attribute_relation_label", [])
    if not label:
        label = operator.get("attribute_relation_labels", [])
    return relation_id_from_label(label) if label else ""


def _numeric_cast(operator: dict[str, Any]) -> bool:
    kind = str(operator.get("type", "")).upper()
    if kind in {"LESS_THAN", "LESS_OR_EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL"}:
        return bool(operator.get("_numeric_comparison_cast"))
    if kind == "EQUAL":
        return bool(operator.get("_numeric_cast_compare"))
    return False


def _node_types(node: str, triples: list[list[str]], ontology: Any) -> set[str]:
    types: set[str] = set()
    for subject, relation, object_ in triples:
        if str(object_) == node:
            value = str(ontology.range_for_relation(str(relation)))
            if value:
                types.add(value)
        if str(subject) == node:
            value = str(ontology.domain_for_relation(str(relation)))
            if value:
                types.add(value)
    return types


def _unique_numeric_leaf(ontology: Any, type_id: str) -> str:
    # Freebase named entities inherit common.topic.  They are not value
    # records, and their numeric attributes must not be chosen implicitly.
    explicit = getattr(ontology, "is_cvt_type", None)
    if callable(explicit):
        if not explicit(type_id):
            return ""
    elif "common.topic" in set(ontology.supertypes(type_id)):
        return ""
    properties = {
        str(relation)
        for relation in ontology.relations_for_domain(type_id)
        if str(ontology.range_for_relation(str(relation))) in _NUMERIC_TYPES
    }
    return next(iter(properties)) if len(properties) == 1 else ""


def schema_safe_numeric_operators(
    *,
    operators: list[dict[str, Any]],
    triples: list[list[str]],
    answer_var: str,
    ontology: Any,
) -> tuple[list[dict[str, Any]], list[list[str]], list[dict[str, Any]]] | None:
    """Return scalar-safe copies, or abstain from a proven unsafe candidate.

    Only numeric-cast operators are inspected.  Existing scalar/text casts and
    targets without local type evidence remain byte-for-byte unchanged.  A
    known nonprimitive target requires a unique numeric leaf of an anonymous
    value-record type; incompatible types or ambiguous leaves cause abstention.
    """

    repaired = deepcopy(operators)
    graph_triples = deepcopy(triples)
    audit: list[dict[str, Any]] = []
    if ontology is None:
        return repaired, graph_triples, audit
    # Multiple constraints can target records of the same type.  Keep this
    # invocation-local cache so a schema closure is computed only once, without
    # retaining stale decisions if the ontology is replaced or updated.
    leaf_by_type: dict[str, str] = {}
    for operator in repaired:
        if not _numeric_cast(operator):
            continue
        input_var = str(operator.get("input_var") or answer_var)
        attribute = _attribute_relation(operator)
        if attribute:
            range_id = str(ontology.range_for_relation(attribute))
            types = {range_id} if range_id else set()
        else:
            types = _node_types(input_var, graph_triples, ontology)
        # Preserve unknown schemas and primitive values (including numeric
        # strings). The targeted bug is casting a proven entity/record URI.
        entity_types = {value for value in types if not value.startswith("type.")}
        if not entity_types:
            continue
        if len(types) != 1:
            return None
        record_type = next(iter(entity_types))
        if record_type not in leaf_by_type:
            leaf_by_type[record_type] = _unique_numeric_leaf(ontology, record_type)
        leaf = leaf_by_type[record_type]
        if not leaf:
            return None
        record_var = input_var
        if attribute:
            existing = sorted({
                str(object_)
                for subject, relation, object_ in graph_triples
                if str(subject) == input_var and str(relation) == attribute
                and str(object_).startswith("V")
            })
            if len(existing) > 1:
                return None
            if existing:
                record_var = existing[0]
            else:
                used = {str(value) for triple in graph_triples for value in (triple[0], triple[2])}
                counter = 0
                while f"VNumericRecord{counter}" in used:
                    counter += 1
                record_var = f"VNumericRecord{counter}"
                graph_triples.append([input_var, attribute, record_var])
        operator["input_var"] = record_var
        operator["attribute_relation_id"] = leaf
        numeric_type = str(ontology.range_for_relation(leaf))
        # The threshold's lexical form does not determine the property's
        # datatype: a float record compared with integer 15 must preserve 15.7,
        # rather than silently truncating it to 15.  Set this only after a
        # record-to-scalar repair; existing scalar paths are left untouched.
        operator["_numeric_cast_kind"] = (
            "float" if numeric_type == "type.float" else "integer"
        )
        audit.append({
            "record_type": record_type,
            "record_var": record_var,
            "numeric_leaf": leaf,
            "numeric_type": numeric_type,
            "numeric_cast_kind": operator["_numeric_cast_kind"],
            "additional_model_calls": 0,
            "additional_endpoint_requests": 0,
        })
    return repaired, graph_triples, audit



__all__ = ['normalize_comparison_constraint_spec', 'parse_english_calendar_date', 'parse_english_number', 'resolve_internal_constraint_spec', 'schema_safe_numeric_operators']
