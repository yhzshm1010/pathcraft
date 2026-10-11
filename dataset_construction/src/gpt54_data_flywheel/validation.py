from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import html
import json
from itertools import product
import re
import shlex
import subprocess
from typing import Any
from difflib import SequenceMatcher

from .schemas import EXECUTABLE_OPERATOR_TYPES, OPERATOR_TYPES


_ANCHOR_RE = re.compile(r"^A\d+$")
_PATH_RE = re.compile(r"^P\d+$")
_PATH_VAR_RE = re.compile(r"^P\d+\.V\d+$")
_PROGRAM_VAR_RE = re.compile(r"^V\d+$")
_ENTITY_RE = re.compile(r"^E\d+$")
_DOTTED_ID_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+){2,}$")
_RDF_LITERAL_RE = re.compile(
    r'^(?P<quoted>"(?:[^"\\]|\\.)*")'
    r'(?:@[A-Za-z0-9-]+|\^\^(?:[A-Za-z][\w-]*:[\w.-]+|<[^>]+>))?$'
)
_ATTRIBUTE_OPERATORS = {
    "ARGMAX",
    "ARGMIN",
    "TC",
    "EQUAL",
}
_VALUE_OPERATORS = {
    "TC",
    "GREATER_THAN",
    "GREATER_OR_EQUAL",
    "LESS_THAN",
    "LESS_OR_EQUAL",
    "EQUAL",
    "NO_EQUAL",
}
_OPTIONAL_SOURCE_OPERATORS = {"EQUAL", "NO_EQUAL"}


_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>{}\[\]]+", re.IGNORECASE)
_FREEBASE_URI_RE = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://)?(?:www\.|rdf\.)?freebase\.com(?:/|\b)",
    re.IGNORECASE,
)
_FREEBASE_MID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:m|g)\.[A-Za-z0-9_]+(?![A-Za-z0-9_])"
)
_FREEBASE_SLASH_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:m|g)/[A-Za-z0-9_]+(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_FREEBASE_NAMESPACE_RE = re.compile(r"(?<![A-Za-z0-9_])ns\s*:", re.IGNORECASE)
_SPARQL_VARIABLE_RE = re.compile(r"(?<![A-Za-z0-9_])[?$][A-Za-z_][A-Za-z0-9_]*\b")
_PROGRAM_ARROW_RE = re.compile(r"(?:<?[-=]+>|<[-=]+|[\u2190-\u21ff])")
_LOWER_DOTTED_RELATION_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[a-z][a-z0-9_]{1,}\.){2,}[a-z][a-z0-9_]{1,}"
    r"(?![A-Za-z0-9_])"
)
_BARE_DOMAIN_RE = re.compile(
    r"\b(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\."
    r"(?:com|org|net|edu|gov|io|ai|co|uk|cn|de|fr|jp|ru|info|biz)\b",
    re.IGNORECASE,
)
_PROGRAM_REFERENCE_RE = re.compile(
    r"(?:\bJOIN\b(?=\s+(?:P\d+\.V\d+|[AEV]\d+|\())|"
    r"\b(?:P\d+\.V\d+|P\d+\.S\d+|[AEV]\d+)\b|"
    r"\b(?:JOIN|ARGMAX|ARGMIN|COUNT|AND)\s*\(|"
    r"\(\s*(?:JOIN|R|AND|COUNT|ARGMAX|ARGMIN)\b)",
    re.IGNORECASE,
)
_PROGRAM_DELIMITER_RE = re.compile(r"[{}]")
_SPARQL_SYNTAX_RE = re.compile(
    r"(?:"
    r"\b(?:select|ask|construct|describe)\b\s+"
    r"(?:(?:distinct|reduced)\b\s*)*(?:[?$*{])"
    r"|\bwhere\b\s*\{"
    r"|\bfilter\b\s*(?:\(|exists\b|not\s+exists\b)"
    r"|\b(?:optional|minus|service|graph)\b\s*\{"
    r"|\bunion\b\s*\{"
    r"|\bvalues\b\s*(?:[?$]|\{)"
    r"|\bbind\b\s*\("
    r"|\b(?:prefix\b\s*(?:[A-Za-z][\w-]*)?\s*:|base\b\s*<)"
    r"|\b(?:order|group)\s+by\b"
    r"|\bhaving\b\s*\("
    r"|\b(?:limit|offset)\b\s+\d+"
    r")",
    re.IGNORECASE,
)
_NATURAL_WORD_RE = re.compile(r"[A-Za-z]+(?:[.'\u2019-][A-Za-z]+)*")
_EXPLICIT_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"\d(?:[\d,\uff0c]*\d)?(?:\.\d+)?(?:st|nd|rd|th)?%?"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)

# These are semantic constraint families, not exact-phrase requirements.  A
# generated subquestion may use a faithful synonym (for example, "above" for
# "greater than") without being rejected.
_CONSTRAINT_CUE_GROUPS: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    (
        "greater-than",
        re.compile(
            r"\b(?:greater|more|higher|larger|older|later)\s+than\b|"
            r"\b(?:above|over)\b(?=\s+(?:[\$\u00a3\u20ac]?\d|the\b))",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:greater|more|higher|larger|older|later)\s+than\b|"
            r"\b(?:above|over|exceed(?:s|ed|ing)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "less-than",
        re.compile(
            r"\b(?:less|fewer|lower|smaller|younger|earlier)\s+than\b|"
            r"\b(?:below|under)\b(?=\s+(?:[\$\u00a3\u20ac]?\d|the\b))",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:less|fewer|lower|smaller|younger|earlier)\s+than\b|"
            r"\b(?:below|under)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "at-least",
        re.compile(r"\b(?:at\s+least|no\s+less\s+than|not\s+less\s+than)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:at\s+least|no\s+less\s+than|not\s+less\s+than|"
            r"minimum(?:\s+of)?|or\s+more)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "at-most",
        re.compile(r"\b(?:at\s+most|no\s+more\s+than|not\s+more\s+than)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:at\s+most|no\s+more\s+than|not\s+more\s+than|"
            r"maximum(?:\s+of)?|or\s+fewer|or\s+less)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "before",
        re.compile(r"\b(?:before|prior\s+to|earlier\s+than)\b", re.IGNORECASE),
        re.compile(r"\b(?:before|prior\s+to|earlier\s+than|preceding)\b", re.IGNORECASE),
    ),
    (
        "after",
        re.compile(r"\b(?:after|following|later\s+than)\b", re.IGNORECASE),
        re.compile(r"\b(?:after|following|later\s+than|subsequent\s+to)\b", re.IGNORECASE),
    ),
    (
        "during",
        re.compile(
            r"\b(?:during|while)\s+(?:the\s+|a\s+|an\s+)?"
            r"(?:year|month|week|day|period|era|\d{1,4}|january|february|march|"
            r"april|may|june|july|august|september|october|november|december)\b|"
            r"\bwithin\s+(?:the\s+)?(?:year|month|week|day|period|era|"
            r"\d{1,4}|january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:during|while|within|in|at|year|date|period)\b", re.IGNORECASE),
    ),
    (
        "since",
        re.compile(r"\bsince\b", re.IGNORECASE),
        re.compile(r"\b(?:since|from|after|starting|beginning)\b", re.IGNORECASE),
    ),
    (
        "until",
        re.compile(r"\b(?:until|up\s+to|through)\b", re.IGNORECASE),
        re.compile(r"\b(?:until|up\s+to|through|before|ending)\b", re.IGNORECASE),
    ),
    (
        "range",
        re.compile(
            r"\bbetween\b|"
            r"\bfrom\s+(?:\d{1,4}|january|february|march|april|may|june|"
            r"july|august|september|october|november|december)\b"
            r"[^.!?]{0,40}\bto\s+(?:\d{1,4}|january|february|march|april|"
            r"may|june|july|august|september|october|november|december)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bbetween\b|\bfrom\b.+?\bto\b|\brange\b", re.IGNORECASE),
    ),
    (
        "current",
        re.compile(r"\b(?:current|currently|now|today|present-day|ongoing)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:current|currently|now|today|present(?:-day)?|ongoing|incumbent)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "maximum-superlative",
        re.compile(
            r"\b(?:most|highest|largest|greatest|longest|latest|newest|youngest)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:most|highest|largest|greatest|longest|latest|newest|youngest|maximum)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "minimum-superlative",
        re.compile(
            r"\b(?:least|lowest|smallest|shortest|earliest|oldest|first)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:least|lowest|smallest|shortest|earliest|oldest|first|minimum)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "last",
        re.compile(r"\blast\b", re.IGNORECASE),
        re.compile(r"\b(?:last|latest|most\s+recent|final)\b", re.IGNORECASE),
    ),
)


def _plain_decomposition_text(value: str) -> str:
    """Return decoded text with ordinary URLs hidden from program-token checks."""
    return _BARE_DOMAIN_RE.sub(" ", _URL_RE.sub(" ", html.unescape(value)))


def _normalized_explicit_numbers(value: str) -> set[str]:
    plain = _plain_decomposition_text(value)
    return {
        match.group(0).casefold().replace(",", "").replace("\uff0c", "")
        for match in _EXPLICIT_NUMBER_RE.finditer(plain)
    }


def _strict_decomposition_item_errors(item: str, index: int) -> list[str]:
    errors: list[str] = []
    decoded = html.unescape(item)
    plain = _plain_decomposition_text(decoded)
    prefix = f"decomposition[{index}]"

    if _FREEBASE_URI_RE.search(decoded):
        errors.append(f"{prefix} contains a Freebase URI")
    if _FREEBASE_NAMESPACE_RE.search(plain):
        errors.append(f"{prefix} contains a Freebase namespace prefix")
    if _FREEBASE_MID_RE.search(plain):
        errors.append(f"{prefix} contains a Freebase MID")
    if _FREEBASE_SLASH_ID_RE.search(plain):
        errors.append(f"{prefix} contains a slash-form Freebase ID")
    if _SPARQL_VARIABLE_RE.search(plain):
        errors.append(f"{prefix} contains a SPARQL variable")
    if _PROGRAM_ARROW_RE.search(plain):
        errors.append(f"{prefix} contains graph/program arrow syntax")
    if "|" in plain:
        errors.append(f"{prefix} contains pipe-style program syntax")
    if _SPARQL_SYNTAX_RE.search(plain):
        errors.append(f"{prefix} contains SPARQL syntax")
    if _LOWER_DOTTED_RELATION_RE.search(plain):
        errors.append(f"{prefix} contains a dotted Freebase relation")
    if _PROGRAM_REFERENCE_RE.search(plain):
        errors.append(f"{prefix} contains executable logical-form notation")
    if _PROGRAM_DELIMITER_RE.search(plain):
        errors.append(f"{prefix} contains program/schema delimiters")

    words = _NATURAL_WORD_RE.findall(plain)
    letter_count = sum(character.isalpha() for character in plain)
    if len(words) < 2 or letter_count < 4:
        errors.append(f"{prefix} must look like a natural-language subquestion")
    return errors


def _decomposition_constraint_errors(items: list[str], question: str) -> list[str]:
    question_text = html.unescape(question)
    decomposition_text = " ".join(items)
    errors: list[str] = []

    question_numbers = _normalized_explicit_numbers(question_text)
    decomposition_numbers = _normalized_explicit_numbers(decomposition_text)
    missing_numbers = sorted(question_numbers - decomposition_numbers)
    if missing_numbers:
        errors.append(
            "decomposition is missing explicit numeric/date constraint(s) from question: "
            + ", ".join(missing_numbers)
        )

    question_months = {match.group(0).casefold() for match in _MONTH_RE.finditer(question_text)}
    decomposition_months = {
        match.group(0).casefold() for match in _MONTH_RE.finditer(decomposition_text)
    }
    missing_months = sorted(question_months - decomposition_months)
    if missing_months:
        errors.append(
            "decomposition is missing explicit month/date constraint(s) from question: "
            + ", ".join(missing_months)
        )

    question_plain = _plain_decomposition_text(question_text)
    decomposition_plain = _plain_decomposition_text(decomposition_text)
    for label, question_pattern, decomposition_pattern in _CONSTRAINT_CUE_GROUPS:
        if question_pattern.search(question_plain) and not decomposition_pattern.search(
            decomposition_plain
        ):
            errors.append(f"decomposition is missing {label} constraint cue from question")
    return errors


def validate_decomposition(
    value: Any,
    *,
    question: str | None = None,
    strict_natural_language: bool = False,
) -> list[str]:
    """Validate the CWQ path-level decomposition contract without using model facts."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["decomposition must be an object"]
    if set(value) != {"decomposition"}:
        errors.append("decomposition must contain exactly decomposition")
    items = value.get("decomposition")
    if not isinstance(items, list) or not items:
        return [*errors, "decomposition must be a non-empty array"]
    seen: set[str] = set()
    valid_items: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"decomposition[{index}] must be a non-empty string")
            continue
        valid_items.append(item)
        normalized = " ".join(item.split()).casefold()
        if normalized in seen:
            errors.append(f"decomposition[{index}] duplicates another path item")
        seen.add(normalized)
        if strict_natural_language:
            errors.extend(_strict_decomposition_item_errors(item, index))
    if strict_natural_language and isinstance(question, str) and question.strip():
        errors.extend(_decomposition_constraint_errors(valid_items, question))
    return errors


def repair_semantic_path_output_variables(graph: Any) -> tuple[Any, list[str]]:
    """Normalize each path output to its structurally required final step variable.

    Compose may still project an intermediate answer variable. ``path_output_var``
    records the end of the semantic traversal, while static mapping and execution
    validation remain responsible for answer semantics.
    """
    if not isinstance(graph, dict) or not isinstance(graph.get("semantic_paths"), list):
        return graph, []
    repaired = deepcopy(graph)
    notes: list[str] = []
    for index, path in enumerate(repaired.get("semantic_paths", [])):
        if not isinstance(path, dict):
            continue
        path_id = str(path.get("id", ""))
        steps = path.get("steps")
        if not _PATH_RE.fullmatch(path_id) or not isinstance(steps, list) or not steps:
            continue
        final_step = steps[-1]
        if not isinstance(final_step, dict):
            continue
        final_variable = str(final_step.get("to", ""))
        if not _PATH_VAR_RE.fullmatch(final_variable) or not final_variable.startswith(
            f"{path_id}."
        ):
            continue
        previous = str(path.get("path_output_var", ""))
        if previous != final_variable:
            path["path_output_var"] = final_variable
            notes.append(
                f"semantic_paths[{index}].path_output_var normalized "
                f"{previous!r} -> {final_variable!r}"
            )
    return repaired, notes


def normalize_operator(operator: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": str(operator.get("type", "")).upper(),
        "inputs": [str(value) for value in operator.get("inputs", []) or []],
        "input_var": str(operator.get("input_var", "")),
        "attribute_relation_label": _label(operator.get("attribute_relation_label", [])),
        "attribute_relation_labels": [
            _label(value)
            for value in operator.get("attribute_relation_labels", []) or []
            if _label(value)
        ],
        "value": str(operator.get("value", "")),
        "value_type": str(operator.get("value_type", "")),
    }


def repair_source_structural_and(
    graph: Any,
    source_graph: Any,
) -> tuple[Any, list[str]]:
    """Restore source-declared structural AND operators over existing paths."""
    if not isinstance(graph, dict) or not isinstance(source_graph, dict):
        return graph, []
    paths = graph.get("semantic_paths")
    operators = graph.get("operators")
    source_operators = source_graph.get("operators")
    if not isinstance(paths, list) or not isinstance(operators, list):
        return graph, []
    if not isinstance(source_operators, list):
        return graph, []
    path_ids = {
        str(path.get("id", ""))
        for path in paths
        if isinstance(path, dict) and _PATH_RE.fullmatch(str(path.get("id", "")))
    }
    existing = {
        tuple(sorted(normalize_operator(operator)["inputs"]))
        for operator in operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() == "AND"
    }
    repaired = deepcopy(graph)
    repaired_operators = repaired.get("operators", [])
    notes: list[str] = []
    for raw in source_operators:
        if not isinstance(raw, dict) or str(raw.get("type", "")).upper() != "AND":
            continue
        operator = normalize_operator(raw)
        inputs = tuple(sorted(operator["inputs"]))
        if len(inputs) < 2 or not set(inputs).issubset(path_ids) or inputs in existing:
            continue
        repaired_operators.append(operator)
        existing.add(inputs)
        notes.append("restored source structural AND over " + ",".join(inputs))
    return repaired, notes


def repair_semantic_graph_cross_path_variables(graph: Any) -> tuple[Any, list[str]]:
    """Repair a common model error: using another path's Pn.Vm as a local output.

    Semantic paths must be path-local. A constraint such as
    P1: Male <- gender <- P0.V1 should be represented as P1.V0 plus a semantic AND
    over P0 and P1, with compose later lowering that AND to one shared global variable.
    """
    if not isinstance(graph, dict):
        return graph, []
    raw_paths = graph.get("semantic_paths", [])
    if not isinstance(raw_paths, list):
        return graph, []

    repaired = deepcopy(graph)
    paths = repaired.get("semantic_paths", [])
    if not isinstance(paths, list):
        return graph, []

    path_outputs: dict[str, str] = {}
    output_to_path: dict[str, str] = {}
    for path in paths:
        if not isinstance(path, dict):
            continue
        path_id = str(path.get("id", ""))
        output_var = str(path.get("path_output_var", ""))
        if _PATH_RE.fullmatch(path_id):
            path_outputs[path_id] = output_var
            if _PATH_VAR_RE.fullmatch(output_var) and output_var.startswith(f"{path_id}."):
                output_to_path[output_var] = path_id

    notes: list[str] = []
    and_inputs: set[str] = set()

    for path_index, path in enumerate(paths):
        if not isinstance(path, dict):
            continue
        path_id = str(path.get("id", ""))
        if not _PATH_RE.fullmatch(path_id):
            continue
        steps = path.get("steps", [])
        if not isinstance(steps, list):
            continue
        aliases: dict[str, str] = {}
        used_local = {
            str(step.get("to", ""))
            for step in steps
            if isinstance(step, dict) and str(step.get("to", "")).startswith(f"{path_id}.")
        }
        next_local_index = 0

        def fresh_local_var() -> str:
            nonlocal next_local_index
            while f"{path_id}.V{next_local_index}" in used_local:
                next_local_index += 1
            value = f"{path_id}.V{next_local_index}"
            used_local.add(value)
            next_local_index += 1
            return value

        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            source_ref = str(step.get("from", ""))
            if source_ref in aliases:
                step["from"] = aliases[source_ref]
                notes.append(
                    f"semantic_paths[{path_index}].steps[{step_index}].from remapped "
                    f"{source_ref} -> {aliases[source_ref]}"
                )

            target_ref = str(step.get("to", ""))
            if not _PATH_VAR_RE.fullmatch(target_ref) or target_ref.startswith(f"{path_id}."):
                continue
            local_ref = aliases.get(target_ref)
            if local_ref is None:
                local_ref = fresh_local_var()
                aliases[target_ref] = local_ref
            step["to"] = local_ref
            notes.append(
                f"semantic_paths[{path_index}].steps[{step_index}].to remapped "
                f"{target_ref} -> {local_ref}"
            )
            source_path_id = output_to_path.get(target_ref)
            if source_path_id and source_path_id != path_id:
                and_inputs.update({source_path_id, path_id})

        output_ref = str(path.get("path_output_var", ""))
        if output_ref in aliases:
            path["path_output_var"] = aliases[output_ref]
            notes.append(
                f"semantic_paths[{path_index}].path_output_var remapped "
                f"{output_ref} -> {aliases[output_ref]}"
            )
            source_path_id = output_to_path.get(output_ref)
            if source_path_id and source_path_id != path_id:
                and_inputs.update({source_path_id, path_id})
        elif _PATH_VAR_RE.fullmatch(output_ref) and not output_ref.startswith(f"{path_id}."):
            local_ref = str(steps[-1].get("to", "")) if steps and isinstance(steps[-1], dict) else fresh_local_var()
            path["path_output_var"] = local_ref
            notes.append(
                f"semantic_paths[{path_index}].path_output_var remapped "
                f"{output_ref} -> {local_ref}"
            )
            source_path_id = output_to_path.get(output_ref)
            if source_path_id and source_path_id != path_id:
                and_inputs.update({source_path_id, path_id})

    if and_inputs:
        ordered_inputs = [
            path_id
            for path_id in path_outputs
            if path_id in and_inputs
        ]
        raw_operators = repaired.get("operators", [])
        if not isinstance(raw_operators, list):
            raw_operators = []
            repaired["operators"] = raw_operators
        existing_and = next(
            (
                operator
                for operator in raw_operators
                if isinstance(operator, dict) and str(operator.get("type", "")).upper() == "AND"
            ),
            None,
        )
        if existing_and is None:
            raw_operators.append(
                {
                    "type": "AND",
                    "inputs": ordered_inputs,
                    "input_var": "",
                    "attribute_relation_label": [],
                    "attribute_relation_labels": [],
                    "value": "",
                    "value_type": "",
                }
            )
            notes.append(f"operators added semantic AND over {ordered_inputs}")
        else:
            merged_inputs = []
            for value in [*existing_and.get("inputs", []), *ordered_inputs]:
                path_id = str(value)
                if path_id in path_outputs and path_id not in merged_inputs:
                    merged_inputs.append(path_id)
            existing_and.update(
                {
                    "type": "AND",
                    "inputs": merged_inputs,
                    "input_var": "",
                    "attribute_relation_label": [],
                    "attribute_relation_labels": [],
                    "value": "",
                    "value_type": "",
                }
            )
            notes.append(f"operators normalized semantic AND over {merged_inputs}")

    return repaired, notes


def repair_unscoped_semantic_operator_variables(graph: Any) -> tuple[Any, list[str]]:
    """Restore a unique ``Pn.Vm`` prefix omitted by a semantic-stage model.

    The Compose representation uses global variables (``V1``), while semantic
    graphs require path-local variables (``P0.V1``).  A model occasionally
    leaks the Compose spelling into a semantic unary operator.  Repair it only
    when exactly one generated semantic path variable has that suffix; an
    ambiguous reference remains untouched for strict validation.
    """
    if not isinstance(graph, dict):
        return graph, []
    paths = graph.get("semantic_paths", [])
    operators = graph.get("operators", [])
    if not isinstance(paths, list) or not isinstance(operators, list):
        return graph, []

    candidates: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if not isinstance(path, dict):
            continue
        for step in path.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            variable = str(step.get("to", ""))
            match = _PATH_VAR_RE.fullmatch(variable)
            if match:
                candidates[variable.rsplit(".", 1)[-1]].append(variable)

    repaired = deepcopy(graph)
    notes: list[str] = []
    for index, operator in enumerate(repaired.get("operators", []) or []):
        if not isinstance(operator, dict):
            continue
        value = str(operator.get("input_var", ""))
        if not re.fullmatch(r"V\d+", value):
            continue
        matches = sorted(set(candidates.get(value, [])))
        if len(matches) != 1:
            continue
        operator["input_var"] = matches[0]
        notes.append(
            f"operators[{index}].input_var restored semantic scope {value} -> {matches[0]}"
        )
    return repaired, notes


def expected_operators(source_compose: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in source_compose.get("operators", []) or []:
        if not isinstance(raw, dict):
            continue
        operator = normalize_operator(raw)
        if operator["type"] in EXECUTABLE_OPERATOR_TYPES:
            output.append(operator)
    return output


def project_program_operators_to_semantic(
    semantic_graph: dict[str, Any],
    program: dict[str, Any],
    operators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate program-scoped operator variables into semantic path variables."""
    mappings = _semantic_program_mappings(semantic_graph, program)
    semantic_operators = [
        normalize_operator(item)
        for item in semantic_graph.get("operators", []) or []
        if isinstance(item, dict) and str(item.get("type", "")).upper() != "AND"
    ]
    path_outputs = {
        str(path.get("path_output_var", ""))
        for path in semantic_graph.get("semantic_paths", []) or []
        if isinstance(path, dict)
    }
    output: list[dict[str, Any]] = []
    for raw in operators:
        operator = normalize_operator(raw)
        source_match = next(
            (
                item
                for item in semantic_operators
                if operator_signature(item) == operator_signature(operator)
            ),
            None,
        )
        if source_match is not None:
            output.append(source_match)
            continue
        projected: dict[str, Any] | None = None
        for mapping in mappings:
            inverse: dict[str, list[str]] = defaultdict(list)
            for semantic_ref, program_ref in mapping.items():
                if _PATH_VAR_RE.fullmatch(semantic_ref):
                    inverse[program_ref].append(semantic_ref)

            def semantic_ref(program_ref: str) -> str:
                candidates = inverse.get(program_ref, [])
                if not candidates:
                    return ""
                return sorted(
                    candidates,
                    key=lambda value: (value not in path_outputs, value),
                )[0]

            input_var = semantic_ref(operator["input_var"])
            inputs = [semantic_ref(value) for value in operator["inputs"]]
            if operator["input_var"] and not input_var:
                continue
            if any(not value for value in inputs):
                continue
            projected = dict(operator)
            projected["input_var"] = input_var
            projected["inputs"] = inputs
            break
        output.append(projected or operator)
    return output


def repair_operator_fidelity(
    value: Any,
    expected: list[dict[str, Any]],
    *,
    location: str,
) -> tuple[Any, list[str]]:
    """Keep generated operators aligned with the source executable contract.

    Models frequently hallucinate a NO_EQUAL/EQUAL constraint even when the
    source program has no executable operators, or preserve the right relation
    while paraphrasing its value.  The source compose row is the authoritative
    operator contract, so normalize those recoverable cases before strict
    validation.  Structural semantic AND is intentionally preserved.
    """
    if not isinstance(value, dict):
        return value, []
    raw_operators = value.get("operators")
    if not isinstance(raw_operators, list):
        return value, []
    repaired = deepcopy(value)
    operators = repaired.get("operators", [])
    expected_normalized = [normalize_operator(item) for item in expected]
    notes: list[str] = []
    kept: list[dict[str, Any]] = []
    used_expected: set[int] = set()
    for index, raw in enumerate(operators):
        if not isinstance(raw, dict):
            kept.append(raw)
            continue
        normalized = normalize_operator(raw)
        if normalized["type"] == "AND":
            kept.append(raw)
            continue
        exact_index = next(
            (
                expected_index
                for expected_index, required in enumerate(expected_normalized)
                if expected_index not in used_expected
                and operator_signature(required) == operator_signature(normalized)
            ),
            None,
        )
        match_index = exact_index
        if match_index is None:
            # Match the relation/type first, then use the source value.  This
            # repairs casing/paraphrase drift without inventing a new operator.
            match_index = next(
                (
                    expected_index
                    for expected_index, required in enumerate(expected_normalized)
                    if expected_index not in used_expected
                    and required["type"] == normalized["type"]
                    and required["attribute_relation_label"]
                    == normalized["attribute_relation_label"]
                    and required["attribute_relation_labels"]
                    == normalized["attribute_relation_labels"]
                ),
                None,
            )
        if match_index is None:
            notes.append(f"{location}.operators[{index}] dropped unsupported executable operator")
            continue
        required = expected_normalized[match_index]
        used_expected.add(match_index)
        if (
            location == "semantic_graph"
            and _PATH_VAR_RE.fullmatch(required["input_var"])
            and normalized["input_var"] != required["input_var"]
        ):
            normalized["input_var"] = required["input_var"]
            notes.append(
                f"{location}.operators[{index}] scoped to source variable "
                f"{required['input_var']}"
            )
        if operator_signature(required) != operator_signature(normalized):
            if not (
                required["type"] == "NO_EQUAL"
                and normalized["value"].casefold() == required["value"].casefold()
            ):
                normalized["value"] = required["value"]
            normalized["value_type"] = required["value_type"]
            notes.append(
                f"{location}.operators[{index}] canonicalized value from source operator"
            )
        kept.append(normalized)
    if location == "semantic_graph":
        semantic_variables = {
            str(step.get("to", ""))
            for path in repaired.get("semantic_paths", []) or []
            if isinstance(path, dict)
            for step in path.get("steps", []) or []
            if isinstance(step, dict)
        }
        for expected_index, required in enumerate(expected_normalized):
            if expected_index in used_expected:
                continue
            if required["input_var"] not in semantic_variables:
                continue
            kept.append(required)
            notes.append(
                f"{location} added missing source-derived {required['type']} operator"
            )
    repaired["operators"] = kept
    return repaired, notes


def repair_compose_literal_equalities(
    program: Any,
    expected: list[dict[str, Any]],
) -> tuple[Any, list[str]]:
    """Lower source-style literal triples to executable EQUAL operators."""
    if not isinstance(program, dict):
        return program, []
    constraints = [
        normalize_operator(item)
        for item in expected
        if str(item.get("type", "")).upper() == "EQUAL"
    ]
    if not constraints:
        return program, []
    repaired = deepcopy(program)
    triples = repaired.get("triples")
    operators = repaired.get("operators")
    if not isinstance(triples, list) or not isinstance(operators, list):
        return program, []
    notes: list[str] = []
    for constraint in constraints:
        matched_subjects: list[str] = []
        kept_triples: list[Any] = []
        for triple in triples:
            if not isinstance(triple, dict):
                kept_triples.append(triple)
                continue
            literal = _quoted_literal(str(triple.get("object", "")))
            if (
                _label(triple.get("relation_label", []))
                == constraint["attribute_relation_label"]
                and literal == constraint["value"]
            ):
                matched_subjects.append(str(triple.get("subject", "")))
                continue
            kept_triples.append(triple)
        if not matched_subjects:
            continue
        triples = kept_triples
        input_var = next(
            (value for value in matched_subjects if _PROGRAM_VAR_RE.fullmatch(value)),
            constraint["input_var"],
        )
        existing = next(
            (
                item
                for item in operators
                if isinstance(item, dict)
                and operator_signature(item) == operator_signature(constraint)
            ),
            None,
        )
        normalized = dict(constraint)
        normalized["input_var"] = input_var
        if existing is None:
            operators.append(normalized)
        else:
            existing.clear()
            existing.update(normalized)
        notes.append(
            "lowered exact literal triple to EQUAL on "
            f"{input_var or '<unknown>'}"
        )
    repaired["triples"] = triples
    return repaired, notes


def repair_compose_entity_surfaces(
    program: Any,
    semantic_graph: Any,
) -> tuple[Any, list[str]]:
    """Canonicalize punctuation and high-confidence spelling drift in entities."""
    if not isinstance(program, dict) or not isinstance(semantic_graph, dict):
        return program, []
    anchors = [
        str(item.get("surface", "")).strip()
        for item in semantic_graph.get("anchors", []) or []
        if isinstance(item, dict) and str(item.get("surface", "")).strip()
    ]
    if not anchors or not isinstance(program.get("entities"), list):
        return program, []

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    repaired = deepcopy(program)
    notes: list[str] = []
    for entity in repaired["entities"]:
        if not isinstance(entity, dict):
            continue
        surface = str(entity.get("surface", "")).strip()
        exact = next((anchor for anchor in anchors if anchor == surface), None)
        if exact is not None:
            continue
        compact = normalized(surface)
        matches = [anchor for anchor in anchors if normalized(anchor) == compact]
        if len(matches) != 1:
            scored = sorted(
                (
                    SequenceMatcher(None, normalized(surface), normalized(anchor)).ratio(),
                    anchor,
                )
                for anchor in anchors
            )
            if scored and scored[-1][0] >= 0.93 and (
                len(scored) == 1 or scored[-1][0] - scored[-2][0] >= 0.04
            ):
                matches = [scored[-1][1]]
        if len(matches) == 1:
            entity["surface"] = matches[0]
            notes.append(f"canonicalized entity surface {surface!r} -> {matches[0]!r}")
    return repaired, notes


def compile_compose_program(
    semantic_graph: dict[str, Any],
    *,
    answer_path_ids: list[str] | None = None,
    source_graph: dict[str, Any] | None = None,
    source_program: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically lower a validated semantic graph into Compose JSON."""
    anchors = [
        item
        for item in semantic_graph.get("anchors", []) or []
        if isinstance(item, dict)
    ]
    paths = [
        item
        for item in semantic_graph.get("semantic_paths", []) or []
        if isinstance(item, dict)
    ]
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    path_outputs = {
        str(path.get("id", "")): str(path.get("path_output_var", ""))
        for path in paths
    }
    for path in paths:
        for step in path.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            for ref in (str(step.get("from", "")), str(step.get("to", ""))):
                if _PATH_VAR_RE.fullmatch(ref):
                    find(ref)

    for operator in semantic_graph.get("operators", []) or []:
        if not isinstance(operator, dict) or str(operator.get("type", "")).upper() != "AND":
            continue
        outputs = [
            path_outputs.get(str(path_id), "")
            for path_id in operator.get("inputs", []) or []
        ]
        outputs = [value for value in outputs if value]
        for value in outputs[1:]:
            union(outputs[0], value)

    if isinstance(source_graph, dict) and isinstance(source_program, dict):
        source_mappings = _semantic_program_mappings(source_graph, source_program)
        if source_mappings:
            refs_by_program: dict[str, list[str]] = defaultdict(list)
            for semantic_ref, program_ref in source_mappings[0].items():
                if semantic_ref in parent and _PROGRAM_VAR_RE.fullmatch(program_ref):
                    refs_by_program[program_ref].append(semantic_ref)
            for refs in refs_by_program.values():
                for value in refs[1:]:
                    union(refs[0], value)

    entities: list[dict[str, str]] = []
    refs: dict[str, str] = {}
    for index, anchor in enumerate(anchors):
        anchor_id = str(anchor.get("id", ""))
        entity_id = f"E{index}"
        refs[anchor_id] = entity_id
        entities.append({"id": entity_id, "surface": str(anchor.get("surface", ""))})

    next_variable = 0

    def program_ref(semantic_ref: str) -> str:
        nonlocal next_variable
        if semantic_ref in refs:
            return refs[semantic_ref]
        root = find(semantic_ref)
        if root not in refs:
            refs[root] = f"V{next_variable}"
            next_variable += 1
        refs[semantic_ref] = refs[root]
        return refs[root]

    triples: list[dict[str, Any]] = []
    seen_triples: set[tuple[str, tuple[str, ...], str]] = set()
    for path in paths:
        for step in path.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            source_ref = str(step.get("from", ""))
            target_ref = str(step.get("to", ""))
            if str(step.get("direction", "")) == "backward":
                subject_ref, object_ref = target_ref, source_ref
            else:
                subject_ref, object_ref = source_ref, target_ref
            relation = _label(step.get("relation_label", []))
            subject = program_ref(subject_ref)
            obj = program_ref(object_ref)
            key = (subject, tuple(relation), obj)
            if key in seen_triples:
                continue
            seen_triples.add(key)
            triples.append(
                {"subject": subject, "relation_label": relation, "object": obj}
            )

    operators: list[dict[str, Any]] = []
    for raw in semantic_graph.get("operators", []) or []:
        if not isinstance(raw, dict):
            continue
        operator = normalize_operator(raw)
        if operator["type"] == "AND":
            continue
        operator["input_var"] = program_ref(operator["input_var"])
        operator["inputs"] = [program_ref(value) for value in operator["inputs"]]
        operators.append(operator)

    requested = set(answer_path_ids or [])
    answer_refs = [
        str(path.get("path_output_var", ""))
        for path in paths
        if str(path.get("id", "")) in requested
    ]
    if not answer_refs:
        answer_refs = [
            str(path.get("path_output_var", ""))
            for path in paths
            if str(path.get("path_output_var", ""))
        ][:1]
    answer_var = program_ref(answer_refs[0]) if answer_refs else ""
    return {
        "entities": entities,
        "triples": triples,
        "operators": operators,
        "answer_var": answer_var,
    }


def semantic_answer_variables(
    graph: Any,
    program: Any,
) -> list[str]:
    """Return semantic path outputs that map to the Compose answer variable."""
    if not isinstance(graph, dict) or not isinstance(program, dict):
        return []
    answer_var = str(program.get("answer_var", ""))
    if not _PROGRAM_VAR_RE.fullmatch(answer_var):
        return []
    path_outputs = [
        str(path.get("path_output_var", ""))
        for path in graph.get("semantic_paths", []) or []
        if isinstance(path, dict)
    ]
    mappings = _semantic_program_mappings(graph, program)
    return [
        output_var
        for output_var in path_outputs
        if output_var and any(mapping.get(output_var) == answer_var for mapping in mappings)
    ]


def repair_answer_exclusion_scopes(
    graph: Any,
    program: Any,
    exclusions: list[str],
    *,
    source_graph: Any | None = None,
    source_program: Any | None = None,
    repair_program: bool = True,
) -> tuple[Any, Any, list[str]]:
    """Bind source-derived NO_EQUAL constraints to the projected answer.

    CWQ expresses some exclusions as FILTER(?answer != ?entity).  The excluded
    entity is an operator value, not a new relation path.  Keep it as an
    anchor/entity, remove model-invented binding paths, and deterministically
    scope the semantic and Compose operators to the answer variables.
    """
    if not isinstance(graph, dict) or not isinstance(program, dict):
        return graph, program, []
    canonical_exclusions = {
        str(value).strip().casefold(): str(value).strip()
        for value in exclusions
        if str(value).strip()
    }
    if not canonical_exclusions:
        return graph, program, []

    repaired_graph = deepcopy(graph)
    repaired_program = deepcopy(program)
    notes: list[str] = []

    source_path_surfaces: set[str] = set()
    if isinstance(source_graph, dict):
        source_anchors = {
            str(anchor.get("id", "")): str(anchor.get("surface", "")).strip().casefold()
            for anchor in source_graph.get("anchors", []) or []
            if isinstance(anchor, dict)
        }
        source_path_surfaces = {
            source_anchors.get(str(path.get("anchor_ref", "")), "")
            for path in source_graph.get("semantic_paths", []) or []
            if isinstance(path, dict)
        }
        source_path_surfaces.discard("")

    anchors = {
        str(anchor.get("id", "")): str(anchor.get("surface", "")).strip().casefold()
        for anchor in repaired_graph.get("anchors", []) or []
        if isinstance(anchor, dict)
    }
    removed_path_ids = {
        str(path.get("id", ""))
        for path in repaired_graph.get("semantic_paths", []) or []
        if isinstance(path, dict)
        and anchors.get(str(path.get("anchor_ref", "")), "") in canonical_exclusions
        and anchors.get(str(path.get("anchor_ref", "")), "") not in source_path_surfaces
    }
    if removed_path_ids:
        all_path_ids = {
            str(path.get("id", ""))
            for path in repaired_graph.get("semantic_paths", []) or []
            if isinstance(path, dict)
        }
        removed_triples = set()
        if repair_program:
            removed_triples = _program_triples_for_paths(
                repaired_graph,
                repaired_program,
                removed_path_ids,
            )
            retained_triples = _program_triples_for_paths(
                repaired_graph,
                repaired_program,
                all_path_ids - removed_path_ids,
            )
            removed_triples.difference_update(retained_triples)
        repaired_graph["semantic_paths"] = [
            path
            for path in repaired_graph.get("semantic_paths", []) or []
            if not isinstance(path, dict) or str(path.get("id", "")) not in removed_path_ids
        ]
        normalized_operators: list[Any] = []
        for operator in repaired_graph.get("operators", []) or []:
            if not isinstance(operator, dict) or str(operator.get("type", "")).upper() != "AND":
                normalized_operators.append(operator)
                continue
            kept_inputs = [
                str(value)
                for value in operator.get("inputs", []) or []
                if str(value) not in removed_path_ids
            ]
            if len(kept_inputs) >= 2:
                normalized = dict(operator)
                normalized["inputs"] = kept_inputs
                normalized_operators.append(normalized)
        repaired_graph["operators"] = normalized_operators
        if removed_triples:
            repaired_program["triples"] = [
                triple
                for index, triple in enumerate(repaired_program.get("triples", []) or [])
                if index not in removed_triples
            ]
        notes.append(
            "removed exclusion-only semantic paths "
            + repr(sorted(removed_path_ids))
        )

    graph_anchors = repaired_graph.get("anchors", [])
    if isinstance(graph_anchors, list):
        present = {
            str(anchor.get("surface", "")).strip().casefold()
            for anchor in graph_anchors
            if isinstance(anchor, dict)
        }
        used_ids = {
            str(anchor.get("id", ""))
            for anchor in graph_anchors
            if isinstance(anchor, dict)
        }
        for key, surface in canonical_exclusions.items():
            if key in present:
                continue
            anchor_id = _next_id("A", used_ids)
            graph_anchors.append({"id": anchor_id, "surface": surface})
            used_ids.add(anchor_id)
            present.add(key)
            notes.append(f"added exclusion value anchor {anchor_id}")

    program_entities = repaired_program.get("entities", [])
    if repair_program and isinstance(program_entities, list):
        present = {
            str(entity.get("surface", "")).strip().casefold()
            for entity in program_entities
            if isinstance(entity, dict)
        }
        used_ids = {
            str(entity.get("id", ""))
            for entity in program_entities
            if isinstance(entity, dict)
        }
        for key, surface in canonical_exclusions.items():
            if key in present:
                continue
            entity_id = _next_id("E", used_ids)
            program_entities.append({"id": entity_id, "surface": surface})
            used_ids.add(entity_id)
            present.add(key)
            notes.append(f"added exclusion value entity {entity_id}")

    answer_vars = semantic_answer_variables(repaired_graph, repaired_program)
    if not answer_vars and isinstance(source_graph, dict) and isinstance(source_program, dict):
        candidate_outputs = {
            str(path.get("path_output_var", ""))
            for path in repaired_graph.get("semantic_paths", []) or []
            if isinstance(path, dict)
        }
        answer_vars = [
            value
            for value in semantic_answer_variables(source_graph, source_program)
            if value in candidate_outputs
        ]
    if not answer_vars:
        path_outputs = [
            str(path.get("path_output_var", ""))
            for path in repaired_graph.get("semantic_paths", []) or []
            if isinstance(path, dict) and str(path.get("path_output_var", ""))
        ]
        if len(path_outputs) == 1:
            answer_vars = path_outputs

    semantic_operators = repaired_graph.get("operators", [])
    if isinstance(semantic_operators, list) and answer_vars:
        _repair_answer_no_equal_operators(
            semantic_operators,
            canonical_exclusions,
            input_var=answer_vars[0],
            notes=notes,
            location="semantic_graph",
        )

    answer_var = str(repaired_program.get("answer_var", ""))
    compose_operators = repaired_program.get("operators", [])
    if (
        repair_program
        and isinstance(compose_operators, list)
        and _PROGRAM_VAR_RE.fullmatch(answer_var)
    ):
        _repair_answer_no_equal_operators(
            compose_operators,
            canonical_exclusions,
            input_var=answer_var,
            notes=notes,
            location="compose_program",
        )

    return repaired_graph, repaired_program, notes


def _repair_answer_no_equal_operators(
    operators: list[Any],
    exclusions: dict[str, str],
    *,
    input_var: str,
    notes: list[str],
    location: str,
) -> None:
    existing: set[str] = set()
    for index, operator in enumerate(operators):
        if not isinstance(operator, dict) or str(operator.get("type", "")).upper() != "NO_EQUAL":
            continue
        key = str(operator.get("value", "")).strip().casefold()
        if key not in exclusions:
            continue
        existing.add(key)
        normalized = {
            "type": "NO_EQUAL",
            "inputs": [],
            "input_var": input_var,
            "attribute_relation_label": [],
            "attribute_relation_labels": [],
            "value": exclusions[key],
            "value_type": "entity",
        }
        if operator != normalized:
            operators[index] = normalized
            notes.append(f"scoped {location} NO_EQUAL to answer variable {input_var}")
    for key, surface in exclusions.items():
        if key in existing:
            continue
        operators.append(
            {
                "type": "NO_EQUAL",
                "inputs": [],
                "input_var": input_var,
                "attribute_relation_label": [],
                "attribute_relation_labels": [],
                "value": surface,
                "value_type": "entity",
            }
        )
        notes.append(f"added source-derived {location} NO_EQUAL on {input_var}")


def _next_id(prefix: str, used: set[str]) -> str:
    index = 0
    while f"{prefix}{index}" in used:
        index += 1
    return f"{prefix}{index}"


def _program_triples_for_paths(
    graph: dict[str, Any],
    program: dict[str, Any],
    path_ids: set[str],
) -> set[int]:
    mappings = _semantic_program_mappings(graph, program)
    if not mappings:
        return set()
    mapping = mappings[0]
    triples = program.get("triples", []) or []
    indexes: set[int] = set()
    for path in graph.get("semantic_paths", []) or []:
        if not isinstance(path, dict) or str(path.get("id", "")) not in path_ids:
            continue
        for step in path.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            source_ref = str(step.get("from", ""))
            target_ref = str(step.get("to", ""))
            if str(step.get("direction", "")) == "backward":
                subject_ref, object_ref = target_ref, source_ref
            else:
                subject_ref, object_ref = source_ref, target_ref
            subject = mapping.get(subject_ref, "")
            obj = mapping.get(object_ref, "")
            relation = _label(step.get("relation_label", []))
            for index, triple in enumerate(triples):
                if not isinstance(triple, dict):
                    continue
                if (
                    str(triple.get("subject", "")) == subject
                    and str(triple.get("object", "")) == obj
                    and _label(triple.get("relation_label", [])) == relation
                ):
                    indexes.add(index)
                    break
    return indexes


def repair_semantic_graph_directions_to_compose(
    graph: Any,
    program: Any,
) -> tuple[Any, list[str]]:
    """Try direction flips when a generated semantic graph cannot map to Compose.

    Relation labels and the generated Compose triples provide a deterministic
    orientation signal.  Search only the binary forward/backward choices while
    retaining all path-local variable names and semantic content.
    """
    if not isinstance(graph, dict) or not isinstance(program, dict):
        return graph, []
    if _semantic_program_mappings(graph, program):
        return graph, []
    paths = graph.get("semantic_paths")
    if not isinstance(paths, list):
        return graph, []
    choices: list[tuple[int, list[str]]] = []
    for index, path in enumerate(paths):
        if not isinstance(path, dict) or not isinstance(path.get("steps"), list):
            continue
        directions = [
            str(step.get("direction", "forward"))
            for step in path["steps"]
            if isinstance(step, dict)
        ]
        choices.append((index, directions))
    if not choices:
        return graph, []
    max_variants = 512
    total_variants = 1
    for _, directions in choices:
        total_variants *= 2 ** len(directions)
    if total_variants > max_variants:
        return graph, []
    repaired = deepcopy(graph)
    for variant in product(*[
        list(product(("forward", "backward"), repeat=len(directions)))
        for _, directions in choices
    ]):
        candidate = deepcopy(graph)
        changed = False
        for (path_index, original), selected in zip(choices, variant):
            path = candidate["semantic_paths"][path_index]
            for step, direction in zip(path.get("steps", []), selected):
                if isinstance(step, dict):
                    changed = changed or step.get("direction") != direction
                    step["direction"] = direction
        if changed and _semantic_program_mappings(candidate, program):
            repaired = candidate
            return repaired, ["semantic path directions aligned to generated compose triples"]
    return graph, []


def repair_superfluous_semantic_and(
    graph: Any,
    program: Any,
) -> tuple[Any, list[str]]:
    """Remove AND only when Compose already expresses the same join.

    A structural AND is valid when every referenced path output lowers to one
    shared Compose variable.  Some model generations instead put AND over
    independent path outputs while Compose correctly joins those paths through
    an intermediate variable.  In that narrow case, keeping AND makes the
    semantic and Compose representations contradictory.
    """
    if not isinstance(graph, dict) or not isinstance(program, dict):
        return graph, []
    raw_operators = graph.get("operators")
    if not isinstance(raw_operators, list):
        return graph, []
    and_operators = [
        item for item in raw_operators
        if isinstance(item, dict) and str(item.get("type", "")).upper() == "AND"
    ]
    if not and_operators:
        return graph, []
    mappings = _semantic_program_mappings(graph, program)
    if not mappings:
        return graph, []
    semantic_and = [normalize_operator(item) for item in and_operators]
    if any(_and_operators_follow_mapping(graph, semantic_and, mapping) for mapping in mappings):
        return graph, []
    candidate = deepcopy(graph)
    candidate["operators"] = [
        item for item in raw_operators
        if not (isinstance(item, dict) and str(item.get("type", "")).upper() == "AND")
    ]
    candidate_mappings = _semantic_program_mappings(candidate, program)
    if not candidate_mappings:
        return graph, []
    return candidate, ["removed semantic AND already represented by a Compose shared-variable join"]


def operator_signature(operator: dict[str, Any]) -> tuple[Any, ...]:
    normalized = normalize_operator(operator)
    value = normalized["value"]
    if normalized["type"] == "NO_EQUAL":
        # Entity labels are case-insensitive in Freebase, while source CWQ
        # composition answers are often lower-cased and model surfaces are
        # title-cased. Keep the generated surface for anchor validation but
        # compare the source contract case-insensitively.
        value = value.casefold()
    return (
        normalized["type"],
        tuple(normalized["attribute_relation_label"]),
        tuple(tuple(label) for label in normalized["attribute_relation_labels"]),
        value,
        normalized["value_type"],
    )


def validate_semantic_graph(
    graph: Any,
    *,
    expected: list[dict[str, Any]],
    decomposition: list[str] | None = None,
    source_graph: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(graph, dict):
        return ["semantic_graph must be an object"]
    if set(graph) != {"anchors", "semantic_paths", "operators"}:
        errors.append("semantic_graph must contain exactly anchors, semantic_paths, and operators")

    raw_anchors = graph.get("anchors", [])
    raw_paths = graph.get("semantic_paths", [])
    raw_operators = graph.get("operators", [])
    if not isinstance(raw_anchors, list):
        errors.append("anchors must be an array")
        raw_anchors = []
    if not isinstance(raw_paths, list) or not raw_paths:
        errors.append("semantic_paths must be a non-empty array")
        raw_paths = []
    if not isinstance(raw_operators, list):
        errors.append("operators must be an array")
        raw_operators = []

    anchors: set[str] = set()
    anchor_surfaces: set[str] = set()
    for index, anchor in enumerate(raw_anchors):
        if not isinstance(anchor, dict):
            errors.append(f"anchors[{index}] must be an object")
            continue
        anchor_id = str(anchor.get("id", ""))
        surface = str(anchor.get("surface", "")).strip()
        if not _ANCHOR_RE.fullmatch(anchor_id) or anchor_id in anchors:
            errors.append(f"anchors[{index}] has invalid or duplicate id {anchor_id!r}")
        if not surface:
            errors.append(f"anchors[{index}] has an empty surface")
        if _DOTTED_ID_RE.fullmatch(surface):
            errors.append(f"anchors[{index}] leaks a dotted KB id")
        anchors.add(anchor_id)
        anchor_surfaces.add(surface)

    path_ids: set[str] = set()
    path_variables: set[str] = set()
    for path_index, path in enumerate(raw_paths):
        if not isinstance(path, dict):
            errors.append(f"semantic_paths[{path_index}] must be an object")
            continue
        path_id = str(path.get("id", ""))
        anchor_ref = str(path.get("anchor_ref", ""))
        if not _PATH_RE.fullmatch(path_id) or path_id in path_ids:
            errors.append(f"semantic_paths[{path_index}] has invalid or duplicate id {path_id!r}")
        path_ids.add(path_id)
        if anchor_ref not in anchors:
            errors.append(f"semantic_paths[{path_index}] refers to unknown anchor {anchor_ref!r}")
        if not str(path.get("goal", "")).strip():
            errors.append(f"semantic_paths[{path_index}] has an empty goal")
        steps = path.get("steps", [])
        if not isinstance(steps, list) or not steps:
            errors.append(f"semantic_paths[{path_index}].steps must be non-empty")
            continue
        current_ref = anchor_ref
        local_variables: set[str] = set()
        for step_index, step in enumerate(steps):
            location = f"semantic_paths[{path_index}].steps[{step_index}]"
            if not isinstance(step, dict):
                errors.append(f"{location} must be an object")
                continue
            expected_step_id = f"{path_id}.S{step_index}"
            if str(step.get("id", "")) != expected_step_id:
                errors.append(f"{location}.id must be {expected_step_id!r}")
            if str(step.get("direction", "")) not in {"forward", "backward"}:
                errors.append(f"{location}.direction is invalid")
            source_ref = str(step.get("from", ""))
            target_ref = str(step.get("to", ""))
            if source_ref != current_ref:
                errors.append(f"{location}.from breaks path connectivity")
            if not _PATH_VAR_RE.fullmatch(target_ref) or not target_ref.startswith(f"{path_id}."):
                errors.append(f"{location}.to must be a variable local to {path_id}")
            _validate_relation_label(step.get("relation_label"), f"{location}.relation_label", errors)
            local_variables.add(target_ref)
            path_variables.add(target_ref)
            current_ref = target_ref
        output_var = str(path.get("path_output_var", ""))
        if output_var != current_ref or output_var not in local_variables:
            errors.append(f"semantic_paths[{path_index}].path_output_var must be the final path variable")

    normalized_operators: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_operators):
        if not isinstance(raw, dict):
            errors.append(f"operators[{index}] must be an object")
            continue
        operator = normalize_operator(raw)
        normalized_operators.append(operator)
        _validate_operator(
            operator,
            path_variables,
            f"operators[{index}]",
            errors,
            allowed_types=OPERATOR_TYPES,
        )
        if operator["type"] == "NO_EQUAL" and operator["value"] not in anchor_surfaces:
            errors.append(f"operators[{index}].value must be a semantic anchor surface for NO_EQUAL")
        if operator["type"] == "AND":
            if any(not _PATH_RE.fullmatch(value) for value in operator["inputs"]):
                errors.append(f"operators[{index}].inputs must use semantic path ids")
        elif operator["inputs"]:
            errors.append(f"operators[{index}].inputs must be empty for unary operators")

    _validate_semantic_and_scope(path_ids, normalized_operators, errors)
    if decomposition is not None:
        if len(raw_paths) != len(decomposition):
            errors.append(
                "semantic_paths count must equal decomposition count "
                f"({len(raw_paths)} != {len(decomposition)})"
            )
        for index, (path, item) in enumerate(zip(raw_paths, decomposition)):
            if not isinstance(path, dict):
                continue
            goal = " ".join(str(path.get("goal", "")).split()).casefold()
            expected_goal = " ".join(str(item).split()).casefold()
            if goal != expected_goal:
                errors.append(
                    f"semantic_paths[{index}].goal must match decomposition[{index}]"
                )
    if isinstance(source_graph, dict):
        source_paths = source_graph.get("semantic_paths", [])
        source_anchors = {
            str(anchor.get("id", "")): " ".join(
                str(anchor.get("surface", "")).split()
            ).casefold()
            for anchor in source_graph.get("anchors", []) or []
            if isinstance(anchor, dict)
        }
        candidate_anchors = {
            str(anchor.get("id", "")): " ".join(
                str(anchor.get("surface", "")).split()
            ).casefold()
            for anchor in raw_anchors
            if isinstance(anchor, dict)
        }
        if isinstance(source_paths, list) and len(raw_paths) != len(source_paths):
            errors.append(
                "semantic_paths count must preserve source semantic path count "
                f"({len(raw_paths)} != {len(source_paths)})"
            )
        if isinstance(source_paths, list):
            for index, (path, source_path) in enumerate(zip(raw_paths, source_paths)):
                if not isinstance(path, dict) or not isinstance(source_path, dict):
                    continue
                source_anchor = source_anchors.get(str(source_path.get("anchor_ref", "")), "")
                candidate_anchor = candidate_anchors.get(str(path.get("anchor_ref", "")), "")
                if source_anchor and candidate_anchor and source_anchor != candidate_anchor:
                    errors.append(
                        f"semantic_paths[{index}].anchor_ref must preserve source path order"
                    )
        source_and = {
            frozenset(str(value) for value in operator.get("inputs", []) or [])
            for operator in source_graph.get("operators", []) or []
            if isinstance(operator, dict) and str(operator.get("type", "")).upper() == "AND"
        }
        candidate_and = {
            frozenset(operator["inputs"])
            for operator in normalized_operators
            if operator["type"] == "AND"
        }
        for inputs in source_and - candidate_and:
            errors.append(
                "semantic_graph is missing source structural AND for paths "
                + ",".join(sorted(inputs))
            )
    _compare_operator_requirements(
        [value for value in normalized_operators if value["type"] != "AND"],
        expected,
        "semantic_graph",
        errors,
    )
    return errors


def validate_compose_program(
    program: Any,
    *,
    semantic_graph: dict[str, Any],
    expected: list[dict[str, Any]],
    source_graph: dict[str, Any] | None = None,
    source_program: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(program, dict):
        return ["compose_program must be an object"]
    if set(program) != {"entities", "triples", "operators", "answer_var"}:
        errors.append("compose_program must contain exactly entities, triples, operators, and answer_var")

    raw_entities = program.get("entities", [])
    raw_triples = program.get("triples", [])
    raw_operators = program.get("operators", [])
    if not isinstance(raw_entities, list):
        errors.append("entities must be an array")
        raw_entities = []
    if not isinstance(raw_triples, list) or not raw_triples:
        errors.append("triples must be a non-empty array")
        raw_triples = []
    if not isinstance(raw_operators, list):
        errors.append("operators must be an array")
        raw_operators = []

    anchor_surfaces = {
        str(anchor.get("surface", "")).strip()
        for anchor in semantic_graph.get("anchors", []) or []
        if isinstance(anchor, dict)
    }
    entity_ids: set[str] = set()
    entity_surfaces: set[str] = set()
    for index, entity in enumerate(raw_entities):
        if not isinstance(entity, dict):
            errors.append(f"entities[{index}] must be an object")
            continue
        entity_id = str(entity.get("id", ""))
        surface = str(entity.get("surface", "")).strip()
        if not _ENTITY_RE.fullmatch(entity_id) or entity_id in entity_ids:
            errors.append(f"entities[{index}] has invalid or duplicate id {entity_id!r}")
        if not surface:
            errors.append(f"entities[{index}] has an empty surface")
        elif surface not in anchor_surfaces:
            errors.append(f"entities[{index}] surface {surface!r} is not a semantic anchor")
        if _DOTTED_ID_RE.fullmatch(surface):
            errors.append(f"entities[{index}] leaks a dotted KB id")
        entity_ids.add(entity_id)
        entity_surfaces.add(surface)

    variables: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    for index, triple in enumerate(raw_triples):
        location = f"triples[{index}]"
        if not isinstance(triple, dict):
            errors.append(f"{location} must be an object")
            continue
        subject = str(triple.get("subject", ""))
        obj = str(triple.get("object", ""))
        for field, term in (("subject", subject), ("object", obj)):
            if _PROGRAM_VAR_RE.fullmatch(term):
                variables.add(term)
            elif _ENTITY_RE.fullmatch(term):
                if term not in entity_ids:
                    errors.append(f"{location}.{field} refers to unknown entity {term!r}")
            elif _quoted_literal(term):
                pass
            else:
                errors.append(f"{location}.{field} has invalid reference {term!r}")
        _validate_relation_label(triple.get("relation_label"), f"{location}.relation_label", errors)
        adjacency.setdefault(subject, set()).add(obj)
        adjacency.setdefault(obj, set()).add(subject)

    answer_var = str(program.get("answer_var", ""))
    if not _PROGRAM_VAR_RE.fullmatch(answer_var) or answer_var not in variables:
        errors.append("answer_var must be a variable present in triples")
    if adjacency:
        visited: set[str] = set()
        pending = [next(iter(adjacency))]
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            pending.extend(adjacency.get(node, set()) - visited)
        if visited != set(adjacency):
            errors.append("compose triples are disconnected")

    normalized_operators: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_operators):
        if not isinstance(raw, dict):
            errors.append(f"operators[{index}] must be an object")
            continue
        operator = normalize_operator(raw)
        normalized_operators.append(operator)
        _validate_operator(
            operator,
            variables,
            f"operators[{index}]",
            errors,
            allowed_types=EXECUTABLE_OPERATOR_TYPES,
        )
        if operator["type"] == "NO_EQUAL" and operator["value"] not in entity_surfaces:
            errors.append(f"operators[{index}].value must be a compose entity surface for NO_EQUAL")
        if operator["inputs"]:
            errors.append(f"operators[{index}].inputs must be empty for unary operators")

    semantic_operators = [
        normalize_operator(value)
        for value in semantic_graph.get("operators", []) or []
        if isinstance(value, dict) and str(value.get("type", "")).upper() != "AND"
    ]
    _compare_operator_requirements(normalized_operators, semantic_operators, "compose_program", errors)
    _compare_operator_requirements(normalized_operators, expected, "compose_program", errors)
    semantic_and_operators = [
        normalize_operator(value)
        for value in semantic_graph.get("operators", []) or []
        if isinstance(value, dict) and str(value.get("type", "")).upper() == "AND"
    ]
    mappings = _semantic_program_mappings(semantic_graph, program)
    if mappings and isinstance(source_graph, dict) and isinstance(source_program, dict):
        source_mappings = _semantic_program_mappings(source_graph, source_program)
        if source_mappings:
            mappings = [
                mapping
                for mapping in mappings
                if any(
                    _mapping_equivalence_matches(mapping, source_mapping)
                    for source_mapping in source_mappings
                )
            ]
    if not mappings:
        errors.append("semantic paths cannot be mapped consistently onto compose triples")
    elif not any(
        _operators_follow_mapping(semantic_operators, normalized_operators, mapping)
        and _and_operators_follow_mapping(semantic_graph, semantic_and_operators, mapping)
        for mapping in mappings
    ):
        errors.append(
            "semantic operator variables or AND lowering are not preserved by the "
            "path-to-program mapping"
        )
    return errors


def run_external_validator(
    command: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> list[str]:
    if not command.strip():
        return []
    try:
        completed = subprocess.run(
            shlex.split(command),
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"external validator could not run: {exc}"]
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return [f"external validator exited {completed.returncode}: {detail[:1000]}"]
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return [f"external validator returned invalid JSON: {exc}"]
    if not isinstance(result, dict) or result.get("valid") is not True:
        raw_errors = result.get("errors", []) if isinstance(result, dict) else []
        if not isinstance(raw_errors, list):
            raw_errors = [str(raw_errors)]
        errors = [str(value) for value in raw_errors]
        if isinstance(result, dict):
            executions = result.get("executions", [])
            if isinstance(executions, list):
                for index, execution in enumerate(executions[:2]):
                    if not isinstance(execution, dict):
                        continue
                    diagnostic = {
                        key: execution.get(key)
                        for key in (
                            "candidate_answers",
                            "error",
                            "sparql",
                            "grounded_entities",
                            "grounded_relations",
                        )
                        if execution.get(key) not in (None, "", [], {})
                    }
                    if diagnostic:
                        errors.append(
                            f"execution[{index}]="
                            + json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)[:4000]
                        )
        return errors or ["external validator rejected candidate"]
    return []


def _validate_operator(
    operator: dict[str, Any],
    variables: set[str],
    location: str,
    errors: list[str],
    *,
    allowed_types: tuple[str, ...],
) -> None:
    operator_type = operator["type"]
    if operator_type not in allowed_types:
        errors.append(f"{location} has unsupported type {operator_type!r}")
    if operator_type == "AND":
        if operator["input_var"]:
            errors.append(f"{location}.input_var must be empty for AND")
        if len(operator["inputs"]) < 2 or len(set(operator["inputs"])) != len(operator["inputs"]):
            errors.append(f"{location}.inputs must contain at least two unique path ids")
        if _operator_has_attribute(operator):
            errors.append(f"{location} must not define attribute relations for AND")
        if operator["value"] or operator["value_type"]:
            errors.append(f"{location} must not define value fields for AND")
        return

    input_var = operator["input_var"]
    if not input_var or input_var not in variables:
        errors.append(f"{location}.input_var must refer to a graph variable")
    for value in operator["inputs"]:
        if value not in variables:
            errors.append(f"{location}.inputs refers to unknown variable {value!r}")
    labels = [
        operator["attribute_relation_label"],
        *operator["attribute_relation_labels"],
    ]
    if operator_type in _ATTRIBUTE_OPERATORS and not any(len(label) >= 2 for label in labels):
        errors.append(f"{location} requires a canonical attribute relation label")
    for label_index, label in enumerate(labels):
        if label:
            _validate_relation_label(
                label,
                f"{location}.attribute_labels[{label_index}]",
                errors,
                min_segments=2,
            )
    if operator_type in _VALUE_OPERATORS and not operator["value"]:
        errors.append(f"{location} requires a value")
    if operator_type in {"COUNT", "ARGMAX", "ARGMIN"} and (
        operator["value"] or operator["value_type"]
    ):
        errors.append(f"{location} must not define value fields for {operator_type}")
    if operator_type == "COUNT" and _operator_has_attribute(operator):
        errors.append(f"{location} must not define attribute relations for COUNT")
    if operator_type == "NO_EQUAL":
        if _operator_has_attribute(operator):
            errors.append(f"{location} must not define attribute relations for NO_EQUAL")
        if operator["value_type"] != "entity":
            errors.append(f"{location}.value_type must be 'entity' for NO_EQUAL")
    if operator_type == "EQUAL" and not operator["value_type"]:
        errors.append(f"{location}.value_type must not be empty for EQUAL")


def _validate_semantic_and_scope(
    path_ids: set[str],
    operators: list[dict[str, Any]],
    errors: list[str],
) -> None:
    and_operators = [value for value in operators if value["type"] == "AND"]
    if not and_operators:
        return
    if len(path_ids) <= 1:
        errors.append("semantic_graph must not emit AND for fewer than two semantic paths")
        return
    if len(and_operators) != 1:
        errors.append("semantic_graph must contain at most one normalized AND")
        return
    unknown_inputs = set(and_operators[0]["inputs"]) - path_ids
    if unknown_inputs:
        errors.append("semantic AND inputs must refer to existing semantic path ids")


def _operator_has_attribute(operator: dict[str, Any]) -> bool:
    return bool(operator["attribute_relation_label"] or operator["attribute_relation_labels"])


def _semantic_program_mappings(
    semantic_graph: dict[str, Any],
    program: dict[str, Any],
) -> list[dict[str, str]]:
    anchor_targets: dict[str, set[str]] = defaultdict(set)
    surfaces = {
        str(anchor.get("id", "")): str(anchor.get("surface", "")).strip()
        for anchor in semantic_graph.get("anchors", []) or []
        if isinstance(anchor, dict)
    }
    for entity in program.get("entities", []) or []:
        if not isinstance(entity, dict):
            continue
        surface = str(entity.get("surface", "")).strip()
        entity_id = str(entity.get("id", ""))
        for anchor_id, anchor_surface in surfaces.items():
            if surface == anchor_surface:
                anchor_targets[anchor_id].add(entity_id)
    for triple in program.get("triples", []) or []:
        if not isinstance(triple, dict):
            continue
        for term in (str(triple.get("subject", "")), str(triple.get("object", ""))):
            literal = _quoted_literal(term)
            if literal is None:
                continue
            for anchor_id, anchor_surface in surfaces.items():
                if literal == anchor_surface:
                    anchor_targets[anchor_id].add(term)

    triples = [triple for triple in program.get("triples", []) or [] if isinstance(triple, dict)]
    edges: list[tuple[str, tuple[str, ...], str]] = []
    for path in semantic_graph.get("semantic_paths", []) or []:
        if not isinstance(path, dict):
            continue
        for step in path.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            source_ref = str(step.get("from", ""))
            target_ref = str(step.get("to", ""))
            if str(step.get("direction", "")) == "backward":
                subject, obj = target_ref, source_ref
            else:
                subject, obj = source_ref, target_ref
            edges.append((subject, tuple(_label(step.get("relation_label", []))), obj))

    candidates: list[list[tuple[str, str]]] = []
    for subject, relation, obj in edges:
        matches: list[tuple[str, str]] = []
        for triple in triples:
            if tuple(_label(triple.get("relation_label", []))) != relation:
                continue
            candidate_subject = str(triple.get("subject", ""))
            candidate_object = str(triple.get("object", ""))
            if _reference_can_map(subject, candidate_subject, anchor_targets) and _reference_can_map(
                obj, candidate_object, anchor_targets
            ):
                matches.append((candidate_subject, candidate_object))
        if not matches:
            return []
        candidates.append(matches)

    order = sorted(range(len(edges)), key=lambda index: len(candidates[index]))
    mappings: list[dict[str, str]] = []

    def visit(position: int, mapping: dict[str, str]) -> None:
        if len(mappings) >= 128:
            return
        if position >= len(order):
            mappings.append(dict(mapping))
            return
        edge_index = order[position]
        subject, _, obj = edges[edge_index]
        for candidate_subject, candidate_object in candidates[edge_index]:
            additions: list[str] = []
            valid = True
            for semantic_ref, program_ref in (
                (subject, candidate_subject),
                (obj, candidate_object),
            ):
                existing = mapping.get(semantic_ref)
                if existing is not None and existing != program_ref:
                    valid = False
                    break
                if existing is None:
                    mapping[semantic_ref] = program_ref
                    additions.append(semantic_ref)
            if valid:
                visit(position + 1, mapping)
            for semantic_ref in additions:
                mapping.pop(semantic_ref, None)

    visit(0, {})
    return mappings


def _mapping_equivalence_matches(
    candidate: dict[str, str],
    source: dict[str, str],
) -> bool:
    """Require candidate variable sharing to agree with an available source mapping."""
    semantic_refs = sorted(
        ref
        for ref in candidate.keys() & source.keys()
        if _PATH_VAR_RE.fullmatch(ref)
    )
    for left_index, left in enumerate(semantic_refs):
        for right in semantic_refs[left_index + 1 :]:
            candidate_shared = candidate.get(left) == candidate.get(right)
            source_shared = source.get(left) == source.get(right)
            if candidate_shared != source_shared:
                return False
    return True


def _reference_can_map(
    semantic_ref: str,
    program_ref: str,
    anchor_targets: dict[str, set[str]],
) -> bool:
    if _ANCHOR_RE.fullmatch(semantic_ref):
        return program_ref in anchor_targets.get(semantic_ref, set())
    if _PATH_VAR_RE.fullmatch(semantic_ref):
        return bool(_PROGRAM_VAR_RE.fullmatch(program_ref))
    return False


def _operators_follow_mapping(
    semantic_operators: list[dict[str, Any]],
    program_operators: list[dict[str, Any]],
    mapping: dict[str, str],
) -> bool:
    remaining = list(program_operators)
    for semantic_operator in semantic_operators:
        signature = operator_signature(semantic_operator)
        mapped_input = mapping.get(semantic_operator["input_var"], "")
        mapped_inputs = [mapping.get(value, "") for value in semantic_operator["inputs"]]
        match_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if operator_signature(candidate) == signature
                and candidate["input_var"] == mapped_input
                and candidate["inputs"] == mapped_inputs
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def _and_operators_follow_mapping(
    semantic_graph: dict[str, Any],
    semantic_and_operators: list[dict[str, Any]],
    mapping: dict[str, str],
) -> bool:
    if not semantic_and_operators:
        return True
    path_outputs = {
        str(path.get("id", "")): str(path.get("path_output_var", ""))
        for path in semantic_graph.get("semantic_paths", []) or []
        if isinstance(path, dict)
    }
    for operator in semantic_and_operators:
        mapped_outputs = {
            mapping.get(path_outputs.get(path_id, ""), "")
            for path_id in operator["inputs"]
        }
        if len(mapped_outputs) != 1 or "" in mapped_outputs:
            return False
    return True


def _compare_operator_requirements(
    actual: list[dict[str, Any]],
    required: list[dict[str, Any]],
    location: str,
    errors: list[str],
) -> None:
    actual_signatures = Counter(operator_signature(value) for value in actual)
    required_signatures = Counter(operator_signature(value) for value in required)
    if actual_signatures == required_signatures:
        return
    missing = list((required_signatures - actual_signatures).elements())
    extra = [
        signature
        for signature in (actual_signatures - required_signatures).elements()
        if signature[0] not in _OPTIONAL_SOURCE_OPERATORS
    ]
    if missing:
        errors.append(f"{location} is missing required operator signatures: {missing!r}")
    if extra:
        errors.append(f"{location} has unsupported extra operator signatures: {extra!r}")


def _validate_relation_label(
    value: Any,
    location: str,
    errors: list[str],
    *,
    min_segments: int = 2,
) -> None:
    if not isinstance(value, list) or len(value) < min_segments:
        errors.append(f"{location} must contain at least {min_segments} canonical segments")
        return
    for segment in value:
        text = str(segment).strip()
        if not text:
            errors.append(f"{location} contains an empty segment")
        if "." in text:
            errors.append(f"{location} contains a dotted KB id")


def _label(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(segment) for segment in value]


def _quoted_literal(value: str) -> str | None:
    match = _RDF_LITERAL_RE.fullmatch(value)
    if match is None:
        return None
    try:
        decoded = json.loads(match.group("quoted"))
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, str) and decoded else None
