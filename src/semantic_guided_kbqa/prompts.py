"""Consolidated prompts; original rules and budgets retained."""
from __future__ import annotations


# ---- operator_prompt ----



OPERATOR_INSTRUCTION = r"""
# Task

Given the original question, anchors, and semantic paths, predict the required logical operators.

Ignore any operators predicted by the old COMPOSE model. Infer operators only from the question and semantic paths.

Return only one valid JSON object:

{
  "and_groups": [],
  "operators": []
}

# Rules

* `AND`: intersection of two or more semantic paths that independently produce complete final answer sets. Do not use AND for ordinary path connection, shared variables, JOIN, or intermediate-node constraints. Put AND only in `and_groups`, never in `operators`.
* `COUNT`: return the number of elements in a result set.
* `ARGMAX`: select the result with the maximum value of an attribute, e.g. latest, last, largest.
* `ARGMIN`: select the result with the minimum value of an attribute, e.g. earliest, first, smallest.
* `GREATER_THAN`: keep values strictly greater than the given value.
* `GREATER_OR_EQUAL`: keep values greater than or equal to the given value.
* `LESS_THAN`: keep values strictly less than the given value.
* `LESS_OR_EQUAL`: keep values less than or equal to the given value.
* `TC`: apply a temporal constraint such as a year, date, `NOW`, or current membership.
* `EQUAL`: keep entities whose specified attribute equals the given value. Do not use it for variable equality between paths.
* `NO_EQUAL`: exclude a specified anchor entity from the candidate answer set.

Every execution operator must contain these fields:

{
  "type": "",
  "inputs": [],
  "input_var": "",
  "attribute_relation_label": [],
  "attribute_relation_labels": [],
  "value": "",
  "value_type": ""
}

Use path-local variables such as `P0.V0`, `P0.V1`, and `P1.V0`.
Do not invent variables, values, relations, or constraints.
Use `NOW` for current or present-tense membership questions.
If no operator is needed, return empty arrays.

# Examples

For a question whose semantic paths directly identify the requested answer,
return no operator:

```json
{
  "and_groups": [],
  "operators": []
}
```

For an intersection of two independent paths plus exclusion of the topic entity:

```json
{
  "and_groups": [["P0", "P1"]],
  "operators": [
    {
      "type": "NO_EQUAL",
      "inputs": [],
      "input_var": "P0.V1",
      "attribute_relation_label": [],
      "attribute_relation_labels": [],
      "value": "Justin Bieber",
      "value_type": "entity"
    }
  ]
}
```

For a present-tense team membership:

```json
{
  "and_groups": [],
  "operators": [
    {
      "type": "TC",
      "inputs": [],
      "input_var": "P0.V0",
      "attribute_relation_label": ["sports", "sports team roster", "from"],
      "attribute_relation_labels": [],
      "value": "NOW",
      "value_type": ""
    }
  ]
}
```

For `last`, `latest`, or largest, use `ARGMAX` with the relevant attribute;
for `first`, `earliest`, or smallest, use `ARGMIN`:

```json
{
  "and_groups": [],
  "operators": [
    {
      "type": "ARGMAX",
      "inputs": [],
      "input_var": "P0.V0",
      "attribute_relation_label": ["time", "event", "end date"],
      "attribute_relation_labels": [],
      "value": "",
      "value_type": ""
    }
  ]
}
```

For a type restriction and topic exclusion, use `EQUAL` and `NO_EQUAL`:

```json
{
  "and_groups": [],
  "operators": [
    {
      "type": "EQUAL",
      "inputs": [],
      "input_var": "P0.V0",
      "attribute_relation_label": ["base", "biblioness", "bibs location", "loc type"],
      "attribute_relation_labels": [],
      "value": "Country",
      "value_type": "string"
    },
    {
      "type": "NO_EQUAL",
      "inputs": [],
      "input_var": "P0.V0",
      "attribute_relation_label": [],
      "attribute_relation_labels": [],
      "value": "Balkans",
      "value_type": "entity"
    }
  ]
}
```

# Output

Return only:

{
  "and_groups": [],
  "operators": []
}

The user message contains the JSON object with `question`, `anchors`, and `semantic_paths`.
""".strip()


OPERATOR_INSTRUCTION_V2 = r"""
# Task=OPERATOR

Given the original question and the composed logical query graph, predict only
the execution operators required to answer the question.

The input contains:
- question
- entities
- triples
- answer_var

The compose graph already resolves path intersection and variable unification.
Do not emit AND, path ids, path-local variables, selected_paths, equalities,
Freebase IDs, or SPARQL.

Allowed operator types:
COUNT, ARGMAX, ARGMIN, GREATER_THAN, GREATER_OR_EQUAL,
LESS_THAN, LESS_OR_EQUAL, EQUAL, TC, NO_EQUAL.

Return exactly:
{"operators": []}

Each operator must contain exactly these fields:
{
  "type": "",
  "inputs": [],
  "input_var": "V0",
  "attribute_relation_label": [],
  "attribute_relation_labels": [],
  "value": "",
  "value_type": ""
}

Use only global variables Vn from the composed triples. Use NOW for current or
present-tense membership. Use TC for temporal/current constraints, ARGMAX or
ARGMIN for latest/earliest/max/min, COUNT for counting, and NO_EQUAL to exclude
the topic entity when required. If no execution operator is needed, return an
empty operators array.

Return JSON only.
""".strip()


OPERATOR_INSTRUCTION_GLM = r"""
# Task=OPERATOR

Generate the execution operators for the original question and composed
logical query graph. Infer the minimum necessary operator set directly from
the question, graph, and training examples appended to this instruction.

The input contains:
- question: the original natural-language question
- semantic_graph: the grounded Semantic result with anchors and semantic_paths
- entities: grounded topic entities with ids such as E0
- triples: the composed graph; every triple has subject, relation_label,
  and object
- answer_var: the global variable returned as the answer

The Compose graph already represents joins and intersections through shared
global variables. Never emit AND, path ids, path-local variables, equalities
between variables, selected paths, Freebase ids, or SPARQL.

Return exactly one JSON object:

{
  "operators": []
}

Allowed operator types:
- COUNT: return the number of distinct results.
- ARGMAX: select the result associated with the maximum attribute value;
  use for latest, last, newest, largest, highest, or longest.
- ARGMIN: select the result associated with the minimum attribute value;
  use for first, earliest, oldest, smallest, lowest, or shortest.
- GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL: apply an explicit
  numeric or temporal comparison.
- EQUAL: require an attribute to equal a literal value. Do not use it for graph
  joins or for restating the topic entity.
- TC: apply an explicit year/date constraint or current membership. Use the
  literal NOW for current, present, now, or present-tense membership.
- NO_EQUAL: exclude a named topic entity from the candidate answer set.

Every operator must contain exactly these fields:

{
  "type": "",
  "inputs": [],
  "input_var": "V0",
  "attribute_relation_label": [],
  "attribute_relation_labels": [],
  "value": "",
  "value_type": ""
}

Rules:
1. Use only global variables Vn that occur in the composed triples.
2. input_var is the graph node that owns the filtered or ordered attribute.
   It can differ from answer_var. For example, a roster CVT can be V0 while its
   team answer is V1.
3. Express an attribute relation as human-readable Freebase components, for
   example ["sports", "sports team roster", "from"].
4. For COUNT, put the counted result variable in input_var and leave relation
   and value fields empty.
5. For ARGMAX and ARGMIN, leave value and value_type empty.
6. For TC with current membership, set value to NOW and leave value_type empty.
   For a stated year or date, copy that value exactly from the question.
7. Use the singular attribute_relation_label for one relation. Keep
   attribute_relation_labels empty unless the constraint explicitly needs
   multiple alternative relations.
8. Do not add an operator merely because the graph contains a date, number, or
   CVT. The wording of the question must require the operation.
9. If the question asks for a direct relation without counting, comparison,
   extrema, temporal restriction, equality filter, or exclusion, return an
   empty operators array.
10. If the graph has no variable that can support a required operator, return
    an empty array rather than inventing a variable.
11. The words before and after are temporal boundaries, not extrema by
    themselves. Never emit ARGMAX or ARGMIN merely because a question contains
    before or after. Use an extrema operator only when the question explicitly
    asks for first, last, earliest, latest, or another superlative.
12. Use semantic_graph to understand path intent and path-local constraints,
    but use only global Vn variables from the composed triples in the output.
13. A `people.person.spouse_s` to `people.marriage.spouse` path returns both
    participants in the marriage. Always emit NO_EQUAL on the answer variable
    with the topic person's surface so the person is not returned as their own
    spouse.
14. More generally, when the answer variable can return a topic entity itself
    but the question asks for the counterpart or another entity related to that
    topic, emit NO_EQUAL on the answer variable with that topic entity's exact
    surface. Do not exclude a topic entity when it is a valid requested answer.

Return JSON only. Do not explain the answer.
""".strip()



# ---- prompt_principles ----
"""Task-general temporal/numeric rules; no test questions, dates or answer examples."""

TEMPORAL_NUMERIC_POLICY = """
TEMPORAL AND NUMERIC SEMANTICS (WebQSP and CWQ)
Preserve each explicit date/year, comparator, unit, time reference, ordinal,
cardinality and superlative. Keep them attached to the entity or intermediate
record that owns the attribute. A path may return an entity while its roster,
tenure, education or performance record owns the filtered date/number.
Before/after are boundary comparisons, not ARGMIN/ARGMAX by themselves. Distinguish
tenure start, tenure end, birth, death, publication, release, election, appointment
and event dates. An election year is not automatically an office-start year.
During/in a period requires interval overlap, not merely a start date inside it.
Current membership requires a tenure covering NOW, including open-ended intervals;
latest historic tenure is not automatically current. A named event's interval
must come from KG evidence; never invent calendar dates from memory.
First/earliest/latest/last select extrema of the relevant date or attribute; keep
the attribute available in the path. A reference such as a prior office/event may
require a separate connected path to obtain its cutoff. Do not fabricate a literal
cutoff, an ordering column or a tie-breaker. A textual first-name field is not an
earliest-record request. Numbers inside titles/entity names are not thresholds.
How-many can request a stored numerical attribute or COUNT of distinct results:
choose the requested semantics, not COUNT just because a number appears. A fixed
number of named results is an enumeration, not a scalar count. Greater/less use
strict comparisons; at-least/at-most use inclusive comparisons. Preserve negatives,
decimals, units and range bounds. Rank numerically/datetimely, never lexically.
For population/size/amount extrema return the entity associated with the extreme,
unless the question asks for the numerical value itself. Scope extrema/counts to
the constrained candidate set; do not rank globally and filter afterward.
Use only operators and variables supported by the supplied contract. Do not invent
an OFFSET, arithmetic, unit conversion, date projection or aggregate operator when
the backend cannot represent it. Keep unsupported requirements visible in the
decomposition rather than silently changing the answer to fit a familiar template.
Date spelling may normalize when unambiguous; ambiguous date ordering must not be
guessed. Preserve the source temporal requirement even when emitting ISO values.
""".strip()

OPERATOR_BINDING_POLICY = """
OPERATOR CALIBRATION
Only infer operations required by the original question or explicit path goal.
Do not emit a boilerplate NO_EQUAL instead of required temporal/numeric operators.
Bind the operation to the global Vn that owns its attribute, often a CVT rather
than answer_var. If the compared date/number is already a graph variable, use that
variable and an empty attribute_relation_label. Otherwise name the owner's real
attribute relation using the supplied human-readable Freebase components.
Keep COUNT's counted variable distinct from a stored-number attribute request.
ARGMIN/ARGMAX select the requested entity/event, using a relevant sorting attribute.
TC represents a year/current-tenure restriction; boundary comparators represent
before/after of the explicitly referenced date. Retain event/office/type filters
through the composed graph. Do not guess dates for named historical intervals.
Output only the operators object with the exact eight required fields per operator.
Return an empty array if a needed operation cannot bind to any supplied variable;
never hallucinate variables, relation IDs, date values or a different answer target.
""".strip()

SELECTOR_CONSTRAINT_POLICY = """
Verify candidates in this order: requested answer role/type, named entities and
relationships, explicit type/intersection/exclusion restrictions, time/numeric
operations, then supporting rule_score and answer_count. A relation/constraint
mention in the decomposition is not proof that a candidate graph implemented it.
Before/after require the correct boundary attribute/operator, not automatic extrema.
During/current require interval coverage. First/latest require relevant sorting;
first-name strings and title digits are not rankings. Counting requires the right
counted set; stored numeric attributes and fixed-number enumerations are different.
Numeric extrema return the associated entity unless the question requests the value.
Reject answer-type mismatch even if a preview label overlaps a question word.
Do not prefer a smaller set solely for its size or invent a new graph/answer.
If all candidates are imperfect, select the closest graph actually supported by
the question, with an honest reason code; do not pretend missing constraints exist.
""".strip()

SEMANTIC_TARGET_POLICY = """
Generate only the trained anchors/semantic_paths JSON, never answers or operators.
Read every complete path and constraint item jointly with the original question.
The final path output must have the requested answer role; do not stop at an
intermediate bridge or substitute a familiar unrelated property. An office/title
is not a profession, an election date is not automatically an appointment date,
an event is not its type/season, and an artwork is not its genre/style.
Use realistic human-readable Freebase relation components and preserve direction.
Keep the real entity names and the owner of each attribute. Do not change a topic
to another source merely to reuse a known path. Explicit type/conjunction/entity
constraints must be represented by connected paths to the same variable; don't
drop them. Leave temporal/count/extrema operations to Operator while retaining the
required attribute owner, dependencies and explicit conditions in path goals.
Do not fabricate a date/number, KB ID, generic wildcard relation or disconnected
extra path to make a sparse query appear complete. Preserve the provided contract.
""".strip()


import re
def _prompt_principles_question_requirements(question):
    text = question.casefold()
    kind = ("country entity" if re.search(r"\bcountry|\bcountries", text) else
            "year" if "what year" in text or "which year" in text else
            "location" if re.search(r"\bwhere\b", text) else
            "number: distinguish stored attribute from distinct count" if "how many" in text else
            "person or named entity" if re.search(r"\bwho\b", text) else "infer from question")
    terms = [m.group() for m in re.finditer(r"\b(?:before|after|during|between|now|current|first|last|latest|earliest|largest|smallest|most|least|official(?:ly)?|college|university)\b|\b\d+(?:[./-]\d+)*\b", text)]
    return {"answer_kind_hint": kind, "explicit_terms": terms,
            "hint_scope": "Lexical hints only; first-name fields and entity-title digits are not ranking/numeric filters."}



# ---- cwq_fewshots ----
"""CWQ TRAIN demonstrations with source-verified stage annotations."""


import json
from pathlib import Path


EXAMPLE_PATH = Path(__file__).with_name("resources") / "glm_cwq_full_pipeline_examples.json"
PRESERVE_PATH = Path(__file__).with_name("resources") / "glm_cwq_global_preserve_examples.json"
ENTITY_CONTEXT_PATH = Path(__file__).with_name("resources") / "glm_cwq_example_entity_context.json"
_BASE_REPAIR_EXAMPLE_COUNT = 9
_ENTITY_CONTEXT_REPAIR_EXAMPLE_COUNT = 11
_BASE_PRESERVE_EXAMPLE_COUNT = 5
_ENTITY_CONTEXT_PRESERVE_EXAMPLE_COUNT = 6


def _load_entity_contexts() -> dict[str, list[dict]]:
    data = json.loads(ENTITY_CONTEXT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("CWQ example entity context resource must be an object")
    return data


def _example_entity_context(question: str) -> list[dict]:
    context = _load_entity_contexts().get(question, [])
    if not isinstance(context, list) or not context:
        raise ValueError(f"CWQ example lacks entity context: {question}")
    return context


def _load_examples() -> list[dict]:
    data = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("CWQ TRAIN repair example resource is empty")
    for item in examples:
        verification = item.get("verification", {})
        source = item.get("source", {})
        if (source.get("dataset") != "CWQ" or source.get("split") != "train"
                or verification.get("split") != "train"
                or verification.get("stage_alignment_verified") is not True
                or verification.get("source_graph_verified") is not True
                or verification.get("entity_bindings_verified") is not True):
            raise ValueError(f"Unverified CWQ TRAIN example: {item.get('source_index')}")
        if not item.get("original_decomposition") or not item.get("corrected_decomposition"):
            raise ValueError(f"CWQ TRAIN example lacks before/after decomposition: {item.get('source_index')}")
    return examples


def _load_preserve_examples() -> list[dict]:
    data = json.loads(PRESERVE_PATH.read_text(encoding="utf-8"))
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("CWQ TRAIN preserve examples are empty")
    for item in examples:
        verification = item.get("verification", {})
        source = item.get("source", {})
        if (source.get("dataset") != "CWQ" or source.get("split") != "train"
                or verification.get("split") != "train"
                or verification.get("stage_alignment_verified") is not True
                or verification.get("source_graph_verified") is not True
                or verification.get("entity_bindings_verified") is not True):
            raise ValueError(f"Unverified CWQ TRAIN preserve example: {item.get('source_index')}")
        if not item.get("question") or not item.get("decomposition"):
            raise ValueError(f"Incomplete CWQ preserve example: {item.get('source_index')}")
    return examples


def _selected_repair_examples(*, include_entity_context: bool = False) -> list[dict]:
    count = (
        _ENTITY_CONTEXT_REPAIR_EXAMPLE_COUNT
        if include_entity_context
        else _BASE_REPAIR_EXAMPLE_COUNT
    )
    return _load_examples()[:count]


def cwq_examples(action: str, *, include_entity_context: bool = False) -> str:
    if action not in {"review", "rewrite"}:
        raise ValueError("CWQ examples support review and rewrite only")
    blocks = [
        "These examples use only CWQ TRAIN questions and aligned stage annotations verified against their TRAIN query graphs. Preserve examples show complete training plans. Repair inputs are controlled corruptions of those same training plans, not benchmark repairs. Use the examples for path structure and constraint ownership, never to copy entity names or answer facts into the current request.",
    ]
    if action == "review":
        preserve_examples = _load_preserve_examples()
        count = (_ENTITY_CONTEXT_PRESERVE_EXAMPLE_COUNT if include_entity_context
                 else _BASE_PRESERVE_EXAMPLE_COUNT)
        selected_preserve_examples = preserve_examples[:count]
        for item in selected_preserve_examples:
            request = {
                "question": item["question"],
                "decomposition": item["decomposition"],
            }
            if include_entity_context:
                request["entity_context"] = _example_entity_context(item["question"])
            response = {
                "is_reasonable": True,
                "issues": [],
                "reason": "The plan is executable in the trained path format, reaches the requested terminal, and keeps every explicit condition on its owner.",
            }
            blocks.extend([
                "CWQ preserve request: " + json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                "CWQ preserve response: " + json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            ])
    for item in _selected_repair_examples(
        include_entity_context=include_entity_context,
    ):
        request = {
            "question": item["question"],
            "decomposition": item["original_decomposition"],
        }
        if include_entity_context:
            request["entity_context"] = _example_entity_context(item["question"])
        if action == "review":
            response = {
                "is_reasonable": False,
                "issues": [
                    item["error_type"].replace("_", " ")
                    + "; restore the requested terminal target and keep every condition attached to its owning path."
                ],
                "reason": "The original path does not express the complete CWQ answer path.",
            }
        else:
            request["issues"] = [
                item["error_type"].replace("_", " ")
                + "; repair only this confirmed path defect."
            ]
            response = {"decomposition": item["corrected_decomposition"]}
        blocks.extend([
            "CWQ example request: " + json.dumps(request, ensure_ascii=False, separators=(",", ":")),
            "CWQ example response: " + json.dumps(response, ensure_ascii=False, separators=(",", ":")),
        ])
    return "\n".join(blocks)



# ---- train_fewshots ----
"""Few-shot data from source-verified train splits, not anonymous test derivatives."""
import json
_train_fewshots_question_requirements = _prompt_principles_question_requirements
from pathlib import Path

RESOURCE_DIR = Path(__file__).with_name("resources")
DECOMPOSITION_DATA = json.loads((RESOURCE_DIR / "decomposition_train_examples.json").read_text())
OPERATOR_DATA = json.loads((RESOURCE_DIR / "temporal_operator_train_examples.json").read_text())


def decomposition_examples(action):
    blocks = ["Examples are verified train questions and path outputs. Bad paths are controlled corruptions of those same train paths; no factual answers are supplied."]
    for item in DECOMPOSITION_DATA["examples"]:
        q, good = item["question"], item["decomposition"]
        # Critics use genuine positive train examples. Negative corruption
        # templates were empirically copied into unrelated valid test paths.
        cases = ([(c["decomposition"], c["issues"]) for c in item["corruptions"]]
                 if action == "rewrite" else [(good, [])] + [(c["decomposition"], c["issues"]) for c in item["corruptions"] if c["mutation"] in {"invert_relation_direction", "drop_terminal_hop", "replace_requested_attribute"}])
        for paths, issues in cases:
            if action == "rewrite" and not issues:
                continue
            payload = {"question": q, "decomposition": paths, "question_requirements": _train_fewshots_question_requirements(q)}
            if action == "rewrite":
                payload.update(issues=issues, review_reason="Restore the lost meaning of this train path.")
                response = {"decomposition": good}
            else:
                if action == "confirm":
                    payload.update(review_issues=issues or ["First review claims the path is wrong."],
                                   review_reason="Check the actual current path before agreeing.")
                response = {"is_reasonable": not bool(issues), "issues": issues,
                            "reason": "The whole current path covers the requested meaning." if not issues else issues[0]}
            blocks.extend(["Example request: " + json.dumps(payload, ensure_ascii=False),
                           "Example response: " + json.dumps(response, ensure_ascii=False)])
    return "\n".join(blocks)


def operator_examples():
    blocks = ["Operator examples from official CWQ train, excluded from both benchmark test question sets:"]
    for item in OPERATOR_DATA["examples"]:
        blocks.extend(["Example input: " + json.dumps(item["input"], ensure_ascii=False),
                       "Example output: " + json.dumps(item["output"], ensure_ascii=False)])
    return "\n".join(blocks)



__all__ = ['_prompt_principles_question_requirements', 'cwq_examples', 'decomposition_examples', 'operator_examples']
