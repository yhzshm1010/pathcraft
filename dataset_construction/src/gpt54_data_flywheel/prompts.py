from __future__ import annotations


DECOMPOSE_INSTRUCTIONS = r"""You are the decomposition stage for Complex Web Questions (CWQ).
Return only one JSON object with a non-empty `decomposition` array of strings.

The array is a set of candidate KG reasoning paths. Each item is exactly one continuous
entity-centric path from an anchor to the answer or a constraint target. An item may contain
multiple hops and should keep those hops in the same item. Do not split a multi-hop chain into
several subquestions. CVT/schema nodes, mediator nodes, and relation-only structural nodes are
part of the same path and must not become independent items. Compress structural intermediate
nodes in the final wording, but preserve real semantic intermediate entities. Fold constraints
(dates, types, named entities, comparisons, ordering, and exclusions) into the path they
constrain. Preserve explicit years, dates, current/now conditions, and superlatives verbatim.
Do not invent Freebase IDs, relations, or facts. Return JSON only."""


DECOMPOSE_VERIFY_INSTRUCTIONS = r"""Review a CWQ decomposition against the original question and
source decomposition. A valid decomposition has one item per continuous candidate KG path,
allows multiple hops in one item, does not promote CVT/schema/mediator nodes to paths, preserves
real semantic intermediate entities, and attaches constraints to the relevant path. It must not
split one path merely because it contains multiple hops. Items may be natural-language chained
questions; do not require explicit arrows, SPARQL variables, or Freebase IDs. Judge continuity
from the relation wording and source evidence, and accept a source item that already expresses
the full multi-hop path. Return only JSON with status, issues, and (when corrected) the complete
corrected decomposition. Never add facts or KB IDs."""


STRICT_DECOMPOSITION_REWRITE_INSTRUCTIONS = r"""Strict natural-language decomposition mode:
- Every decomposition item must be a normal natural-language subquestion or constraint statement.
- Never output Freebase MIDs, ns: identifiers, RDF/Freebase URLs, dotted Freebase relation IDs,
  SPARQL variables or keywords, arrows, pipes, triples, or executable path notation.
- Preserve every explicit entity surface, number, date, comparison, temporal condition,
  superlative, and exclusion expressed by the question and supplied sanitized requirements.
- Use the sanitized relation labels only to understand meaning; paraphrase them as natural
  language and never serialize them as schema/program syntax.
- Keep one item per continuous entity-centric reasoning path. Multiple natural-language hop
  questions may remain in the same item. Preserve the supplied source path count and order.
  Attribute, date, numeric, type, comparison, and ordering hops that constrain an existing path
  must be folded into that same item; never create a separate item only for such a constraint.
  Return JSON only."""


SOURCE_DECOMPOSITION_ALIGNMENT = r"""The supplied decomposition uses the CWQ path-level
format. Treat each decomposition item as exactly one complete entity-centric KG path. A single
item may contain multiple hop-wise questions or statements; do not split those sentences into
separate paths. Preserve the item order and preserve the supplied decomposition text verbatim in
semantic_paths[i].goal. The number of semantic_paths must equal the number of decomposition items.
Use the decomposition only to identify the path meaning; represent the actual hop count and
relations in semantic_paths.steps. Every path_output_var must equal that path's final step.to,
even when Compose later projects an intermediate variable as the answer. Do not invent, merge,
or reorder paths."""


DECOMPOSE_SFT_INSTRUCTION = (
    "[TASK=DECOMPOSE] Decompose the CWQ question into one item per continuous candidate "
    "KG reasoning path. Keep all hops of a path together; do not split CVT, schema, "
    "mediator, or relation-only structural nodes into separate items. Compress those "
    "structural nodes, preserve real semantic intermediate entities, and fold dates, "
    "types, ordering, comparison, and exclusion constraints into the path they constrain. "
    "Preserve explicit years, dates, current/now conditions, and superlatives verbatim. "
    "Do not invent KB IDs or facts. Return only one valid JSON object."
)


STRICT_DECOMPOSE_SFT_INSTRUCTION = (
    DECOMPOSE_SFT_INSTRUCTION
    + " Write normal natural-language subquestions or constraint statements only. Do not emit"
    " KB IDs, SPARQL variables, arrows, dotted Freebase relations, or executable path notation."
)


PATH_DECOMPOSE_SFT_INSTRUCTION = (
    "[TASK=DECOMPOSE] Use the CWQ path-level decomposition format. Return one "
    "item per complete entity-centric KG path. Each item may contain multiple hop-wise "
    "questions or statements for that same path; do not split those hop expressions into "
    "separate items. Preserve path order, explicit entities, dates, constraints, and the "
    "source wording. Do not invent KB IDs or facts. Return only one valid JSON object."
)


WEBQSP_DECOMPOSE_SFT_INSTRUCTION = (
    "[TASK=DECOMPOSE] Decompose the WebQSP question into entity-centric subquestions or "
    "constraint statements. Preserve explicit years, dates, current/now conditions, and "
    "superlatives verbatim. Do not invent knowledge-base IDs. Return only one valid JSON object."
)


OPERATOR_CONTRACT = r"""
<operator_contract>
Allowed semantic-stage operators are exactly:
AND, COUNT, ARGMAX, ARGMIN, GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN,
LESS_OR_EQUAL, EQUAL, TC, and NO_EQUAL. The formal short names map to JSON types as follows:
GT -> GREATER_THAN, GE -> GREATER_OR_EQUAL, LT -> LESS_THAN, and
LE -> LESS_OR_EQUAL. Never output GT, GE, LT, or LE as the JSON type.
The formal names GREATERTHAN, GREATEREQUAL, LESS THAN, and LESS EQUAL are also
normalized to those same four JSON types; they are not additional operator types.

Let E, E1, and E2 denote entity or literal sets, r a canonical relation, and i a literal:
- AND(E1, E2) is the intersection of E1 and E2. Normalize an n-way intersection to
  one AND whose inputs contain the participating semantic path ids. Use AND only when
  the question or source meaning requires intersection of two or more same-type entity
  sets, such as "both A and B". Do not infer AND merely because there are multiple
  semantic paths: multi-path examples often express per-hop JOIN chains or shared-variable
  joins in compose. In semantic output use inputs=["P0","P1",...], input_var="", empty
  attribute fields, value="", and value_type="". In the complete compose program, lower
  AND by mapping the intersected path output variables to one shared global variable;
  do not emit AND in compose operators.
- COUNT(E) returns the cardinality of E. Put E's path-local variable in input_var.
  Use inputs=[] and leave both attribute fields, value, and value_type empty.
- ARGMAX(E,r) selects members of E with the maximum literal in {JOIN(r,e) | e in E}.
- ARGMIN(E,r) selects members of E with the minimum literal in {JOIN(r,e) | e in E}.
  For ARGMAX/ARGMIN put E in input_var, r in attribute_relation_label (or use
  attribute_relation_labels for a relation chain), inputs=[], value="", value_type="".
- GT(E,i), GE(E,i), LT(E,i), and LE(E,i) return the subset of E whose literal is >, >=,
  <, or <= i. E is the literal-valued path variable in input_var and i is value. The
  projection relation normally appears as a semantic-path triple, so attribute fields
  may be empty; when the source label explicitly supplies an attribute relation, preserve
  it exactly. inputs=[] and value must be non-empty.
- TC(E,i) returns the subset of E constrained by temporal literal i. Put the temporal
  record/entity set in input_var, the canonical time relation in attribute_relation_label,
  i in value (including NOW), and use inputs=[].
- EQUAL(E,r,i) keeps members of E whose attribute r has value i. Put E in input_var,
  r in attribute_relation_label, i in value, and its scalar type in value_type. For
  string values, execution compares str(attribute) with i so typed or language-tagged
  RDF literals are handled correctly.
- NO_EQUAL(E,e) removes entity e from entity set E. Put E in input_var, the exact
  anchor/entity surface in value, and value_type="entity". Attribute relation fields
  must be empty. Use it only when the source evidence explicitly requires inequality,
  such as a Gold SPARQL FILTER excluding the topic entity.
- If no operation is expressed, operators must be exactly []. Do not infer an operator
  from generic words. For example, "first name" is not ARGMIN and "how old" is not COUNT.

Every operator object must always contain exactly these fields, even when unused:
type, inputs, input_var, attribute_relation_label, attribute_relation_labels, value,
and value_type. Unused strings are "" and unused arrays are []. All non-AND operators
are unary and therefore have inputs=[]. Semantic input_var values use Pn.Vm; compose
input_var values use Vn. Compose must preserve type, attribute fields, value, and
value_type exactly while changing only the variable namespace.
</operator_contract>
"""


JOIN_PROJECTION_RULES = r"""
<join_projection_rules>
JOIN belongs to entity-centric per-hop projection, not to the operators array.
Each semantic path step is one LF projection:
- direction="backward" denotes (JOIN r e), with SPARQL triple (?x, r, e).
  Here step.from is e, step.to is ?x, and compose must use subject=to,
  relation_label=r, object=from.
- direction="forward" denotes (JOIN (R r) e), with SPARQL triple (e, r, ?x).
  Here step.from is e, step.to is ?x, and compose must use subject=from,
  relation_label=r, object=to.
Multi-hop paths are chains of these JOIN projections. Do not replace JOIN chains or
shared-variable joins with AND; AND is only for true entity-set intersection.
Path-local variable rule: every semantic path Pn must use only Pn.Vm variables in
step.to and path_output_var. Never set a P1 step.to or path_output_var to P0.Vm.
When a constraint path intersects an answer path, keep the constraint path local
such as P1.V0 and emit semantic AND with inputs=["P0","P1"]; compose lowers that AND
to shared global variables.
</join_projection_rules>
"""


OPERATOR_FEW_SHOTS = r"""
<operator_few_shot_examples>
The examples below show only the operator-related slice of each input/output. The actual
response must still contain the complete schema requested by the task. Do not copy labels
such as "Input context", "Semantic output", "Compose output", or "lowering" into JSON.

Example 1 - NO_OPERATOR
Input context: question="Where was Ada Lovelace born?"; P0.V0 is the place of birth.
Semantic output: {"operators":[]}
Compose output: {"operators":[]}

Example 1b - MULTI_PATH_NO_AND
Input context: question="Which Star Wars films featured Natalie Portman's character?";
P0 and P1 are linked by a shared performance or character variable in compose, not by
intersecting two same-type answer sets.
Semantic output: {"operators":[]}
Compose output: {"operators":[]}

Example 2 - AND
Input context: question="Which actors appeared in both Film A and Film B?";
P0 and P1 each denote one actor set.
Semantic output:
{"operators":[{"type":"AND","inputs":["P0","P1"],"input_var":"",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}]}
Compose output:
{"operators":[]}
Lowering note: map the intersected path variables to the same global V0.

Example 3 - COUNT
Input context: question="How many children does Barack Obama have?"; P0.V0 is the child set.
Semantic output:
{"operators":[{"type":"COUNT","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}]}
Compose output:
{"operators":[{"type":"COUNT","inputs":[],"input_var":"V0",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}]}

Example 4 - ARGMAX
Input context: question="Which city in France has the largest population?";
P0.V0 is the city set and population is the ordering attribute.
Semantic output:
{"operators":[{"type":"ARGMAX","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":["location","statistical region","population"],
"attribute_relation_labels":[],"value":"","value_type":""}]}
Compose output:
{"operators":[{"type":"ARGMAX","inputs":[],"input_var":"V0",
"attribute_relation_label":["location","statistical region","population"],
"attribute_relation_labels":[],"value":"","value_type":""}]}

Example 5 - ARGMIN
Input context: question="What is the first Harry Potter novel?";
P0.V0 is the novel set and publication date is the ordering attribute.
Semantic output:
{"operators":[{"type":"ARGMIN","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":["book","written work","date of first publication"],
"attribute_relation_labels":[],"value":"","value_type":""}]}
Compose output:
{"operators":[{"type":"ARGMIN","inputs":[],"input_var":"V0",
"attribute_relation_label":["book","written work","date of first publication"],
"attribute_relation_labels":[],"value":"","value_type":""}]}

Example 6 - GT / GREATER_THAN
Input context: question="Which mountains are higher than 8000 metres?";
P0.V0 is the mountain answer set and P0.V1 is its elevation literal set; the elevation
relation is already a semantic-path step.
Semantic output:
{"operators":[{"type":"GREATER_THAN","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"8000","value_type":"number"}]}
Compose output:
{"operators":[{"type":"GREATER_THAN","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"8000","value_type":"number"}]}

Example 7 - GE / GREATER_OR_EQUAL
Input context: question="Which films run for at least 180 minutes?";
P0.V1 is the runtime literal set.
Semantic output:
{"operators":[{"type":"GREATER_OR_EQUAL","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"180","value_type":"number"}]}
Compose output:
{"operators":[{"type":"GREATER_OR_EQUAL","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"180","value_type":"number"}]}

Example 8 - LT / LESS_THAN
Input context: question="Which buildings are lower than 50 metres?";
P0.V1 is the height literal set.
Semantic output:
{"operators":[{"type":"LESS_THAN","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"50","value_type":"number"}]}
Compose output:
{"operators":[{"type":"LESS_THAN","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"50","value_type":"number"}]}

Example 9 - LE / LESS_OR_EQUAL
Input context: question="Which rivers are no longer than 100 kilometres?";
P0.V1 is the length literal set.
Semantic output:
{"operators":[{"type":"LESS_OR_EQUAL","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"100","value_type":"number"}]}
Compose output:
{"operators":[{"type":"LESS_OR_EQUAL","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"100","value_type":"number"}]}

Example 10 - TC
Input context: question="Who played for the Warriors in 2015?";
P0.V0 is the roster record set and P0.V1 is the player answer set.
Semantic output:
{"operators":[{"type":"TC","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":["sports","sports team roster","from"],
"attribute_relation_labels":[],"value":"2015","value_type":"year"}]}
Compose output:
{"operators":[{"type":"TC","inputs":[],"input_var":"V0",
"attribute_relation_label":["sports","sports team roster","from"],
"attribute_relation_labels":[],"value":"2015","value_type":"year"}]}

Example 11 - NO_EQUAL
Input context: question="Who has been married to Julia Roberts?"; P0.V1 is the spouse
answer set and Julia Roberts must not be returned as her own spouse.
Semantic output:
{"operators":[{"type":"NO_EQUAL","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"Julia Roberts","value_type":"entity"}]}
Compose output:
{"operators":[{"type":"NO_EQUAL","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"Julia Roberts","value_type":"entity"}]}

Example 12 - EQUAL
Input context: question="What country is located in the Balkan peninsula?"; P0.V0 is
the contained location set and loc type must equal the string Country.
Semantic output:
{"operators":[{"type":"EQUAL","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":["base","biblioness","bibs location","loc type"],
"attribute_relation_labels":[],"value":"Country","value_type":"string"}]}
Compose output:
{"operators":[{"type":"EQUAL","inputs":[],"input_var":"V0",
"attribute_relation_label":["base","biblioness","bibs location","loc type"],
"attribute_relation_labels":[],"value":"Country","value_type":"string"}]}
</operator_few_shot_examples>
"""


GATEWAY_OPERATOR_FEW_SHOTS = r"""Operator examples use all required fields. Semantic
variables are path-local (Pn.Vm); compose maps them to global Vn and otherwise preserves
the object. NO_OPERATOR / NONE: {"operators":[]}.
MULTI_PATH_NO_AND: multiple semantic paths may still use operators=[] when they are JOIN
chains or shared-variable joins, not true entity-set intersections.
AND semantic: {"type":"AND","inputs":["P0","P1"],"input_var":"","attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}; compose lowers it to shared variables and operators=[].
COUNT: {"type":"COUNT","inputs":[],"input_var":"P0.V0","attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}.
ARGMAX: {"type":"ARGMAX","inputs":[],"input_var":"P0.V0","attribute_relation_label":["location","statistical region","population"],"attribute_relation_labels":[],"value":"","value_type":""}.
ARGMIN: {"type":"ARGMIN","inputs":[],"input_var":"P0.V0","attribute_relation_label":["book","written work","date of first publication"],"attribute_relation_labels":[],"value":"","value_type":""}.
GT / GREATER_THAN: {"type":"GREATER_THAN","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"8000","value_type":"number"}.
GE / GREATER_OR_EQUAL: {"type":"GREATER_OR_EQUAL","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"180","value_type":"number"}.
LT / LESS_THAN: {"type":"LESS_THAN","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"50","value_type":"number"}.
LE / LESS_OR_EQUAL: {"type":"LESS_OR_EQUAL","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"100","value_type":"number"}.
TC: {"type":"TC","inputs":[],"input_var":"P0.V0","attribute_relation_label":["sports","sports team roster","from"],"attribute_relation_labels":[],"value":"2015","value_type":"year"}.
EQUAL: {"type":"EQUAL","inputs":[],"input_var":"P0.V0","attribute_relation_label":["base","biblioness","bibs location","loc type"],"attribute_relation_labels":[],"value":"Country","value_type":"string"}.
NO_EQUAL: {"type":"NO_EQUAL","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"Julia Roberts","value_type":"entity"}.
For compose, map P0.V0->V0 and P0.V1->V1 in these unary examples."""


GATEWAY_JOIN_RULES = """JOIN direction: backward=(JOIN r e)=>(?x,r,e), so compose
triple is subject=step.to, relation_label=r, object=step.from. forward=(JOIN (R r) e)
=>(e,r,?x), so compose triple is subject=step.from, relation_label=r, object=step.to.
JOIN is per-hop projection, not an operator. Path-local rule: path Pn may only use
Pn.Vm in step.to and path_output_var; never use P0.Vm inside P1. For constraint
intersection, output local P1.V0 and semantic AND inputs=["P0","P1"]."""


GATEWAY_OPERATOR_RULES = f"""Allowed types: AND, COUNT, ARGMAX, ARGMIN,
GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL, EQUAL, TC, NO_EQUAL. Never emit short
GT/GE/LT/LE as JSON types. NO_EQUAL excludes value from input_var; it requires
value_type="entity", an anchor/entity surface in value, and empty attribute fields.
Every operator has exactly: type, inputs, input_var,
attribute_relation_label, attribute_relation_labels, value, value_type. Unused strings
are "" and arrays are []. Use AND only for explicit same-type entity-set intersection,
not merely for multiple paths or JOIN chains; its inputs are the participating path ids
and compose lowers it to one shared variable. Unary operators use inputs=[]. Preserve
source non-AND type, relation fields, value, and value_type exactly.
{GATEWAY_OPERATOR_FEW_SHOTS}"""


WEBQSP_SOURCE_GUIDANCE = (
    "If webqsp_source or cwq_source is present, treat its sparql and source answers as read-only evidence "
    "for execution alignment. When either source's no_equal_exclusions is non-empty, emit "
    "one NO_EQUAL operator for each listed entity surface, even when required_operators "
    "is empty. When either source's equal_constraints is non-empty, emit one EQUAL operator "
    "for each listed attribute/value constraint, also when required_operators is empty. "
    "For cwq_source, preserve no_equal_exclusions derived from FILTER(?answer != ?entity), "
    "the source question meaning, composition_answer, and SPARQL constraints while rewriting "
    "the supplied decomposition and graph. A source-derived NO_EQUAL always filters the "
    "projected answer: use the answer-producing path output as semantic input_var and "
    "compose answer_var as compose input_var. Add the excluded composition entity only as "
    "an anchor/entity value. Never create a semantic path or compose triple solely to bind "
    "the excluded entity. "
    "Do not copy raw SPARQL text or dotted KB IDs into output."
)


GATEWAY_SEMANTIC_INSTRUCTIONS = f"""Rewrite one Freebase KBQA semantic graph without changing
the question meaning. Return only JSON with anchors, semantic_paths, operators. Each
anchor has id,surface. Each path has id,anchor_ref,goal,steps,path_output_var; each step
has id,relation_label,direction,from,to. Use reversible relation-label arrays and Pn.Vm
variables. Preserve expected_operators exactly. Emit AND only for explicit intersection,
not from path count. EQUAL and NO_EQUAL may be added from the corresponding source
derived fields. No KB
IDs or SPARQL. {GATEWAY_JOIN_RULES}
{GATEWAY_OPERATOR_RULES}
{WEBQSP_SOURCE_GUIDANCE}"""


GATEWAY_COMPOSE_INSTRUCTIONS = f"""Build one complete Freebase KBQA compose JSON object from
the verified semantic graph. Return only entities,triples,operators,answer_var. Entities
have id,surface; triples have subject,relation_label,object. Use E0/E1 entities and V0/V1
global variables. Lower AND by sharing variables; map every unary operator input_var from
Pn.Vm to Vn and preserve all other fields. No KB IDs or SPARQL. {GATEWAY_JOIN_RULES}
{GATEWAY_OPERATOR_RULES}
{WEBQSP_SOURCE_GUIDANCE}"""


GATEWAY_VERIFY_INSTRUCTIONS = f"""Strictly verify one rewritten Freebase KBQA training pair
against its question, decomposition, source graphs, joins, directions, answer projection,
and operators. Return only JSON with status,issues,semantic_graph,compose_program. status
is pass only if candidates are unchanged; corrected only with complete reliable corrected
objects; otherwise reject and repeat the candidates. Never add outside facts or KB IDs.
{GATEWAY_JOIN_RULES}
{GATEWAY_OPERATOR_RULES}
{WEBQSP_SOURCE_GUIDANCE}"""


LEAN_OPERATOR_EXAMPLES = r"""O(t,ins,v,a,chain,x,xt) means the required JSON object
{"type":t,"inputs":ins,"input_var":v,"attribute_relation_label":a,
"attribute_relation_labels":chain,"value":x,"value_type":xt}. Examples:
NO_OPERATOR=[]; MULTI_PATH_NO_AND=[] when paths are JOIN chains or shared-variable joins;
AND=O("AND",["P0","P1"],"",[],[],"","") only for true entity-set intersection;
COUNT=O("COUNT",[],"P0.V0",[],[],"","");
ARGMAX=O("ARGMAX",[],"P0.V0",["location","statistical region","population"],[],"","");
ARGMIN=O("ARGMIN",[],"P0.V0",["book","written work","date of first publication"],[],"","");
GT / GREATER_THAN=O("GREATER_THAN",[],"P0.V1",[],[],"8000","number");
GE / GREATER_OR_EQUAL=O("GREATER_OR_EQUAL",[],"P0.V1",[],[],"180","number");
LT / LESS_THAN=O("LESS_THAN",[],"P0.V1",[],[],"50","number");
LE / LESS_OR_EQUAL=O("LESS_OR_EQUAL",[],"P0.V1",[],[],"100","number");
TC=O("TC",[],"P0.V0",["sports","sports team roster","from"],[],"2015","year").
EQUAL=O("EQUAL",[],"P0.V0",["base","biblioness","bibs location","loc type"],[],"Country","string").
NO_EQUAL=O("NO_EQUAL",[],"P0.V1",[],[],"Julia Roberts","entity").
Compose maps Pn.Vm to Vn; AND becomes shared variables and is omitted from compose operators."""


LEAN_JOIN_RULES = (
    "JOIN direction rule: backward=(JOIN r e)=>(?x,r,e), so compose uses "
    "subject=to,object=from; forward=(JOIN (R r) e)=>(e,r,?x), so compose uses "
    "subject=from,object=to. JOIN is a path step, not an operator. Path Pn may only "
    "use Pn.Vm in step.to and path_output_var; never use P0.Vm inside P1. For "
    "constraint intersection, output local P1.V0 and semantic AND inputs=[\"P0\",\"P1\"]."
)


LEAN_SEMANTIC_INSTRUCTIONS = f"""Return only one complete semantic graph JSON with keys
anchors,semantic_paths,operators. Preserve the source meaning; copy valid content and only
fix clear defects. Anchors have id,surface. Paths have id,anchor_ref,goal,steps,
path_output_var. Steps have id,relation_label,direction,from,to. Use Pn.Vm variables and
never output KB IDs or SPARQL. required_operators must be preserved exactly; EQUAL and
NO_EQUAL may be added from the corresponding source derived fields. Use AND only
for explicit same-type entity-set intersection, not for path count. {LEAN_JOIN_RULES}
{LEAN_OPERATOR_EXAMPLES}
{WEBQSP_SOURCE_GUIDANCE}"""


LEAN_COMPOSE_INSTRUCTIONS = f"""Return only one complete compose JSON with keys entities,
triples,operators,answer_var. Build it from semantic_graph while preserving source meaning.
Entities have id,surface. Triples have subject,relation_label,object. Use En and Vn. Lower
AND to shared variables. Map unary operator Pn.Vm to Vn without changing other fields.
Never output KB IDs or SPARQL. {LEAN_JOIN_RULES} {LEAN_OPERATOR_EXAMPLES}
{WEBQSP_SOURCE_GUIDANCE}"""


LEAN_VERIFY_INSTRUCTIONS = (
    """Return only JSON with status and issues. status is pass or reject. Compare the
candidates with the question and source graphs. Reject wrong meaning, paths, joins,
directions, answer projection, relation labels, variables, or operators. Use pass only
when both candidates are answer-equivalent and internally consistent."""
    f"\n{WEBQSP_SOURCE_GUIDANCE}"
)


MICRO_SEMANTIC_INSTRUCTIONS = (
    """Return only a complete semantic graph JSON. Preserve
source meaning and valid content. Use keys anchors,semantic_paths,operators and Pn.Vm
variables. required_ops is exact except for derived EQUAL and NO_EQUAL constraints: copy
every non-AND object. Use AND only
for explicit same-type entity-set intersection, not for path count or JOIN chains.
EQUAL and NO_EQUAL may be added from the corresponding source fields. Every
operator needs type,inputs,input_var,attribute_relation_label,attribute_relation_labels,
value,value_type. Direction: backward=>subject=to, object=from; forward=>subject=from,
object=to. No KB IDs or SPARQL."""
    f"\n{WEBQSP_SOURCE_GUIDANCE}"
)


MICRO_COMPOSE_INSTRUCTIONS = (
    """Return only complete compose JSON with entities,triples,
operators,answer_var. Preserve source meaning. Convert semantic paths to En/Vn triples,
lower AND to shared variables, and map unary operator Pn.Vm to Vn without other changes.
Direction: backward=>subject=to, object=from; forward=>subject=from, object=to.
No KB IDs or SPARQL."""
    f"\n{WEBQSP_SOURCE_GUIDANCE}"
)


MICRO_VERIFY_INSTRUCTIONS = (
    """Return only {"status":"pass|reject","issues":[]}. Pass only if semantic and
compose agree with the question, joins, answer variable and operators. Otherwise reject
with short concrete issues."""
    f"\n{WEBQSP_SOURCE_GUIDANCE}"
)


CWQ_GATEWAY_DECOMPOSE_INSTRUCTIONS = (
    "Return only {\"decomposition\":[\"one continuous KG path per item\"]}. "
    "Keep multi-hop chains together; CVT/schema nodes are not separate paths; compress "
    "structural nodes, retain semantic intermediate entities, and attach constraints to "
    "the path they constrain. Preserve years, dates, now/current, and superlatives."
)


CWQ_GATEWAY_DECOMPOSE_VERIFY_INSTRUCTIONS = (
    "Return only {\"status\":\"pass|corrected|reject\",\"issues\":[],"
    "\"decomposition\":[...]} and enforce one continuous path per item. Natural-language "
    "chained questions are valid; do not demand arrows, SPARQL variables, or KB IDs."
)


SEMANTIC_INSTRUCTIONS = f"""You rewrite supervised Freebase KBQA data for Freebase KBQA.
Return one semantic graph, not an answer to the question. Preserve the source example's
meaning while correcting only clear structural defects. Use canonical reversible
relation_label segments and path-local variables P0.V0, P0.V1, etc. Anchors must be
grounded in the question or source example. Move every operator into the semantic
graph's top-level operators field. The expected_operators list is mandatory for all
non-AND operators: preserve its type, value, value_type, and attribute relation labels
exactly; EQUAL and NO_EQUAL may additionally be emitted from the corresponding
source derived fields.
Use AND only when the source meaning requires same-type entity-set intersection,
not merely because the graph has multiple paths. Do not emit dotted Freebase IDs or SPARQL.
{JOIN_PROJECTION_RULES}
{OPERATOR_CONTRACT}
{OPERATOR_FEW_SHOTS}
{WEBQSP_SOURCE_GUIDANCE}
Now return only the complete semantic graph for the supplied request."""


COMPOSE_INSTRUCTIONS = f"""You rebuild a complete JSON logical query program from a verified
semantic graph. Compose paths by sharing global variables V0, V1, etc. Use E0, E1, etc.
for entity references. Lower semantic AND into shared variables. For every other semantic
operator, change only its path-local input_var to the corresponding global variable and
preserve all remaining fields exactly. Return only entities, triples, operators, and
answer_var. Do not emit dotted Freebase IDs or SPARQL.
{JOIN_PROJECTION_RULES}
{OPERATOR_CONTRACT}
{OPERATOR_FEW_SHOTS}
{WEBQSP_SOURCE_GUIDANCE}
Now return only the complete compose program for the supplied request."""


VERIFY_INSTRUCTIONS = f"""Act as an independent, strict reviewer of one rewritten Freebase KBQA
training pair. Check the original question, decomposition, source labels, rewritten
semantic paths, answer projection, joins, relation directions, entity surfaces, and every
operator. The rewritten pair must be answer-equivalent to the source. Check AND lowering
and require every non-AND semantic operator to be preserved in compose_program after
path-local variables are mapped to global variables. Never add outside facts or KB IDs.
Return status=pass only when no change is needed and repeat both candidates exactly.
Return status=corrected only for a minimal reliable correction and include both complete
corrected objects. Otherwise return status=reject, list concrete issues, and repeat the
uncorrected candidates in the object fields.
{JOIN_PROJECTION_RULES}
{OPERATOR_CONTRACT}
{WEBQSP_SOURCE_GUIDANCE}"""


_SFT_OPERATOR_TYPES = (
    "Allowed semantic operators: AND, COUNT, ARGMAX, ARGMIN, GREATER_THAN, "
    "GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL, EQUAL, TC, NO_EQUAL, or no operator. GT/GE/LT/LE "
    "must be emitted using their long canonical JSON names. AND inputs are participating "
    "path ids only for true entity-set intersection; compose lowers AND to shared variables."
)


SEMANTIC_SFT_INSTRUCTION = """[TASK=SEMANTIC_PATH]

Convert the original question and its decomposition into canonical Freebase semantic paths and semantic operators.

Return exactly one JSON object containing:
- anchors
- semantic_paths
- operators

Each semantic path is an entity-centric chain of JOIN projections.

Direction rules:
- backward: (?x, r, e). step.from=e and step.to=?x.
- forward: (e, r, ?x). step.from=e and step.to=?x.

Each path Pn must use only its own path-local variables Pn.Vm in step.to and path_output_var. Variables from different paths must never be reused directly.

JOIN is represented only by semantic path steps and must not appear in operators.

Allowed operators are exactly:
AND, COUNT, ARGMAX, ARGMIN, GREATER_THAN, GREATER_OR_EQUAL,
LESS_THAN, LESS_OR_EQUAL, EQUAL, TC, NO_EQUAL.

Use AND only for a true intersection of two or more same-type entity sets. Multiple paths, JOIN chains, or later shared-variable joins do not by themselves imply AND.

For COUNT, ARGMAX, ARGMIN, comparison operators, EQUAL, TC, and NO_EQUAL, use the relevant path-local Pn.Vm variable as input_var. EQUAL requires an attribute relation, value, and value_type. NO_EQUAL uses the exact anchor surface as value, value_type="entity", and empty attribute fields. Comparison operators must use their long canonical JSON names.

Every operator must contain exactly:
type, inputs, input_var, attribute_relation_label,
attribute_relation_labels, value, value_type.

Use empty strings or empty arrays for unused fields.
If no operator is expressed, output "operators": [].

Use canonical relation labels only. Do not emit KB IDs, raw SPARQL, or dotted Freebase relation IDs.
Return JSON only."""


PATH_SEMANTIC_SFT_INSTRUCTION = (
    SEMANTIC_SFT_INSTRUCTION
    + "\nThe input decomposition uses CWQ path-level format. Emit exactly one semantic path "
    "for each decomposition item in the same order. A decomposition item may contain multiple "
    "hop-wise questions or statements. Copy decomposition[i] verbatim into "
    "semantic_paths[i].goal; do not split, merge, reorder, or paraphrase path goals."
)


COMPOSE_SFT_INSTRUCTION = (
    "[TASK=COMPOSE] Build the complete JSON logical query program from the original "
    "question and supplied semantic graph. Map path-local variables to shared global "
    "variables and preserve every non-AND operator exactly. "
    f"{_SFT_OPERATOR_TYPES}\n{JOIN_PROJECTION_RULES}\n{OPERATOR_CONTRACT}\n"
    "Return only entities, triples, operators, and answer_var; do not emit KB IDs.\n"
    f"{WEBQSP_SOURCE_GUIDANCE}"
)
