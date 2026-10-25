# PathCraft — CWQ

ComplexWebQuestions (CWQ) question inference and stage-aligned dataset construction in one repository. The inference implementation follows the final Full stability-fix mainline; no ablation or replay implementation is included. GLM calls default to **glm-5.2**.

This release includes **code and the final CWQ/WebQSP stage-aligned TRAIN snapshots** in `data/train/`. It does not contain raw benchmark archives, dev/test exports, model weights, Freebase dumps, retrieval cache outputs, traces, logs, intermediate experiments, or historical predictions/results. Small frozen prompt examples and scoring parameters are required runtime resources, not per-question execution traces. Underlying datasets retain their original terms; see `data/train/NOTICE.md`.

## Dataset size and versioning

The experiments reported in the paper used **24,250 CWQ training examples per
stage**. The dataset is still being refined and expanded; the current public
CWQ TRAIN snapshot contains **25,328 examples per stage (25,000+)**. These
counts refer to stage-aligned training samples, not the evaluation-set size.

The current public snapshot is a later dataset version, not the exact
24,250-example version used in the reported experiments. Dataset updates do
not retroactively change the experimental setup or reported results. See the
[training-data documentation](data/train/README.md) and
[release manifest](data/train/manifest.json) for the published files, row
counts and checksums.

## Layout

- `src/semantic_guided_kbqa/`: decomposition review, Semantic/Grounding, Compose, Operator, KG execution, and selection.
- `configs/pipeline.cwq.json`: Full CWQ configuration; reviewer model is glm-5.2.
- `prompts/`: frozen CWQ reviewer policies.
- `scripts/`: local asset import, question-only decomposition preparation, fresh decomposition inference, and CWQ inference launcher.
- `dataset_construction/`: CWQ training-data generation, Gold-SPARQL construction, static/execution validation, stage splitting and export. See its README.
- `data/train/`: four stage-aligned SFT exports per dataset: CWQ (25,328 rows per stage) and WebQSP (2,651 rows per stage), registration files, checksums and source notices. See its README for the instruction-only publication cleanup.

WebQSP inference configurations/launchers, evaluation experiments and trace replay tools are not distributed; only its TRAIN supervision exports are included. The public data-construction CLI only accepts CWQ.

The inference package is consolidated into **28 Python files**. Closely related
helpers share a module; rules, prompts, score weights and query/model budgets
are unchanged by this file-layout refactor. Independent private helpers and
caches keep separate names to prevent collisions.

| Module | Responsibility |
| --- | --- |
| `pipeline.py`, `cli.py`, `config.py` | Stage orchestration, entry point and configuration |
| `contracts.py`, `data.py`, `evaluation.py`, `validation.py` | Data structures, input loading, metrics and output validation |
| `clients.py`, `decomposition_review.py`, `prompts.py` | Model/KG clients, GLM review and prompt examples |
| `ontology.py`, `grounding.py`, `lowering.py` | Schema lookup, grounding and SPARQL compilation |
| `selectors.py`, `labels.py` | Candidate ranking/selection and answer-label evidence |
| `normalization.py`, `time_constraints.py` | Numeric normalization and temporal constraints |
| `safety.py`, `constraint_repairs.py`, `stability.py`, `runtime_context.py` | Scope guards, bounded repairs, stability checks and per-question budgets |
| `failure_retrieval.py`, `budgeted_schema_repair.py` | Failure-only retrieval and bounded schema repair |
| `graph_reconstruction.py`, `grounding_repair.py`, `template_path_retrieval.py`, `zero_graph_challenger.py` | Graph reconstruction and candidate retrieval |
| `__init__.py` | Package exports |

Legacy helper-module import paths have been replaced by these grouped modules.
Dataset-construction source modules
remain separate because they form a different, independently usable workflow.

## Install

```bash
python3 -m pip install -e .
python3 -m semantic_guided_kbqa.cli --help
python3 -m gpt54_data_flywheel generate --help
```

Python 3.11+ is required. Model servers and Freebase must be provided separately. Configure Semantic, Compose, Operator, Selector, BGE, and Freebase addresses in `configs/pipeline.cwq.json`; defaults use localhost ports 18002, 18003, 18004, 18005, 8008, and 3005 respectively. These inference models must be your CWQ-trained models. No services are automatically started/stopped.

## Runtime inputs

The public SFT exports are documented in [data/train/README.md](data/train/README.md).
They preserve historical questions and targets, with a documented instruction-only
cleanup in CWQ decomposition, and do not automatically replace the runtime
retrieval files below. Publishing these data does not retrain the models or
modify the retrieval configuration.

Put your own CWQ inputs under ignored `assets/`:

```text
assets/
  data/cwq/{semantic_path_train.json,compose_train.json,operator_train.json,
            ComplexWebQuestions_train.json,gold_entities_test.json,
            decompose_test_pred_cwq_25328_vllm.json}
  ontology/{fb_roles,fb_types,reverse_properties,domain_info,domain_dict,
            schema_overlay.tsv,schema_inferred.tsv}
  cache/{train_gold_sparql_path_index_v1.json.gz,
         failure_gold_sparql_template_index_v3.json.gz,
         source_verified_template_consensus_index_v1.json.gz}
```

The TRAIN files/indexes support runtime retrieval; they are not evaluation traces. Use the dataset-construction code to create training arrays, or import an existing local Full bundle without copying its logs/results/traces:

```bash
python3 scripts/import_runtime_assets.py --bundle-root /path/to/your/full-cwq-bundle
```

The importer relocates cache source paths only after checking copied TRAIN content. Assets remain Git-ignored. The config may instead be edited to your own equivalent paths; no dependency on the author's workspace is required.

## Fresh inference

Set `GLM_API_KEY` privately in your shell. No key is stored or automatically loaded from `.env.example`.

```bash
export GLM_API_KEY='YOUR_KEY'
QUESTION_WORKERS=4 bash scripts/run_cwq.sh
```

All Semantic/Compose/Operator/Selector calls are normal model calls, not replay. The launcher uses a decomposition input file, as the original pipeline does. To regenerate decomposition with your CWQ-trained Decomposition model:

```bash
python3 scripts/prepare_decomposition_inputs.py \
  --questions /path/to/ComplexWebQuestions_test.json \
  --gold-entities assets/data/cwq/gold_entities_test.json \
  --training-template /path/to/decompose_train.json \
  --output outputs/decomposition_questions.json
python3 scripts/generate_cwq_decomposition.py \
  --input outputs/decomposition_questions.json \
  --output outputs/fresh_decomposition.json \
  --url http://127.0.0.1:18001 --model decomposition --workers 4
QUESTION_WORKERS=4 bash scripts/run_cwq.sh \
  --decompositions outputs/fresh_decomposition.json
```

Only question and provided entity links enter initial decomposition; source gold answers/SPARQL are not inserted. This project retains the original gold-entity input protocol. The example API names/paths must match your deployed models. If you do not want GLM review, add `--no-decomposition-review`.

The active `two_call/cwq` Review and Rewrite prompts now use only frozen CWQ
TRAIN demonstrations. There are 6 preserve and 11 controlled-repair examples:
17 in Review and 11 in Rewrite. Targets are copied from aligned TRAIN rows;
graph structure, answer-variable identity and entity bindings are checked
against official TRAIN queries/annotations offline. This is not an endpoint
F1 certification. The legacy resource filenames are retained for compatibility;
source hashes and row IDs are in `resources/cwq_review_train_provenance.json`.
No test-question repairs, Gold answer lists or raw SPARQL enter these prompts.

Rebuild/audit the demonstrations using your own official TRAIN inputs:

```bash
python3 scripts/build_cwq_review_train_examples.py \
  --source-train /path/to/ComplexWebQuestions_train.json \
  --train-entity-records /path/to/CWQ_train_extracted.json \
  --heldout-questions /path/to/heldout_questions.json
```

Replacing demonstrations changes prompt content, not the review call budget,
response schemas, inference rules or scoring. Frozen rule/calibration provenance
and existing post-hoc risk disclosures remain intact. This change does not
retroactively validate historical results or saved Review responses; a new
Review run is required to evaluate the TRAIN-only prompt configuration.

Outputs are generated under ignored `outputs/`. Internal in-memory stage evidence remains necessary for the runtime safety/recovery code; no historical trace is shipped. The launcher does not request separate per-question artifact files. The GLM model switch can change predictions: this release does not claim to reproduce previously reported GLM-5.3 metrics without a new run.

## Evaluation and publication cleanup

The primary score contains precision, recall, F1, answer-set Hit (`hit`),
ordered first-answer hit (`hits_at_1`), and exact set match (`exact_match`).
Entity answers use canonical Freebase IDs; literal/text answers use normalized
text sets. There are no auxiliary substring-label scores. Artifact comparisons
and optional baseline-zero filters also use the primary score (`--baseline-f1-zero-only` or
`--baseline-all-f1-zero-only`).

### Hit, first-answer hit, and exact match

These statistics must not be used interchangeably. Let `P` be the normalized,
deduplicated prediction list and `G` the normalized Gold answer set.

| Statistic | Per-question definition |
| --- | --- |
| `hit` (answer-set Hit) | 1 if at least one predicted answer belongs to `G`; 0 otherwise. Partial recall can still yield 1. Answer order does not matter. |
| Current runtime `hits_at_1` (first-answer hit) | 1 if `P` is nonempty and its first answer belongs to `G`; 0 otherwise. A correct answer later in `P` does not count. |
| Current runtime `exact_match` (exact set match) | 1 only if the predicted answer set equals `G`, with neither missing nor extra answers; 0 otherwise. `exact` is its boolean per-row compatibility alias. |

For example, with `G = {a, b}` and `P = [c, a]`, `hit` is 1,
`hits_at_1` is 0, and `exact_match` is 0. With `P = [a]`, `hit`
and `hits_at_1` are both 1, although recall is only 0.5 and exact match is 0.
Recalling every Gold answer is also insufficient for exact match if extra
incorrect answers are returned.

The evaluator preserves the supplied prediction order after normalization and
deduplication; it does not rank answers by confidence. Thus, the first element
of an unordered query result is not automatically a highest-ranked answer.
The runtime now emits `score.hit` for each saved question and `summary.hit`
for the overall answer-set Hit rate. The latter is the number of hit questions
divided by the number of scored questions, **not** mean recall or the fraction
of questions with perfect answers. All numeric metrics are stored in `[0, 1]`;
multiply `summary.hit` by 100 when reporting a percentage.

For the first synthetic example above, the relevant per-question output is:

```json
{"score": {"hit": 1.0, "hits_at_1": 0.0, "exact_match": 0.0}}
```

`summary.hit` follows the any-answer-overlap criterion used by the checked
archived CWQ ID summary. It is distinct from `summary.hits_at_1`; do not rename
one as the other. This agreement in the Hit criterion does not establish that
every historical result uses the same answer matching, denominator or failure
policy. Current entity matching uses canonical IDs, not substring matching.

Metrics are macro-averaged over **all labeled items, including failures**.
A labeled item with a nonempty `failure` receives zero precision, recall, F1,
`hit`, `hits_at_1` and `exact_match`, even if it retains a partial prediction for
diagnostics. Failed items stay in the metric denominator and are also saved
in the separate error output. Only items without evaluable Gold remain
unscored; always report the processed, scored, unlabeled and error counts.
`summary.scored` must equal the full test size before calling the summary a
full-test average. These are the current runtime's aggregation rules, not a
claim about every archived summary's denominator or failure policy.

The evaluation version is `answer-id-primary-v5-hit-failures-zero`. Version 5
adds answer-set Hit while retaining the existing precision/recall/F1,
`hits_at_1`, exact-match, and failure-handling formulas. Supported older
checkpoints can recover missing per-question Hit from their standard primary
F1 scores: positive F1 means at least one matched answer. A macro F1 average
alone cannot recover the Hit rate. When the old Hit aggregate is missing,
continuation therefore requires either a complete set of scored rows or
explicit perfect-row filtering metadata with consistent counters; otherwise
it rejects an ambiguous old checkpoint instead
of treating missing Hit as zero. The existing v3/v2 successful-only denominator
migration still requires the complete paired error file. Missing error records,
unknown versions and mixed main/error versions are rejected. Use a fresh run
directory if the old checkpoint cannot be migrated safely. Existing result
files are not changed until explicitly resumed; archived reports are not
retroactively rewritten or claimed to follow the new output schema.

The publication cleanup generalizes method-specific wording in Review and
construction prompts, neutralizes script/configuration names, and removes
auxiliary label metrics. Adding the `hit` output does not change existing metric
formulas, retrieval budgets, predictions, training data, or archived results.
Prompt wording has changed, so a fresh model run is required
before claiming identical end-to-end predictions. Dataset provenance and
required third-party license notices are deliberately preserved in
[data/train/NOTICE.md](data/train/NOTICE.md).

## Publication boundary

Only code, required configuration/prompt resources, and the explicitly listed final SFT TRAIN snapshots are published. Do not commit runtime assets, secrets, other generated data, logs, model weights, dev/test exports or results. New data updates use ordinary commits; previously published URLs/caches may still contain historical information.
