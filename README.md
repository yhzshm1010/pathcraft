# PathCraft — CWQ

ComplexWebQuestions (CWQ) question inference and stage-aligned dataset construction in one repository. The inference implementation follows the final Full stability-fix mainline; no ablation or replay implementation is included. GLM calls default to **glm-5.2**.

This release includes **code and the final CWQ/WebQSP stage-aligned TRAIN snapshots** in `data/train/`. It does not contain raw benchmark archives, dev/test exports, model weights, Freebase dumps, retrieval cache outputs, traces, logs, intermediate experiments, or historical predictions/results. Small frozen prompt examples and scoring parameters are required runtime resources, not per-question execution traces. Underlying datasets retain their original terms; see `data/train/NOTICE.md`.

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

Only the primary score is produced: precision, recall, F1, Hit (`hits_at_1`),
and exact set match. Entity answers use canonical Freebase IDs; literal/text
answers retain the normalized text-set protocol. There are no auxiliary
substring-label scores. Artifact comparisons and optional baseline-zero
filters also use the primary score (`--baseline-f1-zero-only` or
`--baseline-all-f1-zero-only`).

Metrics are macro-averaged over **all labeled items, including failures**.
An item with a nonempty `failure` receives zero precision, recall, F1, Hit and
exact match, even if it retains a partial prediction for diagnostics. Failed
items stay in the metric denominator and are also saved in the separate error
output. Only items without evaluable Gold remain unscored; always report the
processed, scored, unlabeled and error counts. `summary.scored` must equal the
full test size before calling the summary a full-test average.

The protocol version is `answer-id-primary-v4-failures-zero`. Continuation runs
migrate v3/v2 successful-only aggregates using the complete paired error file,
including when perfect success rows were filtered from JSON. Missing error
records, unknown protocol versions and mixed main/error versions are rejected
instead of silently mixing denominators. Existing result files are not changed
until explicitly resumed; archived experiment reports retain their old scores.

The publication cleanup generalizes method-specific wording in Review and
construction prompts, neutralizes script/configuration names, and removes
auxiliary label metrics. The primary scoring formula and retrieval budgets
are unchanged. Prompt wording has changed, so a fresh model run is required
before claiming identical end-to-end predictions. Dataset provenance and
required third-party license notices are deliberately preserved in
[data/train/NOTICE.md](data/train/NOTICE.md).

## Publication boundary

Only code, required configuration/prompt resources, and the explicitly listed final SFT TRAIN snapshots are published. Do not commit runtime assets, secrets, other generated data, logs, model weights, dev/test exports or results. New data updates use ordinary commits; previously published URLs/caches may still contain historical information.
