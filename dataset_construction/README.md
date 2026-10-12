# CWQ dataset construction

All public commands default to CWQ TRAIN data and GLM-5.2 chat calls. Never use CWQ dev/test gold to create training data. Raw annotations and generated arrays/ledgers stay in ignored inputs/work directories; the explicitly approved final SFT snapshots are separately published under `../data/train/`.

## Source inputs

Provide your licensed/own files under `dataset_construction/inputs/cwq/`:

```text
decompose_train.json
semantic_path_train.json
compose_train.json
operator_train.json                 # Gold-first compiler/template headers
ComplexWebQuestions_train.json      # official TRAIN source, with gold SPARQL
```

Source supervision arrays use the existing Alpaca `instruction/input/output` JSON format. Gold-first/decomposition utilities accept processed TRAIN annotations as a JSON array or source ZIP via `--source-archive`. ZIP members are discovered by the filename `CWQ_train_extracted.json`, without assuming the top-level folder name. Use `--source-member` if the archive contains multiple matches. The raw source annotations must be supplied separately; generated outputs are not a substitute for these source inputs. Upstream attribution is preserved in `../data/train/NOTICE.md`.

## Preflight and GLM generation

Install the root package first:

```bash
python3 -m gpt54_data_flywheel preflight --dataset cwq
export GLM_API_KEY='YOUR_KEY'
python3 -m gpt54_data_flywheel generate \
  --dataset cwq --api-mode chat --model glm-5.2 \
  --workers 4 --require-execution-validation \
  --output dataset_construction/work/cwq_generated
```

The static checks and answer-set execution comparisons are retained. The default validator is `scripts/cwq_execution_validator.py`; configure its external graph executor through `FREEBASE_EXECUTOR_SRC` and `FREEBASE_EXECUTOR_MODULE`, together with your Freebase service. The integration details and required upstream attribution are in `../data/train/NOTICE.md`. The configured module must export `FreebaseConfig`, `FreebaseSparqlClient`, `GoldenEntityLinker`, `FreebaseGroundingBackend`, `FreebaseQueryBackend`, and `query_graph_to_sparql`. Alternatively set `CWQ_EXECUTION_VALIDATOR_COMMAND` to an equivalent JSON-stdin/JSON-stdout validator. The external executor and database are not vendored. Missing execution validation must not be described as execution-verified data.

The public construction CLI accepts only `--api-mode chat` and defaults to
GLM-5.2. For Python integrations, the internal `ExternalResponsesClient`
adapter loads `ResponsesConfig` and `ResponsesClient` from
`EXTERNAL_RESPONSES_MODULE`, with `EXTERNAL_RESPONSES_SRC` if needed. This
adapter is not used by the CWQ CLI.

## Utilities

- `build_gold_first_cwq.py`: construct aligned stages from CWQ TRAIN Gold SPARQL, verify contracts and real endpoint execution. Supply `--source-archive`, `--selection`, `--train-dir`, `--output-dir` explicitly; `--all-appended` may select remaining rows from a retained TRAIN prefix.
- `build_incremental_finetune_dataset.py`: merge accepted records into training arrays.
- `split_operator_dataset.py`: export no-operator Semantic/Compose supervision and standalone Operator supervision.
- `build_source_decomposition_dataset.py`, `build_incremental_source_decomposition.py`: rebuild controlled TRAIN decomposition targets from processed CWQ TRAIN source annotations.
- `auto_correct_cwq_with_glm.py`: repair rejected TRAIN examples with glm-5.2, followed by the same validation gates.
- `validate_and_accept_manual_correction.py`, `batch_accept_cwq_source_fallbacks.py`: accept only validated corrected/source rows.
- `repair_legacy_cwq_train.py`: contract-normalize an existing local TRAIN export.

```bash
python3 dataset_construction/scripts/split_operator_dataset.py \
  --semantic dataset_construction/work/cwq_generated/semantic_path_train.json \
  --compose dataset_construction/work/cwq_generated/compose_train.json \
  --out-dir dataset_construction/work/cwq_sft
```

All construction outputs, accepted/rejected ledgers and correction records remain local and Git-ignored. The selected final CWQ and WebQSP four-stage SFT inputs, their registration files and checksum manifest are published in `../data/train/`; no public WebQSP generation command is added. GLM key/model/endpoint can be configured through the documented environment variables or CLI.
