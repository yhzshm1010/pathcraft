# CWQ dataset construction

All public commands default to CWQ TRAIN data and GLM-5.2 chat calls. Never use CWQ dev/test gold to create training data. Raw annotations and all generated arrays/ledgers stay in ignored inputs/work directories.

## Source inputs

Provide your licensed/own files under `dataset_construction/inputs/cwq/`:

```text
decompose_train.json
semantic_path_train.json
compose_train.json
operator_train.json                 # Gold-first compiler/template headers
ComplexWebQuestions_train.json      # official TRAIN source, with gold SPARQL
```

Source supervision arrays use the existing Alpaca `instruction/input/output` JSON format. Additional Gold-first/decomposition utilities accept your own KaeDe archive. The raw source annotations must be supplied separately; generated outputs are not a substitute for these source inputs.

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

The original static and execution validation pipeline is retained. The default execution validator is `scripts/cwq_execution_validator.py`; it uses an external KaeDe executor configured by `KAEDE_VQG_SRC` plus your Freebase service. Alternatively set `CWQ_EXECUTION_VALIDATOR_COMMAND` to an equivalent JSON-stdin/JSON-stdout validator. The external executor and database are not vendored. Missing execution validation must not be described as execution-verified data.

## Utilities

- `build_gold_first_cwq.py`: construct aligned stages from CWQ TRAIN Gold SPARQL, verify contracts and real endpoint execution. Supply `--kaede-archive`, `--selection`, `--train-dir`, `--output-dir` explicitly; `--all-appended` may select remaining rows from a retained TRAIN prefix.
- `build_incremental_finetune_dataset.py`: merge accepted records into training arrays.
- `split_operator_dataset.py`: export no-operator Semantic/Compose supervision and standalone Operator supervision.
- `build_kaede_decomposition_dataset.py`, `build_incremental_full_kaede_decomposition.py`: rebuild controlled TRAIN decomposition targets from your KaeDe source archive.
- `auto_correct_cwq_with_glm.py`: repair rejected TRAIN examples with glm-5.2, followed by the same validation gates.
- `validate_and_accept_manual_correction.py`, `batch_accept_cwq_source_fallbacks.py`: accept only validated corrected/source rows.
- `repair_legacy_cwq_train.py`: contract-normalize an existing local TRAIN export.

```bash
python3 dataset_construction/scripts/split_operator_dataset.py \
  --semantic dataset_construction/work/cwq_generated/semantic_path_train.json \
  --compose dataset_construction/work/cwq_generated/compose_train.json \
  --out-dir dataset_construction/work/cwq_sft
```

All generated arrays, accepted/rejected ledgers, manifests and correction records are local outputs and Git-ignored. GLM key/model/endpoint can be configured through the documented environment variables or CLI. The underlying shared transport/validation internals retain some legacy names; no WebQSP dataset or public WebQSP generation command is included.
