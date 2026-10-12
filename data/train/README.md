# Final stage-aligned TRAIN snapshots

These stage-aligned TRAIN exports preserve the questions, targets, order and
counts of the retained four-stage fine-tuning inputs. They are not evaluation
predictions or execution traces. The publication cleanup neutralizes only
the instruction wording in the CWQ decomposition file. The other seven files
remain byte-identical to their retained local snapshots. The original and
published hashes of the changed file are recorded in `manifest.json`.

The released CWQ decomposition instruction is therefore not byte-identical
to the historical training prompt. No models are retrained by this cleanup,
and new predictions using the revised wording need not match historical runs.

| Stage | CWQ: 25,328 rows | WebQSP: 2,651 rows |
| --- | --- | --- |
| Decomposition | `cwq/decompose_train.json` | `webqsp/decompose_train.json` |
| Semantic, without Operator | `cwq/semantic_path_train.json` | `webqsp/semantic_train.json` |
| Compose, without Operator | `cwq/compose_train.json` | `webqsp/compose_train.json` |
| Operator | `cwq/operator_train.json` | `webqsp/operator_train.json` |

CWQ corresponds to `cwq_25328_goldfirst_strict_v2`, used by the v42 four-stage
CWQ models. The separate 26,944-row decomposition-only experiment and the older
19,518-row bundle are not substituted for it.
WebQSP corresponds to the `from_base_flywheel_2651` four-stage LoRA training
snapshots. The older three-stage bundle with operators embedded in Semantic
and Compose is not substituted for these decoupled stage files.

Each row has `instruction`, `input`, `output`, and `history`. The input/output
fields contain JSON strings. The stage files have identical question order
within each dataset. Each dataset folder includes a self-contained
`dataset_info.json` for LLaMA-Factory registration.

## Checks performed

- All eight JSON arrays and their nested input/output JSON strings parse.
- All questions match the retained official TRAIN source; WebQSP comparison
  accepts its official `RawQuestion` or `ProcessedQuestion` text.
- Row counts and question order match across all four stages.
- Published-file SHA-256 values match the current manifest; source hashes
  match except for the explicitly recorded instruction-only cleanup.
- No credentials, former publisher username, traces, logs, weights or test
  files are included. Original questions/targets are not rewritten or filtered.

CWQ has 25,327 distinct question strings across 25,328 rows. Two of these
TRAIN question strings also occur in the retained upstream test questions
after whitespace/case normalization. This is an upstream text overlap, not
an appended test export; both belong to the official TRAIN source. They are
retained to preserve the actual fine-tuning inputs. WebQSP has 2,651 distinct
question strings and no such normalized test-text overlap.
This audit checks data membership and structure, not new endpoint execution
or model training. It does not claim every training target is factually correct.

## Registration and integrity

CWQ registration names are `cwq_decomposition_train`, `cwq_semantic_train`,
`cwq_compose_train`, `cwq_operator_train`. WebQSP uses the same names with
the `webqsp_` prefix. Point your own training configuration's dataset directory
at the corresponding folder; model weights and training infrastructure are
not bundled here.

`manifest.json` records dataset versions, row counts, sizes and SHA-256 values.
For example, from this directory:

```bash
sha256sum cwq/*train.json webqsp/*train.json
```

The published text files total approximately 185 MiB. Large files may not have
a GitHub browser preview; use Git clone or the Raw/download link. No Git LFS
pointers, compressed substitutes or external download service are required.

## Attribution and use

See [NOTICE.md](NOTICE.md) for upstream dataset/project sources and terms.
This publication does not relicense the original benchmark content.
