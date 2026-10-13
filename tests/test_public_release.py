"""Offline regression tests for primary scoring and neutral source interfaces."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "dataset_construction/src"),
               str(ROOT / "dataset_construction/scripts")]

from semantic_guided_kbqa.evaluation import (
    EVALUATION_VERSION, canonicalize_answer_id, evaluation_metadata, is_metric_result_row, primary_result_row,
    score_result_row, summarize_result_rows,
)
from semantic_guided_kbqa.cli import (
    _baseline_f1_is_zero, _load_resume_state, _merge_summaries,
    _subtract_summaries, _write_question_artifacts, _write_output, _rewind_resume_state, _question_metric_delta, build_parser,
)
from source_annotations import load_source_annotations
from semantic_guided_kbqa import prompts as review_prompts
from semantic_guided_kbqa.decomposition_review import DecompositionReviewer


class PrimaryScoringTests(unittest.TestCase):
    def test_ids_do_not_depend_on_labels(self):
        row = score_result_row(
            {"answer_entities": {"m.correct": "Correct label"}},
            {"prediction": ["http://rdf.freebase.com/ns/m.correct"],
             "answers": [{"id": "m.correct", "label": "Unrelated label"}]},
        )
        self.assertEqual(row["score_basis"], "answer_ids")
        self.assertEqual(row["score"]["f1"], 1.0)
        self.assertEqual(row["score"]["hits_at_1"], 1.0)
        self.assertEqual(row["score"]["exact_match"], 1.0)

    def test_ordered_hit_differs_from_set_overlap(self):
        row = score_result_row(
            {"answer_entities": {"m.correct": "Correct"}},
            {"prediction": ["m.wrong", "m.correct", "m.correct"]},
        )
        self.assertAlmostEqual(row["score"]["f1"], 2 / 3)
        self.assertEqual(row["score"]["hits_at_1"], 0.0)
        self.assertEqual(row["prediction"], ["m.wrong", "m.correct"])

    def test_empty_prediction_is_zero(self):
        row = score_result_row({"answer_entities": {"m.correct": "Correct"}},
                               {"prediction": []})
        self.assertEqual(row["score"]["f1"], 0.0)
        self.assertEqual(row["score"]["hits_at_1"], 0.0)

    def test_literal_normalization(self):
        row = score_result_row({"answer_entities": {"Value": "2001"}},
                               {"prediction": ['"2001"^^xsd:gYear']})
        self.assertEqual(row["score_basis"], "answer_text")
        self.assertEqual(row["score"]["f1"], 1.0)
        self.assertEqual(canonicalize_answer_id("/m/example"), "m.example")

    def test_auxiliary_scores_are_removed(self):
        row = primary_result_row({"score": {"f1": 0.5},
                                  "legacy_label_score": {"f1": 1.0},
                                  "raw_text_score": {},
                                  "delta": {"score": {"f1": 0.2}, "legacy_label": {}}})
        self.assertEqual(set(row), {"score", "delta"})
        self.assertEqual(set(row["delta"]), {"score"})
        self.assertNotIn("additional_metrics", evaluation_metadata("synthetic"))

    def test_failures_are_included_in_aggregation(self):
        record = {"answer_entities": {"m.correct": "Correct"}}
        good = score_result_row(record, {"prediction": ["m.correct"]})
        wrong = score_result_row(record, {"prediction": ["m.wrong"]})
        error = score_result_row(record, {"prediction": [], "failure": "synthetic error"})
        summary = summarize_result_rows([good, wrong, error])
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["scored"], 3)
        self.assertAlmostEqual(summary["macro_f1"], 1 / 3)
        self.assertTrue(is_metric_result_row(error))
        self.assertFalse(any(key.endswith("_label") for key in summary))
        combined = _merge_summaries(summarize_result_rows([good]), summarize_result_rows([wrong]))
        self.assertEqual(combined["macro_f1"], 0.5)
        self.assertEqual(_subtract_summaries(combined, summarize_result_rows([wrong]))["macro_f1"], 1.0)

    def test_failure_partial_prediction_is_zero_without_mutating_evidence(self):
        record = {"answer_entities": {"m.correct": "Correct"}}
        original = {"prediction": ["m.correct"], "failure": "timeout"}
        row = score_result_row(record, original)
        for metric in ("precision", "recall", "f1", "hits_at_1", "exact_match"):
            self.assertEqual(row["score"][metric], 0.0)
        self.assertFalse(row["score"]["exact"])
        self.assertEqual(row["prediction"], ["m.correct"])
        self.assertNotIn("score", original)

    def test_failed_literal_answer_is_zero(self):
        row = score_result_row({"answer_entities": {"Value": "2001"}},
                               {"prediction": ["2001"], "failure": "timeout"})
        self.assertEqual(row["score"]["f1"], 0.0)
        self.assertTrue(row["score"]["labeled"])

    def test_legacy_failure_score_is_zeroed_during_summary(self):
        good = score_result_row({"answer_entities": {"m.correct": "Correct"}},
                                {"prediction": ["m.correct"]})
        legacy = {**good, "failure": "timeout"}
        self.assertEqual(summarize_result_rows([legacy])["f1"], 0.0)
        self.assertEqual(legacy["score"]["f1"], 1.0)

    def test_missing_gold_is_reported_not_fabricated(self):
        unknown = score_result_row({}, {"prediction": [], "failure": "timeout"})
        self.assertFalse(is_metric_result_row(unknown))
        self.assertIsNone(unknown["score"]["f1"])
        summary = _merge_summaries({}, summarize_result_rows([unknown]))
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["unlabeled"], 1)
        self.assertEqual(summary["scored"], 0)
        self.assertIsNone(summary["f1"])

    def test_failure_gold_without_old_score_is_included(self):
        row = primary_result_row({"failure": "timeout", "gold_answers": ["m.correct"]})
        self.assertTrue(is_metric_result_row(row))
        self.assertEqual(summarize_result_rows([row])["scored"], 1)

    def test_blank_gold_is_not_labeled(self):
        row = primary_result_row({"failure": "timeout", "gold_answers": ["  "]})
        self.assertFalse(is_metric_result_row(row))

    def test_failed_baseline_delta_uses_same_zero_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "000000"
            item.mkdir()
            good = score_result_row({"answer_entities": {"m.correct": "Correct"}},
                {"prediction": ["m.correct"], "index": 0, "question": "synthetic"})
            (item / "evaluation.json").write_text(json.dumps(good))
            (item / "result.json").write_text(json.dumps({"failure": "timeout"}))
            result = _question_metric_delta(root, current=good, index=0, question="synthetic")
            self.assertEqual(result["score"]["f1"], 1.0)
            self.assertEqual(result["evaluation_version"], EVALUATION_VERSION)

    def test_legacy_filtered_resume_migrates_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "results.json", Path(directory) / "errors.json"
            good = score_result_row({"answer_entities": {"m.correct": "Correct"}},
                                    {"prediction": ["m.correct"], "index": 0})
            failed = {**good, "index": 1, "failure": "timeout"}
            # A filtered old checkpoint omitted the perfect row but kept its
            # successful-only aggregate; the partial failed answer scored 1.
            old_summary = summarize_result_rows([good])
            old_summary.update(total=2, unlabeled=1)
            metadata = {"version": "answer-id-primary-v3"}
            output.write_text(json.dumps({"evaluation": metadata, "rows": [],
                "count": 2, "success": 1, "summary": old_summary,
                "storage": {"error_rows": 1, "metric_rows": 1}}))
            errors.write_text(json.dumps({"evaluation": metadata, "rows": [failed],
                                          "count": 1, "processed_count": 2}))
            state = _load_resume_state(output, errors)
            self.assertEqual(state["summary"]["total"], 2)
            self.assertEqual(state["summary"]["scored"], 2)
            self.assertEqual(state["summary"]["unlabeled"], 0)
            self.assertEqual(state["summary"]["f1"], 0.5)
            self.assertEqual(state["metric_rows"], 2)
            self.assertEqual(state["error_rows_data"][0]["score"]["f1"], 0.0)
            _write_output(output, [], config_path="synthetic", evaluation_source="synthetic",
                          write_only_nonperfect=True, errors_path=errors, resume_state=state)
            again = _load_resume_state(output, errors)
            self.assertEqual(again["summary"], state["summary"])
            self.assertEqual(again["metric_rows"], 2)
            self.assertEqual(json.loads(output.read_text())["evaluation"]["version"], EVALUATION_VERSION)

    def test_resume_rejects_missing_error_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "results.json", Path(directory) / "errors.json"
            output.write_text(json.dumps({"evaluation": {"version": "answer-id-primary-v3"},
                "rows": [], "count": 2, "success": 1, "storage": {"error_rows": 1}}))
            with self.assertRaisesRegex(ValueError, "missing/incomplete"):
                _load_resume_state(output, errors)

    def test_resume_rejects_mixed_or_unknown_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "results.json", Path(directory) / "errors.json"
            output.write_text(json.dumps({"evaluation": {"version": EVALUATION_VERSION}, "rows": []}))
            errors.write_text(json.dumps({"evaluation": {"version": "answer-id-primary-v3"}, "rows": []}))
            with self.assertRaisesRegex(ValueError, "versions differ"):
                _load_resume_state(output, errors)
            output.write_text(json.dumps({"evaluation": {"version": "unknown"}, "rows": []}))
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                _load_resume_state(output, errors)

    def test_error_output_is_separate_but_in_summary_and_rewind(self):
        record = {"answer_entities": {"m.correct": "Correct"}}
        good = score_result_row(record, {"prediction": ["m.correct"], "index": 0})
        failed = score_result_row(record, {"prediction": [], "index": 1, "failure": "timeout"})
        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "results.json", Path(directory) / "errors.json"
            _write_output(output, [good, failed], config_path="synthetic", evaluation_source="synthetic",
                          write_only_nonperfect=False, errors_path=errors)
            result = json.loads(output.read_text())
            self.assertEqual(len(result["rows"]), 1)
            self.assertEqual(len(json.loads(errors.read_text())["rows"]), 1)
            self.assertEqual(result["summary"]["scored"], 2)
            self.assertEqual(result["summary"]["f1"], 0.5)
            self.assertEqual(result["storage"]["metric_rows"], 2)
            state = _load_resume_state(output, errors)
            rewound = _rewind_resume_state(state, start=1)
            self.assertEqual(rewound["summary"]["f1"], 1.0)
            self.assertEqual(rewound["metric_rows"], 1)

    def test_all_failures_are_scored(self):
        row = score_result_row({"answer_entities": {"m.correct": "Correct"}},
                               {"prediction": [], "failure": "timeout"})
        summary = summarize_result_rows([row, row])
        self.assertEqual(summary["scored"], 2)
        self.assertEqual(summary["f1"], 0.0)

    def test_rewind_unlabeled_rows_keeps_correct_observed_count(self):
        good = score_result_row({"answer_entities": {"m.correct": "Correct"}},
                                {"prediction": ["m.correct"]})
        unknown = score_result_row({}, {"prediction": [], "failure": "timeout"})
        combined = summarize_result_rows([good, unknown])
        result = _subtract_summaries(combined, summarize_result_rows([unknown]))
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["unlabeled"], 0)
        self.assertEqual(result["f1"], 1.0)

    def test_baseline_filter_uses_primary_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "000000"
            item.mkdir()
            (item / "evaluation.json").write_text(json.dumps({
                "question": "synthetic", "score": {"f1": 0.0},
                "legacy_label_score": {"f1": 1.0}}))
            (item / "result.json").write_text(json.dumps({"failure": None}))
            self.assertTrue(_baseline_f1_is_zero(root, index=0, question="synthetic", include_failures=False))
            self.assertFalse(_baseline_f1_is_zero(root, index=0, question="mismatch", include_failures=False))
        args = build_parser().parse_args(["--config", "synthetic.json", "--output", "synthetic.json",
                                          "--baseline-f1-zero-only"])
        self.assertTrue(args.baseline_f1_zero_only)

    def test_old_primary_resume_drops_auxiliary_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            row = score_result_row({"answer_entities": {"m.correct": "Correct"}},
                                   {"prediction": ["m.correct"], "index": 0})
            row["legacy_label_score"] = {"f1": 1.0}
            summary = summarize_result_rows([row])
            summary["legacy_label"] = {"f1": 1.0}
            output.write_text(json.dumps({"evaluation": {"version": "webqsp-id-aware-v2"},
                                          "rows": [row], "summary": summary, "count": 1}))
            state = _load_resume_state(output, output.with_name("errors.json"))
            self.assertNotIn("legacy_label_score", state["successful_rows"][0])
            self.assertNotIn("legacy_label", state["summary"])
            self.assertEqual(state["summary"]["f1"], 1.0)

    def test_artifact_contains_only_primary_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = score_result_row({"answer_entities": {"m.correct": "Correct"}},
                                   {"prediction": ["m.correct"], "index": 0, "question": "synthetic"})
            _write_question_artifacts(root, row, config_path="synthetic.json", evaluation_source="synthetic")
            result = json.loads((root / "000000/evaluation.json").read_text())
            self.assertIn("score", result)
            self.assertFalse(any(key.endswith("_label_score") for key in result))


class TrainReviewExampleTests(unittest.TestCase):
    def test_only_source_verified_cwq_train_examples_are_loaded(self):
        examples = review_prompts._load_examples() + review_prompts._load_preserve_examples()
        self.assertEqual(len(examples), 17)
        self.assertEqual(len({x['question'] for x in examples}), 17)
        for item in examples:
            self.assertEqual(item['source']['dataset'], 'CWQ')
            self.assertEqual(item['source']['split'], 'train')
            self.assertTrue(item['verification']['source_graph_verified'])
            self.assertTrue(item['verification']['entity_bindings_verified'])
            self.assertNotIn('candidate_f1', item['verification'])
            self.assertFalse(item['verification']['full_pipeline_verified'])

    def test_train_targets_match_the_released_training_rows(self):
        path = ROOT / 'data/train/cwq/decompose_train.json'
        if not path.is_file():
            self.skipTest('TRAIN exports are not installed')
        train = json.loads(path.read_text())
        for item in review_prompts._load_examples() + review_prompts._load_preserve_examples():
            row = train[item['source']['row_index']]
            self.assertEqual(json.loads(row['input'])['question'], item['question'])
            good = item.get('corrected_decomposition', item.get('decomposition'))
            self.assertEqual(json.loads(row['output'])['decomposition'], good)

    def test_original_example_counts_and_context_schema_are_kept(self):
        for include_context, review_count, rewrite_count in ((False,14,9),(True,17,11)):
            for action, count in (('review', review_count), ('rewrite', rewrite_count)):
                text = review_prompts.cwq_examples(action, include_entity_context=include_context)
                requests = [json.loads(line.split(': ',1)[1]) for line in text.splitlines()
                    if line.startswith(('CWQ preserve request: ', 'CWQ example request: '))]
                self.assertEqual(len(requests), count)
                for request in requests:
                    self.assertNotIn('gold_answers', request)
                    self.assertNotIn('sparql', request)
                    self.assertEqual('entity_context' in request, include_context)
                    if include_context:
                        self.assertTrue(request['entity_context'])
                        self.assertTrue(all(set(x) == {'id','name','types'} for x in request['entity_context']))

    def test_nontrain_resource_metadata_is_rejected(self):
        payload = json.loads(review_prompts.EXAMPLE_PATH.read_text())
        payload['examples'][0]['source']['split'] = 'test'
        with patch.object(review_prompts.json, 'loads', return_value=payload):
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                review_prompts._load_examples()

    def test_two_call_budget_and_schemas_are_unchanged(self):
        for reasonable, expected_calls in ((True,1),(False,2)):
            class FakeModel:
                config = types.SimpleNamespace(retries=0, model='offline-stub')
                def __init__(self): self.calls = []
                def generate_json(self, **kwargs):
                    self.calls.append(kwargs)
                    if len(self.calls) == 1:
                        return [{'is_reasonable':reasonable, 'issues':[] if reasonable else ['missing terminal'], 'reason':'offline stub'}]
                    return [{'decomposition':['What is the state of Example University?']}]
            model = FakeModel()
            reviewer = DecompositionReviewer(model, workflow='two_call', prompt_family='cwq')
            result, trace = reviewer.run('What state contains Example University?', ['Where is Example University?'])
            self.assertIsNotNone(result)
            self.assertEqual(len(model.calls), expected_calls)
            self.assertEqual(trace['model_call_count'], expected_calls)
            self.assertTrue(all(call['count'] == 1 for call in model.calls))
            self.assertIn('CWQ TRAIN questions', model.calls[0]['instruction'])

    def test_builder_relation_normalization_matches_runtime(self):
        spec = importlib.util.spec_from_file_location('train_review_builder', ROOT / 'scripts/build_cwq_review_train_examples.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from semantic_guided_kbqa.ontology import relation_id_from_label
        for label in (['sports','sports team','owner s'], ['people','person','place of birth'], [], 'invalid'):
            self.assertEqual(module.relation_id_from_label(label), relation_id_from_label(label))

    def test_builder_checks_graph_direction_and_answer_identity(self):
        spec = importlib.util.spec_from_file_location('train_graph_builder', ROOT / 'scripts/build_cwq_review_train_examples.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        source = {'sparql': 'SELECT DISTINCT ?x WHERE { ns:m.anchor ns:example.person.field ?x . }'}
        compose = {'entities':[{'id':'E0','surface':'Anchor'}], 'answer_var':'V0',
                   'triples':[{'subject':'E0','relation_label':['example','person','field'],'object':'V0'}]}
        self.assertTrue(module.source_graph_matches(compose, source, {'Anchor':'m.anchor'}))
        wrong_answer = {**compose, 'answer_var':'E0'}
        self.assertFalse(module.source_graph_matches(wrong_answer, source, {'Anchor':'m.anchor'}))
        wrong_direction = {**compose, 'triples':[{'subject':'V0','relation_label':['example','person','field'],'object':'E0'}]}
        self.assertFalse(module.source_graph_matches(wrong_direction, source, {'Anchor':'m.anchor'}))

    def test_builder_rejects_wrong_entity_name_binding(self):
        spec = importlib.util.spec_from_file_location('train_entity_builder', ROOT / 'scripts/build_cwq_review_train_examples.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        source = {'sparql': 'SELECT ?x WHERE { ns:m.anchor ns:example.person.field ?x . }'}
        semantic = {'anchors':[{'id':'A0','surface':'Wrong Name'}], 'semantic_paths':[
            {'anchor_ref':'A0','steps':[{'relation_label':['example','person','field'],'direction':'forward'}]}]}
        self.assertEqual(module.entity_context(semantic, source, {'m.anchor':'Canonical Name'}), ([],{}))


class SourceInterfaceTests(unittest.TestCase):
    def test_arbitrary_archive_root_and_json(self):
        rows = [{"question": "synthetic", "reasoning_information": []}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("any-folder/data/processed/CWQ/CWQ_train_extracted.json", json.dumps(rows))
            selected_rows, member, raw = load_source_annotations(archive)
            self.assertEqual(selected_rows, rows)
            self.assertTrue(member.startswith("any-folder/"))
            source = root / "source.json"
            source.write_bytes(raw)
            self.assertEqual(load_source_annotations(source)[0], rows)

    def test_ambiguous_archive_requires_explicit_member(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("a/CWQ_train_extracted.json", "[]")
                handle.writestr("b/CWQ_train_extracted.json", "[]")
            with self.assertRaises(ValueError):
                load_source_annotations(archive)
            self.assertEqual(load_source_annotations(archive, "b/CWQ_train_extracted.json")[0], [])

    def test_executor_is_configurable_and_fails_closed(self):
        spec = importlib.util.spec_from_file_location("public_execution_validator",
                    ROOT / "dataset_construction/scripts/cwq_execution_validator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {"FREEBASE_EXECUTOR_MODULE": "", "FREEBASE_EXECUTOR_SRC": ""}):
            with self.assertRaises(RuntimeError):
                module._execution_backend()
        backend = types.ModuleType("synthetic_executor")
        with patch.dict(sys.modules, {"synthetic_executor": backend}), \
             patch.dict(os.environ, {"FREEBASE_EXECUTOR_MODULE": "synthetic_executor"}):
            self.assertIs(module._execution_backend(), backend)


if __name__ == "__main__":
    unittest.main()
