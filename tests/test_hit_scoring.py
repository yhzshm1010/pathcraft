"""Offline checks for set-overlap Hit and safe evaluation-version migration."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_guided_kbqa.cli import (
    _load_resume_state, _merge_summaries, _question_metric_delta,
    _rewind_resume_state, _subtract_summaries, _write_output,
    _write_question_artifacts,
)
from semantic_guided_kbqa.evaluation import (
    EVALUATION_VERSION, evaluation_metadata, primary_result_row,
    score_result_row, score_webqsp_answers, summarize_result_rows,
)


class SetOverlapHitTests(unittest.TestCase):
    @staticmethod
    def row(prediction, *, gold=("m.correct",), index=0, **extra):
        return score_result_row(
            {"answer_entities": {item: item for item in gold}},
            {"prediction": prediction, "index": index,
             "question": f"synthetic-{index}", **extra},
        )

    def test_version_identifies_the_added_set_hit(self):
        self.assertEqual(EVALUATION_VERSION, "answer-id-primary-v5-hit-failures-zero")
        metadata = evaluation_metadata("synthetic")
        self.assertIn("hit_definition", metadata)
        self.assertIn("hits_at_1_definition", metadata)

    def test_later_correct_id_hits_without_first_answer_hit(self):
        row = self.row(["m.wrong", "http://rdf.freebase.com/ns/m.correct", "m.correct"])
        self.assertEqual(row["prediction"], ["m.wrong", "m.correct"])
        self.assertEqual(row["score"]["hit"], 1.0)
        self.assertEqual(row["score"]["hits_at_1"], 0.0)
        self.assertEqual(row["score"]["exact_match"], 0.0)
        self.assertAlmostEqual(row["score"]["f1"], 2 / 3)

    def test_partial_recall_still_hits(self):
        score = self.row(["m.correct"], gold=("m.correct", "m.other"))["score"]
        self.assertEqual(score["recall"], 0.5)
        self.assertEqual(score["hit"], 1.0)
        self.assertEqual(score["hits_at_1"], 1.0)
        self.assertEqual(score["exact_match"], 0.0)

    def test_extras_do_not_remove_hit_but_prevent_exact_match(self):
        score = self.row(["m.extra", "m.correct"])["score"]
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["precision"], 0.5)
        self.assertEqual(score["hit"], 1.0)
        self.assertEqual(score["hits_at_1"], 0.0)
        self.assertEqual(score["exact_match"], 0.0)

    def test_set_hit_is_order_independent_but_first_hit_is_not(self):
        forward = self.row(["m.wrong", "m.correct"])["score"]
        reverse = self.row(["m.correct", "m.wrong"])["score"]
        self.assertEqual(forward["hit"], reverse["hit"])
        self.assertEqual((forward["hits_at_1"], reverse["hits_at_1"]), (0.0, 1.0))
        self.assertEqual(forward["f1"], reverse["f1"])

    def test_matching_labels_do_not_override_different_entity_ids(self):
        row = score_result_row(
            {"answer_entities": {"m.correct": "Same label"}},
            {"prediction": ["m.wrong"],
             "answers": [{"id": "m.wrong", "label": "Same label"}]},
        )
        self.assertEqual(row["score_basis"], "answer_ids")
        self.assertEqual(row["score"]["hit"], 0.0)

    def test_no_overlap_and_empty_prediction_do_not_hit(self):
        for prediction in (["m.wrong"], []):
            with self.subTest(prediction=prediction):
                score = self.row(prediction)["score"]
                self.assertEqual(score["hit"], 0.0)
                self.assertEqual(score["hits_at_1"], 0.0)
                self.assertEqual(score["f1"], 0.0)

    def test_text_set_hit_uses_normalization_and_all_predictions(self):
        score = score_webqsp_answers(["New York"], ["Boston", "NEW-YORK!", "new york"])
        self.assertTrue(score["labeled"])
        self.assertEqual(score["hit"], 1.0)
        self.assertEqual(score["hits_at_1"], 0.0)
        self.assertAlmostEqual(score["f1"], 2 / 3)

    def test_literal_result_row_emits_set_hit(self):
        row = score_result_row(
            {"answer_entities": {"Value": "2001"}},
            {"prediction": ["2002", '"2001"^^xsd:gYear']},
        )
        self.assertEqual(row["score_basis"], "answer_text")
        self.assertEqual(row["score"]["hit"], 1.0)
        self.assertEqual(row["score"]["hits_at_1"], 0.0)

    def test_missing_or_blank_gold_is_unscored_not_a_hit_zero(self):
        for record in ({}, {"answer_entities": {}}, {"answer_entities": {"Value": "  "}}):
            with self.subTest(record=record):
                score = score_result_row(record, {"prediction": ["m.correct"]})["score"]
                self.assertFalse(score["labeled"])
                self.assertIsNone(score["hit"])
                self.assertIsNone(score["hits_at_1"])
        self.assertIsNone(score_webqsp_answers([], ["answer"])["hit"])
        failed = score_result_row({}, {"prediction": ["m.correct"], "failure": "timeout"})
        self.assertFalse(failed["score"]["labeled"])
        self.assertIsNone(failed["score"]["hit"])

    def test_failure_hit_is_zero_even_when_diagnostic_prediction_overlaps(self):
        for record, prediction in (
            ({"answer_entities": {"m.correct": "Correct"}}, ["m.correct"]),
            ({"answer_entities": {"Value": "2001"}}, ["2001"]),
        ):
            with self.subTest(record=record):
                original = {"prediction": prediction, "failure": "synthetic timeout"}
                row = score_result_row(record, original)
                self.assertEqual(row["score"]["hit"], 0.0)
                self.assertEqual(row["score"]["f1"], 0.0)
                self.assertEqual(row["prediction"], prediction)
                self.assertNotIn("score", original)

    def test_legacy_primary_row_gets_hit_without_changing_other_metrics(self):
        row = self.row(["m.wrong", "m.correct"])
        row["score"].pop("hit", None)
        original = deepcopy(row)
        migrated = primary_result_row(row)
        self.assertEqual(migrated["score"]["hit"], 1.0)
        self.assertEqual({key: value for key, value in migrated["score"].items() if key != "hit"},
                         original["score"])
        self.assertEqual(row, original)
        row["score"]["f1"] = 0.0
        self.assertEqual(primary_result_row(row)["score"]["hit"], 0.0)

    def test_macro_hit_includes_failures_and_excludes_missing_gold(self):
        partial = self.row(["m.wrong", "m.correct"], index=0)
        failed = self.row(["m.correct"], index=1, failure="timeout")
        wrong = self.row(["m.wrong"], index=2)
        unknown = score_result_row({}, {"prediction": ["m.correct"], "index": 3})
        summary = summarize_result_rows([partial, failed, wrong, unknown])
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["scored"], 3)
        self.assertEqual(summary["unlabeled"], 1)
        self.assertAlmostEqual(summary["hit"], 1 / 3)
        self.assertEqual(summary["hits_at_1"], 0.0)
        self.assertAlmostEqual(summary["macro_f1"], 2 / 9)

    def test_summary_adds_legacy_row_hit_without_mutation(self):
        legacy = self.row(["m.wrong", "m.correct"])
        legacy["score"].pop("hit", None)
        original = deepcopy(legacy)
        self.assertEqual(summarize_result_rows([legacy])["hit"], 1.0)
        self.assertEqual(legacy, original)

    def test_labeled_existing_hit_must_be_a_binary_value(self):
        for invalid in (None, -1.0, 0.5, 2.0, float("nan")):
            with self.subTest(hit=invalid):
                row = self.row(["m.correct"])
                row["score"]["hit"] = invalid
                with self.assertRaises(ValueError):
                    summarize_result_rows([row])

    def test_summary_merge_and_subtract_preserve_independent_hit(self):
        first = summarize_result_rows([self.row(["m.wrong", "m.correct"])])
        second = summarize_result_rows([
            self.row(["m.correct"], index=1), self.row(["m.wrong"], index=2),
        ])
        merged = _merge_summaries(first, second)
        self.assertAlmostEqual(merged["hit"], 2 / 3)
        self.assertAlmostEqual(merged["hits_at_1"], 1 / 3)
        remaining = _subtract_summaries(merged, second)
        self.assertAlmostEqual(remaining["hit"], 1.0)
        self.assertAlmostEqual(remaining["hits_at_1"], 0.0)
        empty = _subtract_summaries(first, first)
        self.assertIsNone(empty["hit"])

    def test_aggregate_without_hit_is_not_silently_treated_as_zero(self):
        valid = summarize_result_rows([self.row(["m.wrong", "m.correct"])])
        legacy = {key: value for key, value in valid.items() if key != "hit"}
        with self.assertRaises(ValueError):
            _merge_summaries(legacy, valid)
        with self.assertRaises(ValueError):
            _subtract_summaries(legacy, valid)

    def test_hit_subtraction_roundoff_does_not_prevent_continuation(self):
        total = {"scored": 49, "count": 49, "hit": 1 / 49}
        removed = {"scored": 1, "count": 1, "hit": 1.0}
        remaining = _subtract_summaries(total, removed)
        self.assertEqual(remaining["hit"], 0.0)
        self.assertEqual(_merge_summaries(remaining, {})["hit"], 0.0)
        with self.assertRaises(ValueError):
            _subtract_summaries({"scored": 2, "hit": 0.0}, removed)

    def test_json_output_resume_and_rewind_keep_set_hit(self):
        rows = [self.row(["m.wrong", "m.correct"], index=0),
                self.row(["m.correct"], index=1, failure="timeout")]
        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "run.json", Path(directory) / "errors.json"
            _write_output(output, rows, config_path="synthetic", evaluation_source="synthetic",
                          write_only_nonperfect=False, errors_path=errors)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["evaluation"]["version"], EVALUATION_VERSION)
            self.assertEqual(payload["rows"][0]["score"]["hit"], 1.0)
            self.assertEqual(payload["summary"]["hit"], 0.5)
            self.assertEqual(json.loads(errors.read_text())["rows"][0]["score"]["hit"], 0.0)
            state = _load_resume_state(output, errors)
            self.assertEqual(state["summary"]["hit"], 0.5)
            rewound = _rewind_resume_state(state, start=1)
            self.assertIsNotNone(rewound)
            self.assertEqual(rewound["summary"]["hit"], 1.0)
            self.assertEqual(rewound["summary"]["hits_at_1"], 0.0)

    def test_question_artifact_and_legacy_baseline_delta_emit_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.row(["m.wrong"], index=0)
            _write_question_artifacts(root, baseline, config_path="synthetic",
                                      evaluation_source="synthetic")
            artifact = root / "000000/evaluation.json"
            payload = json.loads(artifact.read_text())
            self.assertEqual(payload["score"]["hit"], 0.0)
            payload["score"].pop("hit")
            artifact.write_text(json.dumps(payload))
            current = self.row(["m.wrong", "m.correct"], index=0)
            delta = _question_metric_delta(root, current=current, index=0,
                                           question="synthetic-0")
            self.assertTrue(delta["baseline_available"])
            self.assertEqual(delta["score"]["hit"], 1.0)
            self.assertEqual(delta["score"]["hits_at_1"], 0.0)

    def test_baseline_failure_uses_zero_hit_despite_retained_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.row(["m.correct"], index=0)
            _write_question_artifacts(root, baseline, config_path="synthetic",
                                      evaluation_source="synthetic")
            result = root / "000000/result.json"
            payload = json.loads(result.read_text())
            payload["failure"] = "timeout"
            result.write_text(json.dumps(payload))
            delta = _question_metric_delta(root, current=self.row(["m.correct"]),
                                           index=0, question="synthetic-0")
            self.assertEqual(delta["score"]["hit"], 1.0)


class LegacyHitResumeTests(unittest.TestCase):
    # Helpers below intentionally generate only synthetic, local checkpoints.
    row = staticmethod(SetOverlapHitTests.row)
    versions = ("answer-id-primary-v4-failures-zero", "answer-id-primary-v3",
                "webqsp-id-aware-v2")

    def checkpoint(self, output, errors, version, *, filtered=False, declared_filter=True):
        overlap = self.row(["m.wrong", "m.correct"], index=0)
        second = self.row(["m.correct"] if filtered else ["m.wrong"], index=1)
        failed = self.row([], index=2, failure="timeout")
        successful = [overlap, second]
        legacy_summary = summarize_result_rows(successful + [failed] if "v4" in version else successful)
        legacy_summary.pop("hit", None)
        if "v4" not in version:
            legacy_summary.update(total=3, unlabeled=1)
        persisted = [overlap] if filtered else successful
        for row in persisted + [failed]:
            row["score"].pop("hit", None)
        storage = {"error_rows": 1, "metric_rows": 3 if "v4" in version else 2}
        if declared_filter:
            storage["row_filter"] = "id_f1_or_recall_nonperfect" if filtered else "all"
        metadata = {"version": version}
        output.write_text(json.dumps({"evaluation": metadata, "rows": persisted,
            "summary": legacy_summary, "count": 3, "success": 2, "storage": storage}))
        errors.write_text(json.dumps({"evaluation": metadata, "rows": [failed],
                                     "count": 1, "processed_count": 3}))

    def test_full_legacy_rows_reconstruct_hit_for_all_known_versions(self):
        for version in self.versions:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                output, errors = Path(directory) / "run.json", Path(directory) / "errors.json"
                self.checkpoint(output, errors, version)
                original = output.read_bytes(), errors.read_bytes()
                state = _load_resume_state(output, errors)
                self.assertAlmostEqual(state["summary"]["hit"], 1 / 3)
                self.assertEqual(state["summary"]["hits_at_1"], 0.0)
                self.assertEqual(state["summary"]["scored"], 3)
                self.assertEqual(state["metric_rows"], 3)
                self.assertEqual(state["successful_rows"][0]["score"]["hit"], 1.0)
                self.assertEqual(state["error_rows_data"][0]["score"]["hit"], 0.0)
                self.assertEqual((output.read_bytes(), errors.read_bytes()), original)

    def test_declared_filtered_perfect_rows_reconstruct_hit(self):
        for version in self.versions:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                output, errors = Path(directory) / "run.json", Path(directory) / "errors.json"
                self.checkpoint(output, errors, version, filtered=True)
                state = _load_resume_state(output, errors)
                self.assertAlmostEqual(state["summary"]["hit"], 2 / 3)
                self.assertAlmostEqual(state["summary"]["hits_at_1"], 1 / 3)
                _write_output(output, [], config_path="synthetic", evaluation_source="synthetic",
                              write_only_nonperfect=True, errors_path=errors, resume_state=state)
                again = _load_resume_state(output, errors)
                self.assertEqual(again["summary"], state["summary"])
                self.assertEqual(json.loads(output.read_text())["evaluation"]["version"], EVALUATION_VERSION)

    def test_missing_filtered_row_evidence_is_rejected_not_counted_as_zero(self):
        for version in self.versions:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                output, errors = Path(directory) / "run.json", Path(directory) / "errors.json"
                self.checkpoint(output, errors, version, filtered=True, declared_filter=False)
                with self.assertRaises(ValueError):
                    _load_resume_state(output, errors)

    def test_claim_all_rows_but_omit_one_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "run.json", Path(directory) / "errors.json"
            self.checkpoint(output, errors, self.versions[0], filtered=True)
            payload = json.loads(output.read_text())
            payload["storage"]["row_filter"] = "all"
            output.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                _load_resume_state(output, errors)

    def test_inconsistent_filtered_metric_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output, errors = Path(directory) / "run.json", Path(directory) / "errors.json"
            self.checkpoint(output, errors, self.versions[0], filtered=True)
            payload = json.loads(output.read_text())
            payload["storage"]["metric_rows"] = 0
            output.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                _load_resume_state(output, errors)

    def test_filtered_aggregate_must_match_the_omitted_perfect_rows(self):
        for version in self.versions:
            for metric in ("f1", "exact"):
                with self.subTest(version=version, metric=metric), tempfile.TemporaryDirectory() as directory:
                    output, errors = Path(directory) / "run.json", Path(directory) / "errors.json"
                    self.checkpoint(output, errors, version, filtered=True)
                    payload = json.loads(output.read_text())
                    # Counts and the explicit perfect-row filter are valid,
                    # but these sums cannot contain the omitted perfect item.
                    if metric == "f1":
                        payload["summary"].update(f1=0.1, macro_f1=0.1)
                    else:
                        payload["summary"].update(exact=0.0, exact_match=0.0)
                    output.write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):
                        _load_resume_state(output, errors)


if __name__ == "__main__":
    unittest.main()
