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
    canonicalize_answer_id, evaluation_metadata, primary_result_row,
    score_result_row, summarize_result_rows,
)
from semantic_guided_kbqa.cli import (
    _baseline_f1_is_zero, _load_resume_state, _merge_summaries,
    _subtract_summaries, _write_question_artifacts, build_parser,
)
from source_annotations import load_source_annotations


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

    def test_aggregation_policy_is_preserved(self):
        record = {"answer_entities": {"m.correct": "Correct"}}
        good = score_result_row(record, {"prediction": ["m.correct"]})
        wrong = score_result_row(record, {"prediction": ["m.wrong"]})
        error = score_result_row(record, {"prediction": [], "failure": "synthetic error"})
        summary = summarize_result_rows([good, wrong, error])
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["scored"], 2)
        self.assertEqual(summary["macro_f1"], 0.5)
        self.assertFalse(any(key.endswith("_label") for key in summary))
        combined = _merge_summaries(summarize_result_rows([good]), summarize_result_rows([wrong]))
        self.assertEqual(combined["macro_f1"], 0.5)
        self.assertEqual(_subtract_summaries(combined, summarize_result_rows([wrong]))["macro_f1"], 1.0)

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
