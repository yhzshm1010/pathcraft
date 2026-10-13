"""Build frozen Review demonstrations from aligned CWQ TRAIN annotations only.

No model or endpoint calls. Correct targets must match the official training
query graph; negative inputs are controlled corruptions, not test repairs.
Use --emit-patch to update packaged resources through apply_patch.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import difflib
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]

TRIPLE = re.compile(r"(\?[A-Za-z_]\w*|ns:[mg]\.[\w]+)\s+ns:([\w.]+)\s+(\?[A-Za-z_]\w*|ns:[mg]\.[\w]+)\s*\.")


def relation_id_from_label(label):
    # The same reversible segment convention as ontology.py; no pipeline
    # imports, so this builder can repair resources that fail runtime checks.
    if not isinstance(label, (list, tuple)):
        return ""
    segments = [re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_") for value in label]
    return ".".join(segments) if segments and all(segments) else ""


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_key(value):
    return " ".join(str(value).casefold().split())


def source_graph_matches(compose, source, bindings):
    """Exact directed triple multiset isomorphism, including answer identity."""
    query = str(source.get("sparql", ""))
    selected = re.search(r"SELECT\s+(?:DISTINCT\s+)?(\?\w+)", query, re.I)
    triples = TRIPLE.findall(query)
    edges = compose.get("triples", [])
    if not selected or not triples or len(triples) != len(edges):
        return False
    candidate = [(str(e.get("subject")), relation_id_from_label(e.get("relation_label")),
                  str(e.get("object"))) for e in edges]
    entities = {str(e["id"]): bindings.get(str(e["surface"])) for e in compose.get("entities", [])}
    used_entities = {x for s, _, o in candidate for x in (s, o) if x in entities}
    if any(not entities[x] for x in used_entities):
        return False
    variables = {x for s, _, o in candidate for x in (s, o) if x not in entities}
    source_variables = {x for s, _, o in triples for x in (s, o) if x.startswith("?")}
    answer = str(compose.get("answer_var", ""))
    if answer not in variables or selected[1] not in source_variables or len(variables) != len(source_variables) or len(variables) > 6:
        return False
    rest, targets = sorted(variables - {answer}), sorted(source_variables - {selected[1]})
    expected = Counter(triples)
    for ordering in permutations(targets):
        mapping = {answer: selected[1], **dict(zip(rest, ordering)),
                   **{key: "ns:" + value for key, value in entities.items() if value}}
        if Counter((mapping[s], rel, mapping[o]) for s, rel, o in candidate) == expected:
            return True
    return False


def entity_context(semantic, source, entity_records):
    triples = TRIPLE.findall(str(source.get("sparql", "")))
    surfaces = {a["id"]: a["surface"] for a in semantic.get("anchors", [])}
    result, bindings = [], {}
    for path in semantic.get("semantic_paths", []):
        steps = path.get("steps", [])
        name = surfaces.get(path.get("anchor_ref"))
        if not steps or not name:
            return [], {}
        step = steps[0]
        relation = relation_id_from_label(step.get("relation_label"))
        backward = step.get("direction") == "backward"
        constants = {o if backward else s for s, rel, o in triples
                     if rel == relation and (o if backward else s).startswith("ns:")}
        if len(constants) != 1:
            return [], {}
        mid = next(iter(constants))[3:]
        if question_key(entity_records.get(mid, "")) != question_key(name):
            # Never associate a TRAIN surface with a different Gold-query MID.
            return [], {}
        if name in bindings and bindings[name] != mid:
            return [], {}
        if name not in bindings:
            # Names come from TRAIN anchors; no types/dates are invented.
            result.append({"id": mid, "name": name, "types": []})
        bindings[name] = mid
    return result, bindings


def corruption(good, semantic):
    # Remove one terminal question while keeping its original anchor chain.
    for index, text in enumerate(good):
        clauses = re.findall(r"[^?]+\?", text)
        if len(clauses) >= 2 and not text.split("?")[-1].strip():
            bad = list(good)
            bad[index] = "".join(clauses[:-1]).strip()
            if bad[index]:
                return bad, "drop_terminal_hop", "The path omits its final hop and stops before the requested terminal."
    # A replacement target is demonstrably wrong for a multi-hop TRAIN graph.
    paths = semantic.get("semantic_paths", [])
    if paths and max(len(p.get("steps", [])) for p in paths) >= 2:
        surfaces = {a["id"]: a["surface"] for a in semantic["anchors"]}
        name = surfaces[paths[0]["anchor_ref"]]
        return [f"What is the name of {name}?"], "replace_requested_attribute", "The plan returns the anchor's name instead of following the required bridge to the requested answer."
    return None


def build(train_dir, source_train, heldout=None, train_entity_records=None):
    names = ("decompose_train.json", "semantic_path_train.json", "compose_train.json", "operator_train.json")
    arrays = [json.loads((train_dir / name).read_text()) for name in names]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("TRAIN stage arrays are not aligned")
    sources = defaultdict(list)
    for row in json.loads(source_train.read_text()):
        sources[str(row.get("question", ""))].append(row)
    entity_path = train_entity_records or ROOT / "assets/source_annotations.zip"
    entity_member = ""
    if entity_path.suffix == ".zip":
        with ZipFile(entity_path) as archive:
            members = [name for name in archive.namelist()
                       if name.endswith("/CWQ_train_extracted.json") or name == "CWQ_train_extracted.json"]
            if len(members) != 1:
                raise ValueError("Need exactly one CWQ TRAIN entity-annotation member")
            entity_member = members[0]
            entity_rows = json.loads(archive.read(entity_member))
    else:
        entity_rows = json.loads(entity_path.read_text())
    links = {str(row.get("question", "")): row.get("golden_entities", row.get("gold_entities", {}))
             for row in entity_rows}
    excluded = set()
    if heldout:
        text = heldout.read_text()
        rows = [json.loads(x) for x in text.splitlines() if x] if heldout.suffix == ".jsonl" else json.loads(text)
        excluded = {question_key(row.get("question", "")) for row in rows}
    candidates, seen = [], set()
    for index, aligned in enumerate(zip(*arrays)):
        inputs, outputs = [], []
        for row in aligned:
            inputs.append(json.loads(row["input"]))
            outputs.append(json.loads(row["output"]))
        question = str(inputs[0].get("question", ""))
        good = outputs[0].get("decomposition", [])
        if question_key(question) in excluded or question in seen or not good or not question.isascii():
            continue
        if any(i.get("question") != question for i in inputs) or inputs[1].get("decomposition") != good:
            continue
        if inputs[2].get("semantic_paths") != outputs[1].get("semantic_paths"):
            continue
        if inputs[3].get("triples") != outputs[2].get("triples"):
            continue
        paths = outputs[1].get("semantic_paths", [])
        if Counter(str(path.get("goal", "")).strip() for path in paths) != Counter(str(text).strip() for text in good):
            continue
        refs = [path.get("anchor_ref") for path in paths]
        if len(refs) != len(set(refs)):
            continue
        anchor_names = {a.get("id"): question_key(a.get("surface", "")) for a in outputs[1].get("anchors", [])}
        used_names = [anchor_names.get(ref) for ref in refs]
        if not all(used_names) or len(used_names) != len(set(used_names)):
            continue
        if any(anchor_names[ref] not in question_key(path.get("goal", "")) for ref, path in zip(refs, paths)):
            continue
        if any(len({relation_id_from_label(step.get("relation_label")) for step in path.get("steps", [])})
               != len(path.get("steps", [])) for path in paths):
            continue
        if any(len([part for part in re.split(r"(?<=[.?])\s+", str(path.get("goal", "")).strip()) if part])
               != len(path.get("steps", [])) for path in paths):
            continue
        negative = corruption(good, outputs[1])
        if not negative:
            continue
        for source in sources.get(question, []):
            context, bindings = entity_context(outputs[1], source, links.get(question, {}))
            if not context or not source_graph_matches(outputs[2], source, bindings):
                continue
            # Deterministic compact examples; no F1-based ranking or selection.
            if len(json.dumps([question, good, context], ensure_ascii=False)) > 1050:
                continue
            provenance = {"dataset": "CWQ", "split": "train", "row_index": index,
                "source_id": source.get("ID"), "source_query_sha256": hashlib.sha256(source["sparql"].encode()).hexdigest(),
                "target_sha256": hashlib.sha256(json.dumps(good, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "stage_alignment_verified": True, "source_graph_verified": True}
            provenance["entity_bindings_verified"] = True
            candidates.append({"question": question, "decomposition": good, "source": provenance,
                "context": context, "negative": negative})
            seen.add(question)
            break
        if len(candidates) == 17:
            break
    if len(candidates) != 17:
        raise ValueError("Need 17 distinct source-verified CWQ TRAIN examples")
    repairs, preserves, contexts, generic = [], [], {}, []
    for n, item in enumerate(candidates):
        question, good, source = item["question"], item["decomposition"], item["source"]
        contexts[question] = item["context"]
        verification = {"split": "train", "stage_alignment_verified": True, "source_graph_verified": True,
                        "entity_bindings_verified": True,
                        "full_pipeline_verified": False, "validation_scope": "offline TRAIN graph alignment, not endpoint F1"}
        if n < 11:
            bad, mutation, issue = item["negative"]
            repairs.append({"source_index": source["row_index"], "question": question,
                "original_decomposition": bad, "corrected_decomposition": good,
                "error_type": issue, "source": source, "verification": verification})
            generic.append({"question": question, "decomposition": good, "source": source,
                "corruptions": [{"decomposition": bad, "mutation": mutation, "issues": [issue]}]})
        else:
            preserves.append({"source_index": source["row_index"], "question": question,
                "decomposition": good, "source": source, "verification": verification})
    policy = "CWQ TRAIN stage annotations verified against official TRAIN query graphs; controlled corruptions only; no test-derived repairs or answer facts."
    manifest = {"schema": "cwq-review-train-provenance-v1", "split": "train", "policy": policy,
        "selection": "first 17 distinct compact source-graph-isomorphic aligned TRAIN rows; no metric-based ranking",
        "repair_count": 11, "preserve_count": 6, "model_calls": 0, "endpoint_calls": 0,
        "sources": [{"file": "data/train/cwq/" + name, "sha256": fingerprint(train_dir / name)} for name in names],
        "official_train": {"file": "assets/data/cwq/ComplexWebQuestions_train.json", "sha256": fingerprint(source_train)},
        "train_entity_annotations": {"file": "provided CWQ TRAIN annotations", "member": entity_member,
                                     "sha256": fingerprint(entity_path), "only_topic_entity_mapping_used": True},
        "heldout_question_exclusion_applied": bool(heldout),
        "heldout_text_overlap": len({question_key(x['question']) for x in candidates} & excluded),
        "examples": [x["source"] for x in candidates]}
    return {
        "glm_cwq_full_pipeline_examples.json": {"dataset": "CWQ", "split": "train", "description": policy, "examples": repairs},
        "glm_cwq_global_preserve_examples.json": {"version": "cwq-train-preserve-v2", "source": "official CWQ TRAIN aligned stage annotations", "examples": preserves},
        "glm_cwq_example_entity_context.json": contexts,
        "decomposition_train_examples.json": {"policy": policy, "examples": generic},
        "cwq_review_train_provenance.json": manifest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/train/cwq")
    parser.add_argument("--source-train", type=Path, default=ROOT / "assets/data/cwq/ComplexWebQuestions_train.json")
    parser.add_argument("--heldout-questions", type=Path)
    parser.add_argument("--train-entity-records", type=Path, default=ROOT / "assets/source_annotations.zip")
    parser.add_argument("--resources", type=Path, default=ROOT / "src/semantic_guided_kbqa/resources")
    parser.add_argument("--emit-patch", action="store_true")
    args = parser.parse_args()
    resources = build(args.train_dir, args.source_train, args.heldout_questions, args.train_entity_records)
    if args.emit_patch:
        parts = ["*** Begin Patch\n"]
        for name, data in resources.items():
            path = args.resources / name
            text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            if path.exists():
                diff = list(difflib.unified_diff(path.read_text().splitlines(), text.splitlines(), lineterm=""))
                if diff:
                    parts.append("*** Update File: " + str(path) + "\n" + "\n".join("@@" if line.startswith("@@ ") else line for line in diff[2:]) + "\n")
            else:
                parts.append("*** Add File: " + str(path) + "\n" + "\n".join("+" + line for line in text.splitlines()) + "\n")
        parts.append("*** End Patch\n")
        print("".join(parts), end="")
    else:
        for name, data in resources.items():
            if json.loads((args.resources / name).read_text()) != data:
                raise SystemExit("Resource differs from TRAIN reconstruction: " + name)
        print(json.dumps({"success": True, **resources["cwq_review_train_provenance.json"]}))


if __name__ == "__main__":
    main()
