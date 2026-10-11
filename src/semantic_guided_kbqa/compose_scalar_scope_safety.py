"""Prove every originating Semantic path survived before a local repair.

Binding a second named topic is not evidence that its path was retained. This
gate compares all original path/variable/direction contracts, not just anchor
names. Extra auto-expanded paths require independent evidence and currently
abstain; no scoring, queries, models or evaluation fields are consulted.
"""
from collections.abc import Mapping


def original_semantic_scope_certificate(grounded, source_semantic, *, ontology=None):
    evidence = {"status": "not_authorized", "additional_model_calls": 0,
                "additional_endpoint_queries": 0, "uses_evaluation_gold": False,
                "all_original_paths_preserved": False,
                "original_predicates_independently_checked": ontology is not None}
    actual = getattr(grounded, "compose_input", None)
    if not isinstance(actual, Mapping) or not isinstance(source_semantic, Mapping):
        return False, {**evidence, "reason": "missing_actual_original_semantic_contract"}
    originals = source_semantic.get("semantic_paths")
    paths = actual.get("semantic_paths")
    if (not isinstance(originals, list) or not originals or not isinstance(paths, list)
            or len(originals) != len(paths)):
        return False, {**evidence, "reason": "original_and_grounded_path_count_differs"}
    by_id = {}
    for path in paths:
        if (not isinstance(path, Mapping) or not isinstance(path.get("id"), str)
                or not path["id"] or path["id"] in by_id):
            return False, {**evidence, "reason": "actual_path_identity_unknown_or_duplicated"}
        by_id[path["id"]] = path
    seen = set()
    for original in originals:
        if (not isinstance(original, Mapping) or not isinstance(original.get("id"), str)
                or not original["id"] or original["id"] in seen):
            return False, {**evidence, "reason": "original_path_identity_unknown_or_duplicated"}
        seen.add(original["id"])
        current = by_id.get(original["id"])
        if current is None:
            return False, {**evidence, "reason": "original_path_not_represented", "path_id": original["id"]}
        for key in ("anchor_ref", "path_output_var"):
            if not isinstance(original.get(key), str) or not original[key] or current.get(key) != original[key]:
                return False, {**evidence, "reason": "original_root_or_output_variable_changed", "path_id": original["id"]}
        before, after = original.get("steps"), current.get("steps")
        if not isinstance(before, list) or not before or not isinstance(after, list) or len(before) != len(after):
            return False, {**evidence, "reason": "original_steps_missing_or_unproved_auto_expansion", "path_id": original["id"]}
        for first, second in zip(before, after):
            if (not isinstance(first, Mapping) or not isinstance(second, Mapping)
                    or first.get("direction") not in {"forward", "backward"}
                    or any(not isinstance(first.get(key), str) or not first[key]
                           or second.get(key) != first[key] for key in ("id", "from", "to", "direction"))):
                return False, {**evidence, "reason": "original_step_variable_identity_or_direction_changed", "path_id": original["id"]}
            if ontology is not None:
                from .budgeted_schema_repair import _same_schema_property
                bindings = getattr(grounded, "relation_bindings", {})
                actual_relation = bindings.get(first["id"]) if isinstance(bindings, Mapping) else None
                if not actual_relation or not _same_schema_property(first.get("relation_label"), actual_relation, ontology):
                    return False, {**evidence, "reason": "original_predicate_replaced_by_unproved_schema_role",
                                   "path_id": original["id"], "step_id": first["id"]}
    return True, {**evidence, "status": "authorized", "reason": "every_original_path_and_variable_contract_preserved",
                  "all_original_paths_preserved": True, "original_path_ids": sorted(seen)}
