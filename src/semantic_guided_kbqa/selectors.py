"""Consolidated selectors; original rules and budgets retained."""
from __future__ import annotations


# ---- selection ----


from copy import deepcopy
import json
from typing import Any

from .clients import ChatClient
from .contracts import ContractError, parse_json_object


def select_candidates(
    model: ChatClient,
    *,
    instruction: str,
    question: str,
    decompositions: list[str],
    candidates: list[dict[str, Any]],
    max_prompt_bytes: int = 0,
    rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select with bounded prompts; compare batch winners until one remains.

    UTF-8 byte length conservatively bounds Llama's byte-level BPE token count.
    The configured budget leaves space for chat template tokens and generation.
    Graph structure and operators are never truncated; answer previews may shrink.
    """
    remaining = deepcopy(candidates)
    if not remaining:
        raise ValueError("selector requires at least one executed candidate")

    def payload_for(items):
        return {"question": question, "decompositions": decompositions, "candidates": items}

    def prompt_size(items):
        text = instruction + json.dumps(payload_for(items), ensure_ascii=False, separators=(",", ":"))
        return len(text.encode("utf-8"))

    def pack(items):
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for item in items:
            if max_prompt_bytes and prompt_size([item]) > max_prompt_bytes:
                return None
            if current and max_prompt_bytes and prompt_size([*current, item]) > max_prompt_bytes:
                batches.append(current)
                current = []
            current.append(item)
        if current:
            batches.append(current)
        return batches

    round_number = 0
    while True:
        batches = pack(remaining)
        # A round consisting entirely of singleton batches cannot reduce the
        # candidate set. Shrink answer previews and try again, with a hard stop.
        while batches is None or (len(remaining) > 1 and len(batches) == len(remaining)):
            changed = False
            for item in remaining:
                for field in ("answers", "answer_ids"):
                    values = item.get(field, [])
                    if len(values) > 1:
                        item[field] = values[:max(1, len(values) // 2)]
                        item["answers_truncated"] = True
                        changed = True
            if not changed:
                raise RuntimeError("SELECTOR_PROMPT_TOO_LARGE: graph structure exceeds prompt byte budget")
            batches = pack(remaining)
        round_number += 1
        winners = []
        for batch in batches:
            if len(batch) == 1 and len(batches) > 1:
                winners.append(batch[0])
                continue
            ids = [item["graph_id"] for item in batch]
            schema = {
                "type": "object",
                "properties": {
                    "selected_graph_id": {"type": "string", "enum": ids},
                    "reason_codes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["selected_graph_id", "reason_codes"],
                "additionalProperties": False,
            }
            payload = payload_for(batch)
            event = {"round": round_number, "candidate_ids": ids,
                     "prompt_bytes": prompt_size(batch), "input": deepcopy(payload)}
            rounds.append(event)
            try:
                outputs = model.generate_json(
                    instruction=instruction, payload=payload, schema=schema, count=1
                )
                decision = parse_json_object(outputs[0] if outputs else {})
                event["model_output"] = deepcopy(decision)
                selected_id = decision.get("selected_graph_id")
                if selected_id not in ids:
                    raise ContractError("selector must choose an executed graph in its current batch")
                reasons = decision.get("reason_codes")
                if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
                    raise ContractError("selector reason_codes must be an array of strings")
                event["status"] = "selected"
            except Exception as exc:
                event.update({"status": "error", "error": str(exc)})
                raise
            if len(batches) == 1:
                return {"selected_graph_id": selected_id, "reason_codes": reasons}
            winners.append(next(item for item in batch if item["graph_id"] == selected_id))
        remaining = winners



# ---- path_alignment_selector ----
"""Frozen Gold-blind path-alignment post-selection gate.

The gate compares each saved Semantic path intent with natural constant-to-
answer paths in already executed candidate graphs.  It deliberately removes
all V0/V1-like variable spellings before local BGE scoring.  No LLM or KG
endpoint is called, and an earlier ``hard_*`` decision always has first
refusal.

Model coefficients and thresholds are frozen from the v12 train/calibration
audit in ``resources/path_alignment_v12_model.json``.  Gold answers are never
accepted by this module and are not part of its API.
"""



from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any, Iterable, Protocol, Sequence

from .contracts import ExecutedGraph as _path_alignment_selector_ExecutedGraph
from .ontology import relation_id_from_label as _path_alignment_selector_relation_id_from_label, relation_label_from_id


_path_alignment_selector_SELECTOR_VERSION = "path-alignment-production-v1"
_path_alignment_selector_REASON_CODE = "path_alignment_gate"
_path_alignment_selector_EPSILON = 1e-12
_path_alignment_selector_VARIABLE_RE = re.compile(r"^\??(?:V|P\d+\.V|v|x|var)[A-Za-z0-9_.-]*$", re.I)
_path_alignment_selector_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MODEL_PATH = Path(__file__).with_name("resources") / "path_alignment_v12_model.json"


class _EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, input_type: str) -> list[list[float]]: ...


def _load_model() -> dict[str, Any]:
    value = json.loads(_MODEL_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != "path-alignment-v12-frozen-model-v1":
        raise ValueError(f"invalid frozen path-alignment model: {_MODEL_PATH}")
    names = value.get("pairwise_feature_names", [])
    if not isinstance(names, list) or not names:
        raise ValueError("path-alignment model has no feature names")
    for key in ("means", "scales", "weights"):
        if not isinstance(value.get(key), list) or len(value[key]) != len(names):
            raise ValueError(f"path-alignment model {key} has wrong length")
    return value


@dataclass(frozen=True, slots=True)
class _path_alignment_selector_SemanticIntent:
    path_id: str
    goal: str
    anchor_surface: str
    anchor_id: str
    relation_ids: tuple[str, ...]
    directions: tuple[str, ...]
    grounded_relations: tuple[str, ...]
    grounded_directions: tuple[str, ...]

    @property
    def query_document(self) -> str:
        relation_path = "; then ".join(
            _direction_phrase(direction, relation)
            for direction, relation in zip(self.directions, self.relation_ids)
        )
        return (
            f"Goal: {self.goal}. Start with the real entity "
            f"{self.anchor_surface or 'the linked entity'}. Intended relationship "
            f"path: {relation_path}. Return the path target."
        )

    @property
    def relation_query_document(self) -> str:
        relation_path = "; then ".join(
            _direction_phrase(direction, relation)
            for direction, relation in zip(self.directions, self.relation_ids)
        )
        return f"Relationship intent for the requested target: {relation_path}."


@dataclass(frozen=True, slots=True)
class _path_alignment_selector_NaturalPath:
    start_id: str
    start_label: str
    relations: tuple[str, ...]
    directions: tuple[str, ...]
    terminal_type: str
    continuity: float
    known_type_coverage: float

    @property
    def document(self) -> str:
        relation_path = "; then ".join(
            _direction_phrase(direction, relation)
            for direction, relation in zip(self.directions, self.relations)
        )
        terminal = self.terminal_type.replace(".", " ").replace("_", " ")
        suffix = f" The returned target has type {terminal}." if terminal else ""
        return (
            f"Start with the real entity or literal {self.start_label or self.start_id}. "
            f"Natural relation path: {relation_path}. Return the target.{suffix}"
        )

    @property
    def relation_document(self) -> str:
        relation_path = "; then ".join(
            _direction_phrase(direction, relation)
            for direction, relation in zip(self.directions, self.relations)
        )
        return f"Concrete relationship path to the answer: {relation_path}."


@dataclass(slots=True)
class _path_alignment_selector_Candidate:
    execution: _path_alignment_selector_ExecutedGraph
    intents: list[_path_alignment_selector_SemanticIntent]
    paths: list[_path_alignment_selector_NaturalPath]
    features: list[float] | None = None


class ThreadSafeEmbeddingCache:
    """Bounded single-flight cache around the shared local embedding client."""

    def __init__(self, client: _EmbeddingClient, *, maximum_entries: int = 32768) -> None:
        self.client = client
        self.maximum_entries = max(256, int(maximum_entries))
        self._values: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
        self._lock = RLock()

    def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
    ) -> tuple[dict[str, tuple[float, ...]], dict[str, int]]:
        unique = list(dict.fromkeys(str(text) for text in texts if str(text)))
        with self._lock:
            hits = 0
            missing: list[str] = []
            for text in unique:
                key = (input_type, text)
                if key in self._values:
                    hits += 1
                    self._values.move_to_end(key)
                else:
                    missing.append(text)
            calls = 0
            if missing:
                vectors = self.client.embed(missing, input_type=input_type)
                if len(vectors) != len(missing):
                    raise RuntimeError("path-alignment embedding response has wrong length")
                calls = 1
                for text, vector in zip(missing, vectors):
                    self._values[(input_type, text)] = tuple(float(value) for value in vector)
                    self._values.move_to_end((input_type, text))
                while len(self._values) > self.maximum_entries:
                    self._values.popitem(last=False)
            return (
                {
                    text: self._values[(input_type, text)]
                    for text in unique
                    if (input_type, text) in self._values
                },
                {"hits": hits, "misses": len(missing), "calls": calls},
            )


def _path_alignment_selector_tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _path_alignment_selector_TOKEN_RE.finditer(str(value)))


def _path_alignment_selector_set_f1(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(map(str, left)), set(map(str, right))
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _path_alignment_selector_is_variable(value: Any) -> bool:
    return bool(_path_alignment_selector_VARIABLE_RE.fullmatch(str(value).strip()))


def _human_relation(relation_id: str) -> str:
    parts = relation_label_from_id(str(relation_id))
    if len(parts) <= 1:
        return parts[0] if parts else str(relation_id)
    return f"{parts[-1]} ({' / '.join(parts[:-1])})"


def _direction_phrase(direction: str, relation_id: str) -> str:
    relation = _human_relation(relation_id)
    if str(direction).casefold() == "backward":
        return f"find the subject whose {relation} points to the current item"
    return f"follow {relation} from subject to object"


def _path_alignment_selector_relation_leaf_tokens(relations: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        token
        for relation in relations
        for token in _path_alignment_selector_tokens(str(relation).split(".")[-1].replace("_", " "))
    )


def _path_alignment_selector_operator_signature(execution: _path_alignment_selector_ExecutedGraph) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(item.get("type", "")).upper()
            for item in execution.graph.operators
            if isinstance(item, dict) and str(item.get("type", "")).strip()
        )
    )


def _question_operator_flags(question: str) -> tuple[int, int, int, int, int]:
    text = " ".join(str(question).casefold().split())
    return (
        int(bool(re.search(r"\b(latest|last|most recent|highest|largest|longest|newest|maximum)\b", text))),
        int(bool(re.search(r"\b(earliest|first|lowest|smallest|shortest|oldest|youngest|minimum)\b", text))),
        int(bool(re.search(r"\b(at least|more than|greater than|over|above|after|later than)\b", text))),
        int(bool(re.search(r"\b(at most|less than|fewer than|under|below|before|prior to)\b", text))),
        int(bool(re.search(r"\b(equal|exactly|same as)\b", text))),
    )


def _operator_compatibility(question: str, execution: _path_alignment_selector_ExecutedGraph) -> float:
    max_q, min_q, high_q, low_q, equal_q = _question_operator_flags(question)
    values = set(_path_alignment_selector_operator_signature(execution))
    requirements = (
        (max_q, "ARGMAX" in values),
        (min_q, "ARGMIN" in values),
        (high_q, "GREATER_THAN" in values),
        (low_q, "LESS_THAN" in values),
        (equal_q, "EQUAL" in values),
    )
    active = [float(present) for requested, present in requirements if requested]
    if active:
        return sum(active) / len(active)
    substantive = values & {"ARGMAX", "ARGMIN", "GREATER_THAN", "LESS_THAN", "EQUAL"}
    return 1.0 if not substantive else 0.0


def _binding_value(binding: Any, field: str) -> str:
    if isinstance(binding, dict):
        return str(binding.get(field, "")).strip()
    return str(getattr(binding, field, "")).strip()


def _grounded_hops(execution: _path_alignment_selector_ExecutedGraph, path_index: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    grounded = execution.graph.provenance.get("grounded_semantic", {})
    hops = grounded.get("hops", []) if isinstance(grounded, dict) else []
    path_hops = hops[path_index] if path_index < len(hops) and isinstance(hops[path_index], list) else []
    relations: list[str] = []
    directions: list[str] = []
    for hop in path_hops:
        if not isinstance(hop, dict):
            continue
        relation = str(hop.get("relation_id", ""))
        direction = str(hop.get("direction", "")).casefold()
        if relation:
            relations.append(relation)
            directions.append(direction if direction in {"forward", "backward"} else "forward")
        second = str(hop.get("second_relation_id", ""))
        if second and second not in relations:
            relations.append(second)
            direction = str(hop.get("second_direction", "")).casefold()
            directions.append(direction if direction in {"forward", "backward"} else "forward")
    return tuple(relations), tuple(directions)


def _path_alignment_selector_semantic_intents(
    execution: _path_alignment_selector_ExecutedGraph,
    semantic_graphs: Sequence[dict[str, Any]],
) -> list[_path_alignment_selector_SemanticIntent]:
    grounded = execution.graph.provenance.get("grounded_semantic", {})
    semantic_index = grounded.get("semantic_graph_index") if isinstance(grounded, dict) else None
    if not isinstance(semantic_index, int) or not (0 <= semantic_index < len(semantic_graphs)):
        return []
    semantic = semantic_graphs[semantic_index]
    anchors = {
        str(item.get("id", "")): item
        for item in semantic.get("anchors", [])
        if isinstance(item, dict)
    }
    bindings = execution.graph.provenance.get("anchor_bindings", {})
    result: list[_path_alignment_selector_SemanticIntent] = []
    for path_index, path in enumerate(semantic.get("semantic_paths", [])):
        if not isinstance(path, dict):
            continue
        anchor_ref = str(path.get("anchor_ref", ""))
        anchor = anchors.get(anchor_ref, {})
        binding = bindings.get(anchor_ref, {}) if isinstance(bindings, dict) else {}
        relation_ids: list[str] = []
        directions: list[str] = []
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                continue
            relation = _path_alignment_selector_relation_id_from_label(step.get("relation_label", []))
            if not relation:
                continue
            relation_ids.append(relation)
            direction = str(step.get("direction", "forward")).casefold()
            directions.append(direction if direction in {"forward", "backward"} else "forward")
        if not relation_ids:
            continue
        grounded_relations, grounded_directions = _grounded_hops(execution, path_index)
        result.append(
            _path_alignment_selector_SemanticIntent(
                path_id=str(path.get("id", f"P{path_index}")),
                goal=str(path.get("goal", "")).strip(),
                anchor_surface=str(anchor.get("surface") or _binding_value(binding, "label")).strip(),
                anchor_id=_binding_value(binding, "id"),
                relation_ids=tuple(relation_ids),
                directions=tuple(directions),
                grounded_relations=grounded_relations,
                grounded_directions=grounded_directions,
            )
        )
    return result


def _type_compatible(left: str, right: str, ontology: Any) -> bool | None:
    if not left or not right:
        return None
    if left == right:
        return True
    return bool(set(ontology.supertypes(left)) & set(ontology.supertypes(right)))


def _path_types(
    relations: Sequence[str], directions: Sequence[str], ontology: Any
) -> tuple[str, float, float]:
    inputs: list[str] = []
    outputs: list[str] = []
    known_steps = 0
    for relation, direction in zip(relations, directions):
        domain = str(ontology.domain_for_relation(relation))
        range_ = str(ontology.range_for_relation(relation))
        input_type, output_type = (range_, domain) if direction == "backward" else (domain, range_)
        inputs.append(input_type)
        outputs.append(output_type)
        known_steps += int(bool(input_type and output_type))
    comparisons = [
        _type_compatible(outputs[index], inputs[index + 1], ontology)
        for index in range(max(0, len(outputs) - 1))
    ]
    known = [value for value in comparisons if value is not None]
    continuity = sum(bool(value) for value in known) / len(known) if known else 1.0
    return outputs[-1] if outputs else "", continuity, known_steps / max(1, len(relations))


def _path_alignment_selector_natural_paths(
    execution: _path_alignment_selector_ExecutedGraph,
    ontology: Any,
    *,
    maximum_hops: int = 7,
) -> list[_path_alignment_selector_NaturalPath]:
    triples = [tuple(map(str, triple)) for triple in execution.graph.triples if len(triple) == 3]
    answer = str(execution.graph.answer_var)
    if not triples or not answer:
        return []
    adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    nodes: set[str] = set()
    for subject, relation, object_ in triples:
        nodes.update((subject, object_))
        adjacency[subject].append((object_, relation, "forward"))
        adjacency[object_].append((subject, relation, "backward"))
    labels: dict[str, str] = {}
    bindings = execution.graph.provenance.get("anchor_bindings", {})
    if isinstance(bindings, dict):
        for binding in bindings.values():
            entity_id = _binding_value(binding, "id")
            label = _binding_value(binding, "label")
            if entity_id and label:
                labels[entity_id] = label
    result: list[_path_alignment_selector_NaturalPath] = []
    signatures: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for start in sorted(node for node in nodes if not _path_alignment_selector_is_variable(node)):
        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...], frozenset[str]]] = deque(
            [(start, (), (), frozenset({start}))]
        )
        best_depth: int | None = None
        found = 0
        while queue and found < 4:
            node, relations, directions, visited = queue.popleft()
            if best_depth is not None and len(relations) > best_depth + 1:
                break
            if node == answer and relations:
                best_depth = len(relations) if best_depth is None else best_depth
                signature = (start, relations, directions)
                if signature in signatures:
                    continue
                signatures.add(signature)
                terminal, continuity, coverage = _path_types(relations, directions, ontology)
                result.append(
                    _path_alignment_selector_NaturalPath(
                        start_id=start,
                        start_label=labels.get(start, start),
                        relations=relations,
                        directions=directions,
                        terminal_type=terminal,
                        continuity=continuity,
                        known_type_coverage=coverage,
                    )
                )
                found += 1
                continue
            if len(relations) >= maximum_hops:
                continue
            for neighbour, relation, direction in adjacency.get(node, []):
                if neighbour in visited:
                    continue
                queue.append((neighbour, (*relations, relation), (*directions, direction), visited | {neighbour}))
    return result


def _expected_terminal_type(intent: _path_alignment_selector_SemanticIntent, ontology: Any) -> str:
    if not intent.relation_ids:
        return ""
    relation, direction = intent.relation_ids[-1], intent.directions[-1]
    return str(
        ontology.domain_for_relation(relation)
        if direction == "backward"
        else ontology.range_for_relation(relation)
    )


def _pair_features(
    intent: _path_alignment_selector_SemanticIntent,
    path: _path_alignment_selector_NaturalPath,
    queries: dict[str, tuple[float, ...]],
    documents: dict[str, tuple[float, ...]],
    ontology: Any,
) -> tuple[float, ...]:
    semantic_pairs = tuple(zip(intent.relation_ids, intent.directions))
    natural_pairs = tuple(zip(path.relations, path.directions))
    grounded_pairs = tuple(zip(intent.grounded_relations, intent.grounded_directions))
    compatible = _type_compatible(_expected_terminal_type(intent, ontology), path.terminal_type, ontology)
    return (
        _cosine(queries.get(intent.query_document, ()), documents.get(path.document, ())),
        _cosine(queries.get(intent.relation_query_document, ()), documents.get(path.relation_document, ())),
        _path_alignment_selector_set_f1(_path_alignment_selector_relation_leaf_tokens(intent.relation_ids), _path_alignment_selector_relation_leaf_tokens(path.relations)),
        _path_alignment_selector_set_f1(intent.relation_ids, path.relations),
        _path_alignment_selector_set_f1(semantic_pairs, natural_pairs),
        _path_alignment_selector_set_f1(intent.grounded_relations, path.relations),
        _path_alignment_selector_set_f1(grounded_pairs, natural_pairs),
        float(bool(intent.anchor_id) and intent.anchor_id == path.start_id),
        _path_alignment_selector_set_f1(_path_alignment_selector_tokens(intent.anchor_surface), _path_alignment_selector_tokens(path.start_label)),
        0.5 if compatible is None else float(compatible),
        path.continuity,
        path.known_type_coverage,
        min(len(intent.relation_ids), len(path.relations))
        / max(1, max(len(intent.relation_ids), len(path.relations))),
    )


def _pair_alignment(values: Sequence[float]) -> float:
    weights = (0.40, 0.19, 0.09, 0.05, 0.04, 0.04, 0.03, 0.05, 0.03, 0.04, 0.02, 0.01, 0.01)
    return sum(weight * value for weight, value in zip(weights, values))


def _optimistic_structural_profile(
    candidate: _path_alignment_selector_Candidate,
    ontology: Any,
) -> tuple[float, float, float, float, float]:
    """BGE-free upper profile used only as a necessary-condition filter.

    Each intent may choose its best natural path independently for each
    structural signal.  This is intentionally optimistic: rejecting a row
    means even that upper profile cannot clear the frozen structural margin.
    """
    if not candidate.intents or len(candidate.paths) < len(candidate.intents):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    feature_indexes = (2, 3, 4, 9, 12)
    totals = [0.0] * len(feature_indexes)
    for intent in candidate.intents:
        pairs = [
            _pair_features(intent, path, {}, {}, ontology)
            for path in candidate.paths
        ]
        for output_index, feature_index in enumerate(feature_indexes):
            totals[output_index] += max(values[feature_index] for values in pairs)
    return tuple(value / len(candidate.intents) for value in totals)


def _cheap_structural_prefilter(
    candidates: Sequence[_path_alignment_selector_Candidate],
    incumbent_index: int,
    eligible: Sequence[int],
    ontology: Any,
    *,
    rule_score_floor: float,
    structural_gain: float = 0.20,
) -> tuple[bool, dict[str, Any]]:
    """Reject obvious non-contenders before any embedding request."""
    incumbent = candidates[incumbent_index]
    incumbent_profile = _optimistic_structural_profile(incumbent, ontology)
    contenders: list[dict[str, Any]] = []
    for index in eligible:
        candidate = candidates[index]
        rule_delta = (
            float(candidate.execution.graph.score)
            - float(incumbent.execution.graph.score)
        )
        if rule_delta + _path_alignment_selector_EPSILON < rule_score_floor:
            continue
        profile = _optimistic_structural_profile(candidate, ontology)
        maximum_gain = max(
            (left - right for left, right in zip(profile, incumbent_profile)),
            default=0.0,
        )
        if maximum_gain + _path_alignment_selector_EPSILON < structural_gain:
            continue
        contenders.append(
            {
                "graph_id": candidate.execution.graph.graph_id,
                "rule_score_delta": rule_delta,
                "maximum_structural_gain": maximum_gain,
            }
        )
    return bool(contenders), {
        "version": "path-alignment-cheap-prefilter-v1",
        "rule_score_floor": rule_score_floor,
        "minimum_optimistic_structural_gain": structural_gain,
        "signals": [
            "semantic_relation_leaf",
            "semantic_exact_relation",
            "semantic_direction",
            "terminal_ontology_type",
            "hop_length",
        ],
        "eligible_candidate_count": len(eligible),
        "contender_count": len(contenders),
        "contenders": contenders,
    }


def _maximum_matching(matrix: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    if not matrix or not matrix[0]:
        return []
    rows, columns = len(matrix), len(matrix[0])
    if columns > 14:
        retained = sorted(
            range(columns),
            key=lambda column: max(matrix[row][column] for row in range(rows)),
            reverse=True,
        )[:14]
        reduced = [[row[column] for column in retained] for row in matrix]
        return [(row, retained[column]) for row, column in _maximum_matching(reduced)]
    states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {0: (0.0, ())}
    for row in range(rows):
        next_states = dict(states)
        for mask, (score, assignments) in states.items():
            for column in range(columns):
                bit = 1 << column
                if mask & bit:
                    continue
                proposed = score + matrix[row][column]
                previous = next_states.get(mask | bit)
                if previous is None or proposed > previous[0] + _path_alignment_selector_EPSILON:
                    next_states[mask | bit] = (proposed, (*assignments, (row, column)))
        states = next_states
    return list(max(states.values(), key=lambda item: (item[0], len(item[1])))[1])


def _path_alignment_selector_candidate_features(
    question: str,
    candidate: _path_alignment_selector_Candidate,
    queries: dict[str, tuple[float, ...]],
    documents: dict[str, tuple[float, ...]],
    ontology: Any,
) -> list[float]:
    pairs = [
        [_pair_features(intent, path, queries, documents, ontology) for path in candidate.paths]
        for intent in candidate.intents
    ]
    alignment_matrix = [[_pair_alignment(values) for values in row] for row in pairs]
    matching = _maximum_matching(alignment_matrix)
    matched = [pairs[row][column] for row, column in matching]
    alignments = [alignment_matrix[row][column] for row, column in matching]
    intent_count = len(candidate.intents)

    def mean(index: int) -> float:
        return sum(values[index] for values in matched) / max(1, intent_count)

    execution = candidate.execution
    return [
        sum(alignments) / max(1, intent_count),
        min(alignments) if len(alignments) == intent_count and alignments else 0.0,
        max(alignments, default=0.0),
        *(mean(index) for index in range(13)),
        len(matching) / max(1, intent_count),
        sum(score >= 0.65 for score in alignments) / max(1, intent_count),
        sum(score >= 0.75 for score in alignments) / max(1, intent_count),
        sum(score >= 0.82 for score in alignments) / max(1, intent_count),
        min(len(candidate.intents), len(candidate.paths))
        / max(1, max(len(candidate.intents), len(candidate.paths))),
        float(execution.graph.score),
        _operator_compatibility(question, execution),
        math.log1p(len(set(map(str, execution.answer_ids)))),
        math.log1p(len(execution.graph.triples)),
    ]


def _pairwise_features(
    candidates: Sequence[_path_alignment_selector_Candidate], incumbent_index: int, candidate_index: int
) -> list[float]:
    candidate = candidates[candidate_index].features or []
    incumbent = candidates[incumbent_index].features or []
    top_alignment = max((item.features or [0.0])[0] for item in candidates)
    top_rule = max((item.features or [0.0] * 22)[21] for item in candidates)
    return [
        *candidate,
        *(left - right for left, right in zip(candidate, incumbent)),
        candidate[0] - top_alignment,
        candidate[21] - top_rule,
        math.log1p(len(candidates)),
    ]


def _probability(values: Sequence[float], model: dict[str, Any]) -> float:
    normalized = [
        (value - float(mean)) / float(scale)
        for value, mean, scale in zip(values, model["means"], model["scales"])
    ]
    raw = max(
        -30.0,
        min(
            30.0,
            float(model["bias"])
            + sum(float(weight) * value for weight, value in zip(model["weights"], normalized)),
        ),
    )
    return 1.0 / (1.0 + math.exp(-raw))


def _has_hard_reason(decision: dict[str, Any]) -> bool:
    return any(str(reason).startswith("hard_") for reason in decision.get("reason_codes", []))


class PathAlignmentSelector:
    """Frozen, local-BGE challenger over already executed graph candidates."""

    def __init__(
        self,
        embedding_client: _EmbeddingClient,
        ontology: Any,
        *,
        complete_answer_limit: int = 100,
        embedding_cache_size: int = 1024,
    ) -> None:
        self.ontology = ontology
        self.complete_answer_limit = max(1, int(complete_answer_limit))
        self.model = _load_model()
        self.cache = ThreadSafeEmbeddingCache(
            embedding_client,
            maximum_entries=embedding_cache_size,
        )

    def challenge(
        self,
        *,
        question: str,
        semantic_graphs: Sequence[dict[str, Any]],
        selected: _path_alignment_selector_ExecutedGraph,
        executions: Sequence[_path_alignment_selector_ExecutedGraph],
        prior_decision: dict[str, Any],
    ) -> tuple[_path_alignment_selector_ExecutedGraph, dict[str, Any]]:
        started = time.perf_counter()
        evidence: dict[str, Any] = {
            "version": _path_alignment_selector_SELECTOR_VERSION,
            "status": "not_applicable",
            "model_calls": 0,
            "endpoint_queries": 0,
            "embedding_calls": 0,
            "uses_evaluation_gold": False,
            "entity_or_relation_whitelist": False,
            "variable_names_sent_to_embedding": False,
            "frozen_model_version": self.model["version"],
            "frozen_gate": dict(self.model["gate"]),
        }
        if _has_hard_reason(prior_decision):
            evidence.update(
                {
                    "status": "skipped_existing_hard_gate",
                    "prior_reason_codes": list(prior_decision.get("reason_codes", [])),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        if self.ontology is None or not semantic_graphs:
            evidence.update(
                {
                    "status": "missing_ontology_or_semantic_graphs",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        unique: list[_path_alignment_selector_ExecutedGraph] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for execution in executions:
            if not execution.answer_ids or len(execution.answer_ids) > self.complete_answer_limit:
                continue
            key = (
                str(execution.graph.graph_id),
                tuple(str(answer_id) for answer_id in execution.answer_ids),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(execution)
        selected_answers = set(map(str, selected.answer_ids))
        incumbent_indexes = [
            index
            for index, execution in enumerate(unique)
            if set(map(str, execution.answer_ids)) == selected_answers
        ]
        if not incumbent_indexes:
            evidence.update(
                {
                    "status": "incumbent_not_materialized_as_complete_candidate",
                    "candidate_count": len(unique),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        incumbent_index = next(
            (
                index
                for index in incumbent_indexes
                if unique[index].graph.graph_id == selected.graph.graph_id
            ),
            incumbent_indexes[0],
        )
        candidates = [
            _path_alignment_selector_Candidate(
                execution=execution,
                intents=_path_alignment_selector_semantic_intents(execution, semantic_graphs),
                paths=_path_alignment_selector_natural_paths(execution, self.ontology),
            )
            for execution in unique
        ]
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if index != incumbent_index
            and set(map(str, candidate.execution.answer_ids)) != selected_answers
            and candidate.intents
            and candidate.paths
        ]
        if not eligible:
            evidence.update(
                {
                    "status": "no_distinct_path_aligned_challenger",
                    "candidate_count": len(candidates),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        prefilter_passed, prefilter = _cheap_structural_prefilter(
            candidates,
            incumbent_index,
            eligible,
            self.ontology,
            rule_score_floor=float(
                self.model["gate"]["minimum_rule_score_delta"]
            ),
        )
        evidence["cheap_prefilter"] = prefilter
        if not prefilter_passed:
            evidence.update(
                {
                    "status": "cheap_structural_prefilter_rejected",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        query_texts = [
            text
            for candidate in candidates
            for intent in candidate.intents
            for text in (intent.query_document, intent.relation_query_document)
        ]
        document_texts = [
            text
            for candidate in candidates
            for path in candidate.paths
            for text in (path.document, path.relation_document)
        ]
        try:
            queries, query_stats = self.cache.embed(query_texts, input_type="query")
            documents, document_stats = self.cache.embed(document_texts, input_type="document")
        except Exception as exc:
            evidence.update(
                {
                    "status": "embedding_unavailable",
                    "embedding_error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        evidence.update(
            {
                "embedding_calls": query_stats["calls"] + document_stats["calls"],
                "embedding_cache_hits": query_stats["hits"] + document_stats["hits"],
                "embedding_cache_misses": query_stats["misses"] + document_stats["misses"],
            }
        )
        for candidate in candidates:
            candidate.features = _path_alignment_selector_candidate_features(
                question,
                candidate,
                queries,
                documents,
                self.ontology,
            )
        probabilities = {
            index: _probability(
                _pairwise_features(candidates, incumbent_index, index),
                self.model,
            )
            for index in eligible
        }
        order = sorted(
            eligible,
            key=lambda index: (
                -probabilities[index],
                -(candidates[index].features or [0.0])[0],
                -float(candidates[index].execution.graph.score),
                str(candidates[index].execution.graph.graph_id),
            ),
        )
        chosen = order[0]
        chosen_features = candidates[chosen].features or []
        incumbent_features = candidates[incumbent_index].features or []
        alignment_values = sorted(
            (candidates[index].features or [0.0])[0]
            for index in [*eligible, incumbent_index]
        )
        top_alignment = alignment_values[-1]
        second_alignment = alignment_values[-2] if len(alignment_values) > 1 else 0.0
        alignment_gap = (
            chosen_features[0] - second_alignment
            if chosen_features[0] >= top_alignment - _path_alignment_selector_EPSILON
            else chosen_features[0] - top_alignment
        )
        proposal = {
            "graph_id": candidates[chosen].execution.graph.graph_id,
            "source_graph_id": selected.graph.graph_id,
            "probability": probabilities[chosen],
            "probability_margin": probabilities[chosen]
            - (probabilities[order[1]] if len(order) > 1 else 0.0),
            "alignment_delta": chosen_features[0] - incumbent_features[0],
            "alignment_gap": alignment_gap,
            "coverage": chosen_features[16],
            "rule_score_delta": chosen_features[21] - incumbent_features[21],
            "candidate_count": len(candidates),
            "challenger_answer_count": len(candidates[chosen].execution.answer_ids),
            "incumbent_answer_count": len(selected.answer_ids),
        }
        gate = self.model["gate"]
        passed = bool(
            proposal["probability"] + _path_alignment_selector_EPSILON >= float(gate["probability"])
            and proposal["probability_margin"] + _path_alignment_selector_EPSILON >= float(gate["probability_margin"])
            and proposal["alignment_delta"] + _path_alignment_selector_EPSILON >= float(gate["alignment_delta"])
            and proposal["alignment_gap"] + _path_alignment_selector_EPSILON >= float(gate["alignment_gap"])
            and proposal["coverage"] + _path_alignment_selector_EPSILON >= float(gate["required_coverage"])
            and proposal["rule_score_delta"] + _path_alignment_selector_EPSILON
            >= float(gate["minimum_rule_score_delta"])
        )
        evidence.update(
            {
                "status": "applied" if passed else "confidence_gate_rejected",
                "reason": "frozen_path_alignment_confidence"
                if passed
                else "frozen_thresholds_not_met",
                "proposal": proposal,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return (candidates[chosen].execution if passed else selected), evidence






# ---- template_consensus_selector ----
"""Source-verified local train-template consensus post-selection gate.

The selector is intentionally independent of the prediction models.  It uses
only already executed graph candidates plus an index built from aligned CWQ
Semantic/Compose/Operator training contracts whose Compose relation multiset
is verified against source-train SPARQL.  Runtime retrieval is sparse BM25 over
an entity-masked question and decomposition; it performs no LLM, embedding, KG
or evaluation-label call.

Thresholds are frozen from the v12 train/calibration audit.  The additional
operator/cardinality guards are generic, but the plural/disjoint guard was
formulated after inspecting a held-out audit error.  That provenance is
preserved in every decision trace and must not be hidden when reporting the
feature as independently validated.
"""



from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import gzip
import heapq
import json
import math
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Iterable, Sequence

from .contracts import ExecutedGraph as _template_consensus_selector_ExecutedGraph
from .ontology import relation_id_from_label as _template_consensus_selector_relation_id_from_label
from .template_path_retrieval import mask_entities as _template_consensus_selector_mask_entities


_template_consensus_selector_INDEX_VERSION = "source-verified-template-index-v1"
_template_consensus_selector_SELECTOR_VERSION = "source-verified-template-consensus-production-v1"
_template_consensus_selector_EPSILON = 1e-12
_template_consensus_selector_TOP_K = 48

# Frozen v12 train/calibration lanes.  Values are structural confidence
# thresholds, never entity/relation/question allow-lists.
SUBSET_SCORE_GAP = 0.07
SUBSET_VOTE_COUNT = 40
DISJOINT_SCORE_GAP = 0.20
DISJOINT_VOTE = 0.20
DISJOINT_RELATION_SUPPORT = 0.30

_template_consensus_selector_TOKEN_RE = re.compile(r"<entity>|[a-z0-9]+", re.I)
_template_consensus_selector_VARIABLE_RE = re.compile(r"^(?:V\d+|P\d+\.V\d+)$", re.I)
_SPARQL_TRIPLE_RE = re.compile(
    r"(?:\?[A-Za-z_][A-Za-z0-9_]*|ns:[mg]\.[A-Za-z0-9_]+)\s+"
    r"ns:([A-Za-z0-9_.]+)\s+"
)
_INDEXES: dict[str, "VerifiedTemplateIndex"] = {}
_INDEXES_LOCK = Lock()


def _template_consensus_selector_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _template_consensus_selector_TOKEN_RE.finditer(str(text))]


def _parse_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_array(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"template consensus source must be a JSON array: {path}")
    return [row for row in value if isinstance(row, dict)]


def _template_consensus_selector_operator_signature(value: Any) -> Counter[str]:
    if isinstance(value, dict) and "output" in value:
        value = _parse_object(value.get("output"))
    operators = value.get("operators", []) if isinstance(value, dict) else []
    return Counter(
        str(operator.get("type", "")).upper()
        for operator in operators
        if isinstance(operator, dict) and str(operator.get("type", "")).strip()
    )


def _multiset_f1(left: Counter[str], right: Counter[str]) -> float:
    left_size, right_size = sum(left.values()), sum(right.values())
    if not left_size or not right_size:
        return float(left_size == right_size)
    return (2.0 * sum((left & right).values())) / (left_size + right_size)


@dataclass(frozen=True, slots=True)
class GraphFeatures:
    triples: tuple[tuple[str, str, str], ...]
    answer_var: str
    relations: Counter[str]
    answer_incidence: Counter[str]
    paths: tuple[tuple[tuple[str, str], ...], ...]
    abstract_paths: tuple[tuple[str, ...], ...]


def _template_consensus_selector_is_variable(value: Any) -> bool:
    return bool(_template_consensus_selector_VARIABLE_RE.fullmatch(str(value)))


def _graph_features(
    triples: Iterable[Sequence[Any]], answer_var: Any
) -> GraphFeatures:
    normalized = tuple(
        (str(triple[0]), str(triple[1]), str(triple[2]))
        for triple in triples
        if isinstance(triple, (list, tuple))
        and len(triple) == 3
        and str(triple[1])
    )
    answer = str(answer_var)
    relations = Counter(relation for _, relation, _ in normalized)
    answer_incidence = Counter(
        f"{relation}|{'out' if subject == answer else 'in' if object_ == answer else 'internal'}"
        for subject, relation, object_ in normalized
    )
    nodes = {
        node for subject, _, object_ in normalized for node in (subject, object_)
    }
    adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for subject, relation, object_ in normalized:
        adjacency[subject].append((object_, relation, "out"))
        adjacency[object_].append((subject, relation, "in"))
    constants = sorted(node for node in nodes if not _template_consensus_selector_is_variable(node))
    paths: list[tuple[tuple[str, str], ...]] = []
    for constant in constants:
        queue: deque[tuple[str, tuple[tuple[str, str], ...]]] = deque(
            [(constant, ())]
        )
        visited = {constant}
        found: tuple[tuple[str, str], ...] | None = None
        while queue:
            node, path = queue.popleft()
            if node == answer:
                found = path
                break
            for neighbour, relation, direction in sorted(adjacency.get(node, [])):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                queue.append((neighbour, (*path, (relation, direction))))
        if found is not None:
            paths.append(found)
    ordered_paths = tuple(sorted(paths))
    return GraphFeatures(
        triples=normalized,
        answer_var=answer,
        relations=relations,
        answer_incidence=answer_incidence,
        paths=ordered_paths,
        abstract_paths=tuple(
            sorted(tuple(direction for _, direction in path) for path in paths)
        ),
    )


def _path_similarity(
    left: tuple[tuple[str, str], ...], right: tuple[tuple[str, str], ...]
) -> float:
    if not left or not right:
        return float(left == right)
    aligned = sum(a == b for a, b in zip(left, right)) / max(len(left), len(right))
    relation = _template_consensus_selector_set_f1((value[0] for value in left), (value[0] for value in right))
    direction = sum(a[1] == b[1] for a, b in zip(left, right)) / max(
        len(left), len(right)
    )
    return 0.55 * aligned + 0.25 * relation + 0.20 * direction


def _template_consensus_selector_set_f1(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return float(not a and not b)
    return (2.0 * len(a & b)) / (len(a) + len(b))


def _path_set_similarity(
    left: tuple[tuple[tuple[str, str], ...], ...],
    right: tuple[tuple[tuple[str, str], ...], ...],
) -> float:
    if not left or not right:
        return float(left == right)

    def directed(source: Any, target: Any) -> float:
        return sum(
            max(_path_similarity(path, other) for other in target)
            for path in source
        ) / len(source)

    return 0.5 * (directed(left, right) + directed(right, left))


def _direction_path_similarity(
    left: tuple[tuple[str, ...], ...], right: tuple[tuple[str, ...], ...]
) -> float:
    if not left or not right:
        return float(left == right)

    def one(source: Sequence[str], target: Sequence[str]) -> float:
        if not source or not target:
            return float(not source and not target)
        return sum(a == b for a, b in zip(source, target)) / max(
            len(source), len(target)
        )

    def directed(source: Any, target: Any) -> float:
        return sum(max(one(path, other) for other in target) for path in source) / len(
            source
        )

    return 0.5 * (directed(left, right) + directed(right, left))


@dataclass(slots=True)
class TemplateDocument:
    source_index: int
    question: str
    tokens: list[str]
    graph: GraphFeatures
    operators: Counter[str]


def _source_manifest(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        stat = path.stat()
        manifest.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return manifest


def _compose_triples(output: dict[str, Any]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for triple in output.get("triples", []):
        if not isinstance(triple, dict):
            continue
        relation = _template_consensus_selector_relation_id_from_label(triple.get("relation_label", []))
        if relation:
            triples.append(
                (
                    str(triple.get("subject", "")),
                    relation,
                    str(triple.get("object", "")),
                )
            )
    return triples


def build_verified_template_cache(
    *,
    semantic_train: str | Path,
    compose_train: str | Path,
    operator_train: str | Path,
    source_train: str | Path,
    cache_path: str | Path,
) -> dict[str, Any]:
    """Build and atomically persist the source-verified local index."""
    paths = (semantic_train, compose_train, operator_train, source_train)
    semantic_rows = _load_array(semantic_train)
    compose_rows = _load_array(compose_train)
    operator_rows = _load_array(operator_train)
    source_rows = _load_array(source_train)
    source_graphs: dict[str, list[Counter[str]]] = defaultdict(list)
    for row in source_rows:
        source_graphs[str(row.get("question", ""))].append(
            Counter(_SPARQL_TRIPLE_RE.findall(str(row.get("sparql", ""))))
        )
    documents: list[dict[str, Any]] = []
    audit: Counter[str] = Counter()
    for source_index, (semantic, compose, operator) in enumerate(
        zip(semantic_rows, compose_rows, operator_rows)
    ):
        semantic_input = _parse_object(semantic.get("input"))
        semantic_output = _parse_object(semantic.get("output"))
        compose_input = _parse_object(compose.get("input"))
        operator_input = _parse_object(operator.get("input"))
        question = str(semantic_input.get("question", ""))
        if not question or question != str(compose_input.get("question", "")):
            audit["semantic_compose_alignment_rejected"] += 1
            continue
        if question != str(operator_input.get("question", "")):
            audit["operator_alignment_rejected"] += 1
            continue
        compose_output = _parse_object(compose.get("output"))
        triples = _compose_triples(compose_output)
        relations = Counter(relation for _, relation, _ in triples)
        sources = source_graphs.get(question, [])
        if not sources:
            audit["source_question_missing"] += 1
            continue
        if not any(not (relations - source) for source in sources):
            audit["source_relation_verification_rejected"] += 1
            continue
        if not triples:
            audit["empty_compose_graph_rejected"] += 1
            continue
        decomposition = [
            str(value)
            for value in semantic_input.get("decomposition", [])
            if str(value).strip()
        ]
        surfaces = [
            str(anchor.get("surface", ""))
            for anchor in semantic_output.get("anchors", [])
            if isinstance(anchor, dict) and str(anchor.get("surface", "")).strip()
        ]
        text = " ".join(
            [
                _template_consensus_selector_mask_entities(question, surfaces),
                *(_template_consensus_selector_mask_entities(value, surfaces) for value in decomposition),
            ]
        )
        tokens = _template_consensus_selector_tokens(text)
        if not tokens:
            audit["empty_document_rejected"] += 1
            continue
        documents.append(
            {
                "source_index": source_index,
                "question": question,
                "tokens": tokens,
                "triples": triples,
                "answer_var": str(compose_output.get("answer_var", "")),
                "operators": list(
                    sorted(_template_consensus_selector_operator_signature(operator).elements())
                ),
            }
        )
        audit["source_verified_documents"] += 1
    payload = {
        "version": _template_consensus_selector_INDEX_VERSION,
        "source_manifest": _source_manifest(paths),
        "audit": dict(audit),
        "documents": documents,
    }
    target = Path(cache_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(target)
    return payload


class VerifiedTemplateIndex:
    def __init__(self, payload: dict[str, Any], *, cache_hit: bool) -> None:
        self.cache_hit = bool(cache_hit)
        self.audit = dict(payload.get("audit", {}))
        self.documents: list[TemplateDocument] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for raw in payload.get("documents", []):
            if not isinstance(raw, dict):
                continue
            tokens = [str(value) for value in raw.get("tokens", [])]
            graph = _graph_features(
                raw.get("triples", []), raw.get("answer_var", "")
            )
            if not tokens or not graph.triples:
                continue
            document_index = len(self.documents)
            document = TemplateDocument(
                source_index=int(raw.get("source_index", -1)),
                question=str(raw.get("question", "")),
                tokens=tokens,
                graph=graph,
                operators=Counter(map(str, raw.get("operators", []))),
            )
            self.documents.append(document)
            frequencies = Counter(tokens)
            document_frequency.update(frequencies)
            for token, frequency in frequencies.items():
                self.postings[token].append((document_index, frequency))
        count = max(1, len(self.documents))
        self.average_length = sum(len(item.tokens) for item in self.documents) / count
        self.idf = {
            token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }
        self.max_posting_length = max(64, int(len(self.documents) * 0.18))

    @classmethod
    def load_or_build(
        cls,
        *,
        semantic_train: str | Path,
        compose_train: str | Path,
        operator_train: str | Path,
        source_train: str | Path,
        cache_path: str | Path,
    ) -> "VerifiedTemplateIndex":
        target = Path(cache_path).expanduser().resolve()
        key = str(target)
        with _INDEXES_LOCK:
            cached_index = _INDEXES.get(key)
            if cached_index is not None:
                return cached_index
            expected_manifest = _source_manifest(
                (semantic_train, compose_train, operator_train, source_train)
            )
            payload: dict[str, Any] | None = None
            if target.is_file():
                try:
                    with gzip.open(target, "rt", encoding="utf-8") as handle:
                        candidate = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    candidate = None
                if (
                    isinstance(candidate, dict)
                    and candidate.get("version") == _template_consensus_selector_INDEX_VERSION
                    and candidate.get("source_manifest") == expected_manifest
                ):
                    payload = candidate
            cache_hit = payload is not None
            if payload is None:
                payload = build_verified_template_cache(
                    semantic_train=semantic_train,
                    compose_train=compose_train,
                    operator_train=operator_train,
                    source_train=source_train,
                    cache_path=target,
                )
            index = cls(payload, cache_hit=cache_hit)
            _INDEXES[key] = index
            return index

    def retrieve(
        self,
        question: str,
        decompositions: Sequence[str],
        surfaces: Sequence[str],
        *,
        top_k: int = _template_consensus_selector_TOP_K,
    ) -> list[tuple[float, TemplateDocument]]:
        text = " ".join(
            [
                _template_consensus_selector_mask_entities(question, surfaces),
                *(_template_consensus_selector_mask_entities(value, surfaces) for value in decompositions),
            ]
        )
        query = Counter(_template_consensus_selector_tokens(text))
        informative = sorted(
            (
                (self.idf.get(token, 0.0), token, frequency)
                for token, frequency in query.items()
                if len(self.postings.get(token, ())) <= self.max_posting_length
            ),
            reverse=True,
        )[:28]
        scores: dict[int, float] = defaultdict(float)
        for _, token, query_frequency in informative:
            inverse_frequency = self.idf.get(token, 0.0)
            for index, frequency in self.postings.get(token, ()):
                document = self.documents[index]
                normalization = 0.25 + 0.75 * len(document.tokens) / max(
                    self.average_length, 1e-9
                )
                scores[index] += (
                    inverse_frequency
                    * ((frequency * 2.2) / (frequency + 1.2 * normalization))
                    * min(query_frequency, 2)
                )
        return [
            (score, self.documents[index])
            for score, index in heapq.nlargest(
                max(1, int(top_k)),
                ((score, index) for index, score in scores.items()),
            )
        ]


@dataclass(slots=True)
class _template_consensus_selector_Candidate:
    execution: _template_consensus_selector_ExecutedGraph
    graph: GraphFeatures
    operators: Counter[str]


def _component_similarity(
    candidate: _template_consensus_selector_Candidate, document: TemplateDocument
) -> tuple[float, float, float, float, float, float]:
    relation = _multiset_f1(candidate.graph.relations, document.graph.relations)
    incidence = _multiset_f1(
        candidate.graph.answer_incidence, document.graph.answer_incidence
    )
    path = _path_set_similarity(candidate.graph.paths, document.graph.paths)
    direction = _direction_path_similarity(
        candidate.graph.abstract_paths, document.graph.abstract_paths
    )
    operators = _multiset_f1(candidate.operators, document.operators)
    total = (
        0.34 * relation
        + 0.24 * incidence
        + 0.22 * path
        + 0.14 * direction
        + 0.06 * operators
    )
    return total, relation, incidence, path, direction, operators


def _answer_relation(
    challenger: Sequence[str], incumbent: Sequence[str]
) -> str:
    left, right = set(map(str, challenger)), set(map(str, incumbent))
    if left == right:
        return "equal"
    if left < right:
        return "subset"
    if left > right:
        return "superset"
    if left & right:
        return "overlap"
    return "disjoint"


def _morphological_plural_request(question: str) -> bool:
    tokens = re.findall(r"[a-z]+", str(question).casefold())
    try:
        start = next(
            index for index, token in enumerate(tokens[:5]) if token in {"which", "what"}
        )
    except StopIteration:
        return False
    return any(
        len(token) > 3 and token.endswith("s") and not token.endswith("ss")
        for token in tokens[start + 1 : start + 3]
    )


def _hard_gate_already_selected(decision: dict[str, Any]) -> bool:
    return any(
        str(reason).startswith("hard_")
        or str(reason) == "path_alignment_gate"
        for reason in decision.get("reason_codes", [])
    )


class SourceVerifiedTemplateConsensusSelector:
    """Frozen, Gold-free KNN consensus over already executed candidates."""

    def __init__(self, index: VerifiedTemplateIndex, *, top_k: int = _template_consensus_selector_TOP_K) -> None:
        self.index = index
        self.top_k = max(1, min(_template_consensus_selector_TOP_K, int(top_k)))

    @classmethod
    def load_or_build(
        cls,
        *,
        semantic_train: str | Path,
        compose_train: str | Path,
        operator_train: str | Path,
        source_train: str | Path,
        cache_path: str | Path,
        top_k: int = _template_consensus_selector_TOP_K,
    ) -> "SourceVerifiedTemplateConsensusSelector":
        return cls(
            VerifiedTemplateIndex.load_or_build(
                semantic_train=semantic_train,
                compose_train=compose_train,
                operator_train=operator_train,
                source_train=source_train,
                cache_path=cache_path,
            ),
            top_k=top_k,
        )

    def challenge(
        self,
        *,
        question: str,
        decompositions: Sequence[str],
        anchor_surfaces: Sequence[str],
        selected: _template_consensus_selector_ExecutedGraph,
        executions: Sequence[_template_consensus_selector_ExecutedGraph],
        prior_decision: dict[str, Any],
    ) -> tuple[_template_consensus_selector_ExecutedGraph, dict[str, Any]]:
        started = time.perf_counter()
        evidence: dict[str, Any] = {
            "version": _template_consensus_selector_SELECTOR_VERSION,
            "status": "not_applicable",
            "model_calls": 0,
            "embedding_calls": 0,
            "endpoint_queries": 0,
            "uses_evaluation_gold": False,
            "entity_or_relation_whitelist": False,
            "source_verified_template_count": len(self.index.documents),
            "index_cache_hit": self.index.cache_hit,
            "posthoc_risk": (
                "plural_disjoint_cardinality_collapse was formulated after held-out "
                "audit diagnosis and requires fresh shadow validation"
            ),
        }
        if _hard_gate_already_selected(prior_decision):
            evidence.update(
                {
                    "status": "skipped_higher_priority_postselection_gate",
                    "prior_reason_codes": list(prior_decision.get("reason_codes", [])),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        unique: list[_template_consensus_selector_ExecutedGraph] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for execution in [*executions, selected]:
            if not execution.answer_ids:
                continue
            key = (
                str(execution.graph.graph_id),
                tuple(sorted(set(map(str, execution.answer_ids)))),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(execution)
        if len(unique) < 2:
            evidence.update(
                {
                    "status": "insufficient_executed_candidates",
                    "candidate_count": len(unique),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        neighbours = self.index.retrieve(
            question,
            decompositions,
            anchor_surfaces,
            top_k=self.top_k,
        )
        if not neighbours:
            evidence.update(
                {
                    "status": "no_verified_template_neighbours",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        candidates = [
            _template_consensus_selector_Candidate(
                execution=execution,
                graph=_graph_features(
                    execution.graph.triples, execution.graph.answer_var
                ),
                operators=_template_consensus_selector_operator_signature(
                    {"operators": execution.graph.operators}
                ),
            )
            for execution in unique
        ]
        top_bm25 = max(score for score, _ in neighbours)
        rows: list[tuple[float, TemplateDocument, list[tuple[float, ...]]]] = []
        for rank, (bm25, document) in enumerate(neighbours):
            weight = (bm25 / max(top_bm25, 1e-9)) / math.log2(rank + 2.0)
            rows.append(
                (
                    weight,
                    document,
                    [
                        _component_similarity(candidate, document)
                        for candidate in candidates
                    ],
                )
            )
        denominator = sum(weight for weight, _, _ in rows) or 1.0
        scored: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            component_means = [
                sum(
                    weight * similarities[candidate_index][component]
                    for weight, _, similarities in rows
                )
                / denominator
                for component in range(6)
            ]
            mean = component_means[0]
            vote = 0.0
            vote_count = 0
            relation_support = 0.0
            exact_support = 0.0
            for weight, document, similarities in rows:
                ranked = sorted(
                    (
                        (item[0], index)
                        for index, item in enumerate(similarities)
                    ),
                    reverse=True,
                )
                winner_score = ranked[0][0]
                winners = [
                    index
                    for score, index in ranked
                    if abs(score - winner_score) <= _template_consensus_selector_EPSILON
                ]
                if candidate_index in winners:
                    vote += weight / len(winners)
                    vote_count += 1
                relation_support += weight * float(
                    candidate.graph.relations == document.graph.relations
                )
                exact_support += weight * float(
                    candidate.graph.relations == document.graph.relations
                    and candidate.graph.answer_incidence
                    == document.graph.answer_incidence
                    and candidate.graph.abstract_paths
                    == document.graph.abstract_paths
                    and candidate.operators == document.operators
                )
            scored.append(
                {
                    "candidate": candidate,
                    "score": mean,
                    "vote": vote / denominator,
                    "vote_count": vote_count,
                    "relation_support": relation_support / denominator,
                    "exact_support": exact_support / denominator,
                    "component_means": component_means,
                }
            )
        ranking = sorted(
            scored,
            key=lambda item: (
                float(item["score"]),
                float(item["vote"]),
                float(item["exact_support"]),
                float(item["candidate"].execution.graph.score),
                str(item["candidate"].execution.graph.graph_id),
            ),
            reverse=True,
        )
        proposal = ranking[0]
        runner = ranking[1] if len(ranking) > 1 else None
        selected_key = (
            str(selected.graph.graph_id),
            frozenset(map(str, selected.answer_ids)),
        )
        incumbent = next(
            (
                item
                for item in scored
                if (
                    str(item["candidate"].execution.graph.graph_id),
                    frozenset(map(str, item["candidate"].execution.answer_ids)),
                )
                == selected_key
            ),
            None,
        )
        if incumbent is None:
            evidence.update(
                {
                    "status": "incumbent_feature_missing",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        challenger = proposal["candidate"].execution
        relation = _answer_relation(challenger.answer_ids, selected.answer_ids)
        score_gap = float(proposal["score"]) - float(incumbent["score"])
        lane = ""
        if (
            relation == "subset"
            and score_gap + _template_consensus_selector_EPSILON >= SUBSET_SCORE_GAP
            and int(proposal["vote_count"]) >= SUBSET_VOTE_COUNT
        ):
            lane = "subset"
        elif (
            relation == "disjoint"
            and score_gap + _template_consensus_selector_EPSILON >= DISJOINT_SCORE_GAP
            and float(proposal["vote"]) + _template_consensus_selector_EPSILON >= DISJOINT_VOTE
            and float(proposal["relation_support"]) + _template_consensus_selector_EPSILON
            >= DISJOINT_RELATION_SUPPORT
        ):
            lane = "disjoint"
        evidence.update(
            {
                "candidate_count": len(candidates),
                "neighbour_count": len(neighbours),
                "source_graph_id": selected.graph.graph_id,
                "proposed_graph_id": challenger.graph.graph_id,
                "proposed_answer_ids": list(map(str, challenger.answer_ids)),
                "answer_relation": relation,
                "source_answer_count": len(set(map(str, selected.answer_ids))),
                "proposed_answer_count": len(set(map(str, challenger.answer_ids))),
                "source_score": float(incumbent["score"]),
                "proposed_score": float(proposal["score"]),
                "score_gap": score_gap,
                "candidate_margin": (
                    float(proposal["score"])
                    - (float(runner["score"]) if runner is not None else 0.0)
                ),
                "vote": float(proposal["vote"]),
                "vote_count": int(proposal["vote_count"]),
                "relation_support": float(proposal["relation_support"]),
                "exact_support": float(proposal["exact_support"]),
                "component_means": {
                    name: float(value)
                    for name, value in zip(
                        (
                            "total",
                            "relation",
                            "answer_incidence",
                            "anchor_answer_path",
                            "anchor_answer_direction",
                            "operator_signature",
                        ),
                        proposal["component_means"],
                    )
                },
                "source_operator_signature": list(
                    sorted(
                        _template_consensus_selector_operator_signature(
                            {"operators": selected.graph.operators}
                        ).elements()
                    )
                ),
                "proposed_operator_signature": list(
                    sorted(
                        _template_consensus_selector_operator_signature(
                            {"operators": challenger.graph.operators}
                        ).elements()
                    )
                ),
                "morphological_plural_request": _morphological_plural_request(
                    question
                ),
                "lane": lane,
                "top_source_template_indexes": [
                    document.source_index for _, document in neighbours[:8]
                ],
            }
        )
        if not lane or challenger is selected:
            evidence.update(
                {
                    "status": "confidence_gate_rejected",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        source_operators = _template_consensus_selector_operator_signature(
            {"operators": selected.graph.operators}
        )
        challenger_operators = _template_consensus_selector_operator_signature(
            {"operators": challenger.graph.operators}
        )
        if lane == "subset" and source_operators != challenger_operators:
            evidence.update(
                {
                    "status": "structural_guard_rejected",
                    "guard_reason": "subset_operator_signature_changed",
                    "source_operator_signature": list(
                        sorted(source_operators.elements())
                    ),
                    "proposed_operator_signature": list(
                        sorted(challenger_operators.elements())
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        if (
            lane == "disjoint"
            and len(set(map(str, challenger.answer_ids)))
            < len(set(map(str, selected.answer_ids)))
            and _morphological_plural_request(question)
        ):
            evidence.update(
                {
                    "status": "structural_guard_rejected",
                    "guard_reason": "plural_disjoint_cardinality_collapse",
                    "morphological_plural_request": True,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        evidence.update(
            {
                "status": "selected_template_consensus_challenger",
                "selected_graph_id": challenger.graph.graph_id,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return challenger, evidence






# ---- underanswer_expansion_selector ----
"""Conservative, model-free repair for a morphology-signalled under-answer.

The selector only considers answers that have already been completely
materialised by normal query-graph execution.  It recognises one narrow,
generic structure: the incumbent returns one value through a singular answer
predicate while a near-top graph changes only that predicate, pluralises the
same question-mentioned relation head, and returns a strict superset.

No evaluation Gold, endpoint, embedding/model call, entity list, relation
list, or sample identifier is available to this module.  It is intentionally
independent of :mod:`pipeline`; callers can place it after the existing hard,
path, template, TRAIN-extrema and numeric/temporal selectors.
"""



from collections import Counter
import itertools
import re
import time
from typing import Any, Iterable, Sequence

from .contracts import ExecutedGraph as _underanswer_expansion_selector_ExecutedGraph, QueryGraphCandidate as _underanswer_expansion_selector_QueryGraphCandidate


_underanswer_expansion_selector_SELECTOR_VERSION = "morphological-underanswer-expansion-production-v2"
_underanswer_expansion_selector_REASON_CODE = "morphological_underanswer_expansion_gate"
EXPLICIT_PLURAL_REASON_CODE = "explicit_plural_underanswer_expansion_gate"
_underanswer_expansion_selector_EPSILON = 1e-12
MIN_ANSWER_COUNT = 2
MAX_ANSWER_COUNT = 8
_underanswer_expansion_selector_MIN_RULE_SCORE_DELTA = -0.08
EXPLICIT_PLURAL_MIN_RULE_SCORE_DELTA = -0.10

_underanswer_expansion_selector_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_underanswer_expansion_selector_VARIABLE_RE = re.compile(r"^(?:V\d+|P\d+\.V\d+)$", re.IGNORECASE)
_ENTITY_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_underanswer_expansion_selector_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DATE_RE = re.compile(r"^-?\d{1,4}(?:-\d{1,2}(?:-\d{1,2})?)?(?:T.*)?$")
_underanswer_expansion_selector_QUOTED_RE = re.compile(r'["“][^"”]*["”]')
_PLURAL_AUXILIARIES = frozenset(
    {"are", "were", "have", "has", "do", "does", "did", "can", "could", "would", "will"}
)

_BLOCKING_REASON_CODES = frozenset(
    {
        "path_alignment_gate",
        "source_verified_template_consensus_gate",
        "train_gold_path_unique_extrema_gate",
        "fixed_slot_numeric_temporal_normalization",
        "entity_kind_fixed_beam",
    }
)


def _raw_tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _underanswer_expansion_selector_TOKEN_RE.finditer(str(value)))


def _is_plural_surface(token: str) -> bool:
    value = str(token).casefold()
    return bool(
        len(value) > 3
        and (
            (len(value) > 4 and value.endswith("ies"))
            or (
                value.endswith("s")
                and not value.endswith(("ss", "us", "is"))
            )
        )
    )


def _singular(token: str) -> str:
    value = str(token).casefold()
    if len(value) > 4 and value.endswith("ies"):
        return f"{value[:-3]}y"
    if _is_plural_surface(value):
        return value[:-1]
    return value


def _normalized_tokens(value: Any) -> frozenset[str]:
    return frozenset(_singular(token) for token in _raw_tokens(value))


def _relation_leaf(relation: str) -> str:
    return str(relation).split(".")[-1].replace("_", " ")


def _triples(graph: _underanswer_expansion_selector_QueryGraphCandidate) -> list[tuple[str, str, str]]:
    return [
        tuple(map(str, triple))
        for triple in graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]


def _constants(graph: _underanswer_expansion_selector_QueryGraphCandidate) -> frozenset[str]:
    return frozenset(
        value
        for subject, _, object_ in _triples(graph)
        for value in (subject, object_)
        if not _underanswer_expansion_selector_VARIABLE_RE.fullmatch(value)
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return str(value)


def _underanswer_expansion_selector_operator_signature(graph: _underanswer_expansion_selector_QueryGraphCandidate) -> tuple[Any, ...]:
    return tuple(sorted((_freeze(operator) for operator in graph.operators), key=repr))


def _value_kind(value: Any) -> str:
    text = str(value)
    if _ENTITY_RE.fullmatch(text):
        return "entity"
    if _underanswer_expansion_selector_NUMBER_RE.fullmatch(text):
        return "number"
    if _DATE_RE.fullmatch(text):
        return "date"
    return "literal"


def _answer_kind(values: Iterable[Any]) -> str:
    kinds = {_value_kind(value) for value in values}
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _type_closure(ontology: Any, value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    try:
        parents = ontology.supertypes(value)
    except (AttributeError, KeyError, TypeError, ValueError):
        parents = ()
    return frozenset((str(value), *(str(item) for item in parents)))


def _terminal_edge(
    graph: _underanswer_expansion_selector_QueryGraphCandidate, ontology: Any
) -> tuple[str, str, str, str, str] | None:
    answer = str(graph.answer_var)
    edges: list[tuple[str, str, str, str, str]] = []
    for subject, relation, object_ in _triples(graph):
        if object_ == answer:
            answer_type = str(ontology.range_for_relation(relation) or "")
            edges.append((subject, relation, object_, "in", answer_type))
        elif subject == answer:
            answer_type = str(ontology.domain_for_relation(relation) or "")
            edges.append((subject, relation, object_, "out", answer_type))
    return edges[0] if len(edges) == 1 else None


def _one_terminal_predicate_edit(
    incumbent: _underanswer_expansion_selector_QueryGraphCandidate,
    challenger: _underanswer_expansion_selector_QueryGraphCandidate,
) -> tuple[str, str] | None:
    if str(incumbent.answer_var) != str(challenger.answer_var):
        return None
    left, right = Counter(_triples(incumbent)), Counter(_triples(challenger))
    removed = list((left - right).elements())
    added = list((right - left).elements())
    answer = str(incumbent.answer_var)
    if not (
        len(removed) == len(added) == 1
        and removed[0][0] == added[0][0]
        and removed[0][2] == added[0][2]
        and removed[0][1] != added[0][1]
        and answer in (removed[0][0], removed[0][2])
        and answer in (added[0][0], added[0][2])
    ):
        return None
    return removed[0][1], added[0][1]


def _alpha_terminal_predicate_edit(
    incumbent: _underanswer_expansion_selector_QueryGraphCandidate,
    challenger: _underanswer_expansion_selector_QueryGraphCandidate,
    ontology: Any,
) -> dict[str, Any] | None:
    """Find one answer-edge predicate edit modulo variable renaming.

    The answer variable is fixed first and at most five remaining variables
    are permuted.  This hard bound keeps the branch well below its runtime
    budget while covering the short CWQ paths for which the gate was designed.
    """

    left, right = _triples(incumbent), _triples(challenger)
    incumbent_answer = str(incumbent.answer_var)
    challenger_answer = str(challenger.answer_var)
    left_variables = sorted(
        {
            value
            for subject, _, object_ in left
            for value in (subject, object_)
            if _underanswer_expansion_selector_VARIABLE_RE.fullmatch(value)
        }
    )
    right_variables = sorted(
        {
            value
            for subject, _, object_ in right
            for value in (subject, object_)
            if _underanswer_expansion_selector_VARIABLE_RE.fullmatch(value)
        }
    )
    if (
        len(left_variables) != len(right_variables)
        or incumbent_answer not in left_variables
        or challenger_answer not in right_variables
    ):
        return None
    left_other = [value for value in left_variables if value != incumbent_answer]
    right_other = [value for value in right_variables if value != challenger_answer]
    if len(left_other) > 5:
        return None
    target = Counter(right)
    for permutation in itertools.permutations(right_other):
        mapping = {
            incumbent_answer: challenger_answer,
            **dict(zip(left_other, permutation)),
        }
        mapped = Counter(
            (mapping.get(subject, subject), relation, mapping.get(object_, object_))
            for subject, relation, object_ in left
        )
        removed = list((mapped - target).elements())
        added = list((target - mapped).elements())
        if not (
            len(removed) == len(added) == 1
            and removed[0][0] == added[0][0]
            and removed[0][2] == added[0][2]
            and removed[0][1] != added[0][1]
            and challenger_answer in (removed[0][0], removed[0][2])
        ):
            continue
        direction = "out" if removed[0][0] == challenger_answer else "in"
        incumbent_type = str(
            (
                ontology.domain_for_relation(removed[0][1])
                if direction == "out"
                else ontology.range_for_relation(removed[0][1])
            )
            or ""
        )
        challenger_type = str(
            (
                ontology.domain_for_relation(added[0][1])
                if direction == "out"
                else ontology.range_for_relation(added[0][1])
            )
            or ""
        )
        if not (
            _type_closure(ontology, incumbent_type)
            & _type_closure(ontology, challenger_type)
        ):
            continue
        return {
            "incumbent_terminal_relation": removed[0][1],
            "challenger_terminal_relation": added[0][1],
            "answer_direction": direction,
            "incumbent_answer_type": incumbent_type,
            "challenger_answer_type": challenger_type,
            "alpha_variable_mapping": dict(sorted(mapping.items())),
        }
    return None


def _explicit_plural_request(question: str) -> dict[str, Any] | None:
    """Recognise a plural interrogative head immediately before an auxiliary.

    This syntactic shape distinguishes a noun such as ``movies`` in
    ``what fantasy movies has ...`` from third-person verbs such as
    ``which country includes ...`` without maintaining a vocabulary list.
    """

    tokens = list(_raw_tokens(_underanswer_expansion_selector_QUOTED_RE.sub(" ", str(question))))
    for question_index, token in enumerate(tokens[:4]):
        if token not in {"what", "which"}:
            continue
        for auxiliary_index in range(question_index + 2, min(len(tokens), question_index + 7)):
            if tokens[auxiliary_index] not in _PLURAL_AUXILIARIES:
                continue
            head = tokens[auxiliary_index - 1]
            if _is_plural_surface(head):
                return {
                    "plural_head": head,
                    "auxiliary": tokens[auxiliary_index],
                    "question_span": tokens[question_index : auxiliary_index + 1],
                }
            break
    return None


def _explicit_plural_expansion_candidates(
    question: str,
    selected: _underanswer_expansion_selector_ExecutedGraph,
    executions: Sequence[_underanswer_expansion_selector_ExecutedGraph],
    ontology: Any,
    *,
    min_answer_count: int,
    max_answer_count: int,
    min_rule_score_delta: float,
) -> list[tuple[_underanswer_expansion_selector_ExecutedGraph, dict[str, Any]]]:
    """Gold-free explicit-plural lane, before the singleton safety guard."""

    plural = _explicit_plural_request(question)
    if plural is None:
        return []
    selected_answers = frozenset(map(str, selected.answer_ids))
    if not selected_answers:
        return []
    eligible: list[tuple[_underanswer_expansion_selector_ExecutedGraph, dict[str, Any]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for execution in executions:
        answers = frozenset(map(str, execution.answer_ids))
        key = (str(execution.graph.graph_id), tuple(sorted(answers)))
        if key in seen:
            continue
        seen.add(key)
        rule_delta = float(execution.graph.score) - float(selected.graph.score)
        if not (
            selected_answers < answers
            and min_answer_count <= len(answers) <= max_answer_count
            and rule_delta + _underanswer_expansion_selector_EPSILON >= min_rule_score_delta
            and _answer_kind(answers) == _answer_kind(selected_answers)
            and _underanswer_expansion_selector_operator_signature(execution.graph)
            == _underanswer_expansion_selector_operator_signature(selected.graph)
            and _constants(execution.graph) == _constants(selected.graph)
        ):
            continue
        alpha_edit = _alpha_terminal_predicate_edit(
            selected.graph, execution.graph, ontology
        )
        if alpha_edit is None:
            continue
        eligible.append(
            (
                execution,
                {
                    "branch": "explicit_plural_singleton",
                    "reason_code": EXPLICIT_PLURAL_REASON_CODE,
                    "graph_id": str(execution.graph.graph_id),
                    "answer_count": len(answers),
                    "rule_score_delta": rule_delta,
                    **alpha_edit,
                    **plural,
                },
            )
        )
    return eligible


def _plural_head_evidence(
    question: str, incumbent_relation: str, challenger_relation: str
) -> dict[str, Any] | None:
    question_tokens = _normalized_tokens(question)
    incumbent_raw = _raw_tokens(_relation_leaf(incumbent_relation))
    challenger_raw = _raw_tokens(_relation_leaf(challenger_relation))
    incumbent_tokens = frozenset(_singular(token) for token in incumbent_raw)
    challenger_tokens = frozenset(_singular(token) for token in challenger_raw)
    challenger_plural_heads = frozenset(
        _singular(token) for token in challenger_raw if _is_plural_surface(token)
    )
    incumbent_plural_heads = frozenset(
        _singular(token) for token in incumbent_raw if _is_plural_surface(token)
    )
    shared_plural_heads = (
        challenger_plural_heads
        & incumbent_tokens
        & challenger_tokens
        & question_tokens
    ) - incumbent_plural_heads
    incumbent_modifiers = (incumbent_tokens - challenger_tokens) & question_tokens
    if not shared_plural_heads or not incumbent_modifiers:
        return None
    return {
        "shared_plural_heads": sorted(shared_plural_heads),
        "question_matched_incumbent_modifiers": sorted(incumbent_modifiers),
    }


def _underanswer_expansion_selector_higher_priority_selected(decision: dict[str, Any]) -> str:
    reasons = {str(reason) for reason in decision.get("reason_codes", [])}
    if any(reason.startswith("hard_") for reason in reasons):
        return "preserve_hard_post_selection_gate"
    blocked = sorted(reasons & _BLOCKING_REASON_CODES)
    return f"preserve_{blocked[0]}" if blocked else ""


class UnderanswerExpansionSelector:
    """Select one uniquely supported, morphology-signalled answer superset."""

    def __init__(
        self,
        ontology: Any,
        *,
        min_answer_count: int = MIN_ANSWER_COUNT,
        max_answer_count: int = MAX_ANSWER_COUNT,
        min_rule_score_delta: float = _underanswer_expansion_selector_MIN_RULE_SCORE_DELTA,
        explicit_plural_min_rule_score_delta: float = EXPLICIT_PLURAL_MIN_RULE_SCORE_DELTA,
    ) -> None:
        self.ontology = ontology
        self.min_answer_count = max(2, int(min_answer_count))
        self.max_answer_count = max(self.min_answer_count, int(max_answer_count))
        self.min_rule_score_delta = float(min_rule_score_delta)
        self.explicit_plural_min_rule_score_delta = float(
            explicit_plural_min_rule_score_delta
        )

    def challenge(
        self,
        *,
        question: str,
        selected: _underanswer_expansion_selector_ExecutedGraph,
        executions: Sequence[_underanswer_expansion_selector_ExecutedGraph],
        prior_decision: dict[str, Any],
    ) -> tuple[_underanswer_expansion_selector_ExecutedGraph, dict[str, Any]]:
        started = time.perf_counter()
        evidence: dict[str, Any] = {
            "version": _underanswer_expansion_selector_SELECTOR_VERSION,
            "status": "not_applicable",
            "model_calls": 0,
            "endpoint_queries": 0,
            "embedding_calls": 0,
            "uses_evaluation_gold": False,
            "entity_relation_or_sample_whitelist": False,
            "frozen_gate": {
                "min_answer_count": self.min_answer_count,
                "max_answer_count": self.max_answer_count,
                "minimum_rule_score_delta": self.min_rule_score_delta,
                "explicit_plural_minimum_rule_score_delta": (
                    self.explicit_plural_min_rule_score_delta
                ),
                "incumbent_answer_count": 1,
                "requires_strict_superset": True,
                "requires_unique_answer_set": True,
                "requires_exact_operator_signature": True,
                "requires_single_terminal_predicate_edit": True,
                "requires_ontology_type_compatibility": True,
                "requires_question_derived_plural_head": True,
            },
        }
        blocked = _underanswer_expansion_selector_higher_priority_selected(prior_decision)
        if blocked:
            evidence.update(
                {
                    "status": "skipped_existing_higher_priority_gate",
                    "reason": blocked,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        selected_answers = frozenset(map(str, selected.answer_ids))
        if len(selected_answers) != 1 or self.ontology is None:
            evidence.update(
                {
                    "status": "incumbent_not_singleton_or_ontology_missing",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        incumbent_terminal = _terminal_edge(selected.graph, self.ontology)
        explicit_plural = _explicit_plural_request(question)

        eligible: list[tuple[_underanswer_expansion_selector_ExecutedGraph, dict[str, Any]]] = []
        explicit_eligible: list[tuple[_underanswer_expansion_selector_ExecutedGraph, dict[str, Any]]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for execution in executions:
            answers = frozenset(map(str, execution.answer_ids))
            key = (str(execution.graph.graph_id), tuple(sorted(answers)))
            if key in seen:
                continue
            seen.add(key)
            if not (
                selected_answers < answers
                and self.min_answer_count <= len(answers) <= self.max_answer_count
                and _answer_kind(answers) == _answer_kind(selected_answers)
                and _underanswer_expansion_selector_operator_signature(execution.graph)
                == _underanswer_expansion_selector_operator_signature(selected.graph)
                and _constants(execution.graph) == _constants(selected.graph)
            ):
                continue
            rule_delta = float(execution.graph.score) - float(selected.graph.score)
            if (
                incumbent_terminal is not None
                and rule_delta + _underanswer_expansion_selector_EPSILON >= self.min_rule_score_delta
            ):
                terminal = _terminal_edge(execution.graph, self.ontology)
                edit = _one_terminal_predicate_edit(selected.graph, execution.graph)
                if (
                    terminal is not None
                    and edit is not None
                    and terminal[3] == incumbent_terminal[3]
                ):
                    incumbent_types = _type_closure(
                        self.ontology, incumbent_terminal[4]
                    )
                    challenger_types = _type_closure(self.ontology, terminal[4])
                    morphology = _plural_head_evidence(question, edit[0], edit[1])
                    if (
                        incumbent_types
                        and challenger_types
                        and incumbent_types & challenger_types
                        and morphology is not None
                    ):
                        eligible.append(
                            (
                                execution,
                                {
                                    "branch": "relation_head_pluralization",
                                    "reason_code": _underanswer_expansion_selector_REASON_CODE,
                                    "graph_id": str(execution.graph.graph_id),
                                    "answer_count": len(answers),
                                    "rule_score_delta": rule_delta,
                                    "incumbent_terminal_relation": edit[0],
                                    "challenger_terminal_relation": edit[1],
                                    "answer_direction": terminal[3],
                                    "incumbent_answer_type": incumbent_terminal[4],
                                    "challenger_answer_type": terminal[4],
                                    **morphology,
                                },
                            )
                        )
        if explicit_plural is not None:
            explicit_eligible = _explicit_plural_expansion_candidates(
                question,
                selected,
                executions,
                self.ontology,
                min_answer_count=self.min_answer_count,
                max_answer_count=self.max_answer_count,
                min_rule_score_delta=self.explicit_plural_min_rule_score_delta,
            )
        answer_sets = {
            frozenset(map(str, execution.answer_ids)) for execution, _ in eligible
        }
        if len(answer_sets) > 1:
            evidence.update(
                {
                    "status": "no_unique_supported_expansion",
                    "eligible_candidate_count": len(eligible),
                    "eligible_answer_set_count": len(answer_sets),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        selected_lane = eligible
        reason = "unique_question_derived_plural_terminal_expansion"
        if not selected_lane:
            explicit_answer_sets = {
                frozenset(map(str, execution.answer_ids))
                for execution, _ in explicit_eligible
            }
            if len(explicit_answer_sets) != 1:
                evidence.update(
                    {
                        "status": "no_unique_supported_expansion",
                        "eligible_candidate_count": len(explicit_eligible),
                        "eligible_answer_set_count": len(explicit_answer_sets),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                return selected, evidence
            selected_lane = explicit_eligible
            reason = "explicit_plural_singleton_alpha_equivalent_expansion"
        chosen, proposal = max(
            selected_lane,
            key=lambda item: (
                float(item[0].graph.score),
                str(item[0].graph.graph_id),
            ),
        )
        evidence.update(
            {
                "status": "applied",
                "reason": reason,
                "reason_code": proposal["reason_code"],
                "eligible_candidate_count": len(selected_lane),
                "proposal": proposal,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return chosen, evidence






# ---- entity_kind_fixed_beam ----
"""Fixed-budget recovery for entity questions with scalar/schema graph slots.

This module is deliberately independent from :mod:`pipeline` so both the
legacy and v2 execution loops can integrate it without a circular import.  A
route is planned before endpoint execution from only:

* the question wh-head;
* ontology types incident on graph answer variables;
* existing semantic/decomposition traces; and
* a cache built from CWQ TRAIN Gold-SPARQL templates.

Prepared raw-SPARQL templates replace existing execution slots.  They never
increase the endpoint-query budget and are excluded from the model selector.
After ordinary selection, a prepared result may replace the incumbent only
when the incumbent answers are all non-entity values and the prepared answer
set is entirely Freebase entities.  Existing entity selections are therefore
a hard non-regression interlock.
"""



from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ExecutedGraph as _entity_kind_fixed_beam_ExecutedGraph, QueryGraphCandidate as _entity_kind_fixed_beam_QueryGraphCandidate
from .ontology import relation_id_from_label as _entity_kind_fixed_beam_relation_id_from_label


VERSION = "entity-kind-fixed-beam-train-template-v1"
ENTITY_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
NS_VALUE_RE = re.compile(r"ns:([A-Za-z0-9_.]+)")
DOUBLE_LITERAL_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
SCALAR_OR_SCHEMA_TYPES = frozenset(
    {
        "type.boolean",
        "type.datetime",
        "type.enumeration",
        "type.float",
        "type.int",
        "type.property",
        "type.rawstring",
        "type.schema",
        "type.text",
        "type.type",
    }
)
SCALAR_WH_HEADS = frozenset(
    {
        "age", "amount", "area", "capacity", "code", "coordinate",
        "cost", "date", "distance", "duration", "height", "id",
        "identifier", "latitude", "length", "longitude", "number",
        "percent", "percentage", "population", "price", "rank", "rate",
        "score", "temperature", "time", "weight", "width", "year",
    }
)
WH_SKIP = frozenset(
    {
        "a", "an", "are", "be", "been", "being", "can", "could",
        "did", "do", "does", "had", "has", "have", "is", "should",
        "the", "was", "were", "will", "would",
    }
)
TOKEN_STOP = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by",
        "did", "do", "does", "for", "from", "had", "has", "have",
        "her", "his", "in", "is", "it", "its", "of", "on", "that",
        "the", "their", "to", "was", "were", "what", "when", "where",
        "which", "who", "with",
    }
)


@dataclass(slots=True)
class PreparedEntityTemplateQuery:
    graph_id: str
    query: str
    answer_var: str
    score: float
    source_index: int
    source_template_rank: int
    mapping_rank: int
    source_question: str
    rank_evidence: dict[str, float]

    def trace_summary(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "query_sha256": hashlib.sha256(self.query.encode()).hexdigest(),
            "answer_var": self.answer_var,
            "fixed_beam_score": self.score,
            "source_index": self.source_index,
            "source_template_rank": self.source_template_rank,
            "mapping_rank": self.mapping_rank,
            "rank_evidence": dict(self.rank_evidence),
        }


@dataclass(slots=True)
class EntityKindFixedBeamPlan:
    ordinary_graphs: list[_entity_kind_fixed_beam_QueryGraphCandidate]
    template_queries: list[PreparedEntityTemplateQuery]
    diagnostics: dict[str, Any]


def _stem(token: str) -> str:
    value = str(token).casefold()
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 4:
        return value[:-1]
    return value


def _words(value: str, *, stop: bool = True) -> list[str]:
    result = [_stem(token) for token in re.findall(r"[a-z]+", str(value).casefold())]
    return [token for token in result if not stop or token not in TOKEN_STOP]


def _dice(left: set[Any], right: set[Any]) -> float:
    return 2.0 * len(left & right) / (len(left) + len(right)) if left and right else 0.0


def question_expects_entity(question: str) -> bool:
    """Recognize entity wh-heads while excluding clear scalar/time heads."""
    text = " ".join(str(question).casefold().split())
    if re.match(r"^(?:when|how\s+(?:many|much|old|long))\b", text):
        return False
    tokens = re.findall(r"[a-z]+", text)
    if not tokens:
        return False
    if tokens[0] in {"who", "where"}:
        return True
    if tokens[0] not in {"what", "which"}:
        return False
    position = 1
    while position < len(tokens) and tokens[position] in WH_SKIP:
        position += 1
    if position >= len(tokens):
        return False
    head = _stem(tokens[position])
    if head in SCALAR_WH_HEADS:
        return False
    if head == "iso":
        return "country" in tokens[position + 1 : position + 7]
    return True


def answer_kind(values: Sequence[Any]) -> str:
    materialized = [str(value) for value in values]
    if not materialized:
        return "empty"
    flags = [bool(ENTITY_RE.fullmatch(value)) for value in materialized]
    if all(flags):
        return "entity"
    if not any(flags):
        return "nonentity"
    return "mixed"


def terminal_types(graph: _entity_kind_fixed_beam_QueryGraphCandidate, ontology: Any) -> frozenset[str]:
    answer = str(graph.answer_var)
    output: set[str] = set()
    for subject, relation, object_ in graph.triples:
        if str(object_) == answer:
            value = str(ontology.range_for_relation(str(relation)))
            if value:
                output.add(value)
        if str(subject) == answer:
            value = str(ontology.domain_for_relation(str(relation)))
            if value:
                output.add(value)
    return frozenset(output)


def is_scalar_schema_answer_slot(graph: _entity_kind_fixed_beam_QueryGraphCandidate, ontology: Any) -> bool:
    values = terminal_types(graph, ontology)
    return bool(values and values <= SCALAR_OR_SCHEMA_TYPES)


def _slot_plan(
    graphs: Sequence[_entity_kind_fixed_beam_QueryGraphCandidate], ontology: Any, maximum: int
) -> tuple[list[int], dict[str, Any]]:
    scalar_indexes = [
        index
        for index, graph in enumerate(graphs)
        if is_scalar_schema_answer_slot(graph, ontology)
    ]
    strongest_index = max(
        scalar_indexes,
        key=lambda index: (float(graphs[index].score), str(graphs[index].graph_id)),
        default=-1,
    )
    replaceable = [index for index in scalar_indexes if index != strongest_index]
    policy = "replace_lower_scored_scalar_schema_slots"
    if not replaceable and strongest_index >= 0 and len(graphs) >= 2:
        top_index = max(
            range(len(graphs)),
            key=lambda index: (float(graphs[index].score), str(graphs[index].graph_id)),
        )
        if top_index == strongest_index:
            replaceable = [
                min(
                    (index for index in range(len(graphs)) if index != strongest_index),
                    key=lambda index: (
                        float(graphs[index].score),
                        str(graphs[index].graph_id),
                    ),
                )
            ]
            policy = "singleton_top_scalar_preserve_sentinel_replace_lowest_other"
    replaceable.sort(
        key=lambda index: (float(graphs[index].score), str(graphs[index].graph_id))
    )
    selected = replaceable[: min(3, max(0, int(maximum)))]
    return selected, {
        "scalar_slot_count": len(scalar_indexes),
        "sentinel_graph_id": (
            str(graphs[strongest_index].graph_id) if strongest_index >= 0 else ""
        ),
        "replacement_slot_count": len(selected),
        "replaced_graph_ids": [str(graphs[index].graph_id) for index in selected],
        "slot_policy": policy if selected else "no_fixed_slot_available",
        "types_by_graph": {
            str(graphs[index].graph_id): sorted(terminal_types(graphs[index], ontology))
            for index in scalar_indexes
        },
    }


def semantic_relation_unions(semantic_graphs: Sequence[Mapping[str, Any]]) -> list[frozenset[str]]:
    output: list[frozenset[str]] = []
    for raw in semantic_graphs:
        graph: Any = raw.get("output", raw) if isinstance(raw, Mapping) else {}
        if not isinstance(graph, Mapping):
            continue
        relation_ids: set[str] = set()
        for path in graph.get("semantic_paths", []):
            if not isinstance(path, Mapping):
                continue
            for step in path.get("steps", []):
                if not isinstance(step, Mapping):
                    continue
                relation_id = _entity_kind_fixed_beam_relation_id_from_label(step.get("relation_label", []))
                if relation_id:
                    relation_ids.add(relation_id)
        frozen = frozenset(relation_ids)
        if frozen and frozen not in output:
            output.append(frozen)
    return output


def _canonical_relation_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _query_relations(query: str) -> frozenset[str]:
    return frozenset(
        value
        for value in NS_VALUE_RE.findall(str(query))
        if not ENTITY_RE.fullmatch(value)
    )


def _query_operator_signature(query: str) -> frozenset[str]:
    text = " ".join(str(query).casefold().split())
    output: set[str] = set()
    if "order by asc" in text:
        output.add("argmin")
    if "order by desc" in text:
        output.add("argmax")
    if re.search(r"filter\s*\([^)]*?\s<\s", text):
        output.add("lt")
    if re.search(r"filter\s*\([^)]*?\s>\s", text):
        output.add("gt")
    if "count(" in text:
        output.add("count")
    return frozenset(output)


def _incompatible_copied_string(query: str, target_text: str) -> bool:
    target_tokens = set(_words(target_text))
    for literal in DOUBLE_LITERAL_RE.findall(query):
        value = bytes(literal, "utf-8").decode("unicode_escape")
        if not value or re.fullmatch(
            r"[-+]?\d+(?:\.\d+)?(?:-\d{2}(?:-\d{2})?)?", value
        ):
            continue
        literal_tokens = set(_words(value))
        if literal_tokens and not literal_tokens <= target_tokens:
            return True
    return False


def _rank_template(
    candidate: dict[str, Any],
    *,
    question: str,
    target_text: str,
    semantic_unions: Sequence[frozenset[str]],
    target_operator: set[str],
) -> tuple[float, dict[str, float]] | None:
    query = str(candidate["query"])
    if _incompatible_copied_string(query, target_text):
        return None
    relations = {
        _canonical_relation_id(value) for value in _query_relations(query)
    }
    relation_alignment = max(
        (
            _dice(
                relations,
                {_canonical_relation_id(item) for item in semantic_union},
            )
            for semantic_union in semantic_unions
        ),
        default=0.0,
    )
    source_operator = set(candidate.get("source_operator_signature", ()))
    if target_operator:
        source_operator.update(_query_operator_signature(query))
    operator_alignment = (
        1.0
        if source_operator == target_operator
        else len(source_operator & target_operator) / len(source_operator | target_operator)
        if source_operator | target_operator
        else 1.0
    )
    source_question = str(candidate.get("source_question", ""))
    token_alignment = _dice(set(_words(source_question)), set(_words(question)))
    source_raw = _words(source_question, stop=False)
    target_raw = _words(question, stop=False)
    bigram_alignment = _dice(
        set(zip(source_raw, source_raw[1:])),
        set(zip(target_raw, target_raw[1:])),
    )
    score = (
        10.0 * relation_alignment
        + 4.0 * operator_alignment
        + 3.0 * token_alignment
        + 2.0 * bigram_alignment
        - 0.01 * int(candidate.get("source_template_rank", 0))
        + 0.001 * float(candidate.get("runtime_score", 0.0))
    )
    return score, {
        "relation_alignment": relation_alignment,
        "operator_alignment": operator_alignment,
        "token_alignment": token_alignment,
        "bigram_alignment": bigram_alignment,
    }


def _answer_values(rows: Sequence[Mapping[str, Any]], answer_var: str) -> list[str]:
    key = str(answer_var).lstrip("?")
    values: list[str] = []
    for row in rows:
        value = row.get(key, row.get("answer", ""))
        if not value:
            value = next(
                (
                    candidate
                    for name, candidate in row.items()
                    if str(name).casefold() == key.casefold()
                ),
                "",
            )
        text = str(value).strip()
        prefix = "http://rdf.freebase.com/ns/"
        if text.startswith(prefix):
            text = text[len(prefix):]
        if text and text not in values:
            values.append(text)
    return values


def _answer_labels(answer_ids: Sequence[str], labels: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    by_id = {
        str(item.get("id", "")): {
            "id": str(item.get("id", "")),
            "label": str(item.get("label", "")),
        }
        for item in labels
        if str(item.get("id", ""))
    }
    return [by_id.get(value, {"id": value, "label": value}) for value in answer_ids]


class EntityKindFixedBeamPlanner:
    """Lazy TRAIN-template planner shared by both pipeline execution modes."""

    def __init__(self, cache_path: str | Path, *, top_k: int = 32) -> None:
        self.cache_path = str(Path(cache_path).expanduser())
        self.top_k = min(64, max(3, int(top_k)))
        self._index_value: Any = None
        self._index_lock = Lock()

    def _index(self) -> Any:
        value = self._index_value
        if value is not None:
            return value
        # Lazy import avoids failure_gold_sparql_retrieval -> pipeline import
        # cycles while reusing the already-audited TRAIN-only cache reader.
        from .failure_retrieval import SourceGoldSparqlIndex

        with self._index_lock:
            if self._index_value is None:
                self._index_value = SourceGoldSparqlIndex(self.cache_path)
            return self._index_value

    def _templates(
        self,
        *,
        question: str,
        decomposition: Sequence[str],
        entities: Sequence[tuple[str, str]],
        semantic_unions: Sequence[frozenset[str]],
        limit: int,
    ) -> list[PreparedEntityTemplateQuery]:
        from .failure_retrieval import _effective_operator_signature, _instantiate, _operator_signature

        index = self._index()
        matches = index.retrieve(
            question=question,
            decomposition=decomposition,
            surfaces=list(dict.fromkeys(label for _, label in entities)),
            entity_count=len(entities),
            top_k=self.top_k,
        )
        values: list[dict[str, Any]] = []
        for match in matches:
            document = match["document"]
            for candidate in _instantiate(match, entities, question, decomposition):
                candidate["source_question"] = document.question
                candidate["source_operator_signature"] = list(
                    _effective_operator_signature(document)
                )
                values.append(candidate)
        values.sort(
            key=lambda item: (
                -float(item["runtime_score"]),
                int(item["source_template_rank"]),
                int(item["mapping_rank"]),
                str(item["key"]),
            )
        )
        pool: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in values:
            signature = " ".join(str(candidate["query"]).split())
            if signature in seen:
                continue
            seen.add(signature)
            pool.append(candidate)
            if len(pool) >= 20:
                break
        ranked: list[tuple[float, dict[str, float], dict[str, Any]]] = []
        target_operator = set(_operator_signature(question))
        target_text = " ".join([question, *map(str, decomposition)])
        for candidate in pool:
            result = _rank_template(
                candidate,
                question=question,
                target_text=target_text,
                semantic_unions=semantic_unions,
                target_operator=target_operator,
            )
            if result is None:
                continue
            score, evidence = result
            ranked.append((score, evidence, candidate))
        ranked.sort(
            key=lambda item: (
                -item[0],
                int(item[2]["source_template_rank"]),
                int(item[2]["mapping_rank"]),
                str(item[2]["key"]),
            )
        )
        return [
            PreparedEntityTemplateQuery(
                graph_id=f"EK{offset}",
                query=str(candidate["query"]),
                answer_var=str(candidate["answer_var"]),
                score=float(score),
                source_index=int(candidate["source_index"]),
                source_template_rank=int(candidate["source_template_rank"]),
                mapping_rank=int(candidate["mapping_rank"]),
                source_question=str(candidate["source_question"]),
                rank_evidence=dict(evidence),
            )
            for offset, (score, evidence, candidate) in enumerate(ranked[: max(0, int(limit))])
        ]

    def prepare(
        self,
        *,
        question: str,
        graphs: Sequence[_entity_kind_fixed_beam_QueryGraphCandidate],
        decomposition: Sequence[str],
        semantic_graphs: Sequence[Mapping[str, Any]],
        entities: Sequence[tuple[str, str]],
        ontology: Any,
        execution_budget: int,
        max_replacements: int = 3,
    ) -> EntityKindFixedBeamPlan:
        budget = min(len(graphs), max(0, int(execution_budget)))
        active = list(graphs[:budget])
        diagnostics: dict[str, Any] = {
            "version": VERSION,
            "status": "not_applied",
            "reason": "",
            "preexecution_trigger": False,
            "query_count_before": budget,
            "query_count_after": budget,
            "additional_endpoint_queries": 0,
            "additional_model_calls": 0,
            "uses_evaluation_gold": False,
            "uses_test_sparql_or_reasoning_information": False,
            "source": "CWQ_TRAIN_gold_sparql_only",
            "entity_or_relation_whitelist": False,
        }
        if ontology is None or budget < 2:
            diagnostics["reason"] = "missing_ontology_or_execution_slots"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        if not question_expects_entity(question):
            diagnostics["reason"] = "question_not_entity_wh_head"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        evicted, slot_diagnostics = _slot_plan(active, ontology, max_replacements)
        diagnostics.update(slot_diagnostics)
        if not evicted:
            diagnostics["reason"] = "no_fixed_replacement_slot"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        if not entities:
            diagnostics["reason"] = "no_linked_entities"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        unions = semantic_relation_unions(semantic_graphs)
        if not unions:
            diagnostics["reason"] = "no_semantic_relation_union"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        started = time.perf_counter()
        templates = self._templates(
            question=question,
            decomposition=decomposition,
            entities=entities,
            semantic_unions=unions,
            limit=len(evicted),
        )
        if not templates:
            diagnostics["reason"] = "no_compatible_train_template"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        # If unmatched-string filtering yields fewer templates than slots,
        # evict exactly the number actually occupied.
        evicted = evicted[: len(templates)]
        evicted_set = set(evicted)
        ordinary = [graph for index, graph in enumerate(active) if index not in evicted_set]
        diagnostics.update(
            {
                "status": "prepared",
                "reason": "entity_wh_head_with_scalar_schema_answer_slot",
                "preexecution_trigger": True,
                "replacement_query_count": len(templates),
                "replaced_graph_ids": [active[index].graph_id for index in evicted],
                "template_graph_ids": [item.graph_id for item in templates],
                "query_count_after": len(ordinary) + len(templates),
                "semantic_relation_unions": [sorted(value) for value in unions],
                "planning_elapsed_seconds": time.perf_counter() - started,
            }
        )
        return EntityKindFixedBeamPlan(ordinary, templates, diagnostics)


def execute_prepared_query(
    prepared: PreparedEntityTemplateQuery,
    endpoint: Any,
    *,
    lookup_labels: bool = True,
) -> tuple[_entity_kind_fixed_beam_ExecutedGraph | None, dict[str, Any]]:
    """Execute one already budgeted raw template query."""
    started = time.perf_counter()
    trace: dict[str, Any] = {
        **prepared.trace_summary(),
        "fixed_execution_slot": True,
        "additional_endpoint_queries": 0,
        "additional_model_calls": 0,
    }
    try:
        rows = endpoint.execute(prepared.query)
        answer_ids = _answer_values(rows, prepared.answer_var)
    except Exception as exc:
        trace.update(
            {
                "error": f"{type(exc).__name__}:{exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return None, trace
    raw_labels: list[dict[str, str]] = []
    if lookup_labels and answer_ids:
        entity_ids = [value for value in answer_ids if ENTITY_RE.fullmatch(value)]
        if entity_ids:
            try:
                raw_labels = list(endpoint.labels(entity_ids))
            except Exception as exc:
                trace["label_error"] = f"{type(exc).__name__}:{exc}"
    graph = _entity_kind_fixed_beam_QueryGraphCandidate(
        graph_id=prepared.graph_id,
        triples=[],
        answer_var=prepared.answer_var,
        operators=[],
        score=prepared.score,
        compose_output={},
        provenance={
            "entity_kind_fixed_beam": {
                **prepared.trace_summary(),
                "source_question": prepared.source_question,
                "additional_endpoint_queries": 0,
                "additional_model_calls": 0,
            }
        },
        sparql=prepared.query,
    )
    trace.update(
        {
            "row_count": len(rows),
            "answer_count": len(answer_ids),
            "answer_kind": answer_kind(answer_ids),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return (
        _entity_kind_fixed_beam_ExecutedGraph(
            graph=graph,
            answer_ids=answer_ids,
            answers=_answer_labels(answer_ids, raw_labels),
            row_count=len(rows),
        ),
        trace,
    )


def is_entity_kind_variant(execution: _entity_kind_fixed_beam_ExecutedGraph) -> bool:
    return isinstance(execution.graph.provenance.get("entity_kind_fixed_beam"), dict)


def postselect_entity_kind_execution(
    selected: _entity_kind_fixed_beam_ExecutedGraph,
    variants: Sequence[_entity_kind_fixed_beam_ExecutedGraph],
) -> tuple[_entity_kind_fixed_beam_ExecutedGraph, dict[str, Any]]:
    """Switch only a pure non-entity incumbent to a pure entity variant."""
    diagnostics: dict[str, Any] = {
        "version": VERSION,
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "source_answer_kind": answer_kind(selected.answer_ids),
        "additional_endpoint_queries": 0,
        "additional_model_calls": 0,
        "entity_or_relation_whitelist": False,
    }
    if diagnostics["source_answer_kind"] != "nonentity":
        diagnostics["reason"] = "existing_answer_is_not_pure_nonentity"
        return selected, diagnostics
    eligible = [
        execution
        for execution in variants
        if execution.answer_ids and answer_kind(execution.answer_ids) == "entity"
    ]
    eligible.sort(
        key=lambda execution: (
            -float(execution.graph.score),
            str(execution.graph.graph_id),
        )
    )
    if not eligible:
        diagnostics["reason"] = "no_nonempty_entity_template"
        return selected, diagnostics
    challenger = eligible[0]
    diagnostics.update(
        {
            "status": "applied",
            "reason": "entity_template_replaces_nonentity_answer",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_answer_count": len(challenger.answer_ids),
            "eligible_graph_ids": [item.graph.graph_id for item in eligible],
        }
    )
    return challenger, diagnostics


def select_empty_entity_kind_execution(
    question: str,
    variants: Sequence[_entity_kind_fixed_beam_ExecutedGraph],
) -> tuple[_entity_kind_fixed_beam_ExecutedGraph | None, dict[str, Any]]:
    """Commit an already-budgeted entity variant only under answer consensus."""

    diagnostics: dict[str, Any] = {
        "version": VERSION,
        "status": "not_applied",
        "source_answer_kind": "empty",
        "additional_endpoint_queries": 0,
        "additional_model_calls": 0,
        "entity_or_relation_whitelist": False,
        "ordinary_nonempty_count": 0,
    }
    if not question_expects_entity(question):
        diagnostics["reason"] = "question_does_not_request_entity"
        return None, diagnostics
    eligible = [
        execution
        for execution in variants
        if execution.answer_ids and answer_kind(execution.answer_ids) == "entity"
    ]
    diagnostics["entity_variant_nonempty_count"] = len(eligible)
    if not eligible:
        diagnostics["reason"] = "no_nonempty_entity_template"
        return None, diagnostics
    answer_sets = {
        tuple(sorted(set(map(str, execution.answer_ids))))
        for execution in eligible
    }
    diagnostics["entity_variant_answer_set_support"] = {
        " | ".join(values): sum(
            tuple(sorted(set(map(str, item.answer_ids)))) == values
            for item in eligible
        )
        for values in sorted(answer_sets)
    }
    if len(answer_sets) != 1:
        diagnostics["reason"] = "divergent_fixed_variant_answers"
        return None, diagnostics
    selected = max(
        eligible,
        key=lambda execution: (
            float(execution.graph.score),
            str(execution.graph.graph_id),
        ),
    )
    diagnostics.update(
        {
            "status": "applied",
            "reason": "ordinary_empty_consistent_entity_fixed_variant",
            "selected_graph_id": selected.graph.graph_id,
            "selected_answer_count": len(selected.answer_ids),
            "eligible_graph_ids": [item.graph.graph_id for item in eligible],
            "fixed_execution_slot": True,
        }
    )
    return selected, diagnostics


def linked_entities_for_question(
    gold_entities: Mapping[str, Mapping[str, str]], question: str
) -> list[tuple[str, str]]:
    """Project configured topic links in their linker-provenance order.

    The source order commonly reflects path/constraint roles.  Re-sorting by
    surface position loses that signal for multi-anchor questions; template
    instantiation already evaluates alternative permutations explicitly.
    """
    normalized = " ".join(str(question).casefold().split())
    mapping = gold_entities.get(normalized, {})
    if not isinstance(mapping, Mapping):
        return []
    return [
        (str(entity_id), str(label))
        for entity_id, label in mapping.items()
    ]






# ---- train_gold_extrema_selector ----
"""Frozen TRAIN-Gold path gate for explicit unique extrema questions.

This selector is deliberately narrow.  It runs after all existing hard,
path-alignment and source-template gates, and immediately abstains when one of
those gates already changed the incumbent.  A challenger must preserve the
complete singleton/operator structure and gain exact top-1 relation-path
support from a local index parsed from *TRAIN* Gold SPARQL.

There are no model, embedding, endpoint, entity-list or relation-list calls.
The evaluation split's Gold answers are not accepted by the API.
"""



from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import heapq
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Sequence

from .contracts import ExecutedGraph as _train_gold_extrema_selector_ExecutedGraph
_train_gold_extrema_selector_NaturalPath = _path_alignment_selector_NaturalPath
_train_gold_extrema_selector_SemanticIntent = _path_alignment_selector_SemanticIntent
_train_gold_extrema_selector_natural_paths = _path_alignment_selector_natural_paths
_train_gold_extrema_selector_operator_signature = _path_alignment_selector_operator_signature
_train_gold_extrema_selector_semantic_intents = _path_alignment_selector_semantic_intents
from .template_path_retrieval import mask_entities as _train_gold_extrema_selector_mask_entities


_train_gold_extrema_selector_SELECTOR_VERSION = "train-gold-path-unique-extrema-production-v1"
_train_gold_extrema_selector_INDEX_VERSION = 2
_train_gold_extrema_selector_REASON_CODE = "train_gold_path_unique_extrema_gate"
_train_gold_extrema_selector_TOP_K = 48
_train_gold_extrema_selector_EPSILON = 1e-12

# Generic high-confidence threshold frozen for the focused lane.  Use a round
# value rather than copying any individual evaluation sample's score.
MIN_ONTOLOGY_VOTE_DELTA = 0.95
MIN_GOLD_SCORE_DELTA = 0.5
_train_gold_extrema_selector_MIN_RULE_SCORE_DELTA = -0.02
RETRIEVAL_RULE_SCORE_FLOOR = -0.12

_train_gold_extrema_selector_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MAXIMUM_RE = re.compile(
    r"\b(?:latest|last|most recent|biggest|largest|highest|maximum|newest|longest)\b",
    re.I,
)
_MINIMUM_RE = re.compile(
    r"\b(?:earliest|least|smallest|lowest|minimum|first|oldest|youngest|shortest)\b",
    re.I,
)
_NON_EXTREMA_NAME_RE = re.compile(r"\b(?:first|last)\s+name\b", re.I)
_train_gold_extrema_selector_QUOTED_RE = re.compile(r'["“][^"”]*["”]')
_FULL_DATE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_YEAR_RE = re.compile(r"(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d.])")
_train_gold_extrema_selector_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")


def _train_gold_extrema_selector_tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _train_gold_extrema_selector_TOKEN_RE.finditer(str(value)))


def _train_gold_extrema_selector_set_f1(left: Iterable[Any], right: Iterable[Any]) -> float:
    one, two = set(left), set(right)
    if not one or not two:
        return 0.0
    return 2.0 * len(one & two) / (len(one) + len(two))


def _train_gold_extrema_selector_relation_leaf_tokens(relations: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        token
        for relation in relations
        for token in _train_gold_extrema_selector_tokens(str(relation).split(".")[-1].replace("_", " "))
    )


def _normalize_scalars(value: Any) -> str:
    text = _FULL_DATE_RE.sub(" date ", str(value))
    text = _YEAR_RE.sub(" year ", text)
    text = _train_gold_extrema_selector_NUMBER_RE.sub(" number ", text)
    return " ".join(text.split())


def _relation_text(relations: Sequence[str]) -> str:
    return " ".join(
        segment.replace("_", " ")
        for relation in relations
        for segment in str(relation).split(".")
    )


def _unique_extrema_operator(question: str) -> str:
    text = _train_gold_extrema_selector_QUOTED_RE.sub(" ", str(question))
    text = _NON_EXTREMA_NAME_RE.sub(" ", text)
    maximum = bool(_MAXIMUM_RE.search(text))
    minimum = bool(_MINIMUM_RE.search(text))
    if maximum == minimum:
        return ""
    return "ARGMAX" if maximum else "ARGMIN"


def _train_gold_extrema_selector_higher_priority_selected(decision: dict[str, Any]) -> bool:
    return any(
        str(reason).startswith("hard_")
        or str(reason) in {
            "path_alignment_gate",
            "source_verified_template_consensus_gate",
        }
        for reason in decision.get("reason_codes", [])
    )


@dataclass(frozen=True, slots=True)
class GoldPathDocument:
    source_index: int
    relations: tuple[str, ...]
    directions: tuple[str, ...]
    operator_signature: tuple[str, ...]
    tokens: tuple[str, ...]
    alignment: tuple[float, ...]


class TrainGoldPathIndex:
    """Read-only sparse index produced by the offline TRAIN parser."""

    def __init__(self, documents: Sequence[GoldPathDocument], *, cache_hit: bool) -> None:
        self.documents = list(documents)
        self.cache_hit = bool(cache_hit)
        self.frequencies = [Counter(document.tokens) for document in self.documents]
        self.average_length = sum(len(document.tokens) for document in self.documents) / max(
            1, len(self.documents)
        )
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document.tokens))
        count = len(self.documents)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, frequencies in enumerate(self.frequencies):
            for token, frequency in frequencies.items():
                self.inverted[token].append((index, frequency))

    @classmethod
    def load(cls, path: str | Path) -> "TrainGoldPathIndex":
        cache_path = Path(path).expanduser().resolve()
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("version") != _train_gold_extrema_selector_INDEX_VERSION:
            raise ValueError(f"invalid TRAIN Gold path cache: {cache_path}")
        documents = []
        for value in payload.get("documents", []):
            if not isinstance(value, dict):
                continue
            relations = tuple(map(str, value.get("relations", [])))
            directions = tuple(map(str, value.get("directions", [])))
            tokens = tuple(map(str, value.get("tokens", [])))
            if not relations or len(relations) != len(directions) or not tokens:
                continue
            documents.append(
                GoldPathDocument(
                    source_index=int(value.get("source_index", -1)),
                    relations=relations,
                    directions=directions,
                    operator_signature=tuple(map(str, value.get("operator_signature", []))),
                    tokens=tokens,
                    alignment=tuple(float(item) for item in value.get("alignment", [])),
                )
            )
        if not documents:
            raise ValueError(f"TRAIN Gold path cache has no documents: {cache_path}")
        return cls(documents, cache_hit=True)

    def retrieve(
        self,
        query: str,
        *,
        operator_signature: tuple[str, ...],
        top_k: int = _train_gold_extrema_selector_TOP_K,
    ) -> list[tuple[float, GoldPathDocument]]:
        query_frequencies = Counter(_train_gold_extrema_selector_tokens(query))
        selected_tokens = sorted(
            query_frequencies,
            key=lambda token: (self.idf.get(token, 0.0), query_frequencies[token], token),
            reverse=True,
        )[:28]
        sparse_scores: dict[int, float] = defaultdict(float)
        for token in selected_tokens:
            query_frequency = query_frequencies[token]
            token_idf = self.idf.get(token, 0.0)
            for index, frequency in self.inverted.get(token, ()):
                length_norm = 0.25 + (
                    0.75 * len(self.documents[index].tokens) / max(1e-9, self.average_length)
                )
                sparse_scores[index] += token_idf * (
                    frequency * 2.2 / (frequency + 1.2 * length_norm)
                ) * min(2, query_frequency)
        preselected = heapq.nlargest(
            max(256, int(top_k) * 4),
            sparse_scores.items(),
            key=lambda item: (item[1], -item[0]),
        )
        target_operators = set(operator_signature) - {"NO_EQUAL", "AND"}
        result: list[tuple[float, int, GoldPathDocument]] = []
        for index, lexical_score in preselected:
            document = self.documents[index]
            score = lexical_score
            source_operators = set(document.operator_signature) - {"NO_EQUAL", "AND"}
            if target_operators and source_operators == target_operators:
                score += 1.25
            elif target_operators and source_operators and source_operators != target_operators:
                score -= 2.0
            alignment = document.alignment
            score += 0.35 * (alignment[1] if len(alignment) > 1 else 0.0)
            score += 0.20 * (alignment[2] if len(alignment) > 2 else 0.0)
            result.append((score, index, document))
        result.sort(key=lambda item: (-item[0], item[2].source_index, item[1]))
        return [(score, document) for score, _, document in result[: max(1, int(top_k))]]


def _query_document(question: str, intents: Sequence[_train_gold_extrema_selector_SemanticIntent]) -> str:
    surfaces = [intent.anchor_surface for intent in intents if intent.anchor_surface]
    masked_question = _train_gold_extrema_selector_mask_entities(_normalize_scalars(question), surfaces)
    goals = [
        _train_gold_extrema_selector_mask_entities(_normalize_scalars(intent.goal), surfaces)
        for intent in intents
        if intent.goal
    ]
    relations = [relation for intent in intents for relation in intent.relation_ids]
    return " ".join(
        [masked_question, masked_question, *(goal for goal in goals for _ in range(2)), _relation_text(relations)]
    )


def _reverse_equivalent(
    left: tuple[str, str], right: tuple[str, str], ontology: Any
) -> bool:
    return (
        left[1] != right[1]
        and right[0] in set(ontology.reverse_for_relation(left[0]))
    )


def _path_match(path: _train_gold_extrema_selector_NaturalPath, document: GoldPathDocument, ontology: Any) -> dict[str, float]:
    candidate_pairs = tuple(zip(path.relations, path.directions))
    gold_pairs = tuple(zip(document.relations, document.directions))
    ontology_exact = len(candidate_pairs) == len(gold_pairs) and all(
        left == right or _reverse_equivalent(left, right, ontology)
        for left, right in zip(candidate_pairs, gold_pairs)
    )
    return {
        "exact": float(candidate_pairs == gold_pairs),
        "ontology_exact": float(ontology_exact),
        "pair_f1": _train_gold_extrema_selector_set_f1(candidate_pairs, gold_pairs),
        "relation_f1": _train_gold_extrema_selector_set_f1(path.relations, document.relations),
        "leaf_f1": _train_gold_extrema_selector_set_f1(
            _train_gold_extrema_selector_relation_leaf_tokens(path.relations),
            _train_gold_extrema_selector_relation_leaf_tokens(document.relations),
        ),
        "length": min(len(path.relations), len(document.relations))
        / max(1, max(len(path.relations), len(document.relations))),
    }


def _train_gold_extrema_selector_candidate_features(
    paths: Sequence[_train_gold_extrema_selector_NaturalPath],
    intents: Sequence[_train_gold_extrema_selector_SemanticIntent],
    retrieved: Sequence[tuple[float, GoldPathDocument]],
    ontology: Any,
) -> dict[str, Any]:
    if not paths or not retrieved:
        return {
            "score": 0.0,
            "exact_vote": 0.0,
            "ontology_vote": 0.0,
            "soft_vote": 0.0,
            "top1_ontology_exact": 0.0,
            "best_match": {},
        }
    anchor_ids = {intent.anchor_id for intent in intents if intent.anchor_id}
    anchored_paths = [path for path in paths if not anchor_ids or path.start_id in anchor_ids]
    compared_paths = anchored_paths or list(paths)
    top_score = retrieved[0][0]
    weights = [
        math.exp(min(0.0, (score - top_score) / 2.5)) / math.log2(rank + 2)
        for rank, (score, _) in enumerate(retrieved)
    ]
    exact_vote = ontology_vote = soft_vote = top1 = 0.0
    best_value = -1.0
    best_match: dict[str, Any] = {}
    for rank, ((retrieval_score, document), weight) in enumerate(zip(retrieved, weights), start=1):
        match, path = max(
            ((_path_match(path, document, ontology), path) for path in compared_paths),
            key=lambda item: (
                item[0]["ontology_exact"],
                item[0]["exact"],
                item[0]["pair_f1"],
                item[0]["relation_f1"],
                item[0]["leaf_f1"],
                item[0]["length"],
            ),
        )
        exact_vote += weight * match["exact"]
        ontology_vote += weight * match["ontology_exact"]
        soft = (
            0.42 * match["pair_f1"]
            + 0.27 * match["relation_f1"]
            + 0.18 * match["leaf_f1"]
            + 0.13 * match["length"]
        )
        soft_vote += weight * soft
        if rank == 1:
            top1 = match["ontology_exact"]
        value = weight * (0.62 * match["ontology_exact"] + 0.38 * soft)
        if value > best_value:
            best_value = value
            best_match = {
                "rank": rank,
                "retrieval_score": retrieval_score,
                "source_index": document.source_index,
                "gold_relations": list(document.relations),
                "gold_directions": list(document.directions),
                "candidate_relations": list(path.relations),
                "candidate_directions": list(path.directions),
                "match": match,
            }
    denominator = sum(weights) or 1.0
    exact_vote /= denominator
    ontology_vote /= denominator
    soft_vote /= denominator
    return {
        "score": 0.54 * ontology_vote + 0.18 * exact_vote + 0.28 * soft_vote,
        "exact_vote": exact_vote,
        "ontology_vote": ontology_vote,
        "soft_vote": soft_vote,
        "top1_ontology_exact": top1,
        "best_match": best_match,
    }


@dataclass(slots=True)
class _train_gold_extrema_selector_Candidate:
    execution: _train_gold_extrema_selector_ExecutedGraph
    intents: list[_train_gold_extrema_selector_SemanticIntent]
    paths: list[_train_gold_extrema_selector_NaturalPath]
    features: dict[str, Any] | None = None


class TrainGoldPathExtremaSelector:
    """Apply the frozen unique-extrema TRAIN-path challenger."""

    def __init__(self, index: TrainGoldPathIndex, ontology: Any, *, top_k: int = _train_gold_extrema_selector_TOP_K) -> None:
        self.index = index
        self.ontology = ontology
        self.top_k = max(1, min(_train_gold_extrema_selector_TOP_K, int(top_k)))

    @classmethod
    def load(
        cls,
        cache_path: str | Path,
        ontology: Any,
        *,
        top_k: int = _train_gold_extrema_selector_TOP_K,
    ) -> "TrainGoldPathExtremaSelector":
        return cls(TrainGoldPathIndex.load(cache_path), ontology, top_k=top_k)

    def challenge(
        self,
        *,
        question: str,
        semantic_graphs: Sequence[dict[str, Any]],
        selected: _train_gold_extrema_selector_ExecutedGraph,
        executions: Sequence[_train_gold_extrema_selector_ExecutedGraph],
        prior_decision: dict[str, Any],
    ) -> tuple[_train_gold_extrema_selector_ExecutedGraph, dict[str, Any]]:
        started = time.perf_counter()
        wanted = _unique_extrema_operator(question)
        evidence: dict[str, Any] = {
            "version": _train_gold_extrema_selector_SELECTOR_VERSION,
            "status": "not_applicable",
            "model_calls": 0,
            "embedding_calls": 0,
            "endpoint_queries": 0,
            "uses_evaluation_gold": False,
            "uses_train_gold_sparql_paths": True,
            "entity_or_relation_whitelist": False,
            "index_document_count": len(self.index.documents),
            "index_cache_hit": self.index.cache_hit,
            "frozen_gate": {
                "minimum_ontology_vote_delta": MIN_ONTOLOGY_VOTE_DELTA,
                "minimum_gold_score_delta": MIN_GOLD_SCORE_DELTA,
                "minimum_rule_score_delta": _train_gold_extrema_selector_MIN_RULE_SCORE_DELTA,
                "retrieval_rule_score_floor": RETRIEVAL_RULE_SCORE_FLOOR,
            },
        }
        if _train_gold_extrema_selector_higher_priority_selected(prior_decision):
            evidence.update(
                {
                    "status": "skipped_higher_priority_postselection_gate",
                    "prior_reason_codes": list(prior_decision.get("reason_codes", [])),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        if not wanted or self.ontology is None or not semantic_graphs:
            evidence.update(
                {
                    "status": "question_or_schema_not_applicable",
                    "extrema_operator": wanted,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        selected_answers = set(map(str, selected.answer_ids))
        selected_signature = _train_gold_extrema_selector_operator_signature(selected)
        if len(selected_answers) != 1 or wanted not in selected_signature:
            evidence.update(
                {
                    "status": "incumbent_singleton_or_operator_guard_rejected",
                    "extrema_operator": wanted,
                    "incumbent_answer_count": len(selected_answers),
                    "incumbent_operator_signature": list(selected_signature),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence

        unique: list[_train_gold_extrema_selector_ExecutedGraph] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for execution in [*executions, selected]:
            answers = tuple(sorted(set(map(str, execution.answer_ids))))
            if not answers:
                continue
            key = (str(execution.graph.graph_id), answers)
            if key in seen:
                continue
            seen.add(key)
            unique.append(execution)
        incumbent_index = next(
            (
                index
                for index, execution in enumerate(unique)
                if execution.graph.graph_id == selected.graph.graph_id
                and set(map(str, execution.answer_ids)) == selected_answers
            ),
            None,
        )
        if incumbent_index is None:
            evidence.update(
                {
                    "status": "incumbent_not_materialized_as_complete_candidate",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        candidates = [
            _train_gold_extrema_selector_Candidate(
                execution=execution,
                intents=_train_gold_extrema_selector_semantic_intents(execution, semantic_graphs),
                paths=_train_gold_extrema_selector_natural_paths(execution, self.ontology),
            )
            for execution in unique
        ]
        incumbent_rule_score = float(selected.graph.score)
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if index != incumbent_index
            and len(set(map(str, candidate.execution.answer_ids))) == 1
            and set(map(str, candidate.execution.answer_ids)) != selected_answers
            and _train_gold_extrema_selector_operator_signature(candidate.execution) == selected_signature
            and float(candidate.execution.graph.score) + _train_gold_extrema_selector_EPSILON
            >= incumbent_rule_score + RETRIEVAL_RULE_SCORE_FLOOR
            and candidate.intents
            and candidate.paths
        ]
        incumbent = candidates[incumbent_index]
        if not eligible or not incumbent.intents or not incumbent.paths:
            evidence.update(
                {
                    "status": "no_singleton_operator_matched_challenger",
                    "candidate_count": len(candidates),
                    "eligible_candidate_count": len(eligible),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        retrieved = self.index.retrieve(
            _query_document(question, incumbent.intents),
            operator_signature=selected_signature,
            top_k=self.top_k,
        )
        if not retrieved:
            evidence.update(
                {
                    "status": "no_train_gold_path_neighbours",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        for index in [incumbent_index, *eligible]:
            candidate = candidates[index]
            candidate.features = _train_gold_extrema_selector_candidate_features(
                candidate.paths,
                candidate.intents,
                retrieved,
                self.ontology,
            )
        chosen_index = max(
            eligible,
            key=lambda index: (
                float((candidates[index].features or {}).get("score", 0.0)),
                float((candidates[index].features or {}).get("ontology_vote", 0.0)),
                float((candidates[index].features or {}).get("soft_vote", 0.0)),
                float(candidates[index].execution.graph.score),
                str(candidates[index].execution.graph.graph_id),
            ),
        )
        challenger = candidates[chosen_index]
        source_features = incumbent.features or {}
        proposed_features = challenger.features or {}
        proposal = {
            "source_graph_id": selected.graph.graph_id,
            "proposed_graph_id": challenger.execution.graph.graph_id,
            "extrema_operator": wanted,
            "source_answer_count": len(selected_answers),
            "proposed_answer_count": len(set(map(str, challenger.execution.answer_ids))),
            "operator_signature": list(selected_signature),
            "source_top1_ontology_exact": float(source_features.get("top1_ontology_exact", 0.0)),
            "proposed_top1_ontology_exact": float(proposed_features.get("top1_ontology_exact", 0.0)),
            "ontology_vote_delta": float(proposed_features.get("ontology_vote", 0.0))
            - float(source_features.get("ontology_vote", 0.0)),
            "gold_score_delta": float(proposed_features.get("score", 0.0))
            - float(source_features.get("score", 0.0)),
            "rule_score_delta": float(challenger.execution.graph.score) - incumbent_rule_score,
            "source_best_match": source_features.get("best_match", {}),
            "proposed_best_match": proposed_features.get("best_match", {}),
            "neighbour_count": len(retrieved),
            "top_source_index": retrieved[0][1].source_index,
        }
        passed = bool(
            proposal["source_top1_ontology_exact"] == 0.0
            and proposal["proposed_top1_ontology_exact"] == 1.0
            and proposal["ontology_vote_delta"] + _train_gold_extrema_selector_EPSILON >= MIN_ONTOLOGY_VOTE_DELTA
            and proposal["gold_score_delta"] + _train_gold_extrema_selector_EPSILON >= MIN_GOLD_SCORE_DELTA
            and proposal["rule_score_delta"] + _train_gold_extrema_selector_EPSILON >= _train_gold_extrema_selector_MIN_RULE_SCORE_DELTA
        )
        evidence.update(
            {
                "status": "selected_train_gold_path_extrema_challenger"
                if passed
                else "confidence_gate_rejected",
                "candidate_count": len(candidates),
                "eligible_candidate_count": len(eligible),
                "proposal": proposal,
                "selected_graph_id": challenger.execution.graph.graph_id if passed else selected.graph.graph_id,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return (challenger.execution if passed else selected), evidence






__all__ = ['EntityKindFixedBeamPlan', 'EntityKindFixedBeamPlanner', 'GoldPathDocument', 'GraphFeatures', 'PathAlignmentSelector', 'PreparedEntityTemplateQuery', 'SourceVerifiedTemplateConsensusSelector', 'TemplateDocument', 'ThreadSafeEmbeddingCache', 'TrainGoldPathExtremaSelector', 'TrainGoldPathIndex', 'UnderanswerExpansionSelector', 'VerifiedTemplateIndex', '_path_alignment_selector_NaturalPath', '_path_alignment_selector_SemanticIntent', 'answer_kind', 'build_verified_template_cache', 'execute_prepared_query', 'is_entity_kind_variant', 'is_scalar_schema_answer_slot', 'linked_entities_for_question', 'postselect_entity_kind_execution', 'question_expects_entity', 'select_candidates', 'select_empty_entity_kind_execution', 'semantic_relation_unions', 'terminal_types']
