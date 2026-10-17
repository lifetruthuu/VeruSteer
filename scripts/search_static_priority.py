#!/usr/bin/env python3
"""Fit a static error-type order by replaying training repair trees.

Python 3.10+ standard library only. No model, embeddings, GPU, or LLM calls.
See docs/experiments/searched_static_priority.md for the experiment protocol.

Fit and replay are separate commands so test outcomes never enter the search:
  python scripts/search_static_priority.py fit --method autoverus \
    --train-path data/paths/train_verusage.jsonl \
    --save-policy /tmp/static_autoverus.json
  python scripts/search_static_priority.py replay \
    --policy /tmp/static_autoverus.json \
    --test-path data/paths/test_autoverus.jsonl \
    --save-log /tmp/static_autoverus.log
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path


SCHEMA_VERSION = 1
DEFAULT_PRIORITY = (
    "MismatchedType", "PreCondFailVecLen", "ArithmeticFlow", "InvFailFront",
)


@dataclass(frozen=True)
class Outcome:
    verified: bool
    rounds: int
    score: tuple[float, float]

    @property
    def objective(self) -> tuple[int, int]:
        return int(self.verified), -self.rounds if self.verified else 0


@dataclass
class Tree:
    tree_id: str
    task: str
    root: str
    # Only structural metadata is retained, never source code or diagnostics.
    reported: dict[str, tuple[str, ...]] = field(default_factory=dict)
    edges: dict[str, dict[str, str]] = field(default_factory=dict)
    leaves: dict[str, tuple[str, Outcome]] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)
    default_path: str | None = None
    paths: int = 0

    def candidates(self, state: str) -> tuple[str, ...]:
        children = self.edges.get(state, {})
        return tuple(sorted(t for t in self.reported[state] if t in children))

    def replay(self, order: tuple[str, ...], trace: bool = False) -> dict:
        rank = {t: i for i, t in enumerate(order)}
        state = self.root
        seen: set[str] = set()
        selections = []
        steps = 0
        while True:
            if state in seen:
                raise ValueError(f"{self.tree_id}: cycle at {state}")
            seen.add(state)
            candidates = self.candidates(state)
            if not candidates:
                if state not in self.leaves:
                    raise ValueError(f"{self.tree_id}: no recorded leaf at {state}")
                break
            # A total order, including a fixed alphabetical fallback for types
            # absent from training. Never break ties using the current default.
            selected = min(candidates, key=lambda t: (rank.get(t, len(rank)), t))
            child = self.edges[state][selected]
            if trace:
                selections.append({
                    "step": steps, "current_state_id": state,
                    "candidate_names": list(candidates), "selected_name": selected,
                    "selected_reason": "searched_static_priority",
                    "default_name": self.defaults.get(state),
                    "next_state_id": child, "dead_end_reason": None,
                })
            steps += 1
            state = child
        path_id, outcome = self.leaves[state]
        return {
            "tree_id": self.tree_id, "selector_mode": "searched_static",
            "num_paths": self.paths, "reached_path_id": path_id,
            "reached_leaf_state_id": state, "dead_end": False,
            "dead_end_reason": None, "steps": steps,
            "selection_trace": selections,
            "outcome": {"verified": outcome.verified, "rounds": outcome.rounds,
                        "score_tuple": list(outcome.score)},
        }


def binary(value: object) -> bool:
    if value in (True, 1, "true", "True", "1"):
        return True
    if value in (False, 0, "false", "False", "0"):
        return False
    raise ValueError(f"Invalid binary label: {value!r}")


def load_trees(path: Path, method: str, exclude_incomplete: bool = False) -> tuple[list[Tree], dict]:
    """Stream path JSONL into compact trees; record its size and coverage."""
    trees: dict[str, Tree] = {}
    selected_rows = 0
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            tid = row["tree_id"]
            if not tid.startswith(method + "-"):
                continue
            if row.get("default_pruned_early_stop"):
                # Default path completed by commands/complete_pruned_default_paths.py. Its end state is
                # interior to the tree (the pruned type has no recorded successor), so no replay reaches it.
                continue
            ids, states = row["path_state_ids"], row["path_states"]
            if (not ids or len(ids) != len(states) or ids[0] != row["root_state_id"]
                    or len(set(ids)) != len(ids)):
                raise ValueError(f"{path}:{line_number}: invalid path structure")
            task = row["tree_key"]["task_name"]
            tree = trees.setdefault(tid, Tree(tid, task, ids[0]))
            if tree.root != ids[0] or tree.task != task:
                raise ValueError(f"{tid}: inconsistent root/task")
            for index, (sid, state) in enumerate(zip(ids, states)):
                reported = state.get("failure_types") or [
                    r["error_type"] for r in state.get("repaired_failure_list", [])
                    if r.get("error_type")
                ]
                tree.reported.setdefault(sid, tuple(sorted(set(reported))))
                if index + 1 == len(ids):
                    continue
                action = state["chosen_failure_type"]
                if not isinstance(action, str) or not action:
                    raise ValueError(f"{tid}: missing action at {sid}")
                edges = tree.edges.setdefault(sid, {})
                if action in edges and edges[action] != ids[index + 1]:
                    raise ValueError(f"{tid}: multiple successors for {(sid, action)}")
                edges[action] = ids[index + 1]
                if state.get("is_default_chose") is True:
                    previous = tree.defaults.setdefault(sid, action)
                    if previous != action:
                        raise ValueError(f"{tid}: conflicting default action at {sid}")
            last = states[-1]
            outcome = Outcome(binary(row["label"]), int(last["round_id"]),
                              tuple(float(x) for x in last["score_tuple"]))
            if len(outcome.score) != 2:
                raise ValueError(f"{tid}: invalid terminal score")
            entry = (row["path_id"], outcome)
            if ids[-1] in tree.leaves and tree.leaves[ids[-1]] != entry:
                raise ValueError(f"{tid}: conflicting terminal paths at {ids[-1]}")
            tree.leaves[ids[-1]] = entry
            if binary(row["default"]):
                if tree.default_path not in (None, row["path_id"]):
                    raise ValueError(f"{tid}: multiple default paths")
                tree.default_path = row["path_id"]
            tree.paths += 1
            selected_rows += 1
    if not trees:
        raise ValueError(f"No {method} trees in {path}")
    ordered = []
    excluded = []
    for tid in sorted(trees):
        tree = trees[tid]
        invalid_states = []
        for sid in tree.reported:
            candidates = tree.candidates(sid)
            if sid in tree.leaves and candidates:
                raise ValueError(f"{tree.tree_id}: terminal state has outgoing actions: {sid}")
            if sid not in tree.leaves and not candidates:
                invalid_states.append(sid)
        if invalid_states:
            if not exclude_incomplete:
                raise ValueError(f"{tree.tree_id}: no available action/leaf at {invalid_states}")
            excluded.append({"tree_id": tid, "states": sorted(invalid_states),
                             "reason": "no legal recorded continuation or terminal path"})
        else:
            ordered.append(tree)
    if not ordered:
        raise ValueError("No replayable trees")
    return ordered, {
        "file_name": path.name, "bytes": path.stat().st_size,
        "loaded_trees": len(trees), "trees": len(ordered), "paths": selected_rows,
        "tasks": sorted({t.task for t in trees.values()}),
        "excluded_trees": excluded,
        "trees_with_default": sum(t.default_path is not None for t in ordered),
    }


class SearchObjective:
    """Exact root-to-leaf objective, cached by induced training decisions.

    Trees whose leaves all have the same objective contribute a constant.
    Removing them from repeated replay changes neither score nor search.
    """

    def __init__(self, trees: list[Tree], vocabulary: tuple[str, ...]):
        self.vocabulary = vocabulary
        type_id = {t: i for i, t in enumerate(vocabulary)}
        self.constant = [0, 0]
        self.compiled = []
        self.candidate_sets: list[tuple[int, ...]] = []
        set_ids: dict[tuple[str, ...], int] = {}
        for tree in trees:
            values = {outcome.objective for _, outcome in tree.leaves.values()}
            if len(values) == 1:
                value = next(iter(values))
                self.constant[0] += value[0]
                self.constant[1] += value[1]
                continue
            ids = {sid: i for i, sid in enumerate(sorted(tree.reported))}
            nodes = []
            for sid in sorted(ids):
                candidates = tree.candidates(sid)
                if not candidates:
                    nodes.append((-1, tree.leaves[sid][1].objective))
                    continue
                if candidates not in set_ids:
                    set_ids[candidates] = len(self.candidate_sets)
                    self.candidate_sets.append(tuple(type_id[t] for t in candidates))
                children = {type_id[t]: ids[tree.edges[sid][t]] for t in candidates}
                nodes.append((set_ids[candidates], children))
            self.compiled.append((ids[tree.root], nodes))
        self.cache: dict[tuple[int, ...], tuple[int, int]] = {}
        self.requests = 0

    def __call__(self, order: tuple[str, ...]) -> tuple[int, int]:
        self.requests += 1
        ranks = {t: i for i, t in enumerate(order)}
        rank_by_id = [ranks[t] for t in self.vocabulary]
        choices = tuple(min(s, key=rank_by_id.__getitem__) for s in self.candidate_sets)
        if choices not in self.cache:
            verified, negative_rounds = self.constant
            for state, nodes in self.compiled:
                while nodes[state][0] >= 0:
                    group, children = nodes[state]
                    state = children[choices[group]]
                value = nodes[state][1]
                verified += value[0]
                negative_rounds += value[1]
            self.cache[choices] = (verified, negative_rounds)
        return self.cache[choices]


def describe_objective(value: tuple[int, int]) -> dict:
    return {"verified_trees": value[0], "successful_rounds_sum": -value[1]}


def search(trees: list[Tree], method: str, seed: int = 0,
           random_starts: int = 20, max_passes: int = 100) -> dict:
    vocabulary = tuple(sorted({a for tree in trees for sid in tree.reported
                               for a in tree.candidates(sid)}))
    priority = DEFAULT_PRIORITY + (("InvFailEnd",) if method == "verusage" else ())
    initial = tuple(t for t in priority if t in vocabulary)
    initial += tuple(t for t in vocabulary if t not in initial)
    rng = random.Random(seed)
    starts = [initial]
    for _ in range(random_starts):
        order = list(vocabulary)
        rng.shuffle(order)
        starts.append(tuple(order))
    objective = SearchObjective(trees, vocabulary)
    best_order, best_score = initial, objective(initial)
    winner = 0
    records = []
    swaps = list(combinations(range(len(vocabulary)), 2))
    for start_id, order in enumerate(starts):
        score = objective(order)
        record = {"start": start_id, "kind": "default_priority" if start_id == 0 else "random",
                  "initial_order": list(order), "initial_objective": describe_objective(score),
                  "improvements": []}
        converged = False
        for pass_id in range(max_passes):
            next_order, next_score, chosen_swap = order, score, None
            for left, right in swaps:
                candidate = list(order)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate = tuple(candidate)
                value = objective(candidate)
                # Equal-quality moves are not accepted. Equal best neighbors
                # keep the first pair of indices in lexicographic order.
                if value > next_score:
                    next_order, next_score = candidate, value
                    chosen_swap = [left, right]
            if chosen_swap is None:
                converged = True
                break
            order, score = next_order, next_score
            record["improvements"].append({"pass": pass_id + 1, "swap": chosen_swap,
                                           "objective": describe_objective(score)})
        record.update(final_order=list(order), final_objective=describe_objective(score),
                      converged=converged)
        records.append(record)
        if score > best_score:
            best_order, best_score, winner = order, score, start_id
        print(f"[{method}] start {start_id:02d}: {score[0]} verified, {-score[1]} successful rounds, "
              f"{len(record['improvements'])} improvements", file=sys.stderr)
    # Check the optimized implementation against an independent plain replay.
    actual = [tree.replay(best_order)["outcome"] for tree in trees]
    check = (sum(o["verified"] for o in actual),
             -sum(o["rounds"] for o in actual if o["verified"]))
    if check != best_score:
        raise AssertionError(f"Search/replay objective mismatch: {best_score} != {check}")
    return {
        "schema_version": SCHEMA_VERSION, "selector": "searched_static", "method": method,
        "objective": ["maximize verified training trees", "minimize successful-rounds sum"],
        "search": {"seed": seed, "random_starts": random_starts, "max_passes": max_passes,
                   "neighborhood": "all pairwise swaps; best strict improvement",
                   "neighbor_tie_break": "first index pair", "restart_tie_break": "first start",
                   "objective_requests": objective.requests,
                   "distinct_training_decisions": len(objective.cache),
                   "variable_objective_trees": len(objective.compiled)},
        "vocabulary": list(vocabulary), "priority_order": list(best_order),
        "unseen_type_rule": "after learned types, in alphabetical order",
        "winning_start": winner, "train_objective": describe_objective(best_score),
        "starts": records,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="search using training paths only")
    fit.add_argument("--train-path", type=Path, required=True)
    fit.add_argument("--method", choices=("autoverus", "verusage"), required=True)
    fit.add_argument("--save-policy", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--random-starts", type=int, default=20)
    fit.add_argument("--max-passes", type=int, default=100)
    replay = sub.add_parser("replay", help="replay a frozen order on disjoint test tasks")
    replay.add_argument("--policy", type=Path, required=True)
    replay.add_argument("--test-path", type=Path, required=True)
    replay.add_argument("--save-log", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        if args.random_starts < 0 or args.max_passes < 1:
            parser.error("random-starts must be nonnegative and max-passes positive")
        trees, metadata = load_trees(args.train_path, args.method, exclude_incomplete=True)
        for entry in metadata["excluded_trees"]:
            print(f"[excluded training tree] {entry['tree_id']}: {entry['states']}", file=sys.stderr)
        policy = search(trees, args.method, args.seed, args.random_starts, args.max_passes)
        policy["training_data"] = metadata
        write_json(args.save_policy, policy)
        print(json.dumps({"method": args.method, **policy["train_objective"],
                          "trees": len(trees), "policy": str(args.save_policy)}))
    else:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        if policy.get("schema_version") != SCHEMA_VERSION or policy.get("selector") != "searched_static":
            parser.error("unsupported policy artifact")
        order = tuple(policy["priority_order"])
        if len(set(order)) != len(order) or set(order) != set(policy["vocabulary"]):
            parser.error("priority order must be a permutation of the training vocabulary")
        trees, metadata = load_trees(args.test_path, policy["method"])
        overlap = set(metadata["tasks"]) & set(policy["training_data"]["tasks"])
        if overlap:
            parser.error(f"training/test task overlap: {len(overlap)} tasks")
        records = [t.replay(order, trace=True) for t in trees]
        args.save_log.parent.mkdir(parents=True, exist_ok=True)
        with args.save_log.open("w", encoding="utf-8") as output:
            for row in records:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(args.save_log.with_suffix(".metadata.json"), {
            "schema_version": SCHEMA_VERSION, "method": policy["method"],
            "policy_file": args.policy.name,
            "test_data": metadata, "train_test_task_overlap": 0,
            "replayed_trees": len(records),
            "verified_trees": sum(r["outcome"]["verified"] for r in records),
            "unseen_types": sorted({t for tree in trees for sid in tree.reported
                                    for t in tree.candidates(sid)} - set(order)),
        })
        print(json.dumps({"method": policy["method"], "trees": len(trees),
                          "verified": sum(r["outcome"]["verified"] for r in records),
                          "log": str(args.save_log)}))


if __name__ == "__main__":
    main()
