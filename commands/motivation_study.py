#!/usr/bin/env python3
"""Reproduce the statistics of the paper's Motivation study (Section 3.2).

All numbers are computed from the merged root-to-leaf paths. In a few AutoVerus
trees the default branch is dropped in its last round (no candidate passes the
safety check after repeated rejected rounds), and the dropped round is not
recorded. That round accepts no patch, so the default trajectory ends with the
program it started the round with; we complete it accordingly. The merged paths
may already carry that default trajectory (rows flagged default_pruned_early_stop,
added by commands/complete_pruned_default_paths.py); the dropped round is then
still counted as a choice of the default rule, as for unflagged data.

Run from the project root; reading the merged file takes a few minutes.
"""

import argparse
from pathlib import Path
import json
from collections import Counter, defaultdict


def is_verified(label) -> bool:
    return label in (True, 1, "1", "true", "True")


def path_key(path):
    """Order paths by Equation 5: success, then fewer rounds, then higher score."""
    last = path["path_states"][-1]
    if is_verified(path["label"]):
        return (1, -last["round_id"], 0, 0)
    x, y = (last.get("score_tuple") or [0, 0])[:2]
    return (0, 0, x, -y)


def keep_best(choices, choice, key):
    if choice not in choices or key > choices[choice]:
        choices[choice] = key


def load_trees(merged_path_file):
    trees = defaultdict(lambda: {"edges": {}, "states": {}, "default": None, "alternatives": []})
    best = defaultdict(lambda: defaultdict(dict))  # tree -> state -> choice -> best key
    with open(merged_path_file) as f:
        for line in f:
            path = json.loads(line)
            tree = trees[path["tree_id"]]
            tree["root"] = path["root_state_id"]
            if path.get("default_pruned_early_stop"):
                tree["pruned_default"] = True
            key = path_key(path)
            states = path["path_states"]
            for state, child in zip(states, states[1:]):
                tree["edges"][state["state_id"], child["state_id"]] = (
                    state["chosen_failure_type"],
                    state["is_default_chose"],
                )
            for state in states:
                sid = state["state_id"]
                tree["states"][sid] = (frozenset(state["failure_types"]), state["score_tuple"])
                if state.get("chosen_failure_type"):
                    keep_best(best[path["tree_id"]][sid], state["chosen_failure_type"], key)
            if path.get("default"):
                tree["default"] = key
            else:
                tree["alternatives"].append(key)
    return trees, best


def default_trajectory(tree, best_choices):
    """Return the (state, choice) steps of the default trajectory, completing a dropped last round."""
    default_edges = defaultdict(list)
    for (src, dst), (choice, is_default) in tree["edges"].items():
        if is_default is True:
            default_edges[src].append((dst, choice))
    steps, node = [], tree["root"]
    while len(default_edges.get(node, [])) == 1:
        dst, choice = default_edges[node][0]
        steps.append((node, choice))
        node = dst
    if tree["default"] is None or tree.get("pruned_default"):
        # The dropped round repairs the same program as the round before, so the
        # default rule targets the same type, and the trajectory ends at `node`.
        x, y = tree["states"][node][1][:2]
        tree["default"] = (0, 0, x, -y)
        steps.append((node, steps[-1][1]))
        for sid, choice in steps:
            keep_best(best_choices[sid], choice, tree["default"])
    return steps


def report_trees(trees, steps):
    choice_trees = [
        tid for tid, tree in trees.items()
        if any(len(tree["states"][sid][0]) >= 2 for sid, _ in steps[tid])
    ]
    n = len(choice_trees)
    print("Trees")
    print(f"  trees: {len(trees)}")
    print(f"  default trajectory reaches a state with >=2 types: {n} ({n / len(trees):.1%})")
    outcome, by_pipeline = Counter(), defaultdict(Counter)
    for tid in choice_trees:
        tree = trees[tid]
        pipeline = tid.split("-", 1)[0]
        by_pipeline[pipeline]["trees"] += 1
        default, alternative = tree["default"], max(tree["alternatives"])
        if alternative <= default:
            outcome["default best"] += 1
        elif default[0] == 0 and alternative[0] == 1:
            outcome["alternative verifies"] += 1
        elif alternative[0] == 1:
            outcome["fewer rounds"] += 1
        else:
            outcome["more progress"] += 1
        by_pipeline[pipeline]["better"] += alternative > default
    better = n - outcome["default best"]
    gains = outcome["alternative verifies"] + outcome["fewer rounds"]
    print(f"  alternative strictly better: {better} ({better / n:.1%})")
    print(f"  default best: {outcome['default best']} ({outcome['default best'] / n:.1%})")
    print(f"  verification or fewer rounds: {gains} ({gains / n:.1%})")
    for name in ("alternative verifies", "fewer rounds", "more progress"):
        print(f"    {name}: {outcome[name]}")
    for pipeline, c in sorted(by_pipeline.items()):
        print(f"  {pipeline}: {c['better']} / {c['trees']} ({c['better'] / c['trees']:.1%})")


def report_decisive_states(trees, best):
    by_set = defaultdict(Counter)
    for tid, states in best.items():
        for sid, choices in states.items():
            if len(choices) < 2:
                continue
            ranked = sorted(choices.items(), key=lambda kv: kv[1], reverse=True)
            if ranked[0][1] == ranked[1][1]:
                continue
            by_set[trees[tid]["states"][sid][0]][ranked[0][0]] += 1
    decisive = sum(sum(c.values()) for c in by_set.values())
    mixed = {s: c for s, c in by_set.items() if len(c) >= 2}
    covered = sum(sum(c.values()) for c in mixed.values())
    static_best = sum(max(c.values()) for c in by_set.values())
    print("Decisive states")
    print(f"  decisive branching states: {decisive}")
    print(f"  candidate sets: {len(by_set)}, with different best choices: {len(mixed)}")
    print(f"  decisive states in those sets: {covered} ({covered / decisive:.1%})")
    print(f"  best fixed set->choice mapping: {static_best} ({static_best / decisive:.1%})")
    example = frozenset({"ArithmeticFlow", "AssertFail"})
    print(f"  {{ArithmeticFlow, AssertFail}}: {dict(by_set.get(example, {}))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-paths", nargs="+", default=sorted((Path(__file__).resolve().parents[1] / "data/paths").glob("*.jsonl")))
    args = parser.parse_args()
    trees, best = {}, {}
    for path in args.merged_paths:
        part_trees, part_best = load_trees(path)
        trees.update(part_trees)
        best.update(part_best)
    steps = {tid: default_trajectory(tree, best[tid]) for tid, tree in trees.items()}
    report_trees(trees, steps)
    report_decisive_states(trees, best)


if __name__ == "__main__":
    main()
