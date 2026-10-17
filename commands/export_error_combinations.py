#!/usr/bin/env python3
"""Export action-conditioned success rates for multi-error combinations.

Frequency is counted over unique dataset nodes `(tree_id, state_id)`.
Default decisions are also counted once per unique node.  When the same
default edge appears in multiple complete paths, the script prefers the path
that follows default choices from that node onward; otherwise it falls back to
the first observed default-edge path.

VeruSteer decisions are counted from SASRec rollout logs, deduplicated by
`(tree_id, state_id)`, and labeled by whether the reached full path verifies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MERGED_PATH = sorted((ROOT / "data/paths").glob("*.jsonl"))
LOG_DIR = ROOT / "results/full_corpus"
LOG_PATHS = [
    LOG_DIR / "autoverus.log",
    LOG_DIR / "verusage.log",
]
OUT_PATH = ROOT / "outputs/analysis/error_combinations.csv"
TEST_PATHS = [
    ROOT / "data/paths/test_autoverus.jsonl",
    ROOT / "data/paths/test_verusage.jsonl",
]


def load_test_tree_ids() -> set[str]:
    """Tree ids of the test split (the test path files list every tree of a split)."""
    ids: set[str] = set()
    for path in TEST_PATHS:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                start = line.index('"tree_id": "') + len('"tree_id": "')
                ids.add(line[start:line.index('"', start)])
    return ids


@dataclass
class Decision:
    combo: str
    action: str
    success: bool


def read_jsonl(path: Path):
    if isinstance(path, list):
        for item in path:
            yield from read_jsonl(item)
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def combo_key(names: list[str] | tuple[str, ...] | set[str]) -> str:
    return " + ".join(sorted({name for name in names if isinstance(name, str) and name}))


def is_multi_error(names: Any) -> bool:
    return isinstance(names, list) and len({name for name in names if isinstance(name, str) and name}) >= 2


def suffix_is_default(states: list[dict[str, Any]], start_idx: int) -> bool:
    """Whether this path follows default choices from states[start_idx] onward."""
    for state in states[start_idx:-1]:
        chosen = state.get("chosen_failure_type")
        if isinstance(chosen, str) and chosen and state.get("is_default_chose") is not True:
            return False
    return True


def pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "N/A"
    return f"{100.0 * numer / denom:.2f}%"


def pp(delta: float | None) -> str:
    if delta is None or math.isnan(delta):
        return "N/A"
    return f"{delta:+.2f} pp"


def format_choice_distribution(counter: Counter[str], denom: int) -> str:
    if denom <= 0:
        return "N/A"
    return ", ".join(
        f"{action} {100.0 * count / denom:.2f}% ({count}/{denom})"
        for action, count in counter.most_common()
    )


def format_action_success(success: Counter[str], total: Counter[str]) -> str:
    if not total:
        return "N/A"
    return ", ".join(
        f"{action} {pct(success[action], count)} ({success[action]}/{count})"
        for action, count in total.most_common()
    )


def add_decision(
    agg: dict[str, dict[str, Any]],
    decision: Decision,
    side: str,
) -> None:
    item = agg[decision.combo]
    item[f"{side}_events"] += 1
    item[f"{side}_choice_counts"][decision.action] += 1
    item[f"{side}_success_total"] += int(decision.success)
    item[f"{side}_action_total"][decision.action] += 1
    item[f"{side}_action_success"][decision.action] += int(decision.success)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", choices=("test", "all"), default="all",
                        help="trees to count; the paper reports the test trees only (default)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    wanted = load_test_tree_ids() if args.splits == "test" else None

    state_combo: dict[tuple[str, str], str] = {}
    path_label: dict[tuple[str, str], bool] = {}
    default_fallback: dict[tuple[str, str], Decision] = {}
    default_suffix: dict[tuple[str, str], Decision] = {}

    print(f"[1/4] Reading merged paths: {MERGED_PATH}")
    for obj in read_jsonl(MERGED_PATH):
        tree_id = str(obj.get("tree_id") or "")
        path_id = str(obj.get("path_id") or "")
        if not tree_id or not path_id:
            continue
        if wanted is not None and tree_id not in wanted:
            continue
        success = normalize_bool(obj.get("label"))
        path_label[(tree_id, path_id)] = success

        states = obj.get("path_states")
        if not isinstance(states, list):
            continue
        for idx, state in enumerate(states[:-1]):
            if not isinstance(state, dict):
                continue
            state_id = str(state.get("state_id") or "")
            if not state_id or not is_multi_error(state.get("failure_types")):
                continue

            combo = combo_key(state.get("failure_types") or [])
            state_key = (tree_id, state_id)
            state_combo.setdefault(state_key, combo)

            action = state.get("chosen_failure_type")
            if state.get("is_default_chose") is True and isinstance(action, str) and action:
                decision = Decision(combo=combo, action=action, success=success)
                default_fallback.setdefault(state_key, decision)
                if suffix_is_default(states, idx):
                    default_suffix.setdefault(state_key, decision)

    combo_frequency = Counter(state_combo.values())
    agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "default_events": 0,
            "default_choice_counts": Counter(),
            "default_success_total": 0,
            "default_action_total": Counter(),
            "default_action_success": Counter(),
            "verusteer_events": 0,
            "verusteer_choice_counts": Counter(),
            "verusteer_success_total": 0,
            "verusteer_action_total": Counter(),
            "verusteer_action_success": Counter(),
        }
    )

    print("[2/4] Aggregating default decisions")
    default_missing = 0
    for state_key, combo in state_combo.items():
        decision = default_suffix.get(state_key) or default_fallback.get(state_key)
        if decision is None:
            default_missing += 1
            continue
        add_decision(agg, decision, "default")

    print("[3/4] Reading VeruSteer rollout logs")
    seen_verusteer_nodes: set[tuple[str, str]] = set()
    verusteer_missing_label = 0
    verusteer_missing_combo = 0
    for log_path in LOG_PATHS:
        for rec in read_jsonl(log_path):
            tree_id = str(rec.get("tree_id") or "")
            reached_path_id = rec.get("reached_path_id")
            if not tree_id or not isinstance(reached_path_id, str):
                continue
            if wanted is not None and tree_id not in wanted:
                continue
            reached_success = path_label.get((tree_id, reached_path_id))
            if reached_success is None:
                verusteer_missing_label += 1
                continue

            trace = rec.get("selection_trace")
            if not isinstance(trace, list):
                continue
            for step in trace:
                if not isinstance(step, dict):
                    continue
                state_id = str(step.get("current_state_id") or "")
                action = step.get("selected_name")
                if not state_id or not isinstance(action, str) or not action:
                    continue
                node_key = (tree_id, state_id)
                if node_key in seen_verusteer_nodes:
                    continue

                combo = state_combo.get(node_key)
                if combo is None:
                    candidates = step.get("candidate_names")
                    if is_multi_error(candidates):
                        combo = combo_key(candidates)
                    else:
                        verusteer_missing_combo += 1
                        continue

                seen_verusteer_nodes.add(node_key)
                add_decision(
                    agg,
                    Decision(combo=combo, action=action, success=reached_success),
                    "verusteer",
                )

    print(f"[4/4] Writing CSV: {args.out}")
    rows: list[dict[str, Any]] = []
    for combo, frequency in combo_frequency.most_common():
        item = agg[combo]
        default_events = item["default_events"]
        verusteer_events = item["verusteer_events"]
        default_rate = item["default_success_total"] / default_events * 100.0 if default_events else None
        verusteer_rate = item["verusteer_success_total"] / verusteer_events * 100.0 if verusteer_events else None
        delta = None
        if default_rate is not None and verusteer_rate is not None:
            delta = verusteer_rate - default_rate

        rows.append(
            {
                "Combination": combo,
                "Frequency_unique_nodes": frequency,
                "Default_nodes": default_events,
                "Default_choice_distribution": format_choice_distribution(
                    item["default_choice_counts"], default_events
                ),
                "Default_action_success_rates": format_action_success(
                    item["default_action_success"], item["default_action_total"]
                ),
                "Default_overall_success": (
                    f"{default_rate:.2f}% ({item['default_success_total']}/{default_events})"
                    if default_events
                    else "N/A"
                ),
                "VeruSteer_nodes": verusteer_events,
                "VeruSteer_choice_distribution": format_choice_distribution(
                    item["verusteer_choice_counts"], verusteer_events
                ),
                "VeruSteer_action_success_rates": format_action_success(
                    item["verusteer_action_success"], item["verusteer_action_total"]
                ),
                "VeruSteer_overall_success": (
                    f"{verusteer_rate:.2f}% ({item['verusteer_success_total']}/{verusteer_events})"
                    if verusteer_events
                    else "N/A"
                ),
                "Success_delta": pp(delta),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"unique multi-error nodes: {sum(combo_frequency.values())}")
    print(f"combinations: {len(combo_frequency)}")
    print(f"default nodes without default decision: {default_missing}")
    print(f"VeruSteer visited multi-error nodes: {len(seen_verusteer_nodes)}")
    print(f"VeruSteer records missing reached label: {verusteer_missing_label}")
    print(f"VeruSteer steps missing combo: {verusteer_missing_combo}")
    print(args.out)


if __name__ == "__main__":
    main()
