#!/usr/bin/env python3
"""
按 v3 六类空间划分评估 SASRec 表现。

输出格式：
SASRec表现（默认空间中） class1 ...
SASRec表现（提升空间中） class1 ...

其中每个格子为 a / b：
- class1:
  - 默认空间中，b 表示同时包含 default 与 non-default 路径、且 default 能通过验证的树数，a 表示 SASRec 能选到一条通过验证路径的树数
  - 提升空间中，b 表示 non-default 能通过且 default 不能通过的树数，a 表示 SASRec 能选到一条通过验证路径的树数
- class2/class3:
  - b: 该设置下落入该类空间的树数
  - a: SASRec 在该类空间中“做对”的数量

三种更优情况：
- class1: 通过 vs 没通过
- class2: 都通过但更快
- class3: 都没通过但是分数更高

python commands/eval_sasrec_log_v3.py \
    --model-log results/rollout_logs/greedy_score.log \
    --merged-path data/paths/test_verusage.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CATEGORY_ORDER: Tuple[str, ...] = (
    "autoverus-deepseek-chat-zero-shot",
    "autoverus-deepseek-chat-few-shot",
    "autoverus-qwen-coder-zero-shot",
    "autoverus-qwen-coder-few-shot",
    "verusage-deepseek-chat-zero-shot",
    "verusage-deepseek-chat-few-shot",
    "verusage-qwen-coder-zero-shot",
    "verusage-qwen-coder-few-shot",
)

SPACE_CLASS_ORDER: Tuple[Tuple[str, str], ...] = (
    ("默认空间", "class1"),
    ("默认空间", "class2"),
    ("默认空间", "class3"),
    ("提升空间", "class1"),
    ("提升空间", "class2"),
    ("提升空间", "class3"),
)


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^L\d+:(.*)$", line)
    if m:
        line = m.group(1)
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def classify_tree_id(tree_id: str) -> str:
    tree_id_lower = tree_id.lower().strip()
    parts = tree_id_lower.split("-")
    if len(parts) < 6:
        return "unknown"
    method = parts[0]
    if method not in {"autoverus", "verusage"}:
        return "unknown"
    result_idx = None
    for i, part in enumerate(parts):
        if part.startswith("result_"):
            result_idx = i
            break
    if result_idx is None:
        for i, part in enumerate(parts):
            if part.isdigit():
                result_idx = i
                break
    if result_idx is None or result_idx < 2:
        return "unknown"
    shot = "-".join(parts[result_idx - 2 : result_idx])
    model = "-".join(parts[result_idx - 4 : result_idx - 2])
    category = f"{method}-{model}-{shot}"
    return category if category in CATEGORY_ORDER else "unknown"


def normalize_binary(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return 1
        if v in {"0", "false", "f", "no", "n"}:
            return 0
    return None


def extract_final_score_tuple(path_obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    path_states = path_obj.get("path_states")
    if not isinstance(path_states, list) or not path_states:
        return None
    last = path_states[-1]
    if not isinstance(last, dict):
        return None
    score = last.get("score_tuple")
    if (
        isinstance(score, list)
        and len(score) >= 2
        and isinstance(score[0], (int, float))
        and isinstance(score[1], (int, float))
    ):
        return float(score[0]), float(score[1])
    return None


def extract_final_round_id(path_obj: Dict[str, Any]) -> Optional[int]:
    path_states = path_obj.get("path_states")
    if not isinstance(path_states, list) or not path_states:
        return None
    last = path_states[-1]
    if not isinstance(last, dict):
        return None
    round_id = last.get("round_id")
    if isinstance(round_id, int):
        return round_id
    if isinstance(round_id, float) and round_id.is_integer():
        return int(round_id)
    return None


def is_passed(path_obj: Dict[str, Any]) -> bool:
    return normalize_binary(path_obj.get("label")) == 1


def second_score_comparable(default_second: float, nondefault_second: float) -> bool:
    return default_second != -1 and nondefault_second != -1


def is_nondefault_better_by_eval(
    nondefault_score: Tuple[float, float], default_score: Tuple[float, float]
) -> bool:
    nd_first, nd_second = nondefault_score
    d_first, d_second = default_score
    if nd_first > d_first:
        return True
    if nd_first < d_first:
        return False
    return second_score_comparable(d_second, nd_second) and nd_second < d_second


def load_model_rows(model_log: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in model_log.open("r", encoding="utf-8"):
        obj = parse_json_line(line)
        if obj is None:
            continue
        tree_id = obj.get("tree_id")
        reached_path_id = obj.get("reached_path_id")
        if isinstance(tree_id, str) and isinstance(reached_path_id, str):
            rows.append({"tree_id": tree_id, "reached_path_id": reached_path_id})
    return rows


def build_merged_index(
    merged_path: Path, wanted_tree_ids: set[str]
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, str],
    Dict[str, List[Dict[str, Any]]],
]:
    row_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    default_path_id_by_tree: Dict[str, str] = {}
    rows_by_tree: Dict[str, List[Dict[str, Any]]] = {}
    for line in merged_path.open("r", encoding="utf-8"):
        obj = parse_json_line(line)
        if obj is None:
            continue
        tid = obj.get("tree_id")
        pid = obj.get("path_id")
        if not isinstance(tid, str) or tid not in wanted_tree_ids or not isinstance(pid, str):
            continue
        row_by_key[(tid, pid)] = obj
        rows_by_tree.setdefault(tid, []).append(obj)
        if tid not in default_path_id_by_tree:
            default_pid = obj.get("default_path_id")
            if isinstance(default_pid, str):
                default_path_id_by_tree[tid] = default_pid
    return row_by_key, default_path_id_by_tree, rows_by_tree


def classify_tree_space_class(tree_rows: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    default_labels: List[int] = []
    nondefault_labels: List[int] = []
    default_scores: List[Tuple[float, float]] = []
    nondefault_scores: List[Tuple[float, float]] = []
    default_pass_rounds: List[int] = []
    nondefault_pass_rounds: List[int] = []

    for row in tree_rows:
        default = normalize_binary(row.get("default"))
        label = normalize_binary(row.get("label"))
        if default is None or label is None:
            continue
        score = extract_final_score_tuple(row)
        final_round = extract_final_round_id(row)
        if default == 1:
            default_labels.append(label)
            if score is not None:
                default_scores.append(score)
            if label == 1 and final_round is not None:
                default_pass_rounds.append(final_round)
        else:
            nondefault_labels.append(label)
            if score is not None:
                nondefault_scores.append(score)
            if label == 1 and final_round is not None:
                nondefault_pass_rounds.append(final_round)

    if not default_labels or not nondefault_labels:
        return None

    default_pass = any(label == 1 for label in default_labels)
    nondefault_pass = any(label == 1 for label in nondefault_labels)

    if default_pass and not nondefault_pass:
        return ("默认空间", "class1")
    if (not default_pass) and nondefault_pass:
        return ("提升空间", "class1")

    if default_pass and nondefault_pass:
        if not default_pass_rounds or not nondefault_pass_rounds:
            return None
        if min(nondefault_pass_rounds) < min(default_pass_rounds):
            return ("提升空间", "class2")
        return ("默认空间", "class2")

    if len(default_scores) != 1 or not nondefault_scores:
        return None
    default_score = default_scores[0]
    if any(is_nondefault_better_by_eval(nd_score, default_score) for nd_score in nondefault_scores):
        return ("提升空间", "class3")
    return ("默认空间", "class3")


def summarize_tree_passes(tree_rows: List[Dict[str, Any]]) -> Tuple[bool, bool, bool]:
    has_nondefault_path = False
    default_pass = False
    nondefault_pass = False
    for row in tree_rows:
        default = normalize_binary(row.get("default"))
        if default is None:
            continue
        passed = is_passed(row)
        if default == 1:
            default_pass = default_pass or passed
        else:
            has_nondefault_path = True
            nondefault_pass = nondefault_pass or passed
    return has_nondefault_path, default_pass, nondefault_pass


def is_model_success_on_space_class(
    space_label: str, class_label: str, default_row: Dict[str, Any], model_row: Dict[str, Any]
) -> bool:
    default_pass = is_passed(default_row)
    model_pass = is_passed(model_row)

    if class_label == "class1":
        if space_label == "提升空间":
            return (not default_pass) and model_pass
        return default_pass and model_pass

    if class_label == "class2":
        default_round = extract_final_round_id(default_row)
        model_round = extract_final_round_id(model_row)
        if default_round is None:
            return False
        if space_label == "提升空间":
            return default_pass and model_pass and model_round is not None and model_round < default_round
        return default_pass and model_pass and model_round is not None and model_round <= default_round

    if class_label == "class3":
        default_score = extract_final_score_tuple(default_row)
        model_score = extract_final_score_tuple(model_row)
        if default_score is None or model_score is None or default_pass or model_pass:
            return False
        if space_label == "提升空间":
            return is_nondefault_better_by_eval(model_score, default_score)
        return not is_nondefault_better_by_eval(default_score, model_score)

    return False


def init_bucket() -> Dict[str, Any]:
    return {
        "total_rows": 0,
        "missing_tree_rows": 0,
        "missing_default_rows": 0,
        "missing_model_rows": 0,
        "space_stats": {
            key: {"tree_count": 0, "success_count": 0} for key in SPACE_CLASS_ORDER
        },
    }


def display_width(text: str) -> int:
    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
    return width


def pad_display(text: str, width: int) -> str:
    return text + (" " * max(0, width - display_width(text)))


def format_row(col1: str, col2: str, values: List[str]) -> str:
    width1 = 24
    width2 = 10
    cell = 12
    parts = [pad_display(col1, width1), pad_display(col2, width2)]
    parts.extend(pad_display(v, cell) for v in values)
    return "".join(parts).rstrip()


def print_summary_table(stats_by_cat: Dict[str, Dict[str, Any]], method_key: str) -> None:
    columns = [
        f"{method_key}-deepseek-chat-zero-shot",
        f"{method_key}-deepseek-chat-few-shot",
        f"{method_key}-qwen-coder-zero-shot",
        f"{method_key}-qwen-coder-few-shot",
    ]
    print("SASRec表现\t\tdeepseek-chat\t\tqwen-coder")
    print("\t\t0-shot\t5-shot\t0-shot\t5-shot")
    prev_group = None
    for space_label, class_label in SPACE_CLASS_ORDER:
        values = []
        for category in columns:
            stats = stats_by_cat.get(category, {}).get("space_stats", {}).get(
                (space_label, class_label), {"success_count": 0, "tree_count": 0}
            )
            values.append(f"{stats['success_count']} / {stats['tree_count']}")
        row_group = (
            f"SASRec表现（{space_label}中）" if space_label != prev_group else ""
        )
        print(format_row(row_group, class_label, values))
        prev_group = space_label
        next_item_starts_new_group = False
        if (space_label, class_label) != SPACE_CLASS_ORDER[-1]:
            next_space_label, _ = SPACE_CLASS_ORDER[SPACE_CLASS_ORDER.index((space_label, class_label)) + 1]
            next_item_starts_new_group = next_space_label != space_label
        else:
            next_item_starts_new_group = True

        if next_item_starts_new_group:
            total_values = []
            for category in columns:
                success_total = 0
                tree_total = 0
                for key, item_stats in (
                    stats_by_cat.get(category, {}).get("space_stats", {}).items()
                ):
                    key_space_label, _ = key
                    if key_space_label != space_label:
                        continue
                    success_total += item_stats.get("success_count", 0)
                    tree_total += item_stats.get("tree_count", 0)
                total_values.append(f"{success_total} / {tree_total}")
            print(format_row("", "total", total_values))


def method_has_data(stats_by_cat: Dict[str, Dict[str, Any]], method_key: str) -> bool:
    for category, bucket in stats_by_cat.items():
        if not category.startswith(f"{method_key}-"):
            continue
        if bucket.get("total_rows", 0) > 0:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="按 v3 六类空间划分评估 SASRec 表现")
    parser.add_argument("--model-log", type=Path, required=True, help="模型日志")
    parser.add_argument("--merged-path", type=Path, required=True, help="合并路径 JSONL")
    parser.add_argument(
        "--only-method",
        type=str,
        choices=("autoverus", "verusage", "all"),
        default="all",
        help="只输出指定方法；默认 all",
    )
    args = parser.parse_args()

    if not args.model_log.is_file():
        print(f"Error: model log not found: {args.model_log}", file=sys.stderr)
        sys.exit(1)
    if not args.merged_path.is_file():
        print(f"Error: merged path not found: {args.merged_path}", file=sys.stderr)
        sys.exit(1)

    model_rows = load_model_rows(args.model_log)
    if not model_rows:
        print("No valid rows in model log.")
        return

    wanted_tree_ids = {row["tree_id"] for row in model_rows}
    row_by_key, default_path_id_by_tree, rows_by_tree = build_merged_index(
        args.merged_path, wanted_tree_ids
    )

    stats_by_cat: Dict[str, Dict[str, Any]] = {name: init_bucket() for name in CATEGORY_ORDER}
    for row in model_rows:
        tid = row["tree_id"]
        mpid = row["reached_path_id"]
        category = classify_tree_id(tid)
        if category not in stats_by_cat:
            continue
        bucket = stats_by_cat[category]
        bucket["total_rows"] += 1

        tree_rows = rows_by_tree.get(tid, [])
        if not tree_rows:
            bucket["missing_tree_rows"] += 1
            continue

        dpid = default_path_id_by_tree.get(tid)
        default_row = row_by_key.get((tid, dpid)) if dpid is not None else None
        if default_row is None:
            bucket["missing_default_rows"] += 1
            continue

        model_row = row_by_key.get((tid, mpid))
        if model_row is None:
            bucket["missing_model_rows"] += 1
            continue

        has_nondefault_path, tree_default_pass, tree_nondefault_pass = summarize_tree_passes(
            tree_rows
        )

        if has_nondefault_path and tree_default_pass:
            stats = bucket["space_stats"][("默认空间", "class1")]
            stats["tree_count"] += 1
            if is_passed(model_row):
                stats["success_count"] += 1

        if tree_nondefault_pass and (not tree_default_pass):
            stats = bucket["space_stats"][("提升空间", "class1")]
            stats["tree_count"] += 1
            if is_passed(model_row):
                stats["success_count"] += 1

        space_class = classify_tree_space_class(tree_rows)
        if space_class is None:
            continue
        if space_class[1] == "class1":
            continue

        stats = bucket["space_stats"][space_class]
        stats["tree_count"] += 1
        if is_model_success_on_space_class(space_class[0], space_class[1], default_row, model_row):
            stats["success_count"] += 1

    if args.only_method == "all":
        methods = [
            method for method in ("autoverus", "verusage")
            if method_has_data(stats_by_cat, method)
        ]
    else:
        methods = [args.only_method]

    for idx, method in enumerate(methods):
        if idx > 0:
            print()
        print_summary_table(stats_by_cat, method)
    print()
    print("三种更优情况：")
    print("class1\t通过vs没通过")
    print("class2\t都通过但更快")
    print("class3\t都没通过但是分数更高")


if __name__ == "__main__":
    main()
