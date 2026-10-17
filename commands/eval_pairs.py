#!/usr/bin/env python3
"""Statistics for pair JSONL, aligned with eval_path-style summaries.

Reports:
- Global counts (trees/tasks/pairs)
- tree_class_id distribution in eval_path-aligned style
- pair_class_id distribution (uses pair_class_id instead of class_id)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistics for nondefault-better pairs JSONL")
    parser.add_argument("--input", required=True, help="Input pairs JSONL path")
    parser.add_argument(
        "--top-trees",
        type=int,
        default=10,
        help="Number of top trees to display by pair count (default: 10)",
    )
    return parser.parse_args()


def extract_task_name_from_pair_row(row: dict[str, Any]) -> str | None:
    task_name = row.get("task_name")
    if isinstance(task_name, str) and task_name.strip():
        return task_name.strip()
    return None


def normalize_tree_class_id(value: Any) -> int | None:
    valid = {0, 1, -1, 2, -2, 3, -3}
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in valid else None
    if isinstance(value, float) and value.is_integer():
        iv = int(value)
        return iv if iv in valid else None
    if isinstance(value, str):
        text = value.strip()
        if text in {"0", "1", "-1", "2", "-2", "3", "-3"}:
            return int(text)
    return None


def normalize_pair_class_id(value: Any) -> int | None:
    valid = {1, 2, 3}
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in valid else None
    if isinstance(value, float) and value.is_integer():
        iv = int(value)
        return iv if iv in valid else None
    if isinstance(value, str):
        text = value.strip()
        if text in {"1", "2", "3"}:
            return int(text)
    return None


def infer_tree_class_id_from_row(row: dict[str, Any], pair_class_id: int | None) -> int | None:
    if pair_class_id is None:
        return None
    positive_default = row.get("positive_default")
    negative_default = row.get("negative_default")
    if positive_default in (0, 1) and negative_default in (0, 1):
        if positive_default == 0 and negative_default == 1:
            return pair_class_id
        if positive_default == 1 and negative_default == 0:
            return -pair_class_id
    return None


def status_logo(ok: bool) -> str:
    return "✅" if ok else "❌"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()

    if not input_path.is_file():
        print(f"[ERROR] input file not found: {input_path}")
        return 1

    rows: list[dict[str, Any]] = []
    total_lines = 0
    invalid_lines = 0

    with input_path.open("r", encoding="utf-8") as rf:
        for line in rf:
            raw = line.strip()
            if not raw:
                continue
            total_lines += 1
            try:
                obj = json.loads(raw)
            except Exception:
                invalid_lines += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)

    if not rows:
        print(f"input={input_path}")
        print("No valid rows found.")
        return 0

    # --- Basic counts ---
    tree_counter: Counter[str] = Counter()
    pair_class_counter: Counter[int] = Counter()
    pair_class_tree_sets: dict[int, set[str]] = {1: set(), 2: set(), 3: set()}
    tree_class_counter: Counter[int] = Counter()
    tree_class_tree_sets: dict[int, set[str]] = {0: set(), 1: set(), -1: set(), 2: set(), -2: set(), 3: set(), -3: set()}
    tree_to_tree_class_ids: dict[str, set[int]] = {}
    unique_task_names: set[str] = set()
    pair_class_task_sets: dict[int, set[str]] = {1: set(), 2: set(), 3: set()}
    missing_tree_class_rows = 0
    missing_pair_class_rows = 0
    inferred_tree_class_rows = 0
    fallback_pair_class_rows = 0

    for row in rows:
        tree_id = row.get("tree_id", "<unknown>")
        tree_counter[tree_id] += 1

        pair_class_id = normalize_pair_class_id(row.get("pair_class_id"))
        if pair_class_id is None:
            pair_class_id = normalize_pair_class_id(row.get("class_id"))
            if pair_class_id is not None:
                fallback_pair_class_rows += 1

        tree_class_id = normalize_tree_class_id(row.get("tree_class_id"))
        if tree_class_id is None:
            inferred_tree_class_id = infer_tree_class_id_from_row(row, pair_class_id)
            if inferred_tree_class_id is not None:
                tree_class_id = inferred_tree_class_id
                inferred_tree_class_rows += 1

        if tree_class_id is None:
            missing_tree_class_rows += 1
        else:
            tree_class_counter[tree_class_id] += 1
            tree_class_tree_sets.setdefault(tree_class_id, set()).add(tree_id)
            tree_to_tree_class_ids.setdefault(tree_id, set()).add(tree_class_id)

        if pair_class_id is None:
            missing_pair_class_rows += 1
        else:
            pair_class_counter[pair_class_id] += 1
            pair_class_tree_sets[pair_class_id].add(tree_id)

        task_name = extract_task_name_from_pair_row(row)
        if task_name is not None:
            unique_task_names.add(task_name)
            if pair_class_id in pair_class_task_sets:
                pair_class_task_sets[pair_class_id].add(task_name)

    multi_tree_class_count = sum(
        1 for class_ids in tree_to_tree_class_ids.values() if len(class_ids) > 1
    )
    has_multi_tree_class = multi_tree_class_count > 0

    # --- Output ---
    print(f"总的树的个数: {len(tree_counter)}")
    print(f"唯一程序/task_name个数: {len(unique_task_names)}")
    print(f"总的pair对数据量: {len(rows)}")
    print()
    print("7 种树的统计如下，格式为 [树个数，pair个数]：")
    class_order = [0, 1, -1, 2, -2, 3, -3]
    class_explanations: dict[int, str] = {
        0: "default 与 non-default 均存在，但未命中 1/-1/2/-2/3/-3 的规则",
        1: "non-default 最终分数优于 default",
        -1: "default 最终分数优于 non-default",
        2: "default 全失败且 non-default 至少一条成功",
        -2: "default 至少一条成功且 non-default 全失败",
        3: "default 与 non-default 均有成功，且 non-default 最快成功轮次更小",
        -3: "default 与 non-default 均有成功，且 default 最快成功轮次更小",
    }

    tree_class_tree_sum = 0
    tree_class_pair_sum = 0
    for class_id in class_order:
        tree_count = len(tree_class_tree_sets.get(class_id, set()))
        pair_count = tree_class_counter.get(class_id, 0)
        tree_class_tree_sum += tree_count
        tree_class_pair_sum += pair_count
        print(
            f"- tree_class_id = {class_id:2d}：[{tree_count}, {pair_count}]"
            f" # {class_explanations[class_id]}"
        )

    tree_sum_matches_top = tree_class_tree_sum == len(tree_counter)
    pair_sum_matches_top = tree_class_pair_sum == len(rows)

    print()
    print("3 种pair的统计如下，格式为 [树个数，pair个数]：")
    for cid in (1, 2, 3):
        pair_count = pair_class_counter.get(cid, 0)
        tree_count = len(pair_class_tree_sets.get(cid, set()))
        print(f"- pair_class_id = {cid:2d}：[{tree_count}, {pair_count}]")

    print()
    no_overlap_ok = not has_multi_tree_class
    print(
        "=> tree_class_id 是否存在重叠（一棵树同时命中多个tree_class_id）: "
        f"{has_multi_tree_class} {status_logo(no_overlap_ok)}"
    )
    print(
        "=> tree_class_id 树个数之和是否与顶部总的树的个数一致: "
        f"{tree_sum_matches_top} {status_logo(tree_sum_matches_top)}"
    )
    print(
        "=> tree_class_id pair个数之和是否与顶部总的pair对数据量一致: "
        f"{pair_sum_matches_top} {status_logo(pair_sum_matches_top)}"
    )
    # print(f"rows_missing_tree_class_id: {missing_tree_class_rows}")
    # print(f"rows_missing_pair_class_id: {missing_pair_class_rows}")
    # print(f"rows_inferred_tree_class_id: {inferred_tree_class_rows}")
    # print(f"rows_fallback_pair_class_id_from_class_id: {fallback_pair_class_rows}")

    # print()
    # print("pair_class_id 详细统计：")
    # for cid in (1, 2, 3):
    #     print(f"pair_class_{cid}_task_count: {len(pair_class_task_sets.get(cid, set()))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
