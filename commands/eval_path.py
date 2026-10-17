#!/usr/bin/env python3
"""
python commands/eval.py --input deepseek-chat_merged_root_to_leaf_paths.jsonl

输出字段中文含义（按打印顺序）：
  - input: 输入 JSONL 文件路径
  - total_records: 输入中非空行总数（不要求 JSON 合法）
  - parsed_records: JSON 解析成功的记录数
  - invalid_json_records: JSON 解析失败的记录数
  - missing_or_invalid_label_records: label 缺失或无法归一化为 0/1 的记录数
  - missing_or_invalid_default_records: default 缺失或无法归一化为 0/1 的记录数
  - missing_or_invalid_id_records: tree_id 缺失、非字符串或空字符串的记录数（兼容旧字段 id）
  - unique_ids: 参与 id 级统计的唯一 id 数（要求 tree_id/id 合法，且 label/default 都可解析）
  - all_ids_have_exactly_one_default_path: 是否每个 id（tree_id）恰好有 1 条 default=1 路径
  - ids_not_exactly_one_default_count: default 路径数不等于 1 的 id（tree_id）个数
  - label_0_paths: label=0 的路径条数
  - label_1_paths: label=1 的路径条数
  - default_paths: default=1 的路径条数
  - non_default_paths: default=0 的路径条数
  - default_paths_label_0: default=1 且 label=0 的路径条数
  - default_paths_label_1: default=1 且 label=1 的路径条数
  - non_default_paths_label_0: default=0 且 label=0 的路径条数
  - non_default_paths_label_1: default=0 且 label=1 的路径条数

  - ids_with_default_and_non_default_paths: 同时拥有 default 与 non-default 路径的 id 个数

  - ids_with_default_fail_and_non_default_pass: 同一 id 下同时出现「default 路径失败(label=0)」和「non-default 路径成功(label=1)」的 id 个数(核心指标1)

  - ids_with_non_default_better_final_score: 在可比 id 中，至少一条 non-default 路径最终分数优于 default 的 id 个数(核心指标2)
      可比条件：该 id 恰有 1 条 default 路径、至少 1 条 non-default 路径，且都能提取最终 score_tuple

  - ids_with_non_default_fewer_final_rounds: 在可比 id 中，至少一条 non-default 路径最终轮次小于 default 的 id 个数(核心指标3)
      可比条件：该 id 恰有 1 条 default 路径、至少 1 条 non-default 路径，且都能提取最终 round_id


分数优于(better final score)判定:
    - 给定 default 分数 (d1, d2) 与 non-default 分数 (n1, n2)：
        先比较第一维，若 n1 > d1 则更优；若 n1 < d1 则不更优；
        仅当 n1 == d1 且第二维可比较(均不为 -1)时，若 n2 < d2 则更优。

轮次更少(fewer final rounds)判定:
  - 给定 default 最终轮次 d_round 与 non-default 最终轮次 n_round，若 n_round < d_round，则认为 non-default 修复轮次更少。


"""

import argparse
import json
from collections import Counter
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count label(0/1) and default/non-default path distributions "
            "from a JSONL file"
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed sections below the separator blocks (default: hidden)",
    )
    return parser.parse_args()


def normalize_binary(value):
    """Normalize value to 0/1 or return None when invalid."""
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


def extract_final_score_tuple(obj):
    """Extract final state's score_tuple as (x, y) or return None."""
    path_states = obj.get("path_states")
    if not isinstance(path_states, list) or len(path_states) == 0:
        return None
    last_state = path_states[-1]
    if not isinstance(last_state, dict):
        return None
    score = last_state.get("score_tuple")
    if (
        isinstance(score, list)
        and len(score) >= 2
        and isinstance(score[0], (int, float))
        and isinstance(score[1], (int, float))
    ):
        return (score[0], score[1])
    return None


def extract_root_score_tuple(obj):
    """Extract root state's score_tuple as (x, y) or return None."""
    path_states = obj.get("path_states")
    if not isinstance(path_states, list) or len(path_states) == 0:
        return None
    first_state = path_states[0]
    if not isinstance(first_state, dict):
        return None
    score = first_state.get("score_tuple")
    if (
        isinstance(score, list)
        and len(score) >= 2
        and isinstance(score[0], (int, float))
        and isinstance(score[1], (int, float))
    ):
        return (score[0], score[1])
    return None


def extract_final_round_id(obj):
    """Extract final state's round_id as int or return None."""
    path_states = obj.get("path_states")
    if not isinstance(path_states, list) or len(path_states) == 0:
        return None
    last_state = path_states[-1]
    if not isinstance(last_state, dict):
        return None
    round_id = last_state.get("round_id")
    if isinstance(round_id, int):
        return round_id
    if isinstance(round_id, float) and round_id.is_integer():
        return int(round_id)
    return None


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_score_pair(scores: list[tuple[float, float]]) -> str:
    if not scores:
        return "N/A"
    first_avg = safe_mean([float(x) for x, _ in scores])
    second_avg = safe_mean([float(y) for _, y in scores])
    return f"({first_avg:.4f}, {second_avg:.4f})"


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{numerator / denominator:.4f}"


def second_score_comparable(default_second: float, nondefault_second: float) -> bool:
    return default_second != -1 and nondefault_second != -1


def is_nondefault_better_by_eval(
    nondefault_score: tuple[float, float], default_score: tuple[float, float]
) -> bool:
    nd_first, nd_second = nondefault_score
    d_first, d_second = default_score
    if nd_first > d_first:
        return True
    if nd_first < d_first:
        return False
    return second_score_comparable(d_second, nd_second) and nd_second < d_second


def extract_tree_id(obj):
    """Prefer tree_id; fallback to legacy id."""
    value = obj.get("tree_id")
    if not isinstance(value, str) or not value.strip():
        value = obj.get("id")
    if isinstance(value, str) and value.strip():
        return value
    return None


def extract_task_name(obj):
    tree_key = obj.get("tree_key")
    if isinstance(tree_key, dict):
        task_name = tree_key.get("task_name")
        if isinstance(task_name, str) and task_name.strip():
            return task_name
    task_name = obj.get("task_name")
    if isinstance(task_name, str) and task_name.strip():
        return task_name
    return None


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.is_file():
        print(f"[ERROR] input file not found: {input_path}")
        return 1

    total_records = 0
    parsed_records = 0
    invalid_json_records = 0
    missing_or_invalid_label_records = 0
    missing_or_invalid_default_records = 0
    missing_or_invalid_id_records = 0
    missing_or_invalid_task_name_records = 0

    label_0 = 0
    label_1 = 0
    default_true = 0
    default_false = 0

    default_true_label_0 = 0
    default_true_label_1 = 0
    default_false_label_0 = 0
    default_false_label_1 = 0

    path_length_counter: Counter[int] = Counter()
    path_length_label_counter: dict[int, Counter[int]] = {}

    # id-level condition:
    # same id has at least one default path with label=0,
    # and at least one non-default path with label=1
    id_flags: dict[str, dict[str, bool]] = {}
    id_default_path_counts: dict[str, int] = {}
    id_default_nondefault_flags: dict[str, dict[str, bool]] = {}
    id_default_final_scores: dict[str, list[tuple[float, float]]] = {}
    id_nondefault_final_scores: dict[str, list[tuple[float, float]]] = {}
    id_default_final_rounds: dict[str, list[int]] = {}
    id_nondefault_final_rounds: dict[str, list[int]] = {}
    id_all_path_lengths: dict[str, list[int]] = {}
    id_default_path_lengths: dict[str, list[int]] = {}
    id_nondefault_path_lengths: dict[str, list[int]] = {}
    id_default_pass_final_rounds: dict[str, list[int]] = {}
    id_nondefault_pass_final_rounds: dict[str, list[int]] = {}
    id_default_labels: dict[str, list[int]] = {}
    id_nondefault_labels: dict[str, list[int]] = {}
    id_task_names: dict[str, set[str]] = {}
    all_tree_ids: set[str] = set()
    all_task_names: set[str] = set()
    unique_task_names_in_eval_rows: set[str] = set()
    root_scores_all_paths: list[tuple[float, float]] = []
    final_scores_all_paths: list[tuple[float, float]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1

            try:
                obj = json.loads(line)
            except Exception:
                invalid_json_records += 1
                continue

            parsed_records += 1
            label = normalize_binary(obj.get("label"))
            default = normalize_binary(obj.get("default"))
            sample_id = extract_tree_id(obj)
            task_name = extract_task_name(obj)
            if sample_id is not None:
                all_tree_ids.add(sample_id)
            if task_name is not None:
                all_task_names.add(task_name)
            if sample_id is None:
                missing_or_invalid_id_records += 1

            if label is None:
                missing_or_invalid_label_records += 1
            else:
                if label == 0:
                    label_0 += 1
                else:
                    label_1 += 1

            if default is None:
                missing_or_invalid_default_records += 1
            else:
                if default == 1:
                    default_true += 1
                else:
                    default_false += 1

            if label is not None and default is not None:
                if task_name is None:
                    missing_or_invalid_task_name_records += 1
                if default == 1 and label == 0:
                    default_true_label_0 += 1
                elif default == 1 and label == 1:
                    default_true_label_1 += 1
                elif default == 0 and label == 0:
                    default_false_label_0 += 1
                elif default == 0 and label == 1:
                    default_false_label_1 += 1

                if sample_id is not None:
                    if task_name is not None:
                        unique_task_names_in_eval_rows.add(task_name)
                        id_task_names.setdefault(sample_id, set()).add(task_name)
                    if sample_id not in id_flags:
                        id_flags[sample_id] = {
                            "has_default_fail": False,
                            "has_nondefault_pass": False,
                        }
                        id_default_path_counts[sample_id] = 0
                    if sample_id not in id_default_nondefault_flags:
                        id_default_nondefault_flags[sample_id] = {
                            "has_default_path": False,
                            "has_non_default_path": False,
                        }

                    if default == 1 and label == 0:
                        id_flags[sample_id]["has_default_fail"] = True
                    if default == 0 and label == 1:
                        id_flags[sample_id]["has_nondefault_pass"] = True
                    if default == 1:
                        id_default_path_counts[sample_id] += 1

                    if default == 1:
                        id_default_nondefault_flags[sample_id]["has_default_path"] = True
                    else:
                        id_default_nondefault_flags[sample_id]["has_non_default_path"] = True

                    path_len = None
                    path_state_ids = obj.get("path_state_ids")
                    path_states = obj.get("path_states")
                    if isinstance(path_state_ids, list):
                        path_len = len(path_state_ids)
                    elif isinstance(path_states, list):
                        path_len = len(path_states)

                    if isinstance(path_len, int) and path_len > 0:
                        path_length_counter[path_len] += 1
                        path_length_label_counter.setdefault(path_len, Counter())
                        path_length_label_counter[path_len][label] += 1

                        id_all_path_lengths.setdefault(sample_id, []).append(path_len)
                        if default == 1:
                            id_default_path_lengths.setdefault(sample_id, []).append(path_len)
                        else:
                            id_nondefault_path_lengths.setdefault(sample_id, []).append(path_len)

                    if default == 1:
                        id_default_labels.setdefault(sample_id, []).append(label)
                    else:
                        id_nondefault_labels.setdefault(sample_id, []).append(label)

                    root_score = extract_root_score_tuple(obj)
                    if root_score is not None:
                        root_scores_all_paths.append(root_score)

                    final_score = extract_final_score_tuple(obj)
                    if final_score is not None:
                        final_scores_all_paths.append(final_score)
                        if default == 1:
                            id_default_final_scores.setdefault(sample_id, []).append(
                                final_score
                            )
                        else:
                            id_nondefault_final_scores.setdefault(sample_id, []).append(
                                final_score
                            )
                    final_round = extract_final_round_id(obj)
                    if final_round is not None:
                        if default == 1:
                            id_default_final_rounds.setdefault(sample_id, []).append(
                                final_round
                            )
                        else:
                            id_nondefault_final_rounds.setdefault(sample_id, []).append(
                                final_round
                            )

                        if label == 1:
                            if default == 1:
                                id_default_pass_final_rounds.setdefault(sample_id, []).append(
                                    final_round
                                )
                            else:
                                id_nondefault_pass_final_rounds.setdefault(sample_id, []).append(
                                    final_round
                                )

    ids_with_default_and_nondefault_paths = {
        sample_id
        for sample_id, flags in id_default_nondefault_flags.items()
        if flags["has_default_path"] and flags["has_non_default_path"]
    }

    ids_with_single_default_only_path = {
        sample_id
        for sample_id, flags in id_default_nondefault_flags.items()
        if flags["has_default_path"]
        and (not flags["has_non_default_path"])
        and id_default_path_counts.get(sample_id, 0) == 1
    }

    ids_with_nondefault_only_path = {
        sample_id
        for sample_id, flags in id_default_nondefault_flags.items()
        if (not flags["has_default_path"]) and flags["has_non_default_path"]
    }

    def classify_tree(sample_id: str) -> int:
        default_labels = id_default_labels.get(sample_id, [])
        nondefault_labels = id_nondefault_labels.get(sample_id, [])
        if not default_labels or not nondefault_labels:
            return 0

        default_pass = any(label == 1 for label in default_labels)
        nondefault_pass = any(label == 1 for label in nondefault_labels)

        if (not default_pass) and nondefault_pass:
            return 2
        if default_pass and (not nondefault_pass):
            return -2

        if default_pass and nondefault_pass:
            default_pass_rounds = id_default_pass_final_rounds.get(sample_id, [])
            nondefault_pass_rounds = id_nondefault_pass_final_rounds.get(sample_id, [])
            if not default_pass_rounds or not nondefault_pass_rounds:
                return 0
            best_default_round = min(default_pass_rounds)
            best_nondefault_round = min(nondefault_pass_rounds)
            if best_nondefault_round < best_default_round:
                return 3
            if best_default_round < best_nondefault_round:
                return -3
            return 0

        default_scores = id_default_final_scores.get(sample_id, [])
        nondefault_scores = id_nondefault_final_scores.get(sample_id, [])
        if len(default_scores) != 1 or len(nondefault_scores) == 0:
            return 0

        default_score = default_scores[0]
        has_nondefault_better = any(
            is_nondefault_better_by_eval(nd_score, default_score)
            for nd_score in nondefault_scores
        )
        if has_nondefault_better:
            return 1

        has_default_better = any(
            is_nondefault_better_by_eval(default_score, nd_score)
            for nd_score in nondefault_scores
        )
        if has_default_better:
            return -1

        return 0

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
    class_to_tree_ids: dict[int, set[str]] = {class_id: set() for class_id in class_order}
    tree_to_class_hits: dict[str, set[int]] = {}

    for sample_id in sorted(ids_with_default_and_nondefault_paths):
        class_id = classify_tree(sample_id)
        class_to_tree_ids[class_id].add(sample_id)
        tree_to_class_hits[sample_id] = {class_id}

    class_overlap_exists = any(len(hits) > 1 for hits in tree_to_class_hits.values())

    tree_path_count: dict[str, int] = {
        sample_id: len(id_all_path_lengths.get(sample_id, []))
        for sample_id in ids_with_default_and_nondefault_paths
    }

    single_default_only_tree_path_count: dict[str, int] = {
        sample_id: len(id_all_path_lengths.get(sample_id, []))
        for sample_id in ids_with_single_default_only_path
    }

    nondefault_only_tree_path_count: dict[str, int] = {
        sample_id: len(id_all_path_lengths.get(sample_id, []))
        for sample_id in ids_with_nondefault_only_path
    }

    print(f"总的树的个数: {len(all_tree_ids)}")
    print(f"唯一程序/task_name个数: {len(all_task_names)}")
    print(f"总的路径数据量: {total_records}")
    print()
    print("9 种树的统计如下，格式为 [树个数，路径个数]：")
    class_tree_sum = 0
    class_path_sum = 0
    for class_id in class_order:
        tree_ids = class_to_tree_ids[class_id]
        path_count = sum(tree_path_count.get(tree_id, 0) for tree_id in tree_ids)
        class_tree_sum += len(tree_ids)
        class_path_sum += path_count
        print(
            f"- tree_class_id = {class_id:2d}：[{len(tree_ids)}, {path_count}]"
            f" # {class_explanations[class_id]}"
        )
    single_default_tree_count = len(ids_with_single_default_only_path)
    single_default_path_sum = sum(single_default_only_tree_path_count.values())
    nondefault_only_tree_count = len(ids_with_nondefault_only_path)
    nondefault_only_path_sum = sum(nondefault_only_tree_path_count.values())
    print(
        "- 单默认路径树统计（仅 1 条 default 路径且无 non-default 路径）: "
        f"[{single_default_tree_count}, {single_default_path_sum}]"
        " # 仅有一条 default 路径"
    )
    print(
        "- 仅非默认路径树统计（存在 non-default 路径且无 default 路径）: "
        f"[{nondefault_only_tree_count}, {nondefault_only_path_sum}]"
        " # 仅有 non-default 路径"
    )

    nine_tree_sum = class_tree_sum + single_default_tree_count + nondefault_only_tree_count
    nine_path_sum = class_path_sum + single_default_path_sum + nondefault_only_path_sum
    tree_sum_matches_top = nine_tree_sum == len(all_tree_ids)
    path_sum_matches_top = nine_path_sum == total_records

    def status_logo(ok: bool) -> str:
        return "✅" if ok else "❌"

    print()
    no_overlap_ok = not class_overlap_exists
    print(
        f"=> 9 种树是否存在重叠: {class_overlap_exists} "
        f"{status_logo(no_overlap_ok)}"
    )
    print(
        "=> 9 种树树个数之和是否与顶部总的树的个数一致: "
        f"{tree_sum_matches_top} {status_logo(tree_sum_matches_top)}"
    )
    print(
        "=> 9 种树路径个数之和是否与顶部总的路径数据量一致: "
        f"{path_sum_matches_top} {status_logo(path_sum_matches_top)}"
    )

    if args.show_details:
        print()
        print("各类tree_id明细：")
        for class_id in class_order:
            ids_sorted = sorted(class_to_tree_ids[class_id])
            print(f"tree_class_id={class_id}: {ids_sorted if ids_sorted else '(none)'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
