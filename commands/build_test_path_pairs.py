#!/usr/bin/env python3
"""Extract class-labeled positive-only samples from root-to-leaf JSONL.

Tree classes are expected from input field `tree_class_id` in
{1,-1,2,-2,3,-3}. Output pair class stays in {1,2,3}:
1) score-better side is positive (both failed)
2) passed side is positive (pass vs fail)
3) fewer-round passed side is positive (pass vs pass)

For tree_class_id in {-1,-2,-3}, selection is mirrored:
positive uses default-path side.

Output keeps only positive sample information.
Each positive sample appears at most once in output (deduplicated by tree_id + path_id).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm


ERROR_MESSAGE_TO_SHORT_NAME: dict[str, str] = {
    "precondition not satisfied": "PreCondFail",
    "postcondition not satisfied": "PostCondFail",
    "invariant not satisfied at end of loop body": "InvFailEnd",
    "invariant not satisfied before loop": "InvFailFront",
    "decreases not satisfied at end of loop": "DecFailEnd",
    "decreases not satisfied at continue": "DecFailCont",
    "split assertion failure": "SplitAssertFail",
    "split precondition failure": "SplitPreFail",
    "split postcondition failure": "SplitPostFail",
    "recommendation not met": "RecommendNotMet",
    "assertion failed": "AssertFail",
    "possible arithmetic underflow/overflow": "ArithmeticFlow",
    "mismatched types": "MismatchedType",
    "unexpected proof block": "UnxProofBlock",
    "panic is not supported (if you used Rust's `assert!` macro, you may have meant to use Verus's `assert` function)": "RustAssert",
    "cannot call function with mode exec": "ExecinGhost",
    "expected `)": "ExpectedParentheses",
    "unsupported for bit-vector": "UnsupportedForBitVector",
    "could not prove termination": "TerminationFail",
    "cannot prove termination": "CannotProveTermination",
    "no method named": "MethodNotFound",
    "loop must have a decreases clause": "LoopNoDec",
    "bitvector assertion not satisfied": "BitVAssertFail",
    "in 'ensures' clause of public function, cannot access any field of a datatype where one or more fields are private": "EnsurePrivate",
    "in 'requires' clause of public function, cannot refer to private function": "RequirePrivate",
    "cannot find macro `verus` in this scope": "MissingImport",
    "type annotations needed": "TypeAnnotation",
    "constructed value may fail to meet its declared type invariant": "ConstructorFailTypeInvariant",
    "cannot call function": "CannotCallFunc",
    "in requires, use `old(self)` to refer to the pre-state of an &mut variable": "RequiresOldSelf",
    "non-private spec function must be marked open or closed": "PubSpecVisibility",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract class-labeled positive-only path rows")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--class1-positive-top-k",
        type=int,
        default=1,
        help="Top-k highest-score positives for class_1 positive-only construction (default: 1)",
    )
    return parser.parse_args()


def normalize_binary(value: Any) -> int | None:
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


def extract_tree_id(obj: dict[str, Any]) -> str | None:
    value = obj.get("tree_id")
    if not isinstance(value, str) or not value.strip():
        value = obj.get("id")
    if isinstance(value, str) and value.strip():
        return value
    return None


def extract_task_name(obj: dict[str, Any]) -> str | None:
    tree_key = obj.get("tree_key")
    if isinstance(tree_key, dict):
        task_name = tree_key.get("task_name")
        if isinstance(task_name, str) and task_name.strip():
            return task_name
    return None


def infer_task_name_from_tree_rows(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        task_name = extract_task_name(row)
        if task_name is not None and task_name.strip():
            return task_name.strip()
        fallback_task_name = row.get("task_name")
        if isinstance(fallback_task_name, str) and fallback_task_name.strip():
            return fallback_task_name.strip()
    return None


def extract_final_score_tuple(obj: dict[str, Any]) -> tuple[float, float] | None:
    path_states = obj.get("path_states")
    if not isinstance(path_states, list) or not path_states:
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
        return float(score[0]), float(score[1])
    return None


def extract_final_round_id(obj: dict[str, Any]) -> int | None:
    path_states = obj.get("path_states")
    if not isinstance(path_states, list) or not path_states:
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


def is_nondefault_better_by_eval(
    nondefault_score: tuple[float, float], default_score: tuple[float, float]
) -> bool:
    nd_first, nd_second = nondefault_score
    d_first, d_second = default_score
    if nd_first > d_first:
        return True
    if nd_first < d_first:
        return False
    return nd_second < d_second


def normalize_tree_class_id(value: Any) -> int | None:
    valid = {1, -1, 2, -2, 3, -3}
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in valid else None
    if isinstance(value, float) and value.is_integer():
        iv = int(value)
        return iv if iv in valid else None
    if isinstance(value, str):
        text = value.strip()
        if text in {"1", "-1", "2", "-2", "3", "-3"}:
            return int(text)
    return None


def infer_tree_class_id_fallback(
    default_rows: list[dict[str, Any]],
    nondefault_rows: list[dict[str, Any]],
) -> int | None:
    if not default_rows or not nondefault_rows:
        return None

    default_labels = [r.get("__norm_label") for r in default_rows if r.get("__norm_label") in (0, 1)]
    nondefault_labels = [
        r.get("__norm_label") for r in nondefault_rows if r.get("__norm_label") in (0, 1)
    ]
    if not default_labels or not nondefault_labels:
        return None

    default_pass = any(label == 1 for label in default_labels)
    nondefault_pass = any(label == 1 for label in nondefault_labels)

    if (not default_pass) and nondefault_pass:
        return 2
    if default_pass and (not nondefault_pass):
        return -2

    if default_pass and nondefault_pass:
        default_pass_rounds = [
            round_id
            for row in default_rows
            if row.get("__norm_label") == 1
            for round_id in [extract_final_round_id(row)]
            if round_id is not None
        ]
        nondefault_pass_rounds = [
            round_id
            for row in nondefault_rows
            if row.get("__norm_label") == 1
            for round_id in [extract_final_round_id(row)]
            if round_id is not None
        ]
        if not default_pass_rounds or not nondefault_pass_rounds:
            return None
        best_default_round = min(default_pass_rounds)
        best_nondefault_round = min(nondefault_pass_rounds)
        if best_nondefault_round < best_default_round:
            return 3
        if best_default_round < best_nondefault_round:
            return -3
        return None

    default_scores = [
        score for row in default_rows for score in [extract_final_score_tuple(row)] if score is not None
    ]
    nondefault_scores = [
        score for row in nondefault_rows for score in [extract_final_score_tuple(row)] if score is not None
    ]
    if len(default_scores) != 1 or not nondefault_scores:
        return None

    default_score = default_scores[0]
    if any(is_nondefault_better_by_eval(nd_score, default_score) for nd_score in nondefault_scores):
        return 1
    if any(is_nondefault_better_by_eval(default_score, nd_score) for nd_score in nondefault_scores):
        return -1

    return None


def select_top_k_rows_by_eval_score(
    rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    scored_rows: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for row in rows:
        score = extract_final_score_tuple(row)
        if score is None:
            continue
        scored_rows.append((score, row))

    selected: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for score, row in scored_rows:
        inserted = False
        for idx, (exist_score, _) in enumerate(selected):
            if is_nondefault_better_by_eval(score, exist_score):
                selected.insert(idx, (score, row))
                inserted = True
                break
        if not inserted:
            selected.append((score, row))
        if len(selected) > top_k:
            selected.pop()

    return [row for _, row in selected]


def drop_key_recursive(value: Any, banned_key: str) -> Any:
    if isinstance(value, dict):
        return {
            k: drop_key_recursive(v, banned_key)
            for k, v in value.items()
            if k != banned_key
        }
    if isinstance(value, list):
        return [drop_key_recursive(v, banned_key) for v in value]
    return value


def replace_error_message_with_short_name(line: str) -> str:
    match = re.search(r"(error(?:\[[^\]]+\])?\s*:\s*)(.*)", line, flags=re.IGNORECASE)
    if not match:
        return line

    prefix = match.group(1)
    message = match.group(2)
    message_lower = message.lower()

    for raw_error, short_name in ERROR_MESSAGE_TO_SHORT_NAME.items():
        raw_error_lower = raw_error.lower()
        if message_lower.startswith(raw_error_lower):
            replaced = short_name + message[len(raw_error) :]
            return f"{line[:match.start(1)]}{prefix}{replaced}"

    for raw_error, short_name in ERROR_MESSAGE_TO_SHORT_NAME.items():
        pattern = re.compile(re.escape(raw_error), flags=re.IGNORECASE)
        if pattern.search(message):
            replaced = pattern.sub(short_name, message, count=1)
            return f"{line[:match.start(1)]}{prefix}{replaced}"

    return line


def build_verus_output_full_replace(value: str) -> str:
    return "\n".join(replace_error_message_with_short_name(line) for line in value.splitlines())


def add_verus_output_full_replace_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            out[key] = add_verus_output_full_replace_recursive(inner)
        full = out.get("verus_output_full")
        if isinstance(full, str):
            out["verus_output_full_replace"] = build_verus_output_full_replace(full)
        return out
    if isinstance(value, list):
        return [add_verus_output_full_replace_recursive(v) for v in value]
    return value


def row_clean_copy(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {k: v for k, v in row.items() if not str(k).startswith("__norm_")}
    cleaned = drop_key_recursive(cleaned, "repaired_failure_list")
    return add_verus_output_full_replace_recursive(cleaned)


def positive_only_row(
    tree_id: str,
    tree_class_id: int,
    pair_class_id: int,
    positive_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tree_id": tree_id,
        "task_name": extract_task_name(positive_row),
        "tree_class_id": tree_class_id,
        "pair_class_id": pair_class_id,
        "class_id": pair_class_id,
        "positive_path_id": positive_row.get("path_id"),
        "positive_final_score": extract_final_score_tuple(positive_row),
        "positive_final_round": extract_final_round_id(positive_row),
        "positive_label": positive_row.get("__norm_label"),
        "positive_default": positive_row.get("__norm_default"),
        "positive_path": row_clean_copy(positive_row),
    }


def positive_dedup_key(tree_id: str, positive_row: dict[str, Any]) -> str:
    path_id = positive_row.get("path_id")
    if isinstance(path_id, str) and path_id:
        return f"{tree_id}::{path_id}"
    return f"{tree_id}::{json.dumps(row_clean_copy(positive_row), ensure_ascii=False, sort_keys=True)}"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.is_file():
        print(f"[ERROR] input file not found: {input_path}")
        return 1

    total_nonempty = 0
    parsed_records = 0
    invalid_json_records = 0
    excluded_records = 0

    rows_by_tree: dict[str, list[dict[str, Any]]] = {}

    with input_path.open("r", encoding="utf-8") as rf:
        for line in rf:
            raw = line.strip()
            if not raw:
                continue
            total_nonempty += 1

            try:
                obj = json.loads(raw)
            except Exception:
                invalid_json_records += 1
                continue

            if not isinstance(obj, dict):
                excluded_records += 1
                continue

            parsed_records += 1
            tree_id = extract_tree_id(obj)
            label = normalize_binary(obj.get("label"))
            default = normalize_binary(obj.get("default"))
            if tree_id is None or label is None or default is None:
                excluded_records += 1
                continue

            obj["__norm_label"] = label
            obj["__norm_default"] = default
            rows_by_tree.setdefault(tree_id, []).append(obj)

    valid_tree_ids = {
        tree_id
        for tree_id, rows in rows_by_tree.items()
        if any(r["__norm_default"] == 1 for r in rows)
        and any(r["__norm_default"] == 0 for r in rows)
    }

    out_rows: list[dict[str, Any]] = []
    seen_positive_keys: set[str] = set()

    class1_tree_ids: set[str] = set()
    class2_tree_ids: set[str] = set()
    class3_tree_ids: set[str] = set()
    class1_rows = 0
    class2_rows = 0
    class3_rows = 0
    trees_skipped_missing_tree_class_id = 0
    trees_skipped_inconsistent_tree_class_id = 0
    trees_fallback_inferred_tree_class_id = 0

    def add_positive(
        tree_id: str,
        tree_class_id: int,
        pair_class_id: int,
        row: dict[str, Any],
    ) -> None:
        nonlocal class1_rows, class2_rows, class3_rows
        key = positive_dedup_key(tree_id, row)
        if key in seen_positive_keys:
            return
        seen_positive_keys.add(key)
        out_rows.append(positive_only_row(tree_id, tree_class_id, pair_class_id, row))
        if pair_class_id == 1:
            class1_rows += 1
        elif pair_class_id == 2:
            class2_rows += 1
        elif pair_class_id == 3:
            class3_rows += 1

    for tree_id in tqdm(sorted(valid_tree_ids), desc="Processing trees"):
        rows = rows_by_tree[tree_id]
        task_name = infer_task_name_from_tree_rows(rows)
        default_rows = [r for r in rows if r["__norm_default"] == 1]
        nondefault_rows = [r for r in rows if r["__norm_default"] == 0]

        if len(default_rows) != 1 or not nondefault_rows:
            continue

        default_row = default_rows[0]
        default_passed = default_row["__norm_label"] == 1
        failed_nondefault_rows = [r for r in nondefault_rows if r["__norm_label"] == 0]
        passed_nondefault_rows = [r for r in nondefault_rows if r["__norm_label"] == 1]

        tree_class_values = {
            normalize_tree_class_id(r.get("tree_class_id"))
            for r in rows
            if normalize_tree_class_id(r.get("tree_class_id")) is not None
        }
        if len(tree_class_values) == 0:
            fallback_tree_class = infer_tree_class_id_fallback(
                default_rows=default_rows,
                nondefault_rows=nondefault_rows,
            )
            if fallback_tree_class is None:
                trees_skipped_missing_tree_class_id += 1
                continue
            tree_class = fallback_tree_class
            trees_fallback_inferred_tree_class_id += 1
        else:
            if len(tree_class_values) > 1:
                trees_skipped_inconsistent_tree_class_id += 1
                continue
            tree_class = next(iter(tree_class_values))

        if tree_class == 1:
            class1_tree_ids.add(tree_id)
            default_score = extract_final_score_tuple(default_row)
            if default_score is not None:
                better_rows: list[dict[str, Any]] = []
                for row in failed_nondefault_rows:
                    nd_score = extract_final_score_tuple(row)
                    if nd_score is None:
                        continue
                    if is_nondefault_better_by_eval(nd_score, default_score):
                        better_rows.append(row)
                for row in select_top_k_rows_by_eval_score(
                    better_rows,
                    args.class1_positive_top_k,
                ):
                    row["task_name"] = task_name
                    add_positive(tree_id, tree_class, 1, row)

        elif tree_class == -1:
            class1_tree_ids.add(tree_id)
            default_score = extract_final_score_tuple(default_row)
            if default_score is not None and default_row["__norm_label"] == 0:
                better_than_nondefault = False
                for row in failed_nondefault_rows:
                    nd_score = extract_final_score_tuple(row)
                    if nd_score is None:
                        continue
                    if is_nondefault_better_by_eval(default_score, nd_score):
                        better_than_nondefault = True
                        break
                if better_than_nondefault:
                    default_row["task_name"] = task_name
                    add_positive(tree_id, tree_class, 1, default_row)

        elif tree_class == 2:
            class2_tree_ids.add(tree_id)
            for row in passed_nondefault_rows:
                row["task_name"] = task_name
                add_positive(tree_id, tree_class, 2, row)

            passed_with_round = []
            for row in passed_nondefault_rows:
                round_id = extract_final_round_id(row)
                if round_id is not None:
                    passed_with_round.append((round_id, row))
            passed_sorted = sorted(passed_with_round, key=lambda item: item[0])
            if len(passed_sorted) >= 2:
                for better_round, better_row in passed_sorted[:-1]:
                    if any(
                        (worse_round is not None) and better_round < worse_round
                        for worse_round, _ in passed_sorted
                    ):
                        better_row["task_name"] = task_name
                        add_positive(tree_id, tree_class, 3, better_row)

        elif tree_class == -2:
            class2_tree_ids.add(tree_id)
            if default_passed and failed_nondefault_rows:
                default_row["task_name"] = task_name
                add_positive(tree_id, tree_class, 2, default_row)

        elif tree_class == 3:
            class3_tree_ids.add(tree_id)
            default_round = extract_final_round_id(default_row)
            if default_round is not None:
                for nd in passed_nondefault_rows:
                    nd_round = extract_final_round_id(nd)
                    if nd_round is None:
                        continue
                    if nd_round < default_round:
                        nd["task_name"] = task_name
                        add_positive(tree_id, tree_class, 3, nd)

            if passed_nondefault_rows and failed_nondefault_rows:
                for row in passed_nondefault_rows:
                    row["task_name"] = task_name
                    add_positive(tree_id, tree_class, 2, row)

        elif tree_class == -3:
            class3_tree_ids.add(tree_id)
            default_round = extract_final_round_id(default_row)
            if default_round is not None and default_passed:
                has_slower_nondefault_pass = False
                for nd in passed_nondefault_rows:
                    nd_round = extract_final_round_id(nd)
                    if nd_round is None:
                        continue
                    if default_round < nd_round:
                        has_slower_nondefault_pass = True
                        break
                if has_slower_nondefault_pass:
                    default_row["task_name"] = task_name
                    add_positive(tree_id, tree_class, 3, default_row)

                if failed_nondefault_rows:
                    default_row["task_name"] = task_name
                    add_positive(tree_id, tree_class, 2, default_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as wf:
        for row in out_rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"input={input_path}")
    print(f"total_nonempty_records={total_nonempty}")
    print(f"parsed_records={parsed_records}")
    print(f"invalid_json_records={invalid_json_records}")
    print(f"excluded_records_by_eval_validity={excluded_records}")
    print(f"valid_tree_count={len(valid_tree_ids)}")
    print(f"trees_skipped_missing_tree_class_id={trees_skipped_missing_tree_class_id}")
    print(f"trees_skipped_inconsistent_tree_class_id={trees_skipped_inconsistent_tree_class_id}")
    print(f"trees_fallback_inferred_tree_class_id={trees_fallback_inferred_tree_class_id}")
    print(f"class1_tree_count={len(class1_tree_ids)}")
    print(f"class2_tree_count={len(class2_tree_ids)}")
    print(f"class3_tree_count={len(class3_tree_ids)}")
    print(f"class1_positive_rows={class1_rows}")
    print(f"class2_positive_rows={class2_rows}")
    print(f"class3_positive_rows={class3_rows}")
    print(f"output_rows={len(out_rows)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
