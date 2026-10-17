#!/usr/bin/env python3
"""
Extract root-to-leaf state sequences from merged_result.jsonl.

Input:
  - merged_result.jsonl where each line represents one transition edge:
    source_state_id -> repaired_state_id

Output:
  - jsonl file, one line per root-to-leaf path.


python extract_root_to_leaf_paths.py \
  --input data/verusage/all.jsonl \
  --output data/verusage_merged_path.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TREE_KEY_FIELDS = (
    "method",
    "dataset_name",
    "task_name",
    "llm_name",
    "shot",
    "generation_candidate_id",
    "repair_llm_name",
)


ROUND_ID_RE = re.compile(r"^r(\d+):")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract root-to-leaf state sequences from merged_result.jsonl."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to merged_result.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output jsonl path. Default: <input_dir>/root_to_leaf_paths.jsonl",
    )
    return parser.parse_args()


def get_tree_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in TREE_KEY_FIELDS)


def round_id_from_state_id(state_id: str | None) -> int | None:
    if not state_id:
        return None
    match = ROUND_ID_RE.match(state_id)
    return int(match.group(1)) if match else None


def make_state_info(
    *,
    state_id: str,
    code_path: str | None,
    code_text: str | None,
    score_tuple: Any,
    failure_types: Any,
    repaired_failure_list: Any,
    verus_output_full: str | None,
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "round_id": round_id_from_state_id(state_id),
        "code_path": code_path,
        "code_text": code_text,
        "score_tuple": score_tuple,
        "failure_types": failure_types,
        "repaired_failure_list": repaired_failure_list,
        "verus_output_full": verus_output_full,
    }


def make_tree_id(tree_key: tuple[Any, ...]) -> str:
    return "-".join("" if v is None else str(v) for v in tree_key)


def make_path_id(tree_key: tuple[Any, ...], state_ids: list[str]) -> str:
    payload = json.dumps(
        {"tree_key": tree_key, "path_state_ids": state_ids},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def is_success_score_tuple(score_tuple: Any) -> bool:
    return (
        isinstance(score_tuple, list)
        and len(score_tuple) >= 2
        and isinstance(score_tuple[0], (int, float))
        and isinstance(score_tuple[1], (int, float))
        and score_tuple[0] > 0
        and score_tuple[1] == 0
    )


def parse_json_objects_from_line(line: str, line_no: int) -> list[dict[str, Any]]:
    """Parse one or more concatenated JSON objects from a single line."""
    decoder = json.JSONDecoder()
    pos = 0
    n = len(line)
    rows: list[dict[str, Any]] = []

    while pos < n:
        while pos < n and line[pos].isspace():
            pos += 1
        if pos >= n:
            break
        try:
            value, end = decoder.raw_decode(line, pos)
        except json.JSONDecodeError as exc:
            snippet_start = max(0, pos - 40)
            snippet_end = min(n, pos + 120)
            snippet = line[snippet_start:snippet_end]
            raise ValueError(
                f"Invalid JSON at line {line_no}, char {pos + 1}: {exc}. "
                f"Nearby content: {snippet!r}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid JSON at line {line_no}, char {pos + 1}: "
                f"expected object, got {type(value).__name__}"
            )
        rows.append(value)
        pos = end

    return rows


def emit_state_snapshot(
    row: dict[str, Any],
    state_info: dict[str, dict[str, Any]],
) -> None:
    source_state_id = row["source_state_id"]
    repaired_state_id = row["repaired_state_id"]

    if source_state_id not in state_info:
        state_info[source_state_id] = make_state_info(
            state_id=source_state_id,
            code_path=row.get("source_code_path"),
            code_text=row.get("source_code_text"),
            score_tuple=row.get("source_score_tuple"),
            failure_types=row.get("source_failure_types"),
            repaired_failure_list=row.get("source_failure_list"),
            verus_output_full=row.get("source_verus_output_full"),
        )

    # A leaf state often appears only as repaired_state_id.
    if repaired_state_id not in state_info:
        state_info[repaired_state_id] = make_state_info(
            state_id=repaired_state_id,
            code_path=row.get("repaired_code_path"),
            code_text=row.get("repaired_code_text"),
            score_tuple=row.get("repaired_score_tuple"),
            failure_types=row.get("repaired_failure_types"),
            repaired_failure_list=row.get("repaired_failure_list"),
            verus_output_full=row.get("repaired_verus_output_full"),
        )


def build_paths_for_group(
    rows: list[dict[str, Any]],
    tree_key: tuple[Any, ...],
) -> list[dict[str, Any]]:
    children: dict[str, list[str]] = defaultdict(list)
    edge_meta: dict[tuple[str, str], dict[str, Any]] = {}
    incoming_count: dict[str, int] = defaultdict(int)
    all_nodes: set[str] = set()
    state_info: dict[str, dict[str, Any]] = {}

    for row in rows:
        src = row["source_state_id"]
        dst = row["repaired_state_id"]
        all_nodes.add(src)
        all_nodes.add(dst)
        children[src].append(dst)
        edge_meta[(src, dst)] = {
            "chosen_failure_type": row.get("chosen_failure_type"),
            "is_default_chose": row.get("is_default_chose"),
        }
        incoming_count[dst] += 1
        incoming_count.setdefault(src, incoming_count.get(src, 0))
        emit_state_snapshot(row, state_info)

    roots = [node for node in all_nodes if incoming_count.get(node, 0) == 0]
    leaves = {node for node in all_nodes if len(children.get(node, [])) == 0}
    path_rows: list[dict[str, Any]] = []

    def build_path_states(state_ids: list[str]) -> list[dict[str, Any]]:
        def find_chosen_failure_info(
            repaired_failure_list: Any, chosen_failure_type: Any
        ) -> Any:
            if not chosen_failure_type or not isinstance(repaired_failure_list, list):
                return None
            for item in repaired_failure_list:
                if isinstance(item, dict):
                    item_failure_type = (
                        item.get("error_type")
                        or item.get("failure_type")
                        or item.get("type")
                    )
                    if item_failure_type == chosen_failure_type:
                        return item.get("trace_text")
            return None

        path_states: list[dict[str, Any]] = []
        for idx, state_id in enumerate(state_ids):
            state_obj = dict(state_info[state_id])
            # Node-level choice metadata should represent the choice made AT this node,
            # i.e., the outgoing edge from current node to next node on the path.
            # Therefore: root has values, leaf has null.
            if idx == len(state_ids) - 1:
                state_obj["chosen_failure_type"] = None
                state_obj["is_default_chose"] = None
            else:
                next_state_id = state_ids[idx + 1]
                meta = edge_meta.get((state_id, next_state_id), {})
                state_obj["chosen_failure_type"] = meta.get("chosen_failure_type")
                state_obj["is_default_chose"] = meta.get("is_default_chose")
            state_obj["chosen_failure_info"] = find_chosen_failure_info(
                state_obj.get("repaired_failure_list"),
                state_obj.get("chosen_failure_type"),
            )
            path_states.append(state_obj)
        return path_states

    def dfs(node: str, stack: list[str], seen: set[str]) -> None:
        if node in seen:
            # Defensive: avoid infinite loops if data is not a strict tree.
            return
        stack.append(node)
        seen.add(node)

        if node in leaves:
            key_obj = dict(zip(TREE_KEY_FIELDS, tree_key))
            state_ids = stack.copy()
            path_states = build_path_states(state_ids)
            label = is_success_score_tuple(path_states[-1].get("score_tuple"))
            non_leaf_defaults = [
                state.get("is_default_chose") for state in path_states[:-1]
            ]
            all_default = len(non_leaf_defaults) > 0 and all(
                value is True for value in non_leaf_defaults
            )
            path_rows.append(
                {
                    "path_id": make_path_id(tree_key, state_ids),
                    "tree_id": make_tree_id(tree_key),
                    "tree_key": key_obj,
                    "root_state_id": state_ids[0],
                    "leaf_state_id": state_ids[-1],
                    "path_length": len(state_ids),
                    "path_state_ids": state_ids,
                    "path_states": path_states,
                    "label": label,
                    "default": all_default,
                }
            )
        else:
            for nxt in children.get(node, []):
                dfs(nxt, stack, seen)

        seen.remove(node)
        stack.pop()

    for root in sorted(roots):
        dfs(root, [], set())

    # Each path should carry the default path id of its tree.
    default_path_ids = sorted(
        row["path_id"] for row in path_rows if row.get("default") is True
    )
    default_path_id = default_path_ids[0] if default_path_ids else None
    for row in path_rows:
        row["default_path_id"] = default_path_id

    return path_rows


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.parent / "root_to_leaf_paths.jsonl"
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8") as rf:
        for idx, line in enumerate(rf, start=1):
            line = line.strip()
            if not line:
                continue
            for row in parse_json_objects_from_line(line, idx):
                groups[get_tree_key(row)].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_paths = 0
    with output_path.open("w", encoding="utf-8") as wf:
        for tree_key, rows in groups.items():
            path_rows = build_paths_for_group(rows, tree_key)
            for out_row in path_rows:
                wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                total_paths += 1

    print(f"[DONE] trees={len(groups)}")
    print(f"[DONE] paths={total_paths}")
    print(f"[DONE] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
