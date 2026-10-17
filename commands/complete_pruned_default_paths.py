#!/usr/bin/env python3
"""给"默认分支被提前剪掉"的树补上默认路径。

AutoVerus 的树搜索里，某个分支的候选连续失败 5 次后会被剪掉（refinement.py 的
`pruned_path_early_stop`，只剪这一个分支）。默认分支恰好在这种情况下，抽取根到叶
路径时它没有叶子，整棵树就没有 default=True 的路径（训练 14 棵、测试 10 棵，都是
AutoVerus，且没有任何一条路径验证成功）。

这些树的默认轨迹没有产生新状态：沿默认边走了 4 轮，每一轮都没有接受补丁（程序不变），
第 5 轮被剪掉。因此默认轨迹就是从根到最后一个默认状态的这一段，最终结果是该状态，
不补造新节点。本脚本给每棵这样的树追加一条路径：

  path_state_ids = 默认边走到的最后一个状态（含根），default = true，
  label 按最后一个状态的分数判定，最后一个状态的选择字段置空（与叶子约定一致），
  path_id 用 extract_root_to_leaf_paths.make_path_id 生成，
  同一棵树所有行的 default_path_id 改为这条路径的 id。

新行只追加在该树已有行之后，所以树的顺序、已有行的内容和顺序都不变；
新行多一个 default_pruned_early_stop 字段作为标记。已含默认路径的树原样输出。
断言默认规则在最后一个状态选的类型没有已记录的后继（即确实是被剪掉的那个分支）。

python commands/complete_pruned_default_paths.py --input X.jsonl --output Y.jsonl
python commands/complete_pruned_default_paths.py --input X.jsonl --check   # 只报告
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_root_to_leaf_paths import (  # noqa: E402
    TREE_KEY_FIELDS,
    is_success_score_tuple,
    make_path_id,
)

PRESET_PRIORITY = ("MismatchedType", "PreCondFailVecLen", "ArithmeticFlow", "InvFailFront")


def default_rule(failure_types):
    """AutoVerus refinement.get_one_failure 的优先级（VeruSAGE 不会出现这种树）。"""
    for name in PRESET_PRIORITY:
        if name in failure_types:
            return name
    return failure_types[0]


def build_default_path(rows):
    """返回 (walk, source_row)，walk 是默认边从根走到的状态序列。"""
    root = rows[0]["root_state_id"]
    children, default_children = defaultdict(set), defaultdict(set)
    for row in rows:
        ids, states = row["path_state_ids"], row["path_states"]
        for i in range(len(ids) - 1):
            children[ids[i]].add(states[i]["chosen_failure_type"])
            if states[i].get("is_default_chose") is True:
                default_children[ids[i]].add(ids[i + 1])
    walk = [root]
    while len(default_children.get(walk[-1], ())) == 1:
        walk.append(next(iter(default_children[walk[-1]])))
    if len(default_children.get(walk[-1], ())) > 1:
        raise ValueError(f"{rows[0]['tree_id']}: ambiguous default edge at {walk[-1]}")
    source = next(r for r in rows if r["path_state_ids"][: len(walk)] == walk)
    last = source["path_states"][len(walk) - 1]
    chosen = default_rule(last["failure_types"])
    if chosen in children[walk[-1]]:
        raise ValueError(f"{rows[0]['tree_id']}: default type {chosen} has a recorded child at {walk[-1]}")
    return walk, source


def complete_tree(rows):
    """返回（可能追加了一行的）rows 和被补全的状态序列（没补则为 None）。"""
    if any(r.get("default") is True for r in rows):
        return rows, None
    walk, source = build_default_path(rows)
    states = copy.deepcopy(source["path_states"][: len(walk)])
    leaf = states[-1]
    leaf["chosen_failure_type"] = None
    leaf["is_default_chose"] = None
    leaf["chosen_failure_info"] = None
    key = tuple(source["tree_key"].get(k) for k in TREE_KEY_FIELDS)
    new_row = {
        "path_id": make_path_id(key, walk),
        "tree_id": source["tree_id"],
        "tree_key": source["tree_key"],
        "root_state_id": walk[0],
        "leaf_state_id": walk[-1],
        "path_length": len(walk),
        "path_state_ids": walk,
        "path_states": states,
        "label": is_success_score_tuple(leaf.get("score_tuple")),
        "default": True,
        "default_path_id": None,
        "default_pruned_early_stop": True,
    }
    out = []
    for row in rows:
        row = dict(row)
        row["default_path_id"] = new_row["path_id"]
        out.append(row)
    new_row["default_path_id"] = new_row["path_id"]
    out.append(new_row)
    return out, walk


def iter_trees(path):
    """按树分块读取；要求同一棵树的行连续。"""
    seen, current, block = set(), None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["tree_id"] != current:
                if block:
                    yield block
                if row["tree_id"] in seen:
                    raise ValueError(f"rows of tree {row['tree_id']} are not contiguous")
                seen.add(row["tree_id"])
                current, block = row["tree_id"], []
            block.append(row)
    if block:
        yield block


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="只报告，不写文件")
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error("需要 --output，或使用 --check")
    if args.output is not None and args.output.resolve() == args.input.resolve():
        parser.error("--output 不能与 --input 相同")

    trees = completed = rows_in = rows_out = 0
    out = None if args.check else open(args.output, "w", encoding="utf-8")
    try:
        for block in iter_trees(args.input):
            trees += 1
            rows_in += len(block)
            new_block, walk = complete_tree(block)
            if walk is not None:
                completed += 1
                print(f"completed {block[0]['tree_id']}: default path of {len(walk)} states ending at {walk[-1]}")
            rows_out += len(new_block)
            if out is not None:
                for row in new_block:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if out is not None:
            out.close()
    print(f"trees={trees} completed={completed} rows {rows_in} -> {rows_out}")


if __name__ == "__main__":
    main()
