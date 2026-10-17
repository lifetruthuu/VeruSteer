#!/usr/bin/env python3
"""
不依赖模型的错误类型选择基线，在测试树上做整树 rollout，输出与
eval_tree_better_path.py 相同格式的选择日志（可直接交给 commands/eval_paired_outcomes.py
或 commands/eval_sasrec_log_v3.py）。

三种策略：
- default：每个状态选 is_default_chose 的动作，用来核对 rollout 能否还原默认路径
- static ：从训练树学一张全局优先级表（每个错误类型在同一分叉状态下的胜率）
- combo  ：按当前候选错误组合查训练树里胜率最高的动作，组合没见过时退回 static

胜负的定义：同一分叉状态下两个动作，各自子树里最好的叶子按
“验证通过 > 轮次更短 > 分数更高”比较（与论文 Eq. trajectory-objective 一致）。
表只按测试树所属 pipeline 的训练树学习。

python scripts/eval_tree_rule_baselines.py \
  --train-path-jsonl data/paths/train_verusage.jsonl \
  --test-path-jsonl data/paths/test_verusage.jsonl \
  --method verusage --selector-mode combo \
  --save-table results/rollout_logs/baselines/rule_table_verusage.json \
  --save-selection-log results/rollout_logs/baselines/combo_verusage.log
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from eval_tree_better_path import TreeBuild, build_tree_structures, is_nondefault_better_by_eval


Outcome = Tuple[bool, int, Tuple[float, float]]


def compare_outcome(a: Outcome, b: Outcome) -> int:
    """a 优于 b 返回 1，劣于返回 -1，无法区分返回 0。"""
    if a[0] != b[0]:
        return 1 if a[0] else -1
    if a[0]:
        return (a[1] < b[1]) - (a[1] > b[1])
    if is_nondefault_better_by_eval(a[2], b[2]):
        return 1
    if is_nondefault_better_by_eval(b[2], a[2]):
        return -1
    return 0


def path_outcome(row: Dict[str, Any]) -> Optional[Outcome]:
    states = row.get("path_states") or []
    if not states:
        return None
    last = states[-1]
    score = last.get("score_tuple")
    if not (isinstance(score, list) and len(score) >= 2):
        return None
    return bool(row.get("label")), int(last.get("round_id")), (float(score[0]), float(score[1]))


def canonical_combination(names: List[str]) -> str:
    return " + ".join(sorted(set(names)))


def learn_tables(train_path_jsonl: str, method: str) -> Dict[str, Any]:
    # 只保留学习需要的字段：每个 (tree, state, action) 子树里最好的叶子结果
    best: Dict[Tuple[str, str, str], Outcome] = {}
    with open(train_path_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading train paths", unit="path"):
            row = json.loads(line)
            tree_id = row.get("tree_id", "")
            if not tree_id.startswith(method + "-"):
                continue
            outcome = path_outcome(row)
            if outcome is None:
                continue
            for sid, state in zip(row["path_state_ids"], row["path_states"]):
                action = state.get("chosen_failure_type")
                if not isinstance(action, str):
                    continue
                key = (tree_id, sid, action)
                if key not in best or compare_outcome(outcome, best[key]) > 0:
                    best[key] = outcome

    actions_by_state: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for tree_id, sid, action in best:
        actions_by_state[(tree_id, sid)].append(action)

    global_wl: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    combo_wl: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    branching_states = 0
    for (tree_id, sid), actions in actions_by_state.items():
        if len(actions) < 2:
            continue
        branching_states += 1
        combo = canonical_combination(actions)
        for i, a in enumerate(actions):
            for b in actions[i + 1:]:
                c = compare_outcome(best[(tree_id, sid, a)], best[(tree_id, sid, b)])
                if c == 0:
                    continue
                winner, loser = (a, b) if c > 0 else (b, a)
                global_wl[winner][0] += 1
                global_wl[loser][1] += 1
                combo_wl[combo][winner][0] += 1
                combo_wl[combo][loser][1] += 1

    return {
        "method": method,
        "branching_states": branching_states,
        "global": {t: {"wins": w, "losses": l} for t, (w, l) in sorted(global_wl.items())},
        "combo": {
            combo: {t: {"wins": w, "losses": l} for t, (w, l) in sorted(table.items())}
            for combo, table in sorted(combo_wl.items())
        },
    }


def smoothed_rate(entry: Optional[Dict[str, int]]) -> float:
    # Laplace 平滑：没有比较证据的类型记 0.5
    if not entry:
        return 0.5
    return (entry["wins"] + 1) / (entry["wins"] + entry["losses"] + 2)


def build_default_actions(tree: TreeBuild) -> Dict[str, str]:
    default_action: Dict[str, str] = {}
    for row in tree.paths:
        for sid, state in zip(row.get("path_state_ids") or [], row.get("path_states") or []):
            action = state.get("chosen_failure_type")
            if isinstance(action, str) and state.get("is_default_chose") is True:
                default_action.setdefault(sid, action)
    return default_action


def choose_action(
    mode: str,
    candidate_names: List[str],
    default_action: Optional[str],
    tables: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], str]:
    if mode == "default":
        return (default_action if default_action in candidate_names else None), "default"

    global_table = tables["global"]
    # 平分时优先保留默认动作，再按名字，保证确定性
    def key_static(name: str) -> Tuple[float, int, str]:
        return smoothed_rate(global_table.get(name)), int(name == default_action), name

    if mode == "combo":
        combo_table = tables["combo"].get(canonical_combination(candidate_names))
        if combo_table:
            def key_combo(name: str) -> Tuple[float, float, int, str]:
                return (smoothed_rate(combo_table.get(name)),) + key_static(name)
            return max(candidate_names, key=key_combo), "combo_table"
    return max(candidate_names, key=key_static), "static_table"


def rule_rollout_to_leaf(
    tree: TreeBuild, mode: str, tables: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not tree.paths:
        return {"status": "empty_tree"}

    root_state_id = tree.paths[0].get("root_state_id")
    if not isinstance(root_state_id, str) or root_state_id not in tree.state_by_id:
        return {"status": "no_root"}

    default_action_by_state = build_default_actions(tree)
    state_ids = [root_state_id]
    steps = 0
    dead_end = False
    dead_end_reason: Optional[str] = None
    selection_trace: List[Dict[str, Any]] = []

    while True:
        current_state_id = state_ids[-1]
        current_state = tree.state_by_id[current_state_id]

        # 与 eval_tree_better_path.py 相同：候选 = failure_types ∩ 树中实际可转移的动作
        candidate_names = list(current_state.get("failure_types", []) or [])
        if not candidate_names:
            repaired = current_state.get("repaired_failure_list", []) or []
            candidate_names = [x.get("error_type", "") for x in repaired if x.get("error_type")]
        valid_actions = set(tree.valid_actions_by_state.get(current_state_id, []))
        candidate_names = [name for name in candidate_names if name in valid_actions]

        if not candidate_names:
            selection_trace.append(
                {
                    "step": steps,
                    "current_state_id": current_state_id,
                    "candidate_names": [],
                    "selected_name": None,
                    "next_state_id": None,
                    "dead_end_reason": "no_candidates",
                }
            )
            break

        pred_name, reason = choose_action(
            mode, candidate_names, default_action_by_state.get(current_state_id), tables
        )
        step_log: Dict[str, Any] = {
            "step": steps,
            "current_state_id": current_state_id,
            "candidate_names": list(candidate_names),
            "selected_name": pred_name,
            "selected_reason": reason,
            "default_name": default_action_by_state.get(current_state_id),
            "next_state_id": None,
            "dead_end_reason": None,
        }
        steps += 1

        next_state_id = tree.children_by_state_and_action.get((current_state_id, pred_name)) if pred_name else None
        if pred_name is None:
            dead_end_reason = "selector_returned_none"
        elif next_state_id is None:
            dead_end_reason = "no_transition_for_selected_action"
        elif next_state_id in state_ids:
            dead_end_reason = "loop_detected"
        elif next_state_id not in tree.state_by_id:
            dead_end_reason = "next_state_missing"
        if dead_end_reason is not None:
            dead_end = True
            step_log["next_state_id"] = next_state_id
            step_log["dead_end_reason"] = dead_end_reason
            selection_trace.append(step_log)
            break

        step_log["next_state_id"] = next_state_id
        selection_trace.append(step_log)
        state_ids.append(next_state_id)

    return {
        "status": "ok",
        "steps": steps,
        "dead_end": dead_end,
        "reached_path_id": tree.path_id_by_state_tuple.get(tuple(state_ids)),
        "reached_leaf_state_id": state_ids[-1],
        "state_ids": state_ids,
        "dead_end_reason": dead_end_reason,
        "selection_trace": selection_trace,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="静态优先级 / 错误组合规则基线的整树 rollout 评估")
    parser.add_argument("--test-path-jsonl", required=True, help="测试集 path jsonl(每行一条根到叶路径)")
    parser.add_argument("--selector-mode", choices=("default", "static", "combo"), required=True)
    parser.add_argument("--method", choices=("autoverus", "verusage"), required=True, help="只评估并学习该 pipeline 的树")
    parser.add_argument("--train-path-jsonl", default=None, help="训练集 path jsonl；static/combo 需要（或用 --load-table）")
    parser.add_argument("--load-table", default=None, help="可选：直接读取已学好的规则表 JSON")
    parser.add_argument("--save-table", default=None, help="可选：保存学到的规则表 JSON")
    parser.add_argument("--save-selection-log", default=None, help="可选：保存每棵树选择轨迹日志到 JSONL 文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tables = None
    if args.selector_mode != "default":
        if args.load_table:
            with open(args.load_table, "r", encoding="utf-8") as f:
                tables = json.load(f)
        elif args.train_path_jsonl:
            tables = learn_tables(args.train_path_jsonl, args.method)
        else:
            print("[Error] static/combo 需要 --train-path-jsonl 或 --load-table")
            return 1
        if tables.get("method") != args.method:
            print(f"[Error] 规则表来自 {tables.get('method')}，与 --method {args.method} 不一致")
            return 1
        print(f"[Info] 训练树分叉状态数: {tables['branching_states']}, 错误组合数: {len(tables['combo'])}")
        if args.save_table:
            os.makedirs(os.path.dirname(args.save_table) or ".", exist_ok=True)
            with open(args.save_table, "w", encoding="utf-8") as wf:
                json.dump(tables, wf, ensure_ascii=False, indent=2)
            print(f"[Info] 规则表已保存到: {args.save_table}")

    tree_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(args.test_path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tree_id = obj.get("tree_id")
            if isinstance(tree_id, str) and tree_id.startswith(args.method + "-"):
                tree_rows[tree_id].append(obj)

    if not tree_rows:
        print("[Error] 未读取到有效树")
        return 1

    reached_default = 0
    dead_ends: Dict[str, int] = defaultdict(int)
    selection_log_wf = None
    if args.save_selection_log:
        os.makedirs(os.path.dirname(args.save_selection_log) or ".", exist_ok=True)
        selection_log_wf = open(args.save_selection_log, "w", encoding="utf-8")

    try:
        for tree_id in tqdm(sorted(tree_rows.keys()), desc="Evaluating trees", unit="tree"):
            rows = tree_rows[tree_id]
            tree = build_tree_structures(tree_id, rows)
            rollout = rule_rollout_to_leaf(tree, args.selector_mode, tables)
            reached_path_id = rollout.get("reached_path_id")
            if reached_path_id is not None and reached_path_id == rows[0].get("default_path_id"):
                reached_default += 1
            if rollout.get("dead_end"):
                dead_ends[rollout.get("dead_end_reason")] += 1

            if selection_log_wf is not None:
                reached_path_better_class = tree.better_classes.get(reached_path_id)
                selection_record = {
                    "tree_id": tree_id,
                    "tree_class_id": tree.tree_class_id,
                    "selector_mode": args.selector_mode,
                    "num_paths": len(rows),
                    "num_better_paths": len(tree.better_path_ids),
                    "better_path_ids": sorted(tree.better_path_ids),
                    "rollout_hit_better": bool(reached_path_id in tree.better_path_ids),
                    "reached_path_id": reached_path_id,
                    "reached_path_better_class": (
                        reached_path_better_class if reached_path_better_class in (1, -1, 2, -2, 3, -3) else "none"
                    ),
                    "reached_leaf_state_id": rollout.get("reached_leaf_state_id"),
                    "dead_end": rollout.get("dead_end"),
                    "dead_end_reason": rollout.get("dead_end_reason"),
                    "steps": rollout.get("steps"),
                    "selection_trace": rollout.get("selection_trace", []),
                }
                selection_log_wf.write(json.dumps(selection_record, ensure_ascii=False) + "\n")
    finally:
        if selection_log_wf is not None:
            selection_log_wf.close()

    print("=" * 100)
    print(f"[{args.selector_mode}] 总树数: {len(tree_rows)}")
    print(f"到达 default 路径的树数: {reached_default}")
    print(f"dead end: {dict(dead_ends)}")
    if args.save_selection_log:
        print(f"[Info] selection log 已保存到: {args.save_selection_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
