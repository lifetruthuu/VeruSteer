#!/usr/bin/env python3
"""
在全部测试树上，把某个选择策略 rollout 到的路径与 default 路径逐树配对比较。

与 eval_sasrec_log_v3.py 的 Retention / Exploration 不同，这里不按空间取子集，分母是
该 pipeline 下所有含 default 路径的测试树：
- 验证结果四格：n01 = default 失败而策略成功（新增），n10 = default 成功而策略失败（丢失），
  净提升 = (n01 - n10) / N，95% 置信区间按 task 做聚类 bootstrap
- 字典序胜/平/负：验证通过 > 轮次更短 > 分数更高，只统计多路径树（单路径树所有策略都一样）

同名 --log 出现多次时视为不同随机种子，汇总行取平均。

python commands/eval_paired_outcomes.py \
  --merged-path data/paths/test_verusage.jsonl \
  --method verusage \
  --log VeruSteer=results/rq1/verusage.log \
  --log Random=results/rq2/random_s21.log \
  --log Random=results/rq2/random_s22.log \
  --save-csv analysis_results/paired_outcomes_verusage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_sasrec_log_v3 import (
    CATEGORY_ORDER,
    classify_tree_id,
    extract_final_round_id,
    extract_final_score_tuple,
    is_nondefault_better_by_eval,
    is_passed,
    parse_json_line,
)

Outcome = Tuple[bool, Optional[int], Optional[Tuple[float, float]]]


def compare_outcome(a: Outcome, b: Outcome) -> int:
    """a 优于 b 返回 1，劣于返回 -1，无法区分返回 0。"""
    if a[0] != b[0]:
        return 1 if a[0] else -1
    if a[0]:
        if a[1] is None or b[1] is None:
            return 0
        return (a[1] < b[1]) - (a[1] > b[1])
    if a[2] is None or b[2] is None:
        return 0
    if is_nondefault_better_by_eval(a[2], b[2]):
        return 1
    if is_nondefault_better_by_eval(b[2], a[2]):
        return -1
    return 0


def load_merged(merged_path: Path, method: str):
    outcome_by_key: Dict[Tuple[str, str], Outcome] = {}
    default_pid: Dict[str, str] = {}
    task_by_tree: Dict[str, str] = {}
    num_paths: Counter = Counter()
    with merged_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = parse_json_line(line)
            if obj is None:
                continue
            tid, pid = obj.get("tree_id"), obj.get("path_id")
            if not isinstance(tid, str) or not tid.startswith(method + "-") or not isinstance(pid, str):
                continue
            outcome_by_key[(tid, pid)] = (
                is_passed(obj),
                extract_final_round_id(obj),
                extract_final_score_tuple(obj),
            )
            num_paths[tid] += 1
            task_by_tree[tid] = obj["tree_key"]["task_name"]
            if obj.get("default") is True or str(obj.get("default")).lower() == "true":
                default_pid[tid] = pid
    return outcome_by_key, default_pid, task_by_tree, num_paths


def load_reached(model_log: Path) -> Dict[str, str]:
    reached: Dict[str, str] = {}
    for line in model_log.open("r", encoding="utf-8"):
        obj = parse_json_line(line)
        if obj is None:
            continue
        tid, rpid = obj.get("tree_id"), obj.get("reached_path_id")
        if isinstance(tid, str) and isinstance(rpid, str):
            reached[tid] = rpid
    return reached


def cluster_bootstrap(per_task: Dict[str, List[int]], n_boot: int, seed: int) -> Tuple[float, float]:
    tasks = sorted(per_task)
    if not tasks:
        return 0.0, 0.0
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        gain = n = 0
        for _ in tasks:
            g, c = per_task[tasks[rng.randrange(len(tasks))]]
            gain += g
            n += c
        values.append(100.0 * gain / n)
    values.sort()
    return values[int(0.025 * n_boot)], values[int(0.975 * n_boot) - 1]


def evaluate_log(model_log: Path, merged, method: str, n_boot: int, seed: int) -> Dict[str, Dict[str, Any]]:
    outcome_by_key, default_pid, task_by_tree, num_paths = merged
    reached = load_reached(model_log)
    stats: Dict[str, Dict[str, Any]] = {}
    per_task: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    missing = 0
    for tid in sorted(default_pid):
        d = outcome_by_key[(tid, default_pid[tid])]
        rpid = reached.get(tid)
        m = outcome_by_key.get((tid, rpid)) if rpid is not None else None
        if m is None:
            missing += 1
            continue
        for cat in (classify_tree_id(tid), "Total"):
            s = stats.setdefault(cat, Counter())
            s["N"] += 1
            s["default_verified"] += d[0]
            s["method_verified"] += m[0]
            s["n01"] += (not d[0]) and m[0]
            s["n10"] += d[0] and (not m[0])
            gain = int(m[0]) - int(d[0])
            per_task[cat][task_by_tree[tid]][0] += gain
            per_task[cat][task_by_tree[tid]][1] += 1
            if num_paths[tid] > 1:
                s["multi_path"] += 1
                c = compare_outcome(m, d)
                s["win" if c > 0 else ("loss" if c < 0 else "tie")] += 1
    out: Dict[str, Dict[str, Any]] = {}
    for cat, s in stats.items():
        lo, hi = cluster_bootstrap(per_task[cat], n_boot, seed)
        row = {k: s.get(k, 0) for k in ("N", "default_verified", "method_verified", "n01", "n10", "multi_path", "win", "tie", "loss")}
        row["net_pp"] = 100.0 * (row["n01"] - row["n10"]) / row["N"]
        row["ci_low_pp"], row["ci_high_pp"] = lo, hi
        row["tasks"] = len(per_task[cat])
        out[cat] = row
    out["Total"]["missing_reached"] = missing
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="全部测试树上的 default vs 策略 配对结果（四格计数、净提升、聚类 bootstrap CI）")
    parser.add_argument("--merged-path", type=Path, required=True, help="测试集 path JSONL")
    parser.add_argument("--method", choices=("autoverus", "verusage"), required=True)
    parser.add_argument("--log", action="append", required=True, help="NAME=PATH，可重复；同名视为多个随机种子")
    parser.add_argument("--bootstrap", type=int, default=10000, help="bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap 随机种子")
    parser.add_argument("--save-csv", type=Path, default=None, help="可选：保存逐配置结果 CSV")
    args = parser.parse_args()

    merged = load_merged(args.merged_path, args.method)
    if not merged[1]:
        print("Error: 没有找到含 default 路径的树", file=sys.stderr)
        sys.exit(1)

    results: Dict[str, List[Tuple[str, Dict[str, Dict[str, Any]]]]] = defaultdict(list)
    for spec in args.log:
        name, _, path = spec.partition("=")
        results[name].append((path, evaluate_log(Path(path), merged, args.method, args.bootstrap, args.seed)))

    cats = [c for c in CATEGORY_ORDER if c.startswith(args.method + "-")] + ["Total"]
    csv_rows = []
    for name, runs in results.items():
        print(f"== {name} ({args.method}, {len(runs)} run(s))")
        for path, res in runs:
            for cat in cats:
                row = res.get(cat)
                if row is None:
                    continue
                csv_rows.append({"selector": name, "log": path, "category": cat, **row})
                if len(runs) == 1:
                    print(
                        f"  {cat:40s} N={row['N']:4d} default={row['default_verified']:3d} method={row['method_verified']:3d} "
                        f"n01={row['n01']:2d} n10={row['n10']:2d} net={row['net_pp']:+.2f}pp "
                        f"CI95=[{row['ci_low_pp']:+.2f}, {row['ci_high_pp']:+.2f}] "
                        f"multi-path W/T/L={row['win']}/{row['tie']}/{row['loss']} (of {row['multi_path']})"
                    )
            if len(runs) == 1 and res["Total"].get("missing_reached"):
                print(f"  [Warn] {res['Total']['missing_reached']} 棵树在日志中没有可用的 reached_path_id")
        if len(runs) > 1:
            totals = [res["Total"] for _, res in runs]
            mean = lambda k: sum(t[k] for t in totals) / len(totals)
            nets = [t["net_pp"] for t in totals]
            print(
                f"  {'Total (mean over runs)':40s} N={totals[0]['N']:4d} default={mean('default_verified'):.1f} "
                f"method={mean('method_verified'):.1f} n01={mean('n01'):.1f} n10={mean('n10'):.1f} "
                f"net={mean('net_pp'):+.2f}pp (range {min(nets):+.2f} .. {max(nets):+.2f}) "
                f"multi-path W/T/L={mean('win'):.1f}/{mean('tie'):.1f}/{mean('loss'):.1f} (of {totals[0]['multi_path']})"
            )

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_csv.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        print(f"[Info] CSV 已保存到: {args.save_csv}")


if __name__ == "__main__":
    main()
