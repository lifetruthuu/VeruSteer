#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

'''
python stats_repair_dataset.py \
  --input /recorded/host-a/workspace/ErrorRanker/autoverus_data/VeriCoding_deepseek-chat_few-shot/repair-dataset-merged-20260304-100000.jsonl \
  --topk 10

'''


COMPILE_FAIL_SCORE = (-1, -1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid json on line {i}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} is not a JSON object")
            rows.append(obj)
    return rows


def tuple2(v: Any) -> tuple[int, int] | None:
    if not isinstance(v, list) or len(v) != 2:
        return None
    if not all(isinstance(x, int) for x in v):
        return None
    return int(v[0]), int(v[1])


def is_compile_ok(score: tuple[int, int] | None) -> bool:
    return score is not None and score != COMPILE_FAIL_SCORE


def mean_median(nums: list[int]) -> dict[str, float | int | None]:
    if not nums:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(nums), 4),
        "median": statistics.median(nums),
        "min": min(nums),
        "max": max(nums),
    }


def safe_rate(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return round(num / den, 4)


def score_relation(
    source: tuple[int, int] | None, repaired: tuple[int, int] | None
) -> str:
    src_ok = is_compile_ok(source)
    rep_ok = is_compile_ok(repaired)

    if not src_ok and not rep_ok:
        return "both_compile_fail"
    if not src_ok and rep_ok:
        return "compile_fail_to_ok"
    if src_ok and not rep_ok:
        return "compile_ok_to_fail"

    assert source is not None and repaired is not None
    sv, se = source
    rv, re = repaired
    verify_cmp = (rv > sv) - (rv < sv)
    error_cmp = (re < se) - (re > se)  # lower error is better

    if verify_cmp == 0 and error_cmp == 0:
        return "both_ok_unchanged"
    if verify_cmp >= 0 and error_cmp >= 0 and (verify_cmp > 0 or error_cmp > 0):
        return "both_ok_pareto_better"
    if verify_cmp <= 0 and error_cmp <= 0 and (verify_cmp < 0 or error_cmp < 0):
        return "both_ok_pareto_worse"
    return "both_ok_tradeoff"


def counter_top(counter: Counter[str], topk: int) -> list[dict[str, Any]]:
    return [{"key": k, "count": c} for k, c in counter.most_common(topk)]


def compare_non_default_against_default(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def source_prefix_from_id(record_id: Any) -> str | None:
        if not isinstance(record_id, str):
            return None
        if "-" not in record_id:
            return None
        return record_id.rsplit("-", 1)[0]

    default_rows_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("is_default_chose") is not True:
            continue
        default_path = r.get("default_path")
        key = default_path if isinstance(default_path, str) and default_path else source_prefix_from_id(r.get("id"))
        if isinstance(key, str) and key:
            default_rows_by_path[key].append(r)

    total_non_default = 0
    matched_non_default = 0
    relation_counter: Counter[str] = Counter()
    comparison_case_counter: Counter[str] = Counter()
    non_default_key_source_counter: Counter[str] = Counter()

    for r in rows:
        if r.get("is_default_chose") is not False:
            continue
        total_non_default += 1

        default_path = r.get("default_path")
        if isinstance(default_path, str) and default_path:
            key = default_path
            non_default_key_source_counter["from_default_path"] += 1
        else:
            key = source_prefix_from_id(r.get("id"))
            if isinstance(key, str) and key:
                non_default_key_source_counter["from_id_prefix"] += 1
            else:
                comparison_case_counter["missing_match_key"] += 1
                continue

        if not isinstance(key, str) or not key:
            comparison_case_counter["missing_match_key"] += 1
            continue

        default_rows = default_rows_by_path.get(key, [])
        if len(default_rows) == 0:
            comparison_case_counter["no_default_match_found"] += 1
            continue
        if len(default_rows) > 1:
            comparison_case_counter["multiple_default_matches"] += 1

        default_row = default_rows[0]
        matched_non_default += 1

        default_score = tuple2(default_row.get("repaired_score_tuple"))
        non_default_score = tuple2(r.get("repaired_score_tuple"))
        relation = score_relation(default_score, non_default_score)
        relation_counter[relation] += 1

    better_count = relation_counter["compile_fail_to_ok"] + relation_counter["both_ok_pareto_better"]
    worse_count = relation_counter["compile_ok_to_fail"] + relation_counter["both_ok_pareto_worse"]
    unchanged_count = relation_counter["both_ok_unchanged"] + relation_counter["both_compile_fail"]
    tradeoff_count = relation_counter["both_ok_tradeoff"]
    non_worse_count = better_count + unchanged_count

    return {
        "total_non_default_records": total_non_default,
        "matched_non_default_records": matched_non_default,
        "matched_rate": safe_rate(matched_non_default, total_non_default),
        "unmatched_non_default_records": total_non_default - matched_non_default,
        "comparison_relation_counts": dict(relation_counter),
        "comparison_case_counts": dict(comparison_case_counter),
        "non_default_match_key_source_counts": dict(non_default_key_source_counter),
        "better_count": better_count,
        "better_rate_among_matched": safe_rate(better_count, matched_non_default),
        "worse_count": worse_count,
        "worse_rate_among_matched": safe_rate(worse_count, matched_non_default),
        "unchanged_count": unchanged_count,
        "unchanged_rate_among_matched": safe_rate(unchanged_count, matched_non_default),
        "tradeoff_count": tradeoff_count,
        "tradeoff_rate_among_matched": safe_rate(tradeoff_count, matched_non_default),
        "non_worse_count": non_worse_count,
        "non_worse_rate_among_matched": safe_rate(non_worse_count, matched_non_default),
    }


def summarize(rows: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"total_records": 0}

    field_counter: dict[str, Counter[str]] = {
        "method": Counter(),
        "dataset_name": Counter(),
        "task_name": Counter(),
        "llm_name": Counter(),
        "shot": Counter(),
        "chosen_failure_type": Counter(),
        "source_failure_types": Counter(),
        "repaired_failure_types": Counter(),
    }

    round_id_counter: Counter[int] = Counter()
    is_default_counter: Counter[bool] = Counter()
    fallback_reason_counter: Counter[str] = Counter()

    src_compile_ok = 0
    rep_compile_ok = 0
    src_verify_values: list[int] = []
    src_error_values: list[int] = []
    rep_verify_values: list[int] = []
    rep_error_values: list[int] = []
    verify_delta_both_ok: list[int] = []
    error_delta_both_ok: list[int] = []

    relation_counter: Counter[str] = Counter()
    id_set = set()
    generation_candidate_id_set = set()

    source_code_len: list[int] = []
    repaired_code_len: list[int] = []
    source_output_len: list[int] = []
    repaired_output_len: list[int] = []
    code_len_delta: list[int] = []
    output_len_delta: list[int] = []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        if isinstance(r.get("id"), str):
            id_set.add(r["id"])
        if isinstance(r.get("generation_candidate_id"), str):
            generation_candidate_id_set.add(r["generation_candidate_id"])

        for k in ("method", "dataset_name", "task_name", "llm_name", "shot", "chosen_failure_type"):
            v = r.get(k)
            if isinstance(v, str):
                field_counter[k][v] += 1

        if isinstance(r.get("round_id"), int):
            round_id_counter[r["round_id"]] += 1
        if isinstance(r.get("is_default_chose"), bool):
            is_default_counter[r["is_default_chose"]] += 1
        if isinstance(r.get("fallback_reason"), str) and r["fallback_reason"].strip():
            fallback_reason_counter[r["fallback_reason"]] += 1

        source_types = r.get("source_failure_types")
        if isinstance(source_types, list):
            for t in source_types:
                if isinstance(t, str):
                    field_counter["source_failure_types"][t] += 1

        repaired_types = r.get("repaired_failure_types")
        if isinstance(repaired_types, list):
            for t in repaired_types:
                if isinstance(t, str):
                    field_counter["repaired_failure_types"][t] += 1

        src_score = tuple2(r.get("source_score_tuple"))
        rep_score = tuple2(r.get("repaired_score_tuple"))
        relation = score_relation(src_score, rep_score)
        relation_counter[relation] += 1

        src_ok = is_compile_ok(src_score)
        rep_ok = is_compile_ok(rep_score)
        if src_ok and src_score is not None:
            src_compile_ok += 1
            src_verify_values.append(src_score[0])
            src_error_values.append(src_score[1])
        if rep_ok and rep_score is not None:
            rep_compile_ok += 1
            rep_verify_values.append(rep_score[0])
            rep_error_values.append(rep_score[1])

        if src_ok and rep_ok and src_score is not None and rep_score is not None:
            verify_delta_both_ok.append(rep_score[0] - src_score[0])
            error_delta_both_ok.append(rep_score[1] - src_score[1])  # <0 means better

        source_code = r.get("source_code_text")
        repaired_code = r.get("repaired_code_text")
        source_out = r.get("source_verus_output_full")
        repaired_out = r.get("repaired_verus_output_full")
        if isinstance(source_code, str):
            source_code_len.append(len(source_code))
        if isinstance(repaired_code, str):
            repaired_code_len.append(len(repaired_code))
        if isinstance(source_code, str) and isinstance(repaired_code, str):
            code_len_delta.append(len(repaired_code) - len(source_code))
        if isinstance(source_out, str):
            source_output_len.append(len(source_out))
        if isinstance(repaired_out, str):
            repaired_output_len.append(len(repaired_out))
        if isinstance(source_out, str) and isinstance(repaired_out, str):
            output_len_delta.append(len(repaired_out) - len(source_out))

        g = r.get("chosen_failure_type")
        if isinstance(g, str):
            grouped[g].append(r)

    grouped_stats: dict[str, dict[str, Any]] = {}
    for g, items in grouped.items():
        g_rel = Counter()
        g_src_ok = 0
        g_rep_ok = 0
        g_both_ok = 0
        g_verify_up = 0
        g_verify_same = 0
        g_verify_down = 0
        g_error_down = 0
        g_error_same = 0
        g_error_up = 0

        for r in items:
            src_score = tuple2(r.get("source_score_tuple"))
            rep_score = tuple2(r.get("repaired_score_tuple"))
            rel = score_relation(src_score, rep_score)
            g_rel[rel] += 1

            if is_compile_ok(src_score):
                g_src_ok += 1
            if is_compile_ok(rep_score):
                g_rep_ok += 1

            if is_compile_ok(src_score) and is_compile_ok(rep_score):
                assert src_score is not None and rep_score is not None
                g_both_ok += 1
                sv, se = src_score
                rv, re = rep_score
                if rv > sv:
                    g_verify_up += 1
                elif rv == sv:
                    g_verify_same += 1
                else:
                    g_verify_down += 1
                if re < se:
                    g_error_down += 1
                elif re == se:
                    g_error_same += 1
                else:
                    g_error_up += 1

        grouped_stats[g] = {
            "count": len(items),
            "source_compile_ok_rate": safe_rate(g_src_ok, len(items)),
            "repaired_compile_ok_rate": safe_rate(g_rep_ok, len(items)),
            "compile_ok_rate_delta": round(
                safe_rate(g_rep_ok, len(items)) - safe_rate(g_src_ok, len(items)), 4
            ),
            "relation_counts": dict(g_rel),
            "both_ok_count": g_both_ok,
            "both_ok_verify_up_rate": safe_rate(g_verify_up, g_both_ok),
            "both_ok_verify_same_rate": safe_rate(g_verify_same, g_both_ok),
            "both_ok_verify_down_rate": safe_rate(g_verify_down, g_both_ok),
            "both_ok_error_down_rate": safe_rate(g_error_down, g_both_ok),
            "both_ok_error_same_rate": safe_rate(g_error_same, g_both_ok),
            "both_ok_error_up_rate": safe_rate(g_error_up, g_both_ok),
        }

    summary: dict[str, Any] = {
        "total_records": n,
        "unique_ids": len(id_set),
        "unique_generation_candidate_ids": len(generation_candidate_id_set),
        "top_fields": {
            "method": counter_top(field_counter["method"], topk),
            "dataset_name": counter_top(field_counter["dataset_name"], topk),
            "task_name": counter_top(field_counter["task_name"], topk),
            "llm_name": counter_top(field_counter["llm_name"], topk),
            "shot": counter_top(field_counter["shot"], topk),
            "round_id": [{"key": str(k), "count": v} for k, v in round_id_counter.most_common(topk)],
            "chosen_failure_type": counter_top(field_counter["chosen_failure_type"], topk),
            "source_failure_types": counter_top(field_counter["source_failure_types"], topk),
            "repaired_failure_types": counter_top(field_counter["repaired_failure_types"], topk),
            "is_default_chose": [{"key": str(k), "count": v} for k, v in is_default_counter.items()],
            "fallback_reason": counter_top(fallback_reason_counter, topk),
        },
        "compile_status": {
            "source_compile_ok_count": src_compile_ok,
            "source_compile_ok_rate": safe_rate(src_compile_ok, n),
            "repaired_compile_ok_count": rep_compile_ok,
            "repaired_compile_ok_rate": safe_rate(rep_compile_ok, n),
            "compile_ok_rate_delta": round(
                safe_rate(rep_compile_ok, n) - safe_rate(src_compile_ok, n), 4
            ),
        },
        "score_relation_counts": dict(relation_counter),
        "score_metrics_compile_ok_only": {
            "source_verify": mean_median(src_verify_values),
            "source_error": mean_median(src_error_values),
            "repaired_verify": mean_median(rep_verify_values),
            "repaired_error": mean_median(rep_error_values),
            "verify_delta_repaired_minus_source_when_both_ok": mean_median(verify_delta_both_ok),
            "error_delta_repaired_minus_source_when_both_ok": mean_median(error_delta_both_ok),
            "both_ok_count": len(verify_delta_both_ok),
        },
        "text_length_metrics": {
            "source_code_text_length": mean_median(source_code_len),
            "repaired_code_text_length": mean_median(repaired_code_len),
            "code_text_length_delta_repaired_minus_source": mean_median(code_len_delta),
            "source_verus_output_length": mean_median(source_output_len),
            "repaired_verus_output_length": mean_median(repaired_output_len),
            "verus_output_length_delta_repaired_minus_source": mean_median(output_len_delta),
        },
        "grouped_by_chosen_failure_type": dict(
            sorted(grouped_stats.items(), key=lambda kv: kv[1]["count"], reverse=True)
        ),
        "non_default_vs_default_path": compare_non_default_against_default(rows),
    }
    return summary


def print_human(summary: dict[str, Any]) -> None:
    relation_name_zh = {
        "both_compile_fail": "源与修复均编译失败",
        "compile_fail_to_ok": "源编译失败 -> 修复编译成功",
        "compile_ok_to_fail": "源编译成功 -> 修复编译失败",
        "both_ok_unchanged": "均编译成功且分数不变",
        "both_ok_pareto_better": "均编译成功且Pareto变好",
        "both_ok_pareto_worse": "均编译成功且Pareto变差",
        "both_ok_tradeoff": "均编译成功但有得有失",
    }

    print(f"总样本数: {summary.get('total_records', 0)}")
    print(f"唯一ID数: {summary.get('unique_ids', 0)}")
    print(
        "编译通过率: "
        f"源={summary['compile_status']['source_compile_ok_rate']}, "
        f"修复={summary['compile_status']['repaired_compile_ok_rate']}, "
        f"变化={summary['compile_status']['compile_ok_rate_delta']}"
    )

    print("修复结果关系分布:")
    for k, v in sorted(summary["score_relation_counts"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {relation_name_zh.get(k, k)}: {v}")

    print("Top失败类型（chosen_failure_type）:")
    for item in summary["top_fields"]["chosen_failure_type"]:
        print(f"  - {item['key']}: {item['count']}")

    print("按chosen_failure_type分组（按数量前10）:")
    gitems = list(summary["grouped_by_chosen_failure_type"].items())[:10]
    for name, g in gitems:
        print(
            "  - "
            f"{name}: 数量={g['count']}, "
            f"源通过率={g['source_compile_ok_rate']}, "
            f"修复通过率={g['repaired_compile_ok_rate']}, "
            f"Pareto变好={g['relation_counts'].get('both_ok_pareto_better', 0)}, "
            f"Pareto变差={g['relation_counts'].get('both_ok_pareto_worse', 0)}, "
            f"有得有失={g['relation_counts'].get('both_ok_tradeoff', 0)}"
        )

    nd = summary.get("non_default_vs_default_path")
    if isinstance(nd, dict):
        print("非默认路线 vs 默认路线（按 default_path 配对，比较 repaired_score_tuple）:")
        print(
            "  - "
            f"非默认总数={nd.get('total_non_default_records', 0)}, "
            f"可配对={nd.get('matched_non_default_records', 0)}, "
            f"配对率={nd.get('matched_rate', 0.0)}"
        )
        print(
            "  - "
            f"优于默认={nd.get('better_count', 0)} ({nd.get('better_rate_among_matched', 0.0)}), "
            f"劣于默认={nd.get('worse_count', 0)} ({nd.get('worse_rate_among_matched', 0.0)}), "
            f"不差于默认={nd.get('non_worse_count', 0)} ({nd.get('non_worse_rate_among_matched', 0.0)})"
        )
        print(
            "  - "
            f"持平={nd.get('unchanged_count', 0)} ({nd.get('unchanged_rate_among_matched', 0.0)}), "
            f"有得有失={nd.get('tradeoff_count', 0)} ({nd.get('tradeoff_rate_among_matched', 0.0)})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "统计 repair dataset 的 jsonl。"
            "评分规则：(-1,-1) 表示编译失败；否则编译成功。"
            "verify_num 越高越好；error_num 越低越好。"
        )
    )
    parser.add_argument("--input", required=True, help="repair-dataset jsonl 文件路径。")
    parser.add_argument("--topk", type=int, default=20, help="频次统计输出的 Top K。")
    parser.add_argument(
        "--output-json",
        default=None,
        help="可选：将完整统计结果保存为 json 的路径。",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"[错误] 输入文件不存在: {input_path}")
        return 1

    rows = load_jsonl(input_path)
    summary = summarize(rows, topk=max(1, args.topk))
    print_human(summary)

    if args.output_json is not None:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[完成] 已保存 json: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
