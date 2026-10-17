#!/usr/bin/env python3
"""
核查 path JSONL 中叶子状态的验证结果：
1. label 是否与叶子保存的 Verus 输出一致（label=True 必须报告 "N verified, 0 errors"）
2. 候选错误类型为空但 label=False 的叶子有多少（不能把“没有可识别错误”当成验证成功）
3. 验证通过的叶子相对根状态是否新增了绕过证明的写法（assume/admit/external_body 等），
   函数签名（含 requires/ensures）或 spec fn 函数体是否被改动（均忽略注释）；
   --show-signature-diffs 打印改动明细

每棵树内同一个 leaf_state_id 只统计一次。

python commands/audit_verified_leaves.py \
  --input data/paths/test_autoverus.jsonl --show-signature-diffs
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict

SUMMARY_RE = re.compile(r"verification results::\s*(\d+) verified,\s*(\d+) errors")
FN_SIGNATURE_RE = re.compile(r"\bfn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^{]*?)\{", re.S)
SPEC_FN_RE = re.compile(r"\bspec\s+fn\s+(\w+)")
BYPASS_TOKENS = (
    "assume(",
    "admit(",
    "external_body",
    "#[verifier::external",
    "assume_specification",
    "ensures false",
    "requires false",
)


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    return re.sub(r"//[^\n]*", " ", code)


def fn_signatures(code: str) -> Dict[str, str]:
    # 从 fn 名到函数体左花括号之间的文本，包含参数、返回值和 requires/ensures；忽略注释
    code = strip_comments(code)
    return {m.group(1): re.sub(r"\s+", " ", m.group(2)).strip() for m in FN_SIGNATURE_RE.finditer(code)}


def spec_fn_bodies(code: str) -> Dict[str, str]:
    # spec fn 的函数体也是规格的一部分；按花括号配对截取，忽略注释和空白
    code = strip_comments(code)
    bodies = {}
    for m in SPEC_FN_RE.finditer(code):
        start = code.find("{", m.end())
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(code)):
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            if depth == 0:
                bodies[m.group(1)] = re.sub(r"\s+", " ", code[start:i + 1]).strip()
                break
    return bodies


def main() -> None:
    parser = argparse.ArgumentParser(description="核查叶子状态的验证结果与证明绕过")
    parser.add_argument("--input", type=Path, required=True, help="path JSONL")
    parser.add_argument("--show-signature-diffs", action="store_true", help="打印签名被改动的验证通过叶子")
    args = parser.parse_args()

    stats: Counter = Counter()
    seen = set()
    signature_changes = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["tree_id"], row["leaf_state_id"])
            if key in seen:
                continue
            seen.add(key)
            root, leaf = row["path_states"][0], row["path_states"][-1]
            label = bool(row.get("label"))
            m = SUMMARY_RE.search(leaf.get("verus_output_full") or "")
            summary = "none" if m is None else ("zero_errors" if m.group(2) == "0" else "has_errors")
            stats[f"leaves label={label}"] += 1
            stats[f"leaves label={label} verus_summary={summary}"] += 1
            if not label and not (leaf.get("failure_types") or []):
                stats["leaves label=False with empty candidate set"] += 1
            if not label:
                continue

            root_code, leaf_code = root.get("code_text") or "", leaf.get("code_text") or ""
            for token in BYPASS_TOKENS:
                if leaf_code.count(token) > root_code.count(token):
                    stats[f"verified leaves adding {token}"] += 1
            root_sigs, leaf_sigs = fn_signatures(root_code), fn_signatures(leaf_code)
            removed = sorted(set(root_sigs) - set(leaf_sigs))
            changed = sorted(k for k in root_sigs if k in leaf_sigs and root_sigs[k] != leaf_sigs[k])
            if removed:
                stats["verified leaves removing a fn"] += 1
            if changed:
                stats["verified leaves changing a fn signature/spec"] += 1
                signature_changes.append((row["tree_id"], row["leaf_state_id"], changed, root_sigs, leaf_sigs))
            root_specs, leaf_specs = spec_fn_bodies(root_code), spec_fn_bodies(leaf_code)
            changed_specs = sorted(k for k in root_specs if k in leaf_specs and root_specs[k] != leaf_specs[k])
            if changed_specs:
                stats["verified leaves changing a spec fn body"] += 1
                signature_changes.append((row["tree_id"], row["leaf_state_id"], changed_specs, root_specs, leaf_specs))

    print(f"== {args.input}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    if args.show_signature_diffs:
        for tree_id, leaf_id, changed, root_sigs, leaf_sigs in signature_changes:
            print(f"-- {tree_id} {leaf_id}")
            for name in changed:
                for d in difflib.ndiff([root_sigs[name]], [leaf_sigs[name]]):
                    print(f"   {name}: {d}")


if __name__ == "__main__":
    main()
