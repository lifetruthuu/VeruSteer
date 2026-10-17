#!/usr/bin/env python3
"""Visualize a tree (built from root-to-leaf JSONL paths) by a given id.

Input JSONL schema is expected to include these fields per line:
- tree_id (preferred) or legacy id
- path_id
- default_path_id
- path_state_ids (list[str])
- path_states (list[dict])
- label
- default

Usage:
  python commands/visualize_tree_by_id.py --input test.jsonl --id <tree_id>
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Node:
    state_id: str
    round_id: int | None = None
    score_tuple: Any = None
    failure_types: Any = None
    children: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize tree structure by id from JSONL.")
    parser.add_argument("--input", required=True, help="Path to root-to-leaf jsonl file")
    parser.add_argument("--id", required=True, help="Tree id to visualize")
    parser.add_argument(
        "--format",
        choices=["text", "mermaid", "html"],
        default="text",
        help="Output format: text (terminal), mermaid, or html",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output file path for mermaid/html. "
            "Default: ./tree_<id>.mmd or ./tree_<id>.html"
        ),
    )
    parser.add_argument(
        "--show-failure-types",
        action="store_true",
        help="Also print node failure_types in labels",
    )
    return parser.parse_args()


def extract_tree_id(obj: dict[str, Any]) -> str | None:
    """Prefer tree_id, fallback to legacy id."""
    value = obj.get("tree_id")
    if not isinstance(value, str) or not value.strip():
        value = obj.get("id")
    if isinstance(value, str) and value.strip():
        return value
    return None


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


def load_paths_by_id(input_path: Path, tree_id: str) -> tuple[list[dict[str, Any]], int, set[str]]:
    rows: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    total = 0

    with input_path.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            obj_id = extract_tree_id(obj)
            if obj_id is not None:
                all_ids.add(obj_id)
            if obj_id == tree_id:
                rows.append(obj)

    return rows, total, all_ids


def _pick_default_path_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best default-path row from latest/legacy root_to_leaf fields."""
    if not rows:
        return None

    # New schema: default_path_id should point to the default path.
    for row in rows:
        path_id = row.get("path_id")
        default_path_id = row.get("default_path_id")
        if isinstance(path_id, str) and isinstance(default_path_id, str) and path_id == default_path_id:
            return row

    # Fallback: explicit default flag on path row.
    for row in rows:
        if normalize_binary(row.get("default")) == 1:
            return row

    return None


def build_tree(rows: list[dict[str, Any]]) -> tuple[dict[str, Node], dict[tuple[str, str], dict[str, Any]], str]:
    nodes: dict[str, Node] = {}
    edge_meta: dict[tuple[str, str], dict[str, Any]] = {}

    root_id = rows[0]["root_state_id"]

    default_row = _pick_default_path_row(rows)
    default_edges: set[tuple[str, str]] = set()
    if default_row is not None:
        default_ids = default_row.get("path_state_ids")
        if isinstance(default_ids, list):
            for i in range(len(default_ids) - 1):
                src = default_ids[i]
                dst = default_ids[i + 1]
                if isinstance(src, str) and isinstance(dst, str):
                    default_edges.add((src, dst))

    for row in rows:
        path_ids = row["path_state_ids"]
        path_states = row["path_states"]

        for st in path_states:
            sid = st["state_id"]
            if sid not in nodes:
                nodes[sid] = Node(
                    state_id=sid,
                    round_id=st.get("round_id"),
                    score_tuple=st.get("score_tuple"),
                    failure_types=st.get("failure_types"),
                )

        for i in range(len(path_ids) - 1):
            src = path_ids[i]
            dst = path_ids[i + 1]
            if dst not in nodes[src].children:
                nodes[src].children.append(dst)

            # chosen_failure_type/is_default_chose is stored on source node for this edge.
            src_state = path_states[i]
            edge_meta[(src, dst)] = {
                "chosen_failure_type": src_state.get("chosen_failure_type"),
                "is_default_chose": src_state.get("is_default_chose"),
                "is_default_path_edge": (src, dst) in default_edges,
            }

    for node in nodes.values():
        node.children.sort()

    return nodes, edge_meta, root_id


def format_node(node: Node, show_failure_types: bool) -> str:
    base = f"{node.state_id} [round={node.round_id}, score={node.score_tuple}]"
    if show_failure_types:
        base += f" [failure_types={node.failure_types}]"
    return base


def print_tree(
    nodes: dict[str, Node],
    edge_meta: dict[tuple[str, str], dict[str, Any]],
    root_id: str,
    show_failure_types: bool,
) -> None:
    def dfs(cur: str, prefix: str, is_last: bool, visited: set[str]) -> None:
        connector = "└── " if is_last else "├── "
        label = format_node(nodes[cur], show_failure_types)
        if prefix:
            print(prefix + connector + label)
        else:
            print(label)

        if cur in visited:
            loop_prefix = prefix + ("    " if is_last else "│   ")
            print(loop_prefix + "└── [cycle detected, stop]")
            return

        visited.add(cur)

        child_prefix = prefix + ("    " if is_last else "│   ")
        children = nodes[cur].children
        for idx, child in enumerate(children):
            child_is_last = idx == len(children) - 1
            meta = edge_meta.get((cur, child), {})
            chosen = meta.get("chosen_failure_type")
            default = meta.get("is_default_chose")
            edge_info = f"edge(choice={chosen}, default={default})"
            edge_connector = "└── " if child_is_last else "├── "
            print(child_prefix + edge_connector + edge_info)
            dfs(
                child,
                child_prefix + ("    " if child_is_last else "│   "),
                True,
                visited,
            )

        visited.remove(cur)

    dfs(root_id, "", True, set())


def summarize(rows: list[dict[str, Any]], nodes: dict[str, Node], root_id: str) -> None:
    leaves = [n.state_id for n in nodes.values() if not n.children]
    success_paths = sum(1 for r in rows if normalize_binary(r.get("label")) == 1)
    default_paths = sum(1 for r in rows if normalize_binary(r.get("default")) == 1)

    print("[SUMMARY]")
    print(f"  root_state_id : {root_id}")
    print(f"  paths         : {len(rows)}")
    print(f"  nodes         : {len(nodes)}")
    print(f"  leaves        : {len(leaves)}")
    print(f"  success_paths : {success_paths}")
    print(f"  default_paths : {default_paths}")


def sanitize_filename(name: str) -> str:
        out = []
        for ch in name:
                if ch.isalnum() or ch in {"-", "_"}:
                        out.append(ch)
                else:
                        out.append("_")
        return "".join(out).strip("_") or "tree"


def _mermaid_escape(text: str) -> str:
        return text.replace('"', "\\\"")


def _node_label(node: Node, show_failure_types: bool) -> str:
        label = f"{node.state_id}<br/>round={node.round_id}<br/>score={node.score_tuple}"
        if show_failure_types:
                label += f"<br/>failure_types={node.failure_types}"
        return _mermaid_escape(label)


def build_mermaid(
        nodes: dict[str, Node],
        edge_meta: dict[tuple[str, str], dict[str, Any]],
        root_id: str,
        show_failure_types: bool,
) -> str:
        # Deterministic traversal order for stable output.
        ordered: list[str] = []
        seen: set[str] = set()

        def dfs(cur: str) -> None:
                if cur in seen:
                        return
                seen.add(cur)
                ordered.append(cur)
                for child in nodes[cur].children:
                        dfs(child)

        dfs(root_id)

        node_id_map = {sid: f"n{idx}" for idx, sid in enumerate(ordered)}
        leaves = {sid for sid in ordered if not nodes[sid].children}

        lines: list[str] = ["graph TD"]
        for sid in ordered:
                nid = node_id_map[sid]
                lines.append(f'    {nid}["{_node_label(nodes[sid], show_failure_types)}"]')

        for sid in ordered:
            src_nid = node_id_map[sid]
            for child in nodes[sid].children:
                if child not in node_id_map:
                    continue
                dst_nid = node_id_map[child]
                meta = edge_meta.get((sid, child), {})
                chosen = meta.get("chosen_failure_type")
                default = meta.get("is_default_chose")
                edge_label = _mermaid_escape(f"choice={chosen}, default={default}")
                lines.append(f'    {src_nid} -->|"{edge_label}"| {dst_nid}')

        lines.append("")
        lines.append("    classDef root fill:#ffe8a3,stroke:#a37200,stroke-width:2px;")
        lines.append("    classDef leaf fill:#d9f7be,stroke:#237804,stroke-width:1.5px;")
        lines.append("    classDef normal fill:#e6f4ff,stroke:#1677ff,stroke-width:1px;")
        lines.append(f"    class {node_id_map[root_id]} root;")

        normal_nodes = [node_id_map[sid] for sid in ordered if sid not in leaves and sid != root_id]
        leaf_nodes = [node_id_map[sid] for sid in ordered if sid in leaves]
        if normal_nodes:
                lines.append(f"    class {','.join(normal_nodes)} normal;")
        if leaf_nodes:
                lines.append(f"    class {','.join(leaf_nodes)} leaf;")

        return "\n".join(lines)


def build_html(mermaid_text: str, tree_id: str) -> str:
    # Mermaid is rendered client-side in browser.
    safe_mermaid_text = html.escape(mermaid_text)
    safe_tree_id = html.escape(tree_id)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Tree Visualization</title>
    <style>
        body {{
            margin: 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: #f7f9fc;
            color: #1f2937;
        }}
        .header {{
            position: sticky;
            top: 0;
            background: #ffffff;
            border-bottom: 1px solid #e5e7eb;
            padding: 10px 16px;
            z-index: 1;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}
        .title {{
            font-size: 14px;
            font-weight: 600;
            word-break: break-all;
        }}
        .toolbar {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }}
        .zoom-btn {{
            border: 1px solid #d1d5db;
            background: #ffffff;
            color: #111827;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 13px;
            cursor: pointer;
        }}
        .zoom-btn:hover {{
            background: #f3f4f6;
        }}
        .zoom-value {{
            min-width: 54px;
            text-align: center;
            font-size: 12px;
            color: #374151;
            font-variant-numeric: tabular-nums;
        }}
        .container {{
            padding: 12px;
            overflow: auto;
            height: calc(100vh - 52px);
        }}
        .mermaid {{
            min-width: max-content;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 12px;
        }}
        .mermaid svg {{
            max-width: none !important;
            height: auto !important;
            display: block;
        }}
    </style>
</head>
<body>
    <div class=\"header\">
        <div class=\"title\">id: {safe_tree_id}</div>
        <div class="toolbar">
            <button id="zoom-out" class="zoom-btn" type="button">-</button>
            <div id="zoom-value" class="zoom-value">100%</div>
            <button id="zoom-in" class="zoom-btn" type="button">+</button>
            <button id="zoom-reset" class="zoom-btn" type="button">Reset</button>
        </div>
    </div>
    <div class=\"container\">
        <pre class=\"mermaid\">{safe_mermaid_text}</pre>
    </div>

    <script type=\"module\">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});

        let scale = 1.0;
        const minScale = 0.3;
        const maxScale = 3.0;
        const step = 0.1;

        const zoomOutBtn = document.getElementById('zoom-out');
        const zoomInBtn = document.getElementById('zoom-in');
        const zoomResetBtn = document.getElementById('zoom-reset');
        const zoomValue = document.getElementById('zoom-value');

        function getSvg() {{
            return document.querySelector('.mermaid svg');
        }}

        function applyZoom() {{
            const svg = getSvg();
            if (!svg) return;
            svg.style.transformOrigin = '0 0';
            svg.style.transform = `scale(${{scale}})`;
            zoomValue.textContent = `${{Math.round(scale * 100)}}%`;
        }}

        zoomInBtn?.addEventListener('click', () => {{
            scale = Math.min(maxScale, +(scale + step).toFixed(2));
            applyZoom();
        }});

        zoomOutBtn?.addEventListener('click', () => {{
            scale = Math.max(minScale, +(scale - step).toFixed(2));
            applyZoom();
        }});

        zoomResetBtn?.addEventListener('click', () => {{
            scale = 1.0;
            applyZoom();
        }});

        // Mermaid render is async; poll briefly until svg is ready.
        let retries = 0;
        const timer = setInterval(() => {{
            retries += 1;
            if (getSvg()) {{
                applyZoom();
                clearInterval(timer);
            }}
            if (retries > 50) {{
                clearInterval(timer);
            }}
        }}, 100);
    </script>
</body>
</html>
"""


def default_output_path(fmt: str, tree_id: str) -> Path:
    stem = f"tree_{sanitize_filename(tree_id)}"
    if fmt == "mermaid":
        return Path.cwd() / f"{stem}.mmd"
    if fmt == "html":
        return Path.cwd() / "assets" / f"{stem}.html"
    return Path.cwd() / f"{stem}.txt"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()

    if not input_path.is_file():
        print(f"[ERROR] input file not found: {input_path}")
        return 1

    rows, total_lines, all_ids = load_paths_by_id(input_path, args.id)
    if not rows:
        print(f"[ERROR] id not found: {args.id}")
        print(f"[INFO] total lines={total_lines}, unique ids={len(all_ids)}")
        sample_ids = sorted(all_ids)[:10]
        if sample_ids:
            print("[INFO] sample ids:")
            for sid in sample_ids:
                print(f"  - {sid}")
        return 2

    nodes, edge_meta, root_id = build_tree(rows)

    if args.format == "text":
        print(f"[ID] {args.id}")
        summarize(rows, nodes, root_id)
        print("\n[TREE]")
        print_tree(nodes, edge_meta, root_id, args.show_failure_types)
        return 0

    mermaid_text = build_mermaid(nodes, edge_meta, root_id, args.show_failure_types)
    out_path = Path(args.out).resolve() if args.out else default_output_path(args.format, args.id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "mermaid":
        out_path.write_text(mermaid_text + "\n", encoding="utf-8")
        print(f"[DONE] mermaid file: {out_path}")
        return 0

    html = build_html(mermaid_text, args.id)
    out_path.write_text(html, encoding="utf-8")
    print(f"[DONE] html file: {out_path}")
    print("[TIP] Open this file in your browser for interactive viewing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
