import json
import os
import tempfile
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproduce paper/fig/score_trend.png.
ROOT = Path(__file__).resolve().parents[1]
(ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

DEFAULT_PATH = ROOT / "outputs/analysis/default_paths.jsonl"
VERUSTEER_PATH = ROOT / "outputs/analysis/selector_paths.jsonl"
ROUNDS = list(range(1, 6))
MULTI_BRANCH = "同时包含默认路径和非默认路径"


def baseline_name(tree_id):
    if tree_id.startswith("autoverus-"):
        return "AutoVerus"
    if tree_id.startswith("verusage-"):
        return "VeruSAGE"
    return "Unknown"


def load_paths(path, path_label):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["_path_label"] = path_label
            row["_baseline"] = baseline_name(row.get("tree_id", ""))
            rows.append(row)
    return rows


def in_multi_branch_test(row):
    return row.get("split") == "test" and row.get("tree_category") == MULTI_BRANCH


def score_at_round(row, round_id):
    best_score = None
    best_round = -1
    for node in row.get("path_nodes") or []:
        current_round = node.get("round_id")
        score = node.get("score_tuple")
        if not isinstance(current_round, int):
            continue
        if current_round > round_id or current_round < best_round:
            continue
        if not (isinstance(score, list) and len(score) >= 2):
            continue
        best_round = current_round
        best_score = (float(score[0]), float(score[1]))
    return best_score


def verifies(row):
    nodes = row.get("path_nodes") or []
    if not nodes:
        return False
    score = nodes[-1].get("score_tuple")
    return isinstance(score, list) and len(score) >= 2 and score[0] > 0 and score[1] == 0


def average_series(rows, baseline, score_index, tree_ids=None, valid_error_only=False):
    selected = [r for r in rows if r["_baseline"] == baseline]
    if tree_ids is not None:
        selected = [r for r in selected if r.get("tree_id") in tree_ids]

    values = []
    for round_id in ROUNDS:
        round_values = []
        for row in selected:
            score = score_at_round(row, round_id)
            if score is None:
                continue
            value = score[score_index]
            if valid_error_only and value < 0:
                continue
            round_values.append(value)
        values.append(mean(round_values) if round_values else None)
    return values


default_rows = [r for r in load_paths(DEFAULT_PATH, "Default") if in_multi_branch_test(r)]
verusteer_rows = [r for r in load_paths(VERUSTEER_PATH, "VeruSteer") if in_multi_branch_test(r)]
rows_by_path = {"Default": default_rows, "VeruSteer": verusteer_rows}

# ==========================================
# Two panels side by side, set in the paper's font (Linux Biolinum)
# ==========================================
from matplotlib import font_manager
from matplotlib.lines import Line2D

_font_dir = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/libertine"
_family = "DejaVu Sans"
for _stem in ("LinBiolinum_R", "LinBiolinum_RB"):
    _path = _font_dir / f"{_stem}.otf"
    if _path.exists():
        font_manager.fontManager.addfont(str(_path))
        _family = "Linux Biolinum O"
plt.rcParams.update({"font.family": _family, "font.weight": "normal", "pdf.fonttype": 42})

INK, MUTED = "#222222", "#5F5F5F"
styles = {
    ("AutoVerus", "VeruSteer"): dict(color="#2166AC", marker="o", linestyle="-"),
    ("AutoVerus", "Default"):   dict(color="#7FB2D6", marker="s", linestyle="--"),
    ("VeruSAGE", "VeruSteer"):  dict(color="#B2182B", marker="^", linestyle="-"),
    ("VeruSAGE", "Default"):    dict(color="#E8935E", marker="D", linestyle="--"),
}
plot_order = [
    ("AutoVerus", "VeruSteer"),
    ("AutoVerus", "Default"),
    ("VeruSAGE", "VeruSteer"),
    ("VeruSAGE", "Default"),
]


def draw_panel(ax, ylabel, score_index, valid_error_only=False):
    for baseline, path_label in plot_order:
        y = average_series(rows_by_path[path_label], baseline, score_index,
                           valid_error_only=valid_error_only)
        ax.plot(ROUNDS, y, linewidth=1.5, markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.6, zorder=3, **styles[(baseline, path_label)])
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK, labelpad=3)
    ax.set_xticks(ROUNDS)
    ax.set_xticklabels(ROUNDS)
    ax.set_xlabel("Repair round", fontsize=9.5, color=INK, labelpad=2)
    ax.tick_params(labelsize=9, length=2.5, colors=MUTED, pad=2)
    ax.grid(True, linestyle=":", color="#D4D4D4", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")


fig, axes = plt.subplots(1, 2, figsize=(5.48, 2.85))
draw_panel(axes[0], "Mean verified proof obligations", score_index=0)
draw_panel(axes[1], "Mean reported errors", score_index=1, valid_error_only=True)

handles = [
    Line2D([0], [0], label=f"{b} \u2013 " + ("Default rule" if p == "Default" else p), linewidth=1.5, markersize=4.5,
           markeredgecolor="white", markeredgewidth=0.6, **styles[(b, p)])
    for b, p in plot_order
]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5,
           handlelength=2.2, columnspacing=2.2, handletextpad=0.5, labelspacing=0.25,
           bbox_to_anchor=(0.54, 1.0))
fig.subplots_adjust(left=0.095, right=0.99, bottom=0.18, top=0.78, wspace=0.3)

fig.savefig(ROOT / "outputs/figures/avg_score.pdf", transparent=False)

import csv
with (ROOT / "outputs/analysis/progress.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["pipeline", "selector", "round", "verified_obligations", "reported_errors"])
    for baseline, path_label in plot_order:
        verified = average_series(rows_by_path[path_label], baseline, 0)
        errors = average_series(rows_by_path[path_label], baseline, 1, valid_error_only=True)
        for round_id, x, y in zip(ROUNDS, verified, errors):
            writer.writerow([baseline, path_label, round_id, x, y])
plt.close(fig)
