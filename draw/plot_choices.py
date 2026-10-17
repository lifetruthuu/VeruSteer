from pathlib import Path
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
CSV_PATH = ROOT / "outputs/analysis/error_combinations.csv"

TOP_N = 10

# [字体放大优化] 提高全局基础字体大小
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

def parse_pct_value(text):
    match = re.search(r"([-+]?\d+(?:\.\d+)?)%", str(text))
    return float(match.group(1)) if match else np.nan

def parse_pp_value(text):
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*pp", str(text))
    return float(match.group(1)) if match else np.nan

def parse_choice_distribution(text):
    if not text or text == "N/A":
        return {}
    dist = {}
    for chunk in str(text).split(", "):
        match = re.match(r"(.+?)\s+([-+]?\d+(?:\.\d+)?)%", chunk.strip())
        if match:
            dist[match.group(1)] = float(match.group(2))
    return dist

def safe_combo_label(combo):
    return combo.replace(" + ", "\n+ ")

rows = []
try:
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            delta = parse_pp_value(row["Success_delta"])
            if not np.isfinite(delta) or delta <= 0:
                continue
            item = dict(row)
            item["frequency"] = int(row["Frequency_unique_nodes"])
            item["delta"] = delta
            item["default_dist"] = parse_choice_distribution(row["Default_choice_distribution"])
            item["verusteer_dist"] = parse_choice_distribution(row["VeruSteer_choice_distribution"])
            rows.append(item)
except FileNotFoundError:
    print(f"Warning: Data file not found at {CSV_PATH}.")

rows = sorted(rows, key=lambda r: r["frequency"], reverse=True)[:TOP_N]

base_action_order = [
    "MismatchedType", "InvFailEnd", "AssertFail", "PostCondFail",
    "InvFailFront", "ArithmeticFlow", "Other", "TypeAnnotation",
    "MethodNotFound", "PreCondFail"
]

custom_colors = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", 
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"
]
color_by_action = {action: custom_colors[i % len(custom_colors)] for i, action in enumerate(base_action_order)}

actions = []
for action in base_action_order:
    if any(action in r["default_dist"] or action in r["verusteer_dist"] for r in rows):
        actions.append(action)

# 重命名长标签
if "MethodNotFound" in actions:
    idx = actions.index("MethodNotFound")
    actions[idx] = "MthdNotFnd"
    color_by_action["MthdNotFnd"] = color_by_action.pop("MethodNotFound")

for r in rows:
    for key in ["default_dist", "verusteer_dist"]:
        if "MethodNotFound" in r[key]:
            r[key]["MthdNotFnd"] = r[key].pop("MethodNotFound")

labels = [safe_combo_label(r["Combination"]) for r in rows]
ROW_GAP = 1.6 # 增加行间距，适应更大字体
y_top = np.arange(len(rows)) * ROW_GAP
y_gain = np.arange(len(rows))

fig = plt.figure(figsize=(11.0, 11))
# wspace 缩小让 a 左右两子图更紧凑，整体腾出宽度
gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], hspace=0.35, wspace=0.06)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
ax_gain = fig.add_subplot(gs[1, :])

# --- 绘制图 (a) ---
for ax, key, title in [(axes[0], "default_dist", "Default rule"), (axes[1], "verusteer_dist", "VeruSteer")]:
    left = np.zeros(len(rows))
    for action in actions:
        values = np.array([r[key].get(action, 0.0) for r in rows])
        if values.sum() == 0:
            continue
        ax.barh(
            y_top, values, left=left, height=0.8,
            color=color_by_action[action], edgecolor="white", linewidth=0.8, label=action, zorder=3
        )
        left += values
        
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Choice distribution (%)", fontweight="bold")
    ax.grid(axis="x", linestyle=":", linewidth=1.2, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    ax.invert_yaxis()

axes[0].set_yticks(y_top)
axes[0].set_yticklabels(labels, fontsize=10.5, fontweight="bold")
for tick in axes[0].get_yticklabels(): tick.set_linespacing(1.2)
axes[1].set_yticks(y_top)
axes[1].set_yticklabels([])

# --- 绘制图 (b) ---
deltas = np.array([r["delta"] for r in rows])
gain_colors = ["#44CC44" if value > 0 else "#C0C0C0" for value in deltas]

ax_gain.barh(y_gain, deltas, height=0.75, color=gain_colors, edgecolor="white", linewidth=0.5, zorder=3)
ax_gain.invert_yaxis()
ax_gain.set_yticks(y_gain)
ax_gain.set_yticklabels([])
ax_gain.set_ylabel("Error-Type Combinations\n(Same order as (a))", 
                   fontweight="bold", labelpad=25, color="#555555", fontsize=11)

ax_gain.axvline(0, color="#777777", linewidth=1.2, zorder=1)
ax_gain.grid(axis="x", linestyle=":", linewidth=1.0, color="#EAEAEA", zorder=0)
ax_gain.set_axisbelow(True)
for spine in ["top", "right", "left"]: ax_gain.spines[spine].set_visible(False)

ax_gain.set_title("(b) Verification Success Gain", fontweight="bold", pad=15)
ax_gain.set_xlabel("Verification-Success Improvement (percentage points)", fontweight="bold")

max_delta = max(float(deltas.max()), 1.0)
ax_gain.set_xlim(0, max_delta * 1.2) 

for idx, row in enumerate(rows):
    ax_gain.text(
        row["delta"] + max_delta * 0.015, y_gain[idx],
        f"+{row['delta']:.2f} pp",
        va="center", ha="left", fontsize=11, fontweight="bold", color="#2E7D32"
    )

# --- Legend & 布局调整 [对齐关键点] ---
handles = [plt.Rectangle((0, 0), 1, 1, color=color_by_action[action]) for action in actions]

# 调整 bbox_to_anchor 确保 Legend 宽度与绘图区域一致
# ncol=5 保持两行排布
legend = fig.legend(
    handles, actions,
    loc="upper center", 
    ncol=5, 
    frameon=True,
    fancybox=False,
    bbox_to_anchor=(0.57, 0.99), # 精调 X 偏移以对齐 subplots
    bbox_transform=fig.transFigure,
    columnspacing=1.0, 
    handletextpad=0.5,
    handlelength=1.2, 
    handleheight=1.0,
    prop={"size": 11.5, "weight": "bold"} 
)
legend.get_frame().set_edgecolor("#CCCCCC")
legend.get_frame().set_linewidth(0.8)

# 统一大标题：位置根据 top 边距动态调整
fig.text(
    0.615, 0.88, "(a) Choice Distribution by Candidate Set",
    ha="center", va="center", fontsize=16, fontweight="bold", color="#333333"
)

# [边距微调] left 增大以容纳长标签，right 配合放大绘图区宽度
# 通过调整 right=0.96 和 left=0.28 扩张了 a 和 b 的横向跨度
fig.subplots_adjust(left=0.28, right=0.96, top=0.83, bottom=0.08)

fig.savefig(ROOT / "outputs/figures/error_type_combination.pdf", bbox_inches="tight")
plt.show()