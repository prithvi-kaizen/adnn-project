"""Render the six course-paper figures from the archived experiment records."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from build_submission_figures import ADAPTER, DESIGN, MODEL, PILOT, ROOT, read, sha

# isort: split
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

# The submission builder configures the Matplotlib overlay before these imports.

STYLE_URL = "https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making"
BASE = Path("results/course_fast_track/native_mac_cpu_v1")
PREDICTIONS = BASE / "probes/test_predictions.jsonl.gz"
STATS = BASE / "statistics.json"
SUMMARY = BASE / "core_summary.json"
STEERING = Path("results/course_fast_track/native_first_decision_steering_v1/summary.json")
DESTINATION = ROOT / "figures/course_submission/figures4papers_v1"

INK = "#272727"
MUTED = "#4D4D4D"
NEUTRAL = "#767676"
GRID = "#E3E3E3"
BLUE = "#0F4D92"
PALE_BLUE = "#E3EDF8"
GREEN = "#27633E"
PALE_GREEN = "#DDF3DE"
RED = "#B64342"
PALE_RED = "#F6CFCB"
VIOLET = "#79558B"


def panel(ax, heading, subtitle=None):
    ax.set_title(heading, loc="left", fontweight="bold", fontsize=9, pad=7 if subtitle is None else 20)
    if subtitle is not None:
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")


def percent_axis(ax, limits=(0, 1), ticks=(0, .25, .5, .75, 1)):
    ax.set(ylim=limits, yticks=ticks)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color=GRID, lw=.55)
    ax.set_axisbelow(True)


def methods_figure():
    rows = [json.loads(line) for line in (ROOT / PILOT).read_text().splitlines()]
    pair = {row["label"]: row for row in rows if row["pair_id"] == "pilot-delete-0"}
    reversible, irreversible = pair["reversible"], pair["irreversible"]
    r, i = [json.loads(row["state_text"]) for row in (reversible, irreversible)]
    action = json.loads(reversible["action_text"])
    assert action == json.loads(irreversible["action_text"]) and action["tool"] == "delete_file"
    assert [key for key in r if r[key] != i[key]] == ["recycle_enabled"]
    assert r["recycle_enabled"] and not i["recycle_enabled"] and not r["backups"] and not r["history"]
    exact = reversible["labels_by_horizon"]["3"]["witness"]
    assert len(exact) == 1 and exact[0]["tool"] == "restore_file" and exact[0]["arguments"]["source"] == "trash"
    assert irreversible["labels_by_horizon"]["3"]["search_exhausted"]

    fig, ax = plt.subplots(figsize=(7.12, 2.06))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def label(x, y, value, **style):
        ax.text(x, y, value, va="center", **style)

    def arrow(x1, x2, y, tint=MUTED):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops={"arrowstyle": "->", "color": tint, "lw": .9, "shrinkA": 0, "shrinkB": 0})

    label(.015, .91, "a   One action, two recovery outcomes", fontweight="bold", fontsize=10)
    label(.590, .91, "b   Three model measurements", fontweight="bold", fontsize=10)
    label(.015, .75, "Shared candidate", color=MUTED)
    label(.171, .75, "delete_file", fontfamily="monospace", color=BLUE, fontweight="bold")
    label(.590, .75, "Shared task, sandbox state and tools", color=MUTED)
    ax.plot([.56, .56], [.08, .91], color=GRID, lw=.8)

    lanes = (
        (.53, "Recycling on", "File in trash", "Restore file", "Restored in 1 action", GREEN, PALE_GREEN),
        (.26, "Recycling off", "File removed", "Search ≤3", "No qualifying recovery", RED, PALE_RED),
    )
    for y, start, state, followup, outcome, color, pale in lanes:
        ax.add_patch(plt.Rectangle((.01, y - .078), .529, .156, color=pale, lw=0))
        label(.021, y, start, fontweight="bold", color=color, fontsize=8)
        arrow(.141, .168, y, color)
        label(.181, y, state, fontsize=7.5)
        arrow(.288, .339, y, color)
        label(.313, y + .047, followup, ha="center", color=color, fontsize=6.8)
        label(.349, y, outcome, fontweight="bold", color=color, fontsize=7.4)
    label(.015, .083, "Oracle search: up to 3 actions. Both cases lack backup and history.",
          fontsize=7.15, color=MUTED)

    for y, tag, inputs, output in (
        (.55, "RQ1", "+ candidate", "Linear probe"),
        (.35, "RQ2", "+ candidate, horizon", "Stated label"),
        (.15, "RQ3", "Chooses actions", "Agent trace"),
    ):
        ax.add_patch(plt.Rectangle((.59, y - .064), .071, .128, color=PALE_BLUE, lw=0))
        label(.6255, y, tag, color=BLUE, ha="center", fontweight="bold", fontsize=7.6)
        label(.681, y, inputs, fontsize=7.65)
        arrow(.847, .870, y, BLUE)
        label(.884, y, output, color=BLUE, fontweight="bold", fontsize=7.75)
    return fig


def layer_figure(stats, summary):
    with gzip.open(ROOT / PREDICTIONS, "rt") as handle:
        raw = {row["id"]: row for row in map(json.loads, handle)}
    measured = {row["id"]: row for row in stats["conditions"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.47), sharey=True)
    fig.subplots_adjust(left=.086, bottom=.31, right=.980, top=.80, wspace=.18)
    for index, (ax, task, title) in enumerate(zip(axes, ("binary", "threeway"), ("Binary labels", "Three classes"), strict=True)):
        prefix = f"s1729_h3_{task}_last_all_"
        result = raw[prefix + "ordinary"]
        layer = result["selected_layer"]
        layers = result["layers"]
        x = range(len(layers))
        y = [layers[str(k)]["metrics"]["accuracy"] for k in x]
        uncertainty = measured[prefix + "ordinary"]["layer_accuracy"]
        interval = np.array([uncertainty[str(k)]["CI95"] or (np.nan, np.nan) for k in x])
        ax.fill_between(x, *interval.T, color=BLUE, alpha=.12, linewidth=0)
        ax.plot(x, y, color=BLUE, linewidth=1.75, label="Action-vector probe")
        ax.plot(layer, y[layer], marker="D", ms=4, color=BLUE, markeredgecolor="white", markeredgewidth=.55,
                linestyle="none", label="Dev-selected layer", zorder=5)
        ax.axhline(result["lexical"]["metrics"]["accuracy"], color=RED, ls="--", label="Action-only")
        ax.axhline(result["majority"]["metrics"]["accuracy"], color=INK, ls=(0, (1, 2)), label="Majority")
        shuffled = [raw[prefix + f"shuffle_{seed}"] for seed in range(101, 111)]
        average = [np.mean([r["layers"][str(k)]["metrics"]["accuracy"] for r in shuffled]) for k in x]
        ax.plot(x, average, color=NEUTRAL, ls="-.", label="Mean of 10 shuffles")
        length = next(r for r in summary["exploratory_length_control"] if r["id"] == f"s1729_{task}")
        ax.axhline(length["metrics"]["accuracy"], color=VIOLET, ls=(0, (5, 2, 1, 2)), label="Length-only (expl.)")
        panel(ax, f"{'ab'[index]}   {title}  ·  n = {layers[str(layer)]['metrics']['N']}")
        ax.set(xlabel="Returned layer", xticks=[0, 4, 8, 12, 16, 20, 24], xlim=(-.25, 24.25))
        percent_axis(ax)
    axes[0].set_ylabel("Held-out accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, 0), ncols=3,
               handlelength=2.4, columnspacing=1.65)
    return fig


def stated_figure(summary):
    record = next(r for r in summary["RQ2"] if r["seed"] == 1729 and r["pooling"] == "last")
    labels = ["reversible", "partially_reversible", "irreversible"]
    matrix = np.array([[record["confusion_truth_to_stated"].get(f"{a}:{b}", 0)
                        for b in labels + ["invalid"]] for a in labels])
    fig, ax = plt.subplots(figsize=(3.42, 1.95))
    fig.subplots_adjust(left=.15, bottom=.26, right=.99, top=.80)
    cmap = LinearSegmentedColormap.from_list("paper_blue", ["#F6F8FA", "#9EBBDB", BLUE])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=matrix.max(), aspect="auto")
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, str(value), ha="center", va="center", color="white" if value > matrix.max() * .60 else INK,
                fontweight="bold" if value else "normal", fontsize=8.5)
    ax.set(xticks=range(4), xticklabels=["R", "P", "I", "Invalid"], yticks=range(3), yticklabels=["R", "P", "I"],
           xlabel="Stated response", ylabel="Oracle label")
    ax.set_xticks(np.arange(-.5, 4), minor=True)
    ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    panel(ax, f"Stated labels  ·  n = {matrix.sum()}")
    return fig


def comparison_figure(stats):
    primary = next(r for r in stats["conditions"] if r["id"] == "s1729_h3_threeway_last_all_ordinary")
    probe = primary["selected"]["metrics"]["accuracy"]
    stated = next(r for r in stats["RQ2"] if r["seed"] == 1729 and r["pooling"] == "last")
    stated = stated["stated_all_ID"]["metrics"]["accuracy"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.12, 2.02))
    fig.subplots_adjust(left=.095, right=.98, bottom=.21, top=.72, wspace=.37)
    panel(left, "a   Probe and stated answer", "Same 360 held-out cases")
    for x, item, color in ((0, probe, BLUE), (1, stated, "#688097")):
        value = item["estimate"]
        left.bar(x, value, color=color, width=.46, edgecolor=INK, linewidth=.55)
        if item["CI95"] is not None:
            lo, hi = item["CI95"]
            left.errorbar(x, value, [[value-lo], [hi-value]], fmt="none", color=INK, capsize=2.8, lw=.85)
        left.text(x, .035, f"{value:.1%}", ha="center", color="white", fontsize=8, fontweight="bold")
    left.set(xticks=(0, 1), xticklabels=("Three-way probe", "Stated answer"), xlim=(-.62, 1.62), ylabel="Accuracy")
    percent_axis(left)

    panel(right, "b   Behavior and the blindness gap", "9 development pairs across 3 templates")
    for x, metric, color in ((0, "K", BLUE), (1, "A", GREEN), (2, "gap", RED)):
        item = stats["RQ3"]["point_and_CI"][metric]
        value = item["mean"]
        if item["CI95"] is not None:
            lo, hi = item["CI95"]
            right.errorbar(x, value, [[value-lo], [hi-value]], fmt="none", color=INK, capsize=3, lw=.85)
        right.plot(x, value, marker="D", ms=4.6, color=color, linestyle="none")
        right.annotate(f"{value:.0%}", (x, value), xytext=(0, 8), textcoords="offset points", ha="center", color=color)
    right.axhline(0, color=NEUTRAL, lw=.75)
    right.set(xticks=(0, 1, 2), xticklabels=("Knowledge K", "Adaptation A", "Gap K − A"),
              xlim=(-.55, 2.55), ylabel="Pair-level mean")
    percent_axis(right, limits=(-1, 1), ticks=(-1, -.5, 0, .5, 1))
    return fig


def steering_figure():
    saved = read(STEERING)
    points = saved["points"]
    baseline = next(p for p in points if p["strength"] == 0)
    assert saved["first_decision_cells"] == 102 and not saved["complete_full_RQ4"]
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 1.84), sharey=True)
    fig.subplots_adjust(left=.095, right=.98, top=.83, bottom=.31, wspace=.25)
    for ax, endpoint, title in zip(axes, ("caution", "goal", "legal"),
                                   ("Caution", "Goal reached", "Legal tool call"), strict=True):
        scores = defaultdict(dict)
        for point in points:
            if point["strength"] != 0:
                scores[point["direction"]][point["strength"]] = (
                    point["counts"][endpoint], point["rates"][endpoint]["CI95"])
        assert set(scores) == {"concept", "random_811", "random_812", "random_813"}
        concept = scores["concept"]
        assert all(values == concept for values in scores.values()), "Overlapping series must have identical counts and intervals"
        ordered = sorted([p for p in points if p["direction"] == "concept" and p["strength"] != 0] + [baseline],
                         key=lambda point: point["strength"])
        x = [p["strength"] for p in ordered]
        y = [p["counts"][endpoint] / 6 for p in ordered]
        lo, hi = np.array([p["rates"][endpoint]["CI95"] for p in ordered]).T
        ax.fill_between(x, lo, hi, color=BLUE, alpha=.13, lw=0)
        ax.plot(x, y, marker="o", color=BLUE, lw=1.35, ms=3.7, clip_on=False)
        ax.axvline(0, color=NEUTRAL, linestyle=":", lw=.8)
        ax.set(xticks=x, xlim=(-2.14, 2.14))
        percent_axis(ax, limits=(-.035, 1.035), ticks=(0, .5, 1))
        panel(ax, title)
    axes[0].set_ylabel("Share of 6 versions")
    fig.text(.54, .118, "Signed strength", ha="center")
    fig.text(.54, .007, "Concept direction and all three random controls overlap.", ha="center", color=MUTED, fontsize=7.5)
    return fig


def outcomes_figure():
    design, adapter, model = read(DESIGN), read(ADAPTER), read(MODEL)
    ids = [row["id"] for row in design["tasks"]]
    assert ids == model["assigned_ids"] and len(ids) == 6
    native = {row["id"]: row["score"] for row in model["records"]}
    policies = (("Scripted direct", "direct"), ("Scripted confirmation", "confirmation_then_shortcut"),
                ("Scripted reversible route", "safer"), ("Native model", None))
    rows = [[adapter["raw_episode_records"][i][route]["score"] if route else native[i] for i in ids]
            for _, route in policies]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.12, 1.93))
    fig.subplots_adjust(left=.200, right=.982, top=.71, bottom=.08, wspace=.24)
    for ax, key, heading, color, tint, expected in (
        (left, "goal_success", "a   Goal reached at stop", GREEN, PALE_GREEN, [6, 6, 6, 3]),
        (right, "executed_irreversible_action", "b   Irreversible action executed", RED, PALE_RED, [3, 4, 0, 4]),
    ):
        matrix = np.array([[r[key] for r in scores] for scores in rows], dtype=int)
        assert matrix.sum(axis=1).tolist() == expected
        ax.axhspan(2.5, 3.5, color=tint, lw=0, alpha=.45)
        for (y, x), value in np.ndenumerate(matrix):
            ax.scatter(x, y, marker="s", s=34, c=color if value else "white",
                       edgecolor=color if value else NEUTRAL, lw=.7, zorder=3)
        for x, family in zip((.5, 2.5, 4.5), ("File", "Mail", "Payment"), strict=True):
            ax.text(x, -1.14, family, ha="center", va="center", fontsize=7.6, fontweight="bold", clip_on=False)
        for x in (1.5, 3.5, 5.5):
            ax.axvline(x, color=GRID, lw=.7)
        ax.set(xlim=(-.5, 6.65), ylim=(3.53, -.55), xticks=range(6), xticklabels=list("RIRIRI"), yticks=range(4))
        ax.tick_params(length=0, pad=1)
        ax.xaxis.tick_top()
        ax.set_yticklabels([name for name, _ in policies] if ax is left else [])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        for y, count in enumerate(expected):
            ax.text(6.12, y, f"{count}/6", ha="center", va="center", color=MUTED, fontsize=7.45)
        ax.text(0, 1.39, heading, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="top")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 2.7, "ytick.major.size": 2.7, "xtick.major.width": .65, "ytick.major.width": .65,
        "legend.frameon": False, "legend.fontsize": 7.1, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "adnn-figures4papers-v1", "lines.linewidth": 1.05,
    })
    stats, summary = read(STATS), read(SUMMARY)
    assert stats["source_sha256"][str(SUMMARY)] == sha(SUMMARY)
    assert stats["source_sha256"][str(PREDICTIONS)] == sha(PREDICTIONS)
    figures = {
        "study_design": methods_figure(), "layer_accuracy": layer_figure(stats, summary),
        "stated_confusion": stated_figure(summary), "knowledge_and_behavior": comparison_figure(stats),
        "first_decision_steering": steering_figure(), "visible_behavior_outcomes": outcomes_figure(),
    }
    files = {}
    for name, figure in figures.items():
        for suffix, metadata in (("pdf", {"CreationDate": None}), ("svg", {"Date": None}), ("png", {})):
            path = args.output / f"{name}.{suffix}"
            figure.savefig(path, dpi=300, metadata=metadata, facecolor="white")
            files[path.name] = sha(path.resolve())
        plt.close(figure)
    sources = (DESIGN, ADAPTER, MODEL, PILOT, STATS, SUMMARY, PREDICTIONS, STEERING,
               Path("scripts/build_submission_figures.py"), Path(__file__).resolve().relative_to(ROOT))
    (args.output / "manifest.json").write_text(json.dumps({
        "style_reference": STYLE_URL, "source_sha256": {str(path): sha(path) for path in sources},
        "figure_sha256": files, "matplotlib_version": mpl.__version__, "png_dpi": 300, "new_model_calls": 0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
