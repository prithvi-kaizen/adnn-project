"""Generate publication figures and tables from saved course statistics only."""

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/course_fast_track/plotting"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/course_fast_track/matplotlib"))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models.feature_store import sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve figure versions; choose a new directory")
    args.output.mkdir(parents=True)
    stats = json.loads(args.statistics.read_text())
    summary = json.loads(args.summary.read_text())
    assert stats["status"] == "STATISTICS_COMPUTED_PENDING_INDEPENDENT_VERIFICATION_AND_FIGURES"
    assert stats["source_sha256"][str(args.summary)] == sha(args.summary)
    with gzip.open(args.probes / "test_predictions.jsonl.gz", "rt") as handle:
        raw = {r["id"]: r for r in map(json.loads, handle)}
    measured = {r["id"]: r for r in stats["conditions"]}
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False})
    outputs = []

    def finish(fig, name):
        for extension in ("pdf", "svg", "png"):
            path = args.output / (name + "." + extension)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            outputs.append({"file": path.name, "sha256": sha(path)})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained", sharey=True)
    for ax, task in zip(axes, ("binary", "threeway")):
        prefix = f"s1729_h3_{task}_last_all_"
        result, uncertainty = raw[prefix + "ordinary"], measured[prefix + "ordinary"]
        x = np.arange(25)
        y = [result["layers"][str(k)]["metrics"]["accuracy"] for k in x]
        intervals = [uncertainty["layer_accuracy"][str(k)]["CI95"] for k in x]
        if all(v is not None for v in intervals):
            ax.fill_between(x, np.array(intervals)[:, 0], np.array(intervals)[:, 1], color="#285f87", alpha=.14)
        ax.plot(x, y, color="#285f87", label="Action-vector probe", linewidth=1.6)
        layer = result["selected_layer"]
        ax.plot(layer, y[layer], "o", color="black", markersize=4, label="Dev-selected layer")
        ax.axhline(result["lexical"]["metrics"]["accuracy"], color="#b24e24", linestyle="--", label="Action-only")
        ax.axhline(result["majority"]["metrics"]["accuracy"], color="black", linestyle=":", label="Majority")
        shuffled = [raw[prefix + f"shuffle_{s}"] for s in range(101, 111)]
        average = [np.mean([r["layers"][str(k)]["metrics"]["accuracy"] for r in shuffled]) for k in x]
        ax.plot(x, average, color="#666666", linestyle="-.", linewidth=1, label="Mean 10 shuffles")
        length = next(r for r in summary["exploratory_length_control"] if r["id"] == f"s1729_{task}")
        ax.axhline(length["metrics"]["accuracy"], color="#775398", linestyle=(0, (4, 2, 1, 2)), label="Length-only (expl.)")
        n = result["layers"][str(layer)]["metrics"]["N"]
        ax.set(title=f"{'R/I' if task == 'binary' else 'R/P/I'}; seed 1729, N={n}", xlabel="Returned layer", ylim=(0, 1.02), xticks=[0, 4, 8, 12, 16, 20, 24])
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Held-out accuracy")
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=7, loc="outside lower center", ncol=3, frameon=False)
    finish(fig, "layer_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), layout="constrained")
    primary = measured["s1729_h3_threeway_last_all_ordinary"]["selected"]["metrics"]["accuracy"]
    stated = next(r for r in stats["RQ2"] if r["seed"] == 1729 and r["pooling"] == "last")["stated_all_ID"]["metrics"]["accuracy"]
    for index, item in enumerate((primary, stated)):
        value = item["estimate"]
        axes[0].bar(index, value, width=.55, color=("#285f87", "#b24e24")[index])
        if item["CI95"] is not None:
            lo, hi = item["CI95"]
            axes[0].vlines(index, lo, hi, color="black", linewidth=1)
            axes[0].plot([index-.07, index+.07], [lo, lo], color="black")
            axes[0].plot([index-.07, index+.07], [hi, hi], color="black")
    axes[0].set(xticks=[0, 1], xticklabels=["Three-way probe", "Stated answer"], ylim=(0, 1.02), ylabel="Accuracy",
                title="Same 360 held-out IDs; seed 1729")
    behavior = stats["RQ3"]["point_and_CI"]
    if behavior:
        for j, key in enumerate(("K", "A", "gap")):
            item = behavior[key]
            axes[1].bar(j, item["mean"], color=["#285f87", "#40745b", "#b24e24"][j], width=.55)
            if item["CI95"] is not None:
                axes[1].vlines(j, *item["CI95"], color="black", linewidth=1.2)
            axes[1].annotate(f"{item['mean']:.2f}", (j, item["mean"]), xytext=(0, 5), textcoords="offset points", ha="center")
        axes[1].set(xticks=[0, 1, 2], xticklabels=["Paired knowledge K", "Adaptation A", "Gap K-A"], ylim=(-1.05, 1.05),
                    title="9 exposed pairs, 3 templates", ylabel="Pair-level mean")
    else:
        axes[1].text(.5, .5, "Incomplete outcomes; see bounds", transform=axes[1].transAxes, ha="center")
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    finish(fig, "knowledge_and_behavior")

    agreement = next(r for r in summary["RQ2"] if r["seed"] == 1729 and r["pooling"] == "last")
    labels = ["reversible", "partially_reversible", "irreversible"]
    columns = labels + ["invalid"]
    matrix = np.array([[agreement["confusion_truth_to_stated"].get(a + ":" + b, 0) for b in columns] for a in labels])
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max()*.55 else "black")
    ax.set(xticks=range(4), xticklabels=["R", "P", "I", "Invalid"], yticks=range(3), yticklabels=["R", "P", "I"],
           xlabel="Stated response", ylabel="Oracle label", title="Stated judgments: seed 1729, N=360")
    finish(fig, "stated_confusion")

    with (args.output / "ablation_results.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "seed", "horizon", "binary", "pooling", "subset", "layer", "N", "accuracy", "balanced_accuracy", "macro_F1", "AUROC"])
        for record in summary["probe_conditions"]:
            c = record["condition"]
            if c["control"] == "ordinary":
                m = record["selected_metrics"]
                writer.writerow([record["id"], c["seed"], c["horizon"], c["binary"], c["pooling"], c["subset"],
                    record["selected_layer"], *[m[k] for k in ("N", "accuracy", "balanced_accuracy", "macro_F1", "AUROC")]])
    outputs.append({"file": "ablation_results.csv", "sha256": sha(args.output / "ablation_results.csv")})
    metadata = {"status": "GENERATED_PENDING_VISUAL_QA", "source_statistics_sha256": sha(args.statistics),
                "source_summary_sha256": sha(args.summary), "source_program_sha256": sha(Path(__file__)),
                "outputs": outputs, "primary_figure_seed": 1729,
                "intervals": "1000 whole-component/template draws; conditional on fixed fits",
                "no_human_annotation": True, "native_quantized_results_combined": False}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Generated", len(outputs), "source-backed figure/table files; visual QA pending")


if __name__ == "__main__":
    main()
