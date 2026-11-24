"""Build original, editable paper figures from verified sandbox records."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/course_fast_track/plotting"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/course_fast_track/matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DESIGN = Path("docs/evidence/course_improvement_v1/i2_behavior_design/verification.json")
ADAPTER = Path("docs/evidence/course_improvement_v1/i2_behavior_adapter/verification.json")
MODEL = Path("artifacts/course_improvement/v1/behavior_design_model_diagnostic/report.json")
PILOT = Path("data/course_fast_track/v2/development_tasks.jsonl")
INK = "#172A38"
MUTED = "#445A66"
BLUE = "#1B6070"
BLUE_PALE = "#EAF5F6"
ORANGE = "#A7431E"
ORANGE_PALE = "#FBEFE8"
LINE = "#CBD6DA"
GRAY = "#F4F7F8"


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def canvas(height):
    figure, axis = plt.subplots(figsize=(7.12, height))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_axis_off()
    return figure, axis


def box(axis, left, bottom, width, height, label, fill=GRAY, border=LINE, color=INK,
        weight="normal", size=8, family=None):
    axis.add_patch(FancyBboxPatch((left, bottom), width, height, boxstyle="round,pad=0,rounding_size=.008",
                                  facecolor=fill, edgecolor=border, linewidth=.7))
    axis.text(left + width / 2, bottom + height / 2, label, ha="center", va="center",
              color=color, fontsize=size, fontweight=weight, fontfamily=family, linespacing=1.3)


def arrow(axis, start, end):
    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": MUTED,
                  "linewidth": .8, "mutation_scale": 9, "shrinkA": 0, "shrinkB": 0})


def methods_figure():
    rows = [json.loads(line) for line in (ROOT / PILOT).read_text().splitlines()]
    matched = {row["label"]: row for row in rows if row["pair_id"] == "pilot-delete-0"}
    reversible, irreversible = (matched[key] for key in ("reversible", "irreversible"))
    left, right = (json.loads(row["state_text"]) for row in (reversible, irreversible))
    assert reversible["action_text"] == irreversible["action_text"]
    assert [key for key in left if left[key] != right[key]] == ["recycle_enabled"]
    assert left["recycle_enabled"] and not right["recycle_enabled"] and not right["backups"] and not right["history"]
    assert reversible["labels_by_horizon"]["3"]["witness"][0]["arguments"]["source"] == "trash"
    assert irreversible["labels_by_horizon"]["3"]["search_exhausted"]
    figure, axis = canvas(2.04)
    axis.text(.015, .95, "A  Matched recovery test", color=INK, fontsize=10, fontweight="bold", va="center")
    axis.text(.577, .95, "B  Separate model readouts", color=INK, fontsize=10, fontweight="bold", va="center")
    axis.plot([.55, .55], [.10, .95], color=LINE, linewidth=.7)
    axis.text(.015, .842, "Change one state property; keep the action fixed.", fontsize=8, color=MUTED, va="center")
    variants = [(.52, "Recycling on", "Trash restore\nrecovers prior state", BLUE, BLUE_PALE),
                (.20, "Recycling off", "No recovery within\nthree legal actions", ORANGE, ORANGE_PALE)]
    for bottom, state, outcome, color, fill in variants:
        box(axis, .015, bottom, .125, .205, state, size=7.8, weight="bold")
        box(axis, .168, bottom, .139, .205, "delete_file", family="DejaVu Sans Mono", size=7.5)
        box(axis, .335, bottom, .193, .205, outcome, fill=fill, border=color, color=color, size=7.8, weight="bold")
        arrow(axis, (.142, bottom + .102), (.163, bottom + .102))
        arrow(axis, (.310, bottom + .102), (.329, bottom + .102))
    axis.text(.015, .078, "No backup or history is available in either state.", fontsize=7.6, color=MUTED)
    box(axis, .577, .65, .399, .167, "Task and initial sandbox state", fill=BLUE_PALE, border=BLUE,
        color=BLUE, weight="bold", size=8.3)
    centers = [.637, .777, .916]
    axis.plot([.777, .777], [.642, .551], color=MUTED, linewidth=.8)
    axis.plot([centers[0], centers[-1]], [.551, .551], color=MUTED, linewidth=.8)
    for center, label, hint in zip(centers, ("Linear\nprobe", "Stated\nlabel", "Agent\ntrace"),
                                  ("+ candidate", "+ candidate", "no candidate"), strict=True):
        arrow(axis, (center, .551), (center, .475))
        box(axis, center - .060, .233, .120, .225, label, weight="bold", size=8.2)
        axis.text(center, .153, hint, color=MUTED, fontsize=7.4, ha="center", va="center")
    axis.text(.577, .078, "Compare outcomes only on aligned case IDs.", fontsize=7.6, color=MUTED)
    return figure


def outcomes_figure():
    design, adapter, model = read(DESIGN), read(ADAPTER), read(MODEL)
    keys = [row["id"] for row in design["tasks"]]
    assert model["assigned_ids"] == keys and len(keys) == 6
    observed = {row["id"]: row["score"] for row in model["records"]}
    policies = [("Scripted direct", "direct"), ("Scripted confirmation", "confirmation_then_shortcut"),
                ("Scripted reversible route", "safer"), ("Native model", None)]
    by_policy = {label: [adapter["raw_episode_records"][key][policy]["score"] if policy else observed[key]
                         for key in keys] for label, policy in policies}
    expected = [(6, 3), (6, 4), (6, 0), (3, 4)]
    for label, (goals, irreversible) in zip(by_policy, expected, strict=True):
        assert sum(row["goal_success"] for row in by_policy[label]) == goals
        assert sum(row["executed_irreversible_action"] for row in by_policy[label]) == irreversible
    figure, axis = canvas(1.88)
    panels = [(0.324, "Goal reached at stop", "goal_success", BLUE),
              (0.674, "Any irreversible action", "executed_irreversible_action", ORANGE)]
    y = [.598, .457, .316, .158]
    axis.axhspan(.093, .229, xmin=.012, xmax=.984, color=GRAY, linewidth=0)
    axis.text(.015, .91, "Policy or model", fontsize=9, fontweight="bold", color=INK, va="center")
    for row_y, (label, policy) in zip(y, policies, strict=True):
        axis.text(.015, row_y, label, fontsize=8.4, color=INK, fontweight="bold" if policy is None else "normal",
                  va="center")
    axis.plot([.638, .638], [.093, .937], color=LINE, linewidth=.7)
    for start, title, metric, tint in panels:
        axis.text(start, .91, title, fontsize=9, fontweight="bold", color=INK, va="center")
        columns = [start + .045 * column for column in range(6)]
        for pair, title in enumerate(("File", "Mail", "Payment")):
            axis.text((columns[2 * pair] + columns[2 * pair + 1]) / 2, .790, title, fontsize=7.8,
                      ha="center", va="center", color=MUTED)
        for column, state in zip(columns, "RIRIRI", strict=True):
            axis.text(column, .690, state, ha="center", va="center", fontsize=7.5, color=MUTED)
        for row_y, scores in zip(y, by_policy.values(), strict=True):
            for column, score in zip(columns, scores, strict=True):
                axis.scatter(column, row_y, s=48, facecolor=tint if score[metric] else "white",
                             edgecolor=tint if score[metric] else "#93A5AE", linewidth=.9)
    axis.text(.015, .016, "Filled: event observed.  R/I: initial shortcut label, not the label of the action taken.",
              fontsize=7.4, color=MUTED, va="bottom")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "adnn-course-submission-v1"})
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, build in (("study_design", methods_figure), ("visible_behavior_outcomes", outcomes_figure)):
        figure = build()
        for extension in ("pdf", "svg", "png"):
            path = args.output / f"{name}.{extension}"
            metadata = {"CreationDate": None} if extension == "pdf" else {"Date": None} if extension == "svg" else {}
            figure.savefig(path, dpi=260, metadata=metadata, facecolor="white")
            files[path.name] = sha(path.resolve())
        plt.close(figure)
    manifest = {"status": "SOURCE_BACKED_VECTOR_EXPORTS", "font": "DejaVu Sans", "new_model_calls": 0,
                "files_sha256": files, "source_sha256": {str(path): sha(path) for path in (
                    DESIGN, ADAPTER, MODEL, PILOT, Path(__file__).resolve().relative_to(ROOT))},
                "layout_references": ["https://arxiv.org/abs/2309.15817v2",
                                      "https://aclanthology.org/2024.acl-long.828/"],
                "external_content_reproduced": False, "svg_text_is_editable": True}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Built six vector/raster exports from verified records.")


if __name__ == "__main__":
    main()
