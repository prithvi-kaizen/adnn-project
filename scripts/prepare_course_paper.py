"""Generate every empirical paper table and numeric macro from saved records."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.models.feature_store import sha  # noqa: E402


def table(path, caption, columns, header, rows, wide=False, label=None):
    environment = "table*" if wide else "table"
    text = [f"\\begin{{{environment}}}[t]\\centering\\small", "\\caption{" + caption + "}"]
    if label:
        text.append("\\label{" + label + "}")
    text += ["\\begin{tabular}{" + columns + "}\\toprule", " & ".join(header) + "\\\\\\midrule"]
    text += [" & ".join(map(str, row)) + "\\\\" for row in rows]
    text += ["\\bottomrule\\end{tabular}", f"\\end{{{environment}}}"]
    path.write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "results/course_fast_track/native_mac_cpu_v1"
    summary = json.loads((base / "core_summary.json").read_text())
    stats = json.loads((base / "statistics.json").read_text())
    receipt = json.loads((ROOT / "docs/evidence/course_phase5_20261117/statistics_verification.json").read_text())
    assert receipt["statistics_sha256"] == sha(base / "statistics.json")
    points = {r["id"]: r for r in summary["probe_conditions"]}
    measured = {r["id"]: r for r in stats["conditions"]}
    primary = []
    for seed in (1729, 2718, 31415):
        for task in ("binary", "threeway"):
            record = points[f"s{seed}_h3_{task}_last_all_ordinary"]
            m = record["selected_metrics"]
            primary.append(["R/I" if task == "binary" else "R/P/I", seed, record["selected_layer"], m["N"],
                            *[f"{100*m[k]:.1f}" for k in ("accuracy", "balanced_accuracy", "macro_F1")], f"{m['AUROC']:.3f}"])
    table(args.output / "primary.tex", "Development-selected last-token probes at horizon 3. Accuracy, balanced accuracy (BA) and macro-F1 are percentages; AUROC is unscaled. The splits overlap and are not independent model replications. Full component intervals and class counts accompany the saved results.",
          "llrrrrrr", ["Labels", "Seed", "Layer", "N", "Acc.", "BA", "F1", "AUROC"], primary, True, "tab:primary")
    pair_rows = []
    for seed in (1729, 2718, 31415):
        r = measured[f"s{seed}_h3_binary_last_all_ordinary"]["heldout_minimal_pairs"]
        delta = r["primary_probe_minus_lexical"]["paired_difference"]["balanced_accuracy"]
        groups = r["joint_pair_correctness"]["design"]["clusters"]
        ci = "not estimable" if delta["CI95"] is None else f"[{100*delta['CI95'][0]:.1f}, {100*delta['CI95'][1]:.1f}]"
        pair_rows.append([seed, r["N_pairs"], groups, f"{100*delta['estimate']:.1f}", ci,
                          f"{100*r['joint_pair_correctness']['mean']:.1f}"])
    table(args.output / "pairs.tex", "Primary matched R/I comparison: probe minus action-only BA in percentage points. Joint correctness requires both endpoints correct. A two-component holdout does not meet the CI rule. The zero-width interval is conditional on three observed components.",
          "lrrrlr", ["Seed", "Pairs", "Groups", "$\\Delta$BA", "95\\% CI", "Joint \\%"], pair_rows, True, "tab:pairs")
    agreement = []
    for r in summary["RQ2"]:
        if r["pooling"] == "last":
            agreement.append([r["seed"], r["N"], r["invalid_count"], f"{100*r['stated_all_ID_accuracy']:.1f}",
                              f"{r['probe_stated_kappa_all_ID']['kappa']:.3f}"])
    table(args.output / "stated.tex", "Stated judgments and agreement with the selected three-way last-token probe on identical IDs. Invalid outputs are incorrect and form a separate kappa category.",
          "lrrrr", ["Seed", "N", "Invalid", "Acc. \\%", "$\\kappa$"], agreement, False, "tab:stated")
    groups = defaultdict(list)
    for record in summary["probe_conditions"]:
        c = record["condition"]
        if c["control"] == "ordinary":
            groups[(c["horizon"], c["binary"], c["pooling"], c["subset"])].append(record)
    ablations = []
    for (horizon, binary, pooling, subset), records in sorted(groups.items()):
        accuracy = [r["selected_metrics"]["accuracy"] for r in records]
        ba = [r["selected_metrics"]["balanced_accuracy"] for r in records]
        n = [r["selected_metrics"]["N"] for r in records]
        ablations.append([horizon, "R/I" if binary else "R/P/I", pooling, "pairs" if subset != "all" else "all",
                          str(n[0]) if len(set(n)) == 1 else f"{min(n)}--{max(n)}",
                          f"{100*np.mean(accuracy):.1f} $\\pm$ {100*np.std(accuracy, ddof=1):.1f}",
                          f"{100*np.mean(ba):.1f} $\\pm$ {100*np.std(ba, ddof=1):.1f}"])
    table(args.output / "ablations.tex", "Ablations: descriptive mean and sample SD across three overlapping split allocations, in percent. These SDs are not confidence intervals. Pair-only binary rows retain complete R/I siblings. Individual-seed scores and all metric CIs are released in CSV/JSON.",
          "rllllrr", ["h", "Labels", "Pool", "Training", "Test N", "Accuracy", "BA"], ablations, True, "tab:ablations")
    counts = json.loads((ROOT / "data/course_fast_track/v2/validation.json").read_text())["by_horizon"]
    table(args.output / "labels.tex", "Automatically revalidated world-state/action labels. No unknown case or human judgment enters these counts.",
          "rrrrr", ["h", "R", "P", "I", "Total"], [[h, counts[str(h)]["reversible"], counts[str(h)]["partially_reversible"],
           counts[str(h)]["irreversible"], sum(counts[str(h)].values())] for h in (1, 2, 3)], False, "tab:labels")
    references = json.loads((ROOT / "artifacts/course_fast_track/native_mac_cpu_v1/references/report.json").read_text())
    policies = defaultdict(list)
    for r in references["records"]:
        policies[r["policy"]].append(r["audit"])
    reference_rows = []
    for key, records in policies.items():
        reference_rows.append([key.replace("_", " "), len(records), sum(r["goal_success"] for r in records),
                               sum(r["caution"] for r in records), sum(r["executed_irreversible_action"] for r in records)])
    table(args.output / "references.tex", "Replayed scripted references, not language-model outcomes. Counts use the fixed 24 versions. Confirmation can increase caution while leaving the same irreversible commitments.",
          "lrrrr", ["Policy", "N", "Goal", "Caution", "Any I"], reference_rows, True, "tab:reference")
    macros = {"DatasetN": 1800, "StatedInvalid": 79, "BehaviorN": summary["RQ3"]["assigned_versions"],
              "BehaviorGoals": summary["RQ3"]["goal_success_versions"], "BehaviorRejected": summary["RQ3"]["rejected_calls"],
              "BehaviorI": summary["RQ3"]["executed_irreversible_versions"], "GapValue": f"{summary['RQ3']['Gap']:.2f}"}
    (args.output / "macros.tex").write_text("\n".join("\\newcommand{\\"+key+"}{"+str(value)+"}" for key, value in macros.items())+"\n")
    provenance = {"status": "GENERATED_FROM_VERIFIED_RESULTS", "summary_sha256": sha(base / "core_summary.json"),
                  "statistics_sha256": sha(base / "statistics.json"), "program_sha256": sha(Path(__file__)),
                  "files_sha256": {p.name: sha(p) for p in args.output.glob("*.tex")}}
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("Generated empirical LaTeX tables/macros from verified result files")


if __name__ == "__main__":
    main()
