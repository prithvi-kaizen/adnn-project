"""Render percentage plots for the ADNN presentation from saved results."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BG = "#F7F5EF"
INK = "#132A2A"
MUTED = "#536663"
GRID = "#CED7D1"
TEAL = "#07654F"
SLATE = "#547083"
RED = "#A63424"
SCALE = 2
W, H = 1280, 520


def read(relative):
    return json.loads((ROOT / relative).read_text())


class Plot:
    def __init__(self):
        self.image = Image.new("RGB", (W * SCALE, H * SCALE), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = {}

    def text(self, xy, content, size=22, color=INK, bold=False, anchor="lm"):
        key = (size, bold)
        if key not in self.fonts:
            family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            try:
                self.fonts[key] = ImageFont.truetype(family, size * SCALE)
            except OSError:
                family = "Arial Bold.ttf" if bold else "Arial.ttf"
                self.fonts[key] = ImageFont.truetype(
                    "/System/Library/Fonts/Supplemental/" + family, size * SCALE
                )
        self.draw.text(self.coords(xy), content, color, font=self.fonts[key], anchor=anchor)

    @staticmethod
    def coords(values):
        return tuple(round(v * SCALE) for v in values)

    def line(self, xy, color=GRID, width=1):
        self.draw.line(self.coords(xy), fill=color, width=width * SCALE)

    def rect(self, box, color):
        self.draw.rectangle(self.coords(box), fill=color)

    def dot(self, x, y, color=TEAL, size=7):
        self.draw.ellipse(self.coords((x-size, y-size, x+size, y+size)), fill=color)

    def x_axis(self, left=415, right=1135, top=70, bottom=381):
        def xp(value):
            return left + (right - left) * value

        for fraction in (0, .25, .5, .75, 1):
            x = xp(fraction)
            self.line((x, top, x, bottom))
            self.text((x, bottom + 24), f"{fraction:.0%}", 17, MUTED, anchor="mm")
        self.line((left, bottom, right, bottom), MUTED)
        return xp

    def y_axis(self, left=142, right=1160, top=75, bottom=376):
        def yp(value):
            return bottom - (bottom - top) * value

        for fraction in (0, .25, .5, .75, 1):
            y = yp(fraction)
            self.line((left, y, right, y))
            self.text((left - 15, y), f"{fraction:.0%}", 17, MUTED, anchor="rm")
        self.line((left, bottom, right, bottom), MUTED)
        return yp

    def legend(self, items, x=143, y=37, step=287):
        for label, color in items:
            self.rect((x, y-7, x+22, y+7), color)
            self.text((x+33, y), label, 19)
            x += step

    def foot(self, *lines):
        for i, line in enumerate(lines):
            self.text((75, 465+i*27), line, 18, MUTED)

    def save(self, out, slide):
        self.image.save(out / f"adnn-plot-{slide:02d}.png", optimize=True)

    def hbar(self, xp, y, value, color):
        if value:
            self.rect((xp(0), y-15, xp(value), y+15), color)
        else:
            self.dot(xp(0), y, color, 5)
        label_x = xp(value) + 12 if value < .88 else xp(value) - 8
        anchor = "lm" if value < .88 else "rm"
        self.text((label_x, y), percent(value), 19, INK if value < .88 else "white", True, anchor)

    def interval(self, xp, y, estimate, ci, color):
        start, end = (xp(limit) for limit in ci)
        self.line((start, y, end, y), color, 3)
        self.line((start, y-8, start, y+8), color, 2)
        self.line((end, y-8, end, y+8), color, 2)
        self.dot(xp(estimate), y, color)
        self.text((xp(estimate), y-26), percent(estimate), 20, INK, True, "mm")


def percent(value):
    return f"{value*100:.1f}".removesuffix(".0") + "%"


def compare_probe(out, stats):
    p = Plot()
    yp = p.y_axis(left=135, right=640, top=108, bottom=366)
    p.legend([("Hidden-state probe", TEAL), ("Action-text baseline", SLATE)], x=135, y=75, step=228)
    p.text((76, 23), "Result  ·  Balanced accuracy on matched pairs", 23, INK, True)
    rows = []
    for seed in (1729, 2718, 31415):
        condition = next(c for c in stats["conditions"] if c["id"] == f"s{seed}_h3_binary_last_all_ordinary")
        rows.append(condition["heldout_minimal_pairs"])
    for i, (row, x) in enumerate(zip(rows, (234, 389, 544)), 1):
        metrics = row["primary_probe_minus_lexical"]
        probe = metrics["metrics"]["balanced_accuracy"]["estimate"]
        lexical = probe - metrics["paired_difference"]["balanced_accuracy"]["estimate"]
        for position, value, color in ((x-30, probe, TEAL), (x+30, lexical, SLATE)):
            p.rect((position-25, yp(value), position+25, yp(0)), color)
            p.text((position, yp(value)-16), percent(value), 18, INK, True, "mm")
        p.text((x, 397), f"Split {i}", 19, INK, False, "mm")
        p.text((x, 423), f"{row['N_pairs']} pairs", 16, MUTED, False, "mm")
    p.line((683, 22, 683, 440))
    p.text((716, 23), "Ablation  ·  Which action tokens?", 23, INK, True)
    p.text((716, 68), "Three-split mean balanced accuracy", 19, MUTED)
    xp = p.x_axis(left=888, right=1167, top=106, bottom=298)
    for y, pooling, label, color in [(155, "last", "Last token", TEAL),
                                     (234, "mean", "Mean of tokens", SLATE)]:
        r = [c["selected"]["metrics"]["balanced_accuracy"]["estimate"] for c in stats["conditions"]
             if c["id"].endswith(f"_h3_binary_{pooling}_all_ordinary")]
        assert len(r) == 3
        p.text((716, y), label, 19, INK, True)
        p.hbar(xp, y, mean(r), color)
    p.text((716, 351), "All binary cases, N = 330 per split", 18, MUTED)
    p.text((716, 392), "Also tested: pairs-only training, horizons 1–3", 17, MUTED)
    p.text((716, 418), "and two vs three labels. Transfer remains open.", 17, MUTED)
    p.foot("Matched difference 95% CI: not estimable in splits 1–2. Split 3: [0, 0] percentage points over three groups.",
           "The two panels use different test populations. Controls and full ablations are in the Q&A appendix.")
    p.save(out, 5)


def stated_results(out, stats, core, format_check):
    p = Plot()
    p.text((75, 19), "Question  ·  Can Qwen state whether the action can be undone?", 22, INK, True)
    p.text((75, 55), "Method  ·  Show state and action. Require one of three labels as JSON.", 20, MUTED)
    p.text((75, 103), "Result and majority baseline", 22, INK, True)
    xp = p.x_axis(left=299, right=649, top=127, bottom=344)
    stated = next(r for r in stats["RQ2"] if r["seed"] == 1729 and r["pooling"] == "last")
    majority = next(r for r in stats["conditions"] if r["id"] == "s1729_h3_threeway_last_all_ordinary")
    rows = [("Qwen stated labels", stated["stated_all_ID"]["metrics"]["accuracy"], TEAL),
            ("Training-majority label", majority["majority"]["metrics"]["accuracy"], SLATE)]
    for y, (label, metric, color) in zip((185, 286), rows):
        p.text((75, y), label, 19, INK, True)
        p.interval(xp, y, metric["estimate"], metric["CI95"], color)
    p.text((475, 402), "Accuracy with 95% confidence intervals", 18, MUTED, anchor="mm")
    p.line((683, 96, 683, 430))
    p.text((716, 103), "Ablation  ·  Remove the example", 22, INK, True)
    p.text((716, 143), "Responses passing the JSON parser", 18, MUTED)
    xp = p.x_axis(left=893, right=1167, top=167, bottom=344)
    for y, key, label, color in [(212, "baseline", "With example", TEAL),
                                 (287, "candidate", "Without example", SLATE)]:
        row = format_check[key]
        p.text((716, y), label, 19, INK, True)
        p.hbar(xp, y, row["valid"]/row["N"], color)
    p.text((1030, 402), f"Same {format_check['baseline']['N']} exposed cases per format", 18, MUTED, anchor="mm")
    data = next(r for r in core["RQ2"] if r["seed"] == 1729 and r["pooling"] == "last")
    valid = sum(v for k, v in data["confusion_truth_to_stated"].items() if k.endswith(":reversible"))
    p.foot(f"Held-out split, N = {stated['assigned_N']}: all {valid} valid answers said “reversible”.",
           "Removing the example changed the response format. The literal label stayed “reversible”.")
    p.save(out, 6)


def behavior(out, core, steer):
    p = Plot()
    p.text((75, 19), "Question  ·  Does the acting agent become more careful when undoing is impossible?", 22, INK, True)
    p.text((75, 55), "Method  ·  Give a goal and replay each tool call on both versions of the task.", 20, MUTED)
    xp = p.x_axis(top=97)
    row = core["RQ3"]
    n = row["assigned_versions"]
    completed = sum(steer["selected_terminal_status_counts"].values())
    assert completed == n
    for y, (label, val, color) in zip(
        (123, 193, 263, 333),
        [("Goal reached", row["goal_success_versions"], TEAL),
         ("Irreversible action executed", row["executed_irreversible_versions"], RED),
         ("Valid finish", steer["selected_terminal_status_counts"].get("finish", 0), SLATE),
         ("Invalid-call stop", steer["selected_terminal_status_counts"]["consecutive_invalid_abort"], SLATE)],
    ):
        p.text((75, y), label, 22, INK, True)
        p.hbar(xp, y, val/n, color)
    p.text((775, 434), "Percentage of assigned versions", 19, MUTED, anchor="mm")
    p.foot(f"N = {n} exposed Qwen development episodes. Outcomes can overlap.",
           "A goal can be reached before an episode ends on invalid calls.")
    p.save(out, 7)


def gap(out, stats, refs):
    p = Plot()
    p.text((75, 23), "Result  ·  The paired blindness gap", 23, INK, True)
    xp = p.x_axis(left=307, right=648, top=90, bottom=315)
    for y, name, label, color in (
        (140, "K", "K   Both labels correct", TEAL),
        (247, "A", "A   More cautious on\n     the irreversible side", SLATE),
    ):
        metric = stats["RQ3"]["point_and_CI"][name]
        p.text((75, y-10), label, 20, INK, True)
        p.hbar(xp, y, metric["mean"], color)
    value = stats["RQ3"]["point_and_CI"]["gap"]
    p.text((75, 384), f"K − A = {100*value['mean']:.0f} percentage points", 25, TEAL, True)
    p.text((75, 418), f"N = {value['N']} exposed pairs · conditional 95% CI [0, 0] points", 17, MUTED)
    p.line((683, 22, 683, 439))
    p.text((716, 23), "Baseline  ·  Scripted policies", 23, INK, True)
    p.text((716, 65), "Episodes with an irreversible action", 19, MUTED)
    xp = p.x_axis(left=893, right=1167, top=90, bottom=315)
    for y, key, label, color in [
        (128, "direct_shortcut", "Act immediately", RED),
        (201, "confirmation_then_goal", "Confirm then act", RED),
        (273, "always_refuse", "Always refuse", SLATE),
    ]:
        row = refs[key]
        p.text((716, y), label, 18, INK, True)
        p.hbar(xp, y, row["irreversible"]/row["N"], color)
    p.text((716, 384), "Confirmation left irreversible execution unchanged.", 17, MUTED)
    p.text((716, 418), f"N = {refs['direct_shortcut']['N']} exposed versions per scripted policy", 17, MUTED)
    p.foot("Neither correct paired knowledge nor extra caution was observed in the model pilot.",
           "The planned neutral-vs-caution prompt ablation has no completed agent result.")
    p.save(out, 9)


def steering(out, data):
    p = Plot()
    p.text((75, 19), "Question  ·  Can steering change caution or goal attainment?", 22, INK, True)
    p.text((75, 55), "Method  ·  Add a learned direction to one layer. Vary its signed strength.", 20, MUTED)
    yp = p.y_axis(top=122, bottom=360)
    xs = {strength: 202 + i*223 for i, strength in enumerate((-2, -1, 0, 1, 2))}
    endpoints = [("Goal reached", "goal", TEAL), ("Caution before action", "caution", SLATE)]
    for label, endpoint, color in endpoints:
        coords = []
        for strength, x in xs.items():
            matches = [r[endpoint]/r["N"] for r in data["conditions"] if r["strength"] == strength]
            assert len(set(matches)) == 1
            coords.append((x, yp(matches[0])))
        for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
            p.line((x0, y0, x1, y1), color, 4)
        for x, y in coords:
            p.dot(x, y, color)
        p.text((1130, coords[-1][1]-15), percent(matches[0]), 19, color, True, "rm")
    p.legend([(label, color) for label, _, color in endpoints], y=94, step=259)
    for strength, x in xs.items():
        p.text((x, 390), str(strength).replace("-", "−"), 20, anchor="mm")
    p.text((650, 425), "Signed steering strength", 19, MUTED, anchor="mm")
    p.foot(f"Baseline: zero steering and three same-norm random directions. Curves overlap. N = {data['total_episodes']} development episodes.",
           "All trajectories matched zero strength. All episodes ended at the invalid-call limit.")
    p.save(out, 10)


def uncertainty(out, data):
    p = Plot()
    p.text((75, 19), "Question  ·  Does prediction uncertainty flag irreversible actions?", 22, INK, True)
    p.text((75, 55), "Method  ·  Predict the next game state. Score how surprising the generated tokens are.", 20, MUTED)
    xp = p.x_axis(top=125)
    for y in range(125, 379, 12):
        p.line((xp(.5), y, xp(.5), min(y+6, 381)), SLATE)
    p.text((xp(.5), 104), "Chance baseline", 19, MUTED, anchor="mm")
    scores = data["metrics_irreversible_positive"]["mean_nll"]
    for y, (key, label, color) in zip((201, 309), [
        ("aggregate", "All game cases", SLATE),
        ("within_game_and_verb", "Sensitivity check", TEAL),
    ]):
        row = scores[key]
        p.text((75, y-14), label, 22, INK, True)
        subgroup = "Same game and verb · " if key == "within_game_and_verb" else ""
        p.text((75, y+17), f"{subgroup}N = {row['scored_case_count']}", 18, MUTED)
        p.interval(xp, y, row["auroc"], row["CI95"], color)
    p.text((775, 434), "AUROC shown as a percentage with 95% confidence intervals", 19, MUTED, anchor="mm")
    p.foot(f"No valid structured next-state predictions among {data['scored_cases']} cases.",
           "AUROC ranks cases using uncertainty in the generated text.")
    p.save(out, 11)


def recovery(out, data):
    p = Plot()
    yp = p.y_axis(top=75, bottom=376)
    p.legend([("Goal reached", TEAL), ("Irreversible action executed", RED)], step=244)
    for x, (key, label) in zip((328, 650, 972), [
        ("baseline", "Original model traces"),
        ("veto", "Block and stop"),
        ("veto_with_planner", "Block and plan"),
    ]):
        row = data["outcomes"][key]
        for pos, endpoint, color in ((x-53, "goal", TEAL), (x+53, "irreversible", RED)):
            val = row[endpoint]/data["N"]
            if val:
                p.rect((pos-42, yp(val), pos+42, yp(0)), color)
            else:
                p.dot(pos, yp(0), color, size=5)
            p.text((pos, yp(val)-17), percent(val), 19, INK, True, "mm")
        p.text((x, 410), label, 20, anchor="mm")
    p.foot(f"Same {data['N']} saved development episodes in every condition.",
           "An exact sandbox check triggered on three traces. The planner backed up the file before deleting.")
    p.save(out, 12)


def verified_checks():
    stored = ROOT / "deliverables/results/talk-comparison-checks.json"
    format_path = ROOT / "docs/evidence/course_improvement_v1/i2_formats/verification.json"
    ref_path = ROOT / "artifacts/course_fast_track/native_mac_cpu_v1/references/report.json"
    if not (format_path.exists() and ref_path.exists()):
        return json.loads(stored.read_text())

    fmt = json.loads(format_path.read_text())
    ref = json.loads(ref_path.read_text())
    assert fmt["status"] == "PASS_DEVELOPMENT_STATED_RECHECK" and fmt["holdout_calls"] == 0
    assert not fmt["agent_authorized"] and not ref["is_model_result"]
    checks = {
        "stated_format": {key: {field: value[field] for field in ("N", "valid", "correct")}
                          for key, value in fmt["counts"].items()},
        "scripted_policies": {},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (format_path, ref_path)},
    }
    for policy in ("direct_shortcut", "confirmation_then_goal", "always_refuse"):
        records = [r["audit"] for r in ref["records"] if r["policy"] == policy]
        assert len(records) == ref["assigned_versions"]
        checks["scripted_policies"][policy] = {
            "N": len(records),
            "irreversible": sum(r["executed_irreversible_action"] for r in records),
            "goal": sum(r["goal_success"] for r in records),
        }
    if stored.exists():
        assert checks == json.loads(stored.read_text())
    else:
        stored.write_text(json.dumps(checks, indent=2) + "\n")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    paths = "results/course_fast_track/native_mac_cpu_v1/"
    stats = read(paths + "statistics.json")
    core = read(paths + "core_summary.json")
    evidence = "deliverables/results/"
    checks = verified_checks()
    compare_probe(out, stats)
    stated_results(out, stats, core, checks["stated_format"])
    behavior(out, core, stats["RQ3"])
    gap(out, stats, checks["scripted_policies"])
    steering(out, read(evidence + "full-episode-steering.json"))
    uncertainty(out, read(evidence + "bytesized32-uncertainty.json"))
    recovery(out, read(evidence + "recovery-veto.json"))


if __name__ == "__main__":
    main()
