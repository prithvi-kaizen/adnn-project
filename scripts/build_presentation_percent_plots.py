"""Render percentage plots for the ADNN presentation from saved results."""
import argparse
import json
from pathlib import Path

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
    yp = p.y_axis()
    p.legend([("Hidden-state probe", TEAL), ("Action-text baseline", SLATE)])
    rows = []
    for seed in (1729, 2718, 31415):
        condition = next(c for c in stats["conditions"] if c["id"] == f"s{seed}_h3_binary_last_all_ordinary")
        rows.append(condition["heldout_minimal_pairs"])
    for i, (row, x) in enumerate(zip(rows, (328, 650, 972)), 1):
        metrics = row["primary_probe_minus_lexical"]
        probe = metrics["metrics"]["balanced_accuracy"]["estimate"]
        lexical = probe - metrics["paired_difference"]["balanced_accuracy"]["estimate"]
        for position, value, color in ((x-53, probe, TEAL), (x+53, lexical, SLATE)):
            p.rect((position-42, yp(value), position+42, yp(0)), color)
            p.text((position, yp(value)-16), percent(value), 19, INK, True, "mm")
        p.text((x, 408), f"Split {i}  ·  {row['N_pairs']} pairs", 20, INK, False, "mm")
    p.foot("Balanced accuracy on matched reversible and irreversible examples.",
           "Difference CI: not estimable for splits 1–2. Split 3: [0, 0] percentage points, conditional on three groups.")
    p.save(out, 5)


def stated_results(out, stats, core):
    p = Plot()
    xp = p.x_axis()
    stated = next(r for r in stats["RQ2"] if r["seed"] == 1729 and r["pooling"] == "last")
    majority = next(r for r in stats["conditions"] if r["id"] == "s1729_h3_threeway_last_all_ordinary")
    rows = [("Qwen stated labels", stated["stated_all_ID"]["metrics"]["accuracy"], TEAL),
            ("Training-majority label", majority["majority"]["metrics"]["accuracy"], SLATE)]
    for y, (label, metric, color) in zip((152, 287), rows):
        p.text((75, y), label, 23, INK, True)
        p.interval(xp, y, metric["estimate"], metric["CI95"], color)
    p.text((775, 434), "Three-class accuracy with 95% confidence intervals", 19, MUTED, anchor="mm")
    data = next(r for r in core["RQ2"] if r["seed"] == 1729 and r["pooling"] == "last")
    valid = sum(v for k, v in data["confusion_truth_to_stated"].items() if k.endswith(":reversible"))
    p.foot(f"Held-out split, N = {stated['assigned_N']}: all {valid} valid answers said “reversible”.",
           "The accuracy denominator includes invalid responses.")
    p.save(out, 6)


def behavior(out, core, steer):
    p = Plot()
    xp = p.x_axis(top=54)
    row = core["RQ3"]
    n = row["assigned_versions"]
    completed = sum(steer["selected_terminal_status_counts"].values())
    assert completed == n
    for y, (label, val, color) in zip(
        (104, 177, 250, 323),
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


def gap(out, stats):
    p = Plot()
    xp = p.x_axis(top=61, bottom=335)
    for y, name, label, color in (
        (131, "K", "K   Both labels correct", TEAL),
        (251, "A", "A   More cautious on\n     the irreversible side", SLATE),
    ):
        metric = stats["RQ3"]["point_and_CI"][name]
        p.text((75, y-11), label, 22, INK, True)
        p.hbar(xp, y, metric["mean"], color)
    value = stats["RQ3"]["point_and_CI"]["gap"]
    p.text((75, 412), f"Blindness gap  K − A  =  {100*value['mean']:.0f} percentage points", 26, TEAL, True)
    p.foot(f"N = {value['N']} exposed pairs. The gap interval is [0, 0] points over three templates.",
           "Here, the zero gap comes from zero correct pair judgments and zero increases in caution.")
    p.save(out, 9)


def steering(out, data):
    p = Plot()
    yp = p.y_axis(top=76, bottom=360)
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
    p.legend([(label, color) for label, _, color in endpoints], step=259)
    for strength, x in xs.items():
        p.text((x, 390), str(strength).replace("-", "−"), 20, anchor="mm")
    p.text((650, 425), "Signed steering strength", 19, MUTED, anchor="mm")
    p.foot(f"Concept and three same-norm random directions overlap. N = {data['total_episodes']} development episodes.",
           "All trajectories matched zero strength. All episodes ended at the invalid-call limit.")
    p.save(out, 10)


def uncertainty(out, data):
    p = Plot()
    xp = p.x_axis()
    for y in range(71, 379, 12):
        p.line((xp(.5), y, xp(.5), min(y+6, 381)), SLATE)
    p.text((xp(.5), 52), "Chance", 19, MUTED, anchor="mm")
    scores = data["metrics_irreversible_positive"]["mean_nll"]
    for y, (key, label, color) in zip((152, 287), [
        ("aggregate", "All game cases", SLATE),
        ("within_game_and_verb", "Same game and action verb", TEAL),
    ]):
        row = scores[key]
        p.text((75, y-14), label, 22, INK, True)
        p.text((75, y+17), f"N = {row['scored_case_count']} cases", 18, MUTED)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    paths = "results/course_fast_track/native_mac_cpu_v1/"
    stats = read(paths + "statistics.json")
    core = read(paths + "core_summary.json")
    evidence = "deliverables/results/"
    compare_probe(out, stats)
    stated_results(out, stats, core)
    behavior(out, core, stats["RQ3"])
    gap(out, stats)
    steering(out, read(evidence + "full-episode-steering.json"))
    uncertainty(out, read(evidence + "bytesized32-uncertainty.json"))
    recovery(out, read(evidence + "recovery-veto.json"))


if __name__ == "__main__":
    main()
