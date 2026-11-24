"""Build the two-page results supplement from the verified extension receipt."""

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/course_improvement_v1/final/i3_i5_evidence.json"
INK = colors.HexColor("#172532")
MUTED = colors.HexColor("#475A69")
ACCENT = colors.HexColor("#175B64")
LINE = colors.HexColor("#D7E1E6")
WASH = colors.HexColor("#F2F6F8")

STYLES = {
    "eyebrow": ParagraphStyle("Eyebrow", fontName="Helvetica-Bold", fontSize=9,
                              leading=12, textColor=ACCENT, spaceAfter=8),
    "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=25,
                            leading=29, textColor=INK, spaceAfter=9),
    "deck": ParagraphStyle("Deck", fontSize=11, leading=16, textColor=MUTED, spaceAfter=14),
    "section": ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=13,
                              leading=16, textColor=INK, spaceBefore=12, spaceAfter=6),
    "body": ParagraphStyle("Body", fontSize=10.5, leading=15, textColor=INK, spaceAfter=8),
    "cell": ParagraphStyle("Cell", fontSize=9.5, leading=13, textColor=INK),
    "head": ParagraphStyle("Head", fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=colors.white),
    "note": ParagraphStyle("Note", fontSize=9, leading=13, textColor=MUTED, spaceBefore=4, spaceAfter=6),
}


def para(text, role="body"):
    return Paragraph(text, STYLES[role])


def table(headers, rows, widths):
    value = Table([[para(cell, "head") for cell in headers]] +
                  [[para(cell, "cell") for cell in row] for row in rows],
                  colWidths=widths, hAlign="LEFT", repeatRows=1)
    value.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WASH, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, -1), (-1, -1), .5, LINE),
    ]))
    return value


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(52, 45, 560, 45)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(52, 31, "ADNN course project  |  local evidence supplement")
    canvas.drawRightString(560, 31, str(document.page))
    canvas.restoreState()


def deterministic_canvas(*args, **kwargs):
    return Canvas(*args, **{**kwargs, "invariant": 1})


def build(output):
    evidence = json.loads(EVIDENCE.read_text())
    development = evidence["I4"]["development"]["totals"]
    statistics = evidence["I5"]
    data = evidence["draft_data"]
    assert not evidence["outstanding"]["original_60_family_suite_complete"]
    assert statistics["matched_primary_balanced_accuracy_deltas"] == [0, 0, 0]
    assert evidence["outstanding"]["model_review_returns"] == 0
    number = development["N"]
    rows = [("Correct stated labels", f"{development['correct_stated']} / {number}"),
            ("Goals reached at stop", f"{development['goal']} / {number}"),
            ("Episodes with an irreversible action", f"{development['any_irreversible']} / {number}"),
            ("Episodes stopped by consecutive invalid outputs", f"{development['invalid_abort']} / {number}")]
    content = [
        para("RESULTS SUPPLEMENT  /  19 NOVEMBER 2026  /  IST", "eyebrow"),
        para("Action reversibility in a tool sandbox", "title"),
        para("Additional evidence for the reduced ADNN course study. "
             "Read alongside the verified eight-page paper and presentation.", "deck"),
        para("The main matched-pair result is unchanged", "section"),
        para("The native Qwen2.5-0.5B-Instruct CPU probe had a balanced-accuracy advantage of zero "
             "over the action-only baseline on matched held-out pairs in all three splits. Two intervals were "
             "not estimable because there were too few independent components. The third was [0, 0] under "
             "the observed resampling design. This does not establish equivalence on other tasks or models."),
        para(f"The extension reports interval availability across {statistics['layer_records']:,} saved layer records "
             f"and adds {statistics['selectivity_intervals']} paired selectivity intervals. One hundred layer records "
             "have too few independent components for intervals. These pointwise analyses use saved predictions."),
        para("Tool reliability still limits the behavior test", "section"),
        table(("Development outcome", "Observed"), rows, [365, 143]),
        Spacer(1, 8),
        para(f"All six visible versions received a valid stated JSON answer of reversible. The world runner "
             f"rejected {development['parse_rejected']} malformed responses and {development['state_or_schema_rejected']} "
             "tool requests outside the legal state or schema. The paired knowledge score and caution contrast "
             "were both zero, so their zero difference does not demonstrate informed action."),
        para("Protocol scope: native float32 on Mac CPU; greedy decoding; scripted safety references and model "
             "episodes kept separate. No confirmatory evaluation used the new tasks or drafts.", "note"),
        PageBreak(),
        para("DATA AND READINESS", "eyebrow"),
        para("What is ready for the next run", "title"),
        para(f"The candidate pool now contains {data['sandbox_candidates']:,} sandbox examples. "
             f"{100 * data['paired_fraction']:.1f}% are in verified minimal pairs. It contains zero certified "
             "ScienceWorld cases. The added cases are development evidence until a new freeze passes."),
        para(f"Three draft allocations each place {data['draft_test_rows']} rows in the test partition. All cover "
             "every required class at horizons 1 to 3, but any two share 450 test rows. "
             "Keep the nine-component test union outside prompt or steering development; do not count the three "
             "splits as independent replications."),
        para("Four-step checkpoint", "section"),
        table(("Phase", "Verified state"), [
            ("I3: native readiness", "Five cached checkpoint files match. Full A100 test-only forecast: "
             "16 December at 08:02 IST, after the operational deadline."),
            ("I4: core evaluation", "Six development model traces replay. No new same-ID probe records were "
             "collected for these tasks. New holdout evaluation remains unopened."),
            ("I5: statistics and steering", "Saved uncertainty verified. The earlier one-decision steering "
             "diagnostic remains separate; full-episode steering and uncertainty testing did not run."),
            ("I6: local delivery", "This supplement accompanies the original paper, slides and offline demo. "
             "Use the handoff guide and package receipt to verify the files."),
        ], [152, 356]),
        para("What still gates the original study", "section"),
        para("Certify ScienceWorld states under the frozen rule or an approved amendment, complete the behavior "
             "suite, and qualify the prescribed native checkpoints on development tasks before scaling inference. "
             "The blinded 100-case model review packets are ready; no external returns were received. "
             "No LLM output is counted as human annotation."),
        para("Traceable inputs: saved native statistics, the layerwise verification, the six-case model replay "
             "and the paid-recovery draft verification. Paths and reproduction commands are in the local "
             "improvement handoff guide.", "note"),
    ]
    document = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=46, rightMargin=46,
                                 topMargin=47, bottomMargin=57,
                                 title="Action reversibility in a tool sandbox: evidence supplement",
                                 author="ADNN Course Project")
    document.build(content, onFirstPage=footer, onLaterPages=footer, canvasmaker=deterministic_canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new PDF path")
    build(args.output)


if __name__ == "__main__":
    main()
