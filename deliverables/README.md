# ADNN submission

## Present

- [Editable PowerPoint](reversibility-talk.pptx) with code-generated percentage charts, clickable navigation, fade transitions, and presenter notes.
- [Clickable PDF backup](reversibility-talk.pdf).
- [Timed speaker notes](speaker-notes.md). The 13 main slides total 10:20, including the demo. The talk moves from the problem through RQ1–RQ5 and the recovery planner. Each section gives the method, result, baseline and ablation or sensitivity check when available. Use slides 14–20 for the separate three-minute Q&A.
- [36-second replay video](demo-replay.mp4) and [original transcript](demo-replay.txt). The video captions come from a fresh execution of the saved-model replay. The live option runs the command below in front of the audience. It replays frozen outputs in an isolated sandbox.
- [Image credits and license](visual-credits.md).

## Submit and verify

| Required item | File or source |
|---|---|
| ACL-style paper and editable LaTeX | [Report](report.pdf) · [Portable source](paper-source.zip) |
| Data and splits | [Sandbox dataset](../data/course_fast_track/v2/sandbox_pool.jsonl) · [Grouped splits](../data/course_fast_track/v2/splits) · [Freeze manifest](../data/course_fast_track/v2/freeze_manifest.json) · [Dataset card](../docs/COURSE_DATA_CARD.md) |
| Baselines, ablations and statistics | [Rubric evidence map](rubric-evidence.md) · [Probe and stated results](../results/course_fast_track/native_mac_cpu_v1/core_summary.json) · [Full statistical results](../results/course_fast_track/native_mac_cpu_v1/statistics.json) |
| Code, dependencies and seeds | [Repository code](../src) · [Executable scripts](../scripts) · [Observed environment](../requirements-course-cpu-observed.txt) · [Experiment configuration](../configs/course_execution_native_mac_cpu_v1.json) |
| Recorded executions and bonuses | [Native behavior trace directory](../artifacts/course_fast_track/native_mac_cpu_v1/behavior) · [Additional results](results) |
| Contribution log | [Existing contribution record](../docs/CONTRIBUTIONS.md). Team members still supply their own author and contribution attestations. |

From the repository root, run:

```sh
python3 scripts/demo_course_project.py --pair pilot-delete-0
```

To rebuild the paper, extract the portable LaTeX archive and run `latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex` in the extracted directory. The companion results folder contains public aggregates for later work. The paper's limits apply to the original prescribed study, including ScienceWorld certification, the prescribed BF16 comparison and the independent 60-family evaluation.

The release excludes model weights, credentials, private annotation packets and raw upstream ByteSized32 inputs.
