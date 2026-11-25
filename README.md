# Do AI Agents Know Which Actions Cannot Be Undone?

This ADNN course project studies whether language models represent action reversibility and act on it. In the synthetic sandbox, changing one state property can make the same action reversible in one case and irreversible in the other. The project compares linear probe results, stated judgments, and replayed agent behavior.

## Course deliverables

The latest [submission folder](deliverables/) contains the ACL paper, the 11-minute interactive presentation, the recorded replay, and the rubric evidence map.

- [PowerPoint](deliverables/reversibility-talk.pptx) · [PDF backup](deliverables/reversibility-talk.pdf) · [Timed notes](deliverables/speaker-notes.md)
- [Replay video](deliverables/demo-replay.mp4) · [ACL paper](deliverables/report.pdf) · [Portable LaTeX](deliverables/paper-source.zip)
- [Dataset and splits](data/course_fast_track/v2) · [Rubric evidence](deliverables/rubric-evidence.md) · [Additional result receipts](deliverables/results)

## Replay the demos

Run these commands from the repository root with Python 3. They replay saved outputs in the simulated sandbox and require no model weights.

```sh
python3 scripts/demo_course_project.py --pair pilot-delete-0
python3 scripts/demo_course_improvement.py
python3 scripts/run_course_full_episode_diagnostic.py \
  artifacts/course_fast_track/native_full_episode_steering_v1 \
  --verify-to /tmp/adnn-steering-replay.json
```

Use the bundled LaTeX sources to rebuild the paper. The release includes selected saved traces, probe predictions, plots, and configuration files. Full model reruns require the separately installed dependencies and the checkpoints named in the experiment records.

## Scope

The paper reports the measured course study. The original two-checkpoint BF16 comparison, ScienceWorld label certification, and 60-family behavior evaluation remain open. The later Mac uncertainty diagnostic is documented separately. The public release includes aggregate evidence for the blinded review and ByteSized32. The raw records for those studies are retained privately.

[Experimental protocol](docs/EXPERIMENT_PROTOCOL.md) · [Original-study status](docs/ORIGINAL_STUDY_RESTART_20261120.md)
