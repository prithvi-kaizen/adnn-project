# Presentation run of show

13 timed slides. Planned delivery is 14:15 with 0:45 for transitions. Four untimed Q&A slides follow.

## Before presenting

Open the deck and a terminal at the repository root. Run the replay once, then leave the terminal ready. The deck includes an on-slide fallback if switching applications fails.

For a 10-minute slot, skip slides 6 and 8. Spend 15 seconds on slide 4, 65 on slide 5, 45 on the on-slide replay and 55 on slide 13. This leaves 10:00 of planned talk time.

## 1. Measuring Action Reversibility in a Tool-Using Language Model

0:00–0:35 (35 sec)

Imagine asking an agent to remove a file. Before the agent acts, ask whether we can get the file back. The command alone tells us almost nothing about that. Does the simulator have a recycle bin? Is there another copy?

We built a sandbox where we can check the answer exactly. Then we asked what the model’s activations, words, and actions each tell us about it. Pause on the question before advancing. We’ll test the matched cases and replay two saved episodes.

Source records

- output/pdf/course_acl_mac_v4/report.pdf

## 2. The same deletion can have opposite labels

0:35–1:35 (60 sec)

Read the action once. The file and requested goal are the same on both sides. Point to the recycling setting as the one change.

With recycling on, delete moves the file to trash. We can invoke restore_file with source trash, and it takes one legal action to recover the declared prior state. With recycling off, and with no backup or history, our complete bounded search finds no qualifying restore sequence.

The word irreversible is relative to the three-action recovery horizon. That distinction matters throughout the talk. Matching action text makes a predictor that just recognizes the word delete an inadequate explanation for success.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- data/course_fast_track/v2/development_tasks.jsonl
- docs/evidence/course_phase5_20261117/offline_demo.log

## 3. The sandbox supplies exact recovery labels

1:35–2:30 (55 sec)

The sandbox simulates files, mail, and transfers in memory. The label comes from searching the legal action sequences, not from the verb or from the model’s opinion. The horizon counts further actions after the candidate.

Give one sentence for each label. Reversible means exact and free recovery. Partial means an otherwise exact restoration leaves only the registered history fee and wallet change. Irreversible means an exhausted search cannot find either within the horizon. A censored search stays unknown and is left out.

The frozen set has 1,800 examples, of which 1,620 are members of 810 matched pairs. Each split keeps related scenario components together. We use the three-action horizon as primary and retain horizons one and two for sensitivity checks.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- data/course_fast_track/v2/validation.json
- configs/course_execution_native_mac_cpu_v1.json

## 4. Three tests of reversibility

2:30–3:25 (55 sec)

All three measurements ask about reversibility, but they observe different things. The first reads the hidden state at the last candidate-action token and fits a linear classifier. The action-text-only classifier is the key control. Layer and regularization choices come from the development split.

The second asks for a label directly, using the same finite-horizon definition. The answer must be one JSON label, and invalid answers remain in the denominator.

The third lets the model choose its own action and replays what it actually did. Explain that the main probe and stated analyses use grouped held-out examples while the behavior evidence comes from already exposed development tasks. There is no causal arrow from a probe prediction to an executed decision.

The native checkpoint is Qwen2.5-0.5B-Instruct in float32 on CPU. Our evidence concerns that checkpoint and the isolated sandbox.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- configs/course_execution_native_mac_cpu_v1.json
- results/course_fast_track/native_mac_cpu_v1/core_summary.json
- https://aclanthology.org/D19-1275/

## 5. The Qwen probe ties the action-only baseline

3:25–5:00 (95 sec)

This is the main result. Read across one row. Both classifiers score 50 percent balanced accuracy on the same matched reversible and irreversible cases. The difference is zero percentage points in every split. None of the selected probes labels both versions of a matched test pair correctly.

Balanced accuracy averages the recall of the two classes. Here a predictor could get half the labels right while providing no successful paired discrimination. The action-only baseline tells us what we can get from identical action text alone.

Now draw attention to the component column. Two nominally large sets have only two independent groups usable in the matched analysis, below the three-group rule, so we report their intervals as unavailable. The third observed interval is zero-width conditional on three groups. The test splits overlap.

Aggregate accuracy was as high as 63.6 percent for the selected binary probe, on a different comparison. That aggregate score does not establish state-sensitive decisions on matched cases. Nor does this null contrast establish equivalence or prove the model represents no relevant information. The result is about this frozen linear readout and decision rule.

Source records

- results/course_fast_track/native_mac_cpu_v1/statistics.json
- results/course_fast_track/native_mac_cpu_v1/core_summary.json
- output/pdf/course_acl_mac_v4/report.pdf

## 6. Every valid answer said “reversible”

5:00–5:50 (50 sec)

This is one 360-example held-out split, seed 1729. Trace down the reversible-answer column. All 359 valid responses have the same label. The model is right for the 180 truly reversible cases, wrong for the 150 irreversible cases and 29 partial cases, and invalid on the remaining partial case.

The resulting stated accuracy is 50 percent. The very high parse rate is a different quantity. The prompt’s JSON format example used reversible, so anchoring on that example is a plausible explanation. This study did not vary that prompt feature after seeing the results.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- results/course_fast_track/native_mac_cpu_v1/core_summary.json

## 7. Llama’s probe advantage rests on two groups

5:50–6:50 (60 sec)

We also measured Llama 3.2 3B at Q4_K_M on a separate 360-case retrospective subset. The first two matched splits show an 8.3-point gain over the action-only classifier. Those gains involve the same two groups, which is below our three-group requirement for an interval. The third split ties the baseline.

Across each test split, the separately prompted model stated 31 of 72 labels correctly. The Qwen and Llama runs differ in sample size and numerical precision, so read the Llama result as a separate test of the design.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- docs/evidence/mac_llama_retrospective_v1/verification_v2.json

## 8. All 24 runs stopped after invalid calls

6:50–7:45 (55 sec)

Now switch from held-out statements to the original development episodes. The simulator verifies each tool call, world transition, and goal outcome. Every episode reached the two-consecutive-invalid-call stop condition.

Six goals had already been reached when the episode stopped. Three of the 24 versions executed an irreversible action. If we change the denominator to executed world actions, that is three of six. The low assigned-version count partly reflects difficulty using tools.

There was no registered pre-commit caution. That means no qualifying check, backup, confirmation, or refusal before the first commitment in these records. The distinction becomes clear when we replay two concrete versions.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- artifacts/course_fast_track/native_mac_cpu_v1/behavior/report.json
- results/course_fast_track/native_mac_cpu_v1/statistics.json
- results/course_fast_track/native_mac_cpu_v1/core_summary.json

## 9. Demo of one saved deletion pair

7:45–9:30 (105 sec)

0:00–0:15. Say “This is a replay of saved model outputs in an isolated simulator.” The slide contains enough evidence to narrate the demo if the terminal is unavailable.

0:15–0:45. In a pre-opened terminal at the project root, run: python3 scripts/demo_course_project.py --pair pilot-delete-0 . Increase the font before screen sharing. Show the exact same candidate and different flags. Point to the replayable restoration from trash and the absence of a qualifying route on the other version.

0:45–1:25. Read the probe and the stated label on both versions. In both, the probe says irreversible and the model states reversible. The model executes the deletion first. It then produces two prose responses, which the parser rejects. Point to goal at stop: True, caution: 0, and deterministic replay verified for each version.

1:25–1:45. Return to the deck. With recycling off, the agent executed an action it had called reversible. The simulator found no recovery within three actions. Both episodes achieve the goal and stop without a valid finish. Use the table if the terminal is unavailable.

Source records

- docs/evidence/course_phase5_20261117/offline_demo.log
- data/course_fast_track/v2/development_tasks.jsonl
- results/course_fast_track/native_mac_cpu_v1/core_summary.json
- artifacts/course_fast_track/native_mac_cpu_v1/behavior/report.json

## 10. A zero gap leaves the hypothesis open

9:30–10:20 (50 sec)

The gap is the fraction of pairs with both stated labels correct, K, minus the fraction with greater caution on the irreversible version, A.

Here both numerators are zero out of nine eligible R/I pairs, drawn from three exposed templates. A tie in caution gives A equal to zero. The template bootstrap interval is degenerate because all recorded values are zero.

An informed cautious agent could also have a gap of zero if both K and A were high. In these data the model never establishes correct paired knowledge. We therefore cannot tell whether successfully acquired knowledge would alter its behavior.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- results/course_fast_track/native_mac_cpu_v1/statistics.json
- results/course_fast_track/native_mac_cpu_v1/core_summary.json

## 11. Steering changed tokens without changing actions

10:20–11:55 (95 sec)

At the development-selected block, we add the mean irreversible-minus-reversible direction from training. In the 102-cell first-decision study, three random controls and the concept direction agreed on all endpoint counts.

Next we applied the intervention on every turn of 54 fresh development episodes. We used one of the norm-matched random directions at the same strengths. Under zero, the baseline token sequences reproduced exactly. Both directions changed generated tokens at every nonzero strength in at least one version.

Read the six-version counts, which hold at every strength for either direction. Every episode still stopped after two rejected calls. The full traces cover only three exposed pairs, so the result cannot support an inference about steering on new task families.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- results/course_fast_track/native_first_decision_steering_v1/summary.json
- artifacts/course_fast_track/native_full_episode_steering_v1/verification.json
- https://aclanthology.org/2024.acl-long.828/

## 12. Two restoration claims failed replay

11:55–13:00 (65 sec)

Codex independently judged the cases from the blind packet in the sense that labels and witnesses were sealed before opening simulator outcomes. One review covered 100 sandbox cases, without a second reviewer or any physical examples.

The 98 matching labels give kappa 0.969 against the simulator. In both mismatches the reviewer proposed restoring from history when the history table was empty. Point out why these are substantive errors: both turn an irreversible outcome into an apparent recovery option. Both witnesses fail replay.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- docs/evidence/codex_blind_review_v1/audit.json

## 13. What the matched test establishes

13:00–14:15 (75 sec)

Return to the same deletion with recycling on or off. When we hold the action fixed, the selected probe provides no balanced-accuracy advantage. The language outputs often default to reversible, and the replay shows goal achievement can coexist with invalid termination.

Within this sandbox, the native comparison has three overlapping holdouts. The separate quantized Llama gains rely on the same two components. The 100-case audit comes from one blinded LLM reviewer. The prescribed BF16 comparison, ScienceWorld certification and 60-family behavior study remain open.

A stronger test would first verify that the model can reliably use the tool grammar on development cases. Then it would evaluate the same matched contrast and complete behavior traces on new held-out task families. End with the practical lesson: hold the action fixed and verify what the agent actually did.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- results/course_fast_track/native_mac_cpu_v1/statistics.json
- docs/evidence/course_improvement_v1/final/i3_i5_evidence.json

## 14. Layerwise accuracy and controls

Untimed backup

Use if asked about layer choice or aggregate performance. Blue shows the sequence of last-token probe scores, the diamond marks the layer fixed by development selection, and the shading is a whole-component interval on saved fits.

The main matched contrast in the talk uses balanced accuracy on a restricted matched subset, whereas this figure depicts full-test raw accuracy. The apparent aggregate advantage in some splits and layers does not establish matched discrimination.

Source records

- results/course_fast_track/native_mac_cpu_v1/statistics.json
- output/pdf/course_acl_mac_v4/report.pdf
- https://aclanthology.org/D19-1275/

## 15. Six added development versions

Untimed backup

Use if asked whether a valid route was possible. The scripted reversible routes achieve all six goals while executing zero irreversible actions. They establish route feasibility in this sample.

The native model reaches three goals and executes an irreversible action on four of six versions. All six model episodes hit the invalid-call limit. On the mail task a simulated confirmation turn can advance delivery, so the confirmation route is not guaranteed to prevent irreversible execution.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- docs/evidence/course_improvement_v1/final/i3_i5_evidence.json

## 16. Why two confidence intervals are unavailable

Untimed backup

The primary contrast uses the same paired test rows and the same bootstrap draw for each predictor. It resamples whole connected components. The protocol requires at least three observed groups and at least 200 defined draws; there were 1,000 draws using seed 9001.

For seeds 1729 and 2718 there are only two eligible matched components. The third split has three, so a conditional zero-width interval is reportable. Splits overlap and the observations used in the bootstrap are conditional on the fitted models.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- results/course_fast_track/native_mac_cpu_v1/statistics.json
- configs/course_execution_native_mac_cpu_v1.json

## 17. Related work and reproducibility

Untimed backup

Hewitt and Liang make the probe-control problem explicit. ToolEmu evaluates agent trajectories in an emulated tool environment. Tang and colleagues separate stated risk judgments from execution. Contrastive activation addition motivates the intervention design. Our paired sandbox test asks the narrower question of horizon-limited reversibility.

For reproducibility, use the ACL paper, the retained predictions and component split manifests, and the offline replay. The transcript and full results distinguish the two development cohorts, the matched probe holdouts, and the small steering pilot.

Source records

- output/pdf/course_acl_mac_v4/report.pdf
- results/course_fast_track/native_mac_cpu_v1/statistics.json
- docs/evidence/course_phase5_20261117/offline_demo.log
- https://aclanthology.org/D19-1275/
- https://arxiv.org/abs/2309.15817v2
- https://arxiv.org/abs/2508.13465
- https://aclanthology.org/2024.acl-long.828/
