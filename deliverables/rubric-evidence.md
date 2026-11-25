# Rubric evidence for sections 7.2–7.4

This evidence map describes the measured sandbox study, its local Q4_K_M follow-ups and the missing original experiments. It accompanies the slides and the paper.

## 7.2 Baselines and controls

| Requirement | Verified course evidence |
|---|---|
| Majority class | Training-majority classification for the frozen native probe setups. Slide 6 also gives the comparable three-class majority baseline of 41.7% for the primary displayed stated split. |
| Lexical baseline | Bag-of-words logistic regression on action text. On the primary matched binary test, balanced accuracy is 50.0% for both the probe and lexical baseline in all three splits. |
| Shuffled-label probe | Ten seeded training-label permutations. Example full binary test for seed 1729 gives 50.2% averaged accuracy over permutations. |
| Fixed random labels by verb | Train on a fixed random label for each action verb. Ordinary probe accuracy minus control-task accuracy gives −32.1 percentage points on the same illustrative split. |
| Layer 0 | The embedding-layer probe supplies the starting comparison. It has 45.5% accuracy on the example split. |
| Same-norm random steering | Three random vectors normalized to the concept-direction norm, four signed nonzero strengths, and the shared zero. All 102 full development episodes hit the invalid-call stop. No simulator trajectory differed from zero. |

[Native conditions and intervals](../results/course_fast_track/native_mac_cpu_v1/statistics.json) · [Full steering grid](results/full-episode-steering.json)

## 7.3 Ablations

The paper's ablation table gives the mean and sample standard deviation for 16 executed sandbox configurations.

| Axis | Status |
|---|---|
| Last action token vs mean over action tokens | Measured in the sandbox |
| Minimal-pairs-only vs all data | Measured in the sandbox. The eligible test populations differ by setup |
| Sandbox → ScienceWorld and ScienceWorld → Sandbox | Open. Zero physical labels passed state certification |
| Recovery horizon 1, 2 and 3 | Measured in the sandbox |
| Binary vs three-class | Measured in the sandbox |
| Prompt caution vs neutral | No final paired estimate for the original scope |
| 7B vs 14B size | No controlled size estimate. Cached alternatives differ in model identity and quantization |

## 7.4 Statistics

- Scenario-grouped split seeds are 1729, 2718 and 31415. Report the mean and sample SD across the three allocations. The headline matched balanced-accuracy comparison is 50.0% ± 0.0 percentage points for both classifiers. The splits overlap.
- The uncertainty pipeline resamples whole scenario components with 1,000 bootstrap draws. The first two matched allocations have two groups, below the threshold for an interval. The third has three groups and a conditional zero-width interval.
- The development caution comparison gives exact McNemar p = 1.00 on nine eligible pairs, with zero discordant pairs. Related tasks share templates. This describes the pilot.
- The study specified metrics, split and tuning constraints before test assessment. The register preserves unsuccessful studies and failure reasons. Invalid model calls remain in the reported denominators. [Experiment protocol](../docs/EXPERIMENT_PROTOCOL.md) · [Experiment registry](../results/registry.jsonl)

## Recorded scope

The Mac handoff contains a frozen 1,800-example sandbox dataset, a 100-case single blinded LLM label audit, native small-Qwen results, separate quantized results, both bonus diagnostics and a scripted recovery replay. The prescribed BF16 model comparison, certified joint dataset and original 60-family holdout remain open. The audit covers 98/100 correct labels with two failed recovery witnesses. No human judgments or team attestations are inferred.
