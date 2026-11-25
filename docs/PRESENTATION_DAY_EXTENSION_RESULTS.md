# Presentation-day experiments

25 November 2026 IST. Both analyses use exposed sandbox development cases. The original held-out behavior and steering requirements remain partial.

## A recovery veto can preserve task completion with a planner

We replayed all 24 saved native Qwen development episodes. At the first executed action certified irreversible by the sandbox oracle, we compared stopping with trying a verified route to the same goal. Goal attainment is measured at termination, and we count an irreversible execution at any point in the episode.

| Condition | Goal attained | At least one irreversible action |
|---|---:|---:|
| Saved model traces | 6/24 | 3/24 |
| Veto and stop | 3/24 | 0/24 |
| Veto and plan | 6/24 | 0/24 |

The veto fired in three recycling-disabled cases. In each case, the planner found and replayed the same two-step pattern: back up the file, then delete it. Each step had a simulator-certified recovery route. The planner had access to the exact simulator state and the existing recovery oracle. This retrospective intervention measures the combined system on a cohort with three triggers, without further model inference. [Protocol](PRESENTATION_RECOVERY_VETO_PROTOCOL.md) · [Per-episode evidence](evidence/presentation_day_extension_v1/recovery_veto.json)

## Additional random steering controls

The 48 fresh Qwen2.5-0.5B FP32 CPU episodes complete the 102-cell development grid across six task versions. We included a shared zero, the concept direction, and three norm-matched random directions. Each nonzero direction used four signed strengths. Across all 17 conditions, the six versions always yielded two goals attained, zero valid finishes, zero instances of caution and one episode with irreversible execution. Some output token sequences changed, including in both the concept and random conditions. No simulator trajectory changed relative to zero. All 102 episodes reached the consecutive-invalid-call stop. This failure limits what the intervention can establish about more reliable agents.

The Mac guard recorded no added swap, minimum sampled free memory of 35%, and a process peak RSS of 2.32 GiB. We independently recounted the saved traces and reproduced both aggregate reports byte for byte. [Protocol](PRESENTATION_STEERING_CONTROLS_PROTOCOL.md) · [Condition-level evidence](evidence/presentation_day_extension_v1/steering_controls.json)

## Assignment scope

The written brief assigns 40 base marks and up to two bonus marks. A controlled RQ4 or RQ5 result can earn the bonus when the report meets the brief's submission-readiness criterion. The brief explicitly allows clean negative results. The strongest bonus evidence here is the full development steering control grid with its measured failure mode. The original BF16 comparison, ScienceWorld certification and held-out evaluation of 60 independent families remain open. The Codex label review is one blinded audit. Any grading adjustment for the reported course accommodations is the instructor's decision.
