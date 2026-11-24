# Single blinded Codex review of the sandbox

20 November 2026 IST. The researcher reports that the instructor accepted the previously discussed accommodations and asked the team to do its best within the constraints. The researcher also explicitly asked Codex to perform the LLM review. This report records the review and its limits. [Course scope record](COURSE_SCOPE_CONFIRMATION_20261120.md)

Codex reviewed the 100 sandbox cases in packet A in a single session. The packet contains the simulator rules, complete starting states and candidate actions. It excludes the oracle labels and previously computed recovery witnesses. All five first-pass decision files and their structured equivalents were sealed with SHA-256 hashes before the oracle comparison. The model designation is the one provided by the researcher. The scorer checks the source packet, validates every returned ID, recomputes all 300 oracle horizon labels and replays the proposed witnesses. [Scoring receipt](evidence/codex_blind_review_v1/audit.json)

| Measure | Result |
|---|---:|
| Agreement with the simulator oracle | 98/100 |
| Whole-component 95% interval for agreement | 96.6% to 100% |
| Cohen's κ against the oracle | 0.969 |
| Whole-component 95% interval for κ | 0.949 to 1.000 |
| Valid exact-recovery witnesses | 34 |
| Valid paid-recovery witnesses | 24 |
| Illegal recovery witnesses and false recoverability claims | 2 |

The interval uses 1,000 resamples of the 15 scenario components. Recall was 34/34 for reversible, 24/24 for partially reversible and 40/42 for irreversible. Codex assumed a history entry was available in two cases where the history table was empty. Both witnesses failed at `restore_file`. The positive history fee and sufficient wallet balance did not establish that a stored version existed. The first-pass files and all simulator labels remain intact. The review passed the numerical agreement and recall thresholds and failed the zero-false-recoverable and zero-invalid-witness checks.

This is one model-versus-oracle audit of a sandbox sample. Its five batches shared a conversation, so the run supplies one reviewer and zero human judgments. No reviewer-to-reviewer agreement or joint ScienceWorld audit is available. The earlier two-reviewer rule therefore remains unfulfilled. The academic accommodation permits reporting the completed audit as part of the constrained course submission.

## Offline reproduction

The scorer verifies the first-pass seal before loading the case key. It replays the labels and witnesses without running a model. Use the original verified sandbox checkout or the nested course handoff, then extract the review supplement on top.

```sh
python scripts/score_codex_blind_review.py \
  --review data/annotation/llm_pilot_v1/private/codex_single_a_v1 \
  --output /tmp/adnn-codex-blind-replay.json
```
