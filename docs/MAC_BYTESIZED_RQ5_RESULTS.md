# Mac ByteSized32 uncertainty results

20 November 2026 IST. The Mac ran the cached native Qwen2.5-0.5B-Instruct checkpoint in float32 on CPU. The run used the 82 deduplicated cases from five source-audited ByteSized32 games. It scored all 37 reversible and 45 irreversible cases. The user reports no authorized GPU route. The Mac inventory showed 18 GiB of unified memory and 8.2 GiB free on disk before the run. The two prescribed checkpoint weight sets require 15.23 and 16.06 GB each, before runtime and outputs. The locally cached alternative runs constitute a separate condition. [Protocol](MAC_BYTESIZED_RQ5_PROTOCOL.md) · [Checkpoint and data provenance](ORIGINAL_STUDY_RESTART_20261120.md)

| Score with irreversible as the positive class | All 82 cases, 5 games | Same game and verb, 25 cases in 3 games |
|---|---:|---:|
| Mean generated-token NLL | 0.262 [0.130, 0.428] | 0.500 [0.167, 0.750] |
| Generated content length | 0.755 [0.629, 0.905] | 0.574 [0.167, 1.000] |
| Gold number of changed objects | 0.185 [0.026, 0.343] | 0.500 [0.500, 0.500] |

Intervals are percentiles from 1,000 whole-game bootstrap draws. The within-game, same-verb comparison uses 34 opposite-label pairs. Aggregate NLL ranks irreversible outputs lower on average. The within-stratum estimate lies at chance, with a wide interval. The two diagnostic controls show why the pooled association warrants caution. The data contain five game groups, and only three supply both labels for the same action verb.

None of the 82 predictions parsed as a single JSON object or matched the complete gold delta. Twenty-nine responses ended with EOS and 53 hit the token cap without EOS. The NLL therefore measures confidence in the generated text. It cannot establish confidence in successful next-state predictions in this run.

On the synthetic masking check, cached generation scores agreed with a full shifted-logit forward to within 0.0000192 nats per token. The longest study prompt used 1,010 tokens. The guarded job took about 6 minutes 10 seconds, used 2.70 GiB peak process RSS, and added no swap. The lowest system-wide free-memory reading was 45%. Raw prompts, predictions and token scores are kept in the ignored private workspace tree. Only aggregate evidence and hashes enter Git. [Verified aggregate receipt](evidence/mac_bytesized_rq5_v1/summary.json)

This completes the measured Mac RQ5 diagnostic. The original BF16 comparison, ScienceWorld certification, independent 60-family evaluation and held-out RQ4 remain open. The existing course release retains its earlier measured scope.
