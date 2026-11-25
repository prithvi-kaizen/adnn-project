# Full-episode steering control extension

This extension completes the control grid on the six previously exposed development versions. It runs the two saved norm-matched random vectors with seeds 812 and 813 at strengths −2, −1, 1 and 2. The existing run provides the concept direction, the seed-811 control and the shared zero condition. No condition or task is selected from the new outcomes.

Reuse the pinned Qwen2.5-0.5B float32 CPU runner, block 13, raw direction vectors, greedy decoding, and the standard 15-turn loop. Keep the consecutive-invalid stop. The 48 new episodes and the verified 54 existing episodes form 102 cells across 17 conditions. Run with the established Mac memory, swap and disk guards and a 15-minute wall limit. Preserve partial records if a guard fires.

Replay every episode and independently count pre-commit caution, goal attainment, valid finish, irreversible execution, rejected calls and changes in full output-token sequences or simulator state traces relative to zero. Report the denominator of six versions per condition and the three exposed pairs. The small exposed cohort and the invalid-call stop limit causal interpretation about agent behavior beyond this setting.
