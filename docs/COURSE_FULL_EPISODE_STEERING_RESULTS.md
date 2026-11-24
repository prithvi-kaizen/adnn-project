# Full-episode steering results

[Protocol frozen before inference](COURSE_FULL_EPISODE_STEERING_DIAGNOSTIC.md)

The guarded job completed 54 episodes and 126 model turns. All episodes replayed. Every zero-strength episode reproduced the saved baseline token sequences. Every condition gave 0/6 cautious versions, 2/6 goals at termination and 1/6 irreversible executions. All six episodes in each condition stopped at the two-invalid-call limit, with 12 rejected calls and zero valid finishes. Greater caution on the irreversible version occurred in 0/3 pairs in each condition.

| Strength | Versions with changed token sequence, concept | Versions with changed token sequence, random |
|---:|---:|---:|
| −2 | 6/6 | 6/6 |
| −1 | 1/6 | 2/6 |
| 0 | 0/6 | 0/6 |
| 1 | 2/6 | 2/6 |
| 2 | 3/6 | 4/6 |

The executed tool calls and per-turn world hashes matched the zero condition in every episode. The intervention changed generation without changing the measured behavior on these six versions. The model's invalid-call stop limits what this result says about a model that can sustain tool use.

The process took 184.1 seconds from model initialization through its assigned episodes and peaked at 2.70 GiB process RSS. Across 484 guard readings, added swap stayed at zero, the lowest free-memory reading was 44%, and at least 13.47 GiB of disk remained. The checkpoint was reused from the local cache. [Raw traces and replay](../artifacts/course_fast_track/native_full_episode_steering_v1/) · [Resource samples](../artifacts/course_fast_track/native_full_episode_steering_guard_v1/)
