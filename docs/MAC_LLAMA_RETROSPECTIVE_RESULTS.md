# Cached Llama sandbox evaluation

Later on 20 November, one Codex reviewer completed the blinded sandbox audit under the accommodations reported by the researcher. [Review and remaining boundaries](CODEX_BLINDED_SANDBOX_REVIEW.md)

20 November 2026 IST. The 360-case retrospective sandbox evaluation completed for cached Llama 3.2 3B Instruct Q4_K_M. Its results add a second model family to the reduced sandbox work. The original dual-environment BF16 comparison remains open.

## Verified execution

Metal extraction completed all 360 cases in about 311 seconds. It saved two pooling methods, 28 block inputs and the final normalized output at width 3,072. The 162 matched pairs shared exact action-token IDs. The extractor rebuilt each UTF-8 prompt from its token pieces and required exact byte equality. The excluded Unicode and causal suffix tests passed. Peak added swap was zero and the lowest sampled memory-pressure free reading was 32% during the study extraction. The stated run also recorded zero added swap and unloaded its only resident checkpoint. [Feature preflight](evidence/mac_llama_retrospective_v1/preflight.json) · [Independent result checks](evidence/mac_llama_retrospective_v1/verification_v2.json)

The initial dense fit was stopped before scoring test examples because of its projected runtime. The replacement solves the same standardized linear L2 objective in the span of the training rows. Orthogonal coordinates preserve the L2 penalty. Training and development fits chose the regularization and layer before the test access receipt was written. The time guard stopped the revised selection after 144 fits. A checked resume used the saved fit and record hashes, finished the remaining 36, and evaluated all 180 conditions. The verifier checked 5,220 layer-level test records. The controls include majority, lexical, embedding layer, label shuffles and fixed verb labels. Experiments also cover last/mean pooling, binary/three-way targets, minimal pairs and the three horizons. The retrospective split samples share 60 of 72 test IDs pairwise.

## Outcomes

| Seed | Binary matched-pair cases | Selected layer | Probe BA | Action-only BA | Pairs with both endpoints correct | Three-way stated accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 48 | 1 | 58.3% | 50.0% | 4/24 | 31/72 |
| 2718 | 48 | 1 | 58.3% | 50.0% | 4/24 | 31/72 |
| 31415 | 60 | 21 | 50.0% | 50.0% | 0/30 | 31/72 |

The 8.3-point gains occur on the same two matched components in the first two splits. Each falls below the registered minimum of three components for an interval. The third split has three matched components and its conditional interval for the difference is [0, 0]. These descriptive results are too dependent on a few cases to establish a population improvement. On the full three-way splits, probe balanced accuracy was 46.7%, 56.9% and 61.1%, and lexical balanced accuracy was 61.1%, 63.3% and 49.4%, respectively.

All 96 unique stated responses passed strict JSON parsing. Llama called 84 reversible, 11 irreversible and one partially reversible. Accuracy was 43.1% on each test split. The whole-component 95% intervals were [22.2%, 58.3%], [22.2%, 58.3%] and [22.2%, 50.0%]. Agreement with the selected three-way readout was κ = 0.386, 0.023 and −0.138. The exact grouped intervals and outcome records are in the verification receipt. Feature extraction used the pinned Metal build and stated generation used the same cached checkpoint through Ollama. The numerical runtimes are separate.

## Original-study gates

Gate 3 now has a completed retrospective second-family quantized evaluation. The original fixed checkpoints and precision remain open. The ScienceWorld causal-state definition still needs the instructor decision and per-family validation. The blind 100-case packet remains unopened because neither local reviewer qualified. The 60-family test, full-episode controlled steering and licensed executable ByteSized32 evaluation remain incomplete. The final author and reading attestations require the team. These boundaries are recorded in the current report and handoff.
