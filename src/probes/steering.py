"""Training-only raw-site mean differences and norm-matched random controls."""

import numpy as np


def raw_site_vectors(features, block):
    if type(block) is not int or not 1 <= block <= 24:
        raise ValueError("The fixed Qwen condition requires a transformer block in 1..24")
    return features["raw_final_last"] if block == 24 else features["last"][block]


def directions(vectors, labels, training_indices):
    values = np.asarray(vectors)
    labels = np.asarray(labels)
    training = np.asarray(training_indices, dtype=int)
    if (values.ndim != 2 or len(values) != len(labels) or len(training) != len(set(training.tolist())) or
            len(training) == 0 or training.min() < 0 or training.max() >= len(values) or not np.isfinite(values).all()):
        raise ValueError("Finite vectors and distinct in-range training rows required")
    selected = labels[training]
    r, i = training[selected == "reversible"], training[selected == "irreversible"]
    if not len(r) or not len(i):
        raise ValueError("Both binary training classes required")
    concept = (values[i].astype(np.float64).mean(0) - values[r].astype(np.float64).mean(0)).astype(np.float32)
    norm = float(np.linalg.norm(concept.astype(np.float64)))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("A zero/nonfinite direction cannot support norm-matched steering")
    result = {"concept": concept}
    for seed in (811, 812, 813):
        raw = np.random.default_rng(seed).normal(size=concept.shape)
        result[f"random_{seed}"] = (raw * norm / np.linalg.norm(raw)).astype(np.float32)
    assert all(np.isclose(np.linalg.norm(v.astype(np.float64)), norm, rtol=1e-6) for v in result.values())
    return result, {"concept_norm": norm, "training_R_count": len(r), "training_I_count": len(i),
                    "partial_training_examples_ignored": int((selected == "partially_reversible").sum()),
                    "mean_accumulation_dtype": "float64_then_float32_direction",
                    "Gaussian_generator": "numpy_default_rng_PCG64", "random_seeds": [811, 812, 813]}


def resource_choice(worst_full_cap_seconds, remaining_seconds, policy):
    if not np.isfinite(worst_full_cap_seconds) or worst_full_cap_seconds <= 0:
        raise ValueError("A positive measured full-cap runtime is required")
    candidates = []
    selected = None
    usable = min(policy["maximum_sweep_seconds"], remaining_seconds - policy["delivery_reserve_seconds"])
    for name, n in (("all_24_versions", 24), ("one_RI_pair_per_template_6_versions", 6)):
        projection = policy["runtime_multiplier"] * n * 17 * 15 * worst_full_cap_seconds
        candidate = {"scope": name, "versions": n, "projected_seconds": projection,
                     "fits": projection <= usable, "assumes_early_parse_failure": False}
        candidates.append(candidate)
        if selected is None and candidate["fits"]:
            selected = candidate
    return {"selected": selected, "usable_sweep_seconds": usable, "candidates": candidates,
            "selection_uses_accuracy_caution_goal_or_output_text": False}
