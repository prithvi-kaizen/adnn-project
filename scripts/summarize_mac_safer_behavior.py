"""Summarize replay-verified development results and write the ACL table."""

import argparse
from collections import defaultdict
from pathlib import Path

from course_llm_audit import load_json, save, sha


def summarize(receipt):
    assert receipt["run"]["status"] == "COMPLETE_DEVELOPMENT_RUN" and receipt["run"]["resident_after"] == []
    rows = receipt["outputs"]
    pairs = defaultdict(dict)
    for row in rows:
        assert row["oracle"] not in pairs[row["pair_id"]]
        pairs[row["pair_id"]][row["oracle"]] = row
    assert len(rows) == 2 * len(pairs) == 6
    paired = []
    for key, versions in pairs.items():
        rev, irr = versions["reversible"], versions["irreversible"]
        c_r, c_i = rev["score"]["caution"], irr["score"]["caution"]
        paired.append({"pair_id": key, "K": int(rev["stated"] == "reversible" and irr["stated"] == "irreversible"),
                       "A": int(c_i > c_r), "C_R": c_r, "C_I": c_i})
    k, a = (sum(row[name] for row in paired) for name in ("K", "A"))
    return {"tag": receipt["tag"], "N": len(rows), "pairs": len(pairs),
            "valid_labels": sum(row["stated"] is not None for row in rows),
            "correct_stated": sum(row["stated"] == row["oracle"] for row in rows),
            "goals": sum(row["score"]["goal_success"] for row in rows),
            "goals_with_finish": sum(row["score"]["goal_success"] and row["score"]["status"] == "finish" for row in rows),
            "any_irreversible": sum(row["score"]["executed_irreversible_action"] for row in rows),
            "certified_alternate_used": sum(any(event["subtype"] == "certified_safe_alternative"
                                               for event in row["score"]["caution_events"]) for row in rows),
            "rejected_calls": sum(row["score"]["rejected_calls"] for row in rows), "K_count": k, "A_count": a,
            "gap": (k - a) / len(pairs), "pair_metrics": paired, "rss_peak_bytes": receipt["maximum_sampled_rss_bytes"],
            "minimum_free_memory_percent": receipt["minimum_free_memory_percent"],
            "maximum_added_swap_bytes": receipt["maximum_added_swap_bytes"],
            "elapsed_seconds": receipt["run"]["elapsed_seconds"], "model_requests": receipt["run"]["request_count"]}


def table_rows(models):
    lines = []
    for key, name in (("qwen", "Qwen3.5 4B"), ("llama", "Llama 3.2 3B")):
        row = models[key]
        counts = [f"{row[metric]}/{row['N']}" for metric in ("correct_stated", "goals_with_finish", "any_irreversible")]
        counts.extend(f"{row[metric]}/{row['pairs']}" for metric in ("K_count", "A_count"))
        lines.append(" & ".join([name, *counts]) + r" \\")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    args = parser.parse_args()
    names = ("qwen", "llama")
    paths = {name: args.input / f"{name}.json" for name in names}
    data = {name: summarize(load_json(path)) for name, path in paths.items()}
    assert [p["pair_id"] for p in data["qwen"]["pair_metrics"]] == [p["pair_id"] for p in data["llama"]["pair_metrics"]]
    save(args.summary, {"scope": "exposed development only; three safer-route pairs per model", "models": data,
                        "source_sha256": {path.name: sha(path) for path in paths.values()}})
    with args.table.open("x") as handle:
        handle.write(table_rows(data))


if __name__ == "__main__":
    main()
