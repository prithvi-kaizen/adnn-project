"""Prepare pinned Llama chat prompts without putting target labels in model input."""

import argparse
import hashlib
import json
from pathlib import Path

from course_llm_audit import ROOT, load_json, save, sha

from src.models.course_native import feature_messages

DATA = ROOT / "data/course_fast_track/v2"
TEMPLATE_DIGEST = "966de95ca8a62200913e3f8bfbf84c8494536f1b94b49166851e76644e966396"
MODEL_DIGEST = "dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"
TEMPLATE = Path.home() / f".ollama/models/blobs/sha256-{TEMPLATE_DIGEST}"
ARM_STAGE = "/tmp/adnn-llamacpp-build.YzagxSlf"


def render(system, user):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\n\n" + system + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def selection(mode, pilot):
    published = DATA / "freeze_manifest.json"
    hashes = load_json(published)["files"]
    if mode == "development":
        path = DATA / "development_tasks.jsonl"
        assert sha(path) == hashes[path.name]["sha256"]
        inputs = [r for r in rows(path) if r["pair_id"] in {f"pilot-{kind}-0" for kind in ("delete", "edit", "mail", "transfer")}]
        assert len(inputs) == 8
        inputs.extend([{**inputs[0], "id": "unicode_alignment_only", "action_text":
                       '{"tool":"delete_file","arguments":{"path":"résumé-文件.txt"}}'},
                       {**inputs[0], "id": "causal_suffix_only", "append_suffix": " Longer post-action suffix."}])
        return inputs, {"mode": mode, "data_sha256": sha(path)}
    if not pilot:
        raise ValueError("A verified resource receipt is required for the retrospective study")
    resource = load_json(pilot)
    assert resource["status"] == "PASS_EXCLUDED_ALIGNMENT_RESOURCE_PREFLIGHT"
    assert resource["projected_360_seconds"] < 1200 and resource["max_added_swap_bytes"] <= 402653184
    path = DATA / "sandbox_pool.jsonl"
    subset = DATA / "subset_360.json"
    assert all(sha(p) == hashes[p.name]["sha256"] for p in (path, subset))
    ids = set(load_json(subset)["ids"])
    inputs = [r for r in rows(path) if r["id"] in ids]
    assert len(inputs) == len(ids) == 360
    return inputs, {"mode": mode, "data_sha256": sha(path), "subset_sha256": sha(subset), "preflight_sha256": sha(pilot)}


def prepare(mode, directory, pilot):
    assert sha(TEMPLATE) == TEMPLATE_DIGEST
    inputs, receipt = selection(mode, pilot)
    directory.mkdir(parents=True, exist_ok=False)
    cases = []
    with (directory / "spans.tsv").open("x") as spans:
        for i, row in enumerate(inputs):
            messages = feature_messages(row)
            system, user = (m["content"] for m in messages)
            action = row["action_text"]
            if not user.endswith(action) or not action:
                raise ValueError("Frozen user message must end with the candidate action")
            prefix = render(system, user[:-len(action)])
            start = prefix.index("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
            prompt = render(system, user) + row.get("append_suffix", "")
            begin = len(prefix[:start].encode())
            end = begin + len(action.encode())
            body = prompt.encode()
            assert body[begin:end] == action.encode()
            (directory / f"{i}.prompt").write_bytes(body)
            spans.write(f"{i}\t{begin}\t{end}\n")
            cases.append({"index": i, "id": row["id"], "prompt_sha256": hashlib.sha256(body).hexdigest(),
                          "action_utf8_span": [begin, end], "pair": row["minimal_pair_id"] if "minimal_pair_id" in row else row["pair_id"]})
    receipt.update(cases=cases, cached_template_sha256=sha(TEMPLATE), model_blob_sha256=MODEL_DIGEST,
                   study="cached_llama_quantized", code_sha256=sha(Path(__file__)))
    save(directory / "manifest.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("development", "retrospective"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    prepare(args.mode, args.directory, args.pilot)
