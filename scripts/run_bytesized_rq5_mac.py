"""Run the preregistered cached Qwen CPU/FP32 ByteSized32 diagnostic."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

from run_bytesized_rq5_native import MAX_NEW, ROOT, read_cases, serialize, sha, strict_prediction

SOURCE = ROOT / "docs/evidence/original_restart_20261120/bytesized_combined_design.json"
PLAN = ROOT / "docs/MAC_BYTESIZED_RQ5_PROTOCOL.md"
EXECUTION = ROOT / "configs/course_execution_native_mac_cpu_v1.json"


def measure(runner, ids, minimum=0):
    import torch

    prompt = torch.tensor([ids])
    eos = runner.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    before = time.perf_counter()
    with torch.inference_mode():
        generated = runner.model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                          generation_config=runner.generation_config, max_new_tokens=MAX_NEW,
                                          min_new_tokens=minimum, return_dict_in_generate=True, output_logits=True)
        sequence = generated.sequences[0]
        suffix = sequence[len(ids):]
        assert suffix.numel() == len(generated.logits)
        terminal_eos = bool(suffix.numel() and suffix[-1].item() in eos_ids)
        content_ids = suffix[:-1] if terminal_eos else suffix
        assert not any(token in eos_ids for token in content_ids.tolist())
        nll = [float(-logits[0].float().log_softmax(0)[token].item())
               for logits, token in zip(generated.logits, content_ids.tolist())]
    return {"ids": suffix.tolist(), "terminal_eos": terminal_eos, "cap_without_eos": not terminal_eos and len(suffix) == MAX_NEW,
            "text": runner.tokenizer.decode(suffix.tolist(), skip_special_tokens=True),
            "nll": nll, "seconds": time.perf_counter() - before, **runner.memory()}


def run(case_path, directory):
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="0")
    import torch
    import transformers

    from scripts.stage_course_checkpoint import file_receipt
    from src.models.continuation_scoring import continuation_nll
    from src.models.course_native import NativeCourseRunner, append_record, assert_context, sha_text
    from src.models.ollama_dev import api

    spec = json.loads(SOURCE.read_text())
    cases = read_cases(case_path, spec["private_design_sha256"])
    assert len(cases) == 82
    config = json.loads(EXECUTION.read_text())
    settings = json.loads((ROOT / config["pilot_settings"]).read_text())
    receipt = json.loads((ROOT / config["checkpoint_receipt"]).read_text())
    model_files = json.loads((ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json").read_text())["files"]
    snapshot = Path(receipt["snapshot"])
    assert (platform.system(), settings["device"], settings["precision"]) == ("Darwin", "cpu", "float32")
    if api("/api/ps", timeout=5).get("models", []):
        raise RuntimeError("Another model is resident in Ollama")
    weights = {item["path"]: file_receipt(snapshot / item["path"], item) for item in model_files}
    torch.manual_seed(1729)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = directory / "records.jsonl"
    runner = NativeCourseRunner(snapshot, settings, directory)
    data = []
    for row in cases:
        rendered, encoded = runner.encode(row["messages"])
        ids = encoded["input_ids"]
        assert_context([len(ids)], settings["context_limit_including_generation"], MAX_NEW)
        data.append((row, rendered, ids))
    lock = {"type": "lock", "scope": "exploratory_mac_native_fp32_rq5", "provenance_sha256": sha(SOURCE),
            "cohort_sha256": sha(case_path), "protocol_sha256": sha(PLAN), "code_sha256": sha(Path(__file__)),
            "scorer_sha256": sha(ROOT / "src/models/continuation_scoring.py"), "model": settings["model"],
            "revision": settings["revision"], "weight_files": weights, "assigned": len(data),
            "chat_template_sha256": sha_text(runner.tokenizer.chat_template), "torch": torch.__version__,
            "transformers": transformers.__version__, "platform": platform.platform(), "seed": 1729,
            "device": "cpu", "dtype": "float32", "max_new_tokens": MAX_NEW, "do_sample": False, "threads": 4}
    append_record(path, lock)
    toy = [{"role": "user", "content": "Predict the next light state. Off is toggled once. Return JSON."}]
    toy_ids = runner.encode(toy)[1]["input_ids"]
    small = measure(runner, toy_ids)
    full = torch.tensor([toy_ids + small["ids"]])
    eos = runner.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    with torch.inference_mode():
        values, _ = continuation_nll(full[0], runner.model(full, use_cache=False).logits[0], len(toy_ids), eos_ids)
    torch.testing.assert_close(torch.tensor(small["nll"]), values, atol=1e-4, rtol=1e-5)
    longest = max(len(ids) for _, _, ids in data)
    timings = []
    for token in (" x", " y"):
        fixture = [{"role": "user", "content": "Synthetic benchmark text." + token * longest}]
        ids = runner.encode(fixture)[1]["input_ids"]
        assert len(ids) >= longest
        assert_context([len(ids)], settings["context_limit_including_generation"], MAX_NEW)
        timing = measure(runner, ids, minimum=MAX_NEW)
        assert len(timing["ids"]) == MAX_NEW
        timings.append({"seconds": timing["seconds"], "prompt_tokens": len(ids)})
    projected = 2 * len(data) * max(row["seconds"] for row in timings)
    append_record(path, {"type": "preflight", "timings": timings, "longest_case_tokens": longest,
                         "seconds_with_margin": projected, "raw_score_max_error": (values - torch.tensor(small["nll"])).abs().max().item()})
    if projected > 1200:
        raise RuntimeError("Measured full-cap projection exceeds the declared diagnostic budget")
    for index, (row, prompt, ids) in enumerate(data, 1):
        append_record(path, {"type": "request", **row, "prompt": prompt, "input_ids": ids})
        measured = measure(runner, ids)
        prediction = strict_prediction(measured["text"])
        append_record(path, {"type": "score", "id": row["id"], **measured, "json": prediction is not None,
                             "exact_delta": prediction == serialize(row["gold_delta"])})
        if index % 10 == 0 or index == len(data):
            print(f"Scored {index}/{len(data)} assigned transitions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.cases, args.output)
