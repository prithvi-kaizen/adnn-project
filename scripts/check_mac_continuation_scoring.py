"""Compare shifted-logit NLL with generation scores on a synthetic next-state prompt."""

import argparse
import platform
import time
from pathlib import Path

import torch
from benchmark_course_mac_cached import guard
from course_llm_audit import ROOT, load_json, save, sha

from src.models.continuation_scoring import continuation_nll
from src.models.course_native import NativeCourseRunner, sha_text
from src.models.ollama_dev import system_memory

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
SNAPSHOT = (ROOT / ".cache/course_fast_track/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION)
MESSAGES = [{"role": "system", "content": "Apply the transition to the toy state. Return only compact JSON."},
            {"role": "user", "content": 'State: {"light":"off"}. Action: toggle light once. Next state?'}]


def verify():
    if sha(SNAPSHOT / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("Cached weight digest differs from the native pilot")
    settings = load_json(ROOT / "configs/course_execution_native_mac_cpu_v1.json")
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(1729)
    baseline = system_memory()
    limits = {"min_free_memory_percent": 10, "max_added_swap_bytes": 536870912}
    deadline = time.monotonic() + 90

    def resource_check():
        return guard(limits, baseline, deadline)

    runner = NativeCourseRunner(SNAPSHOT, settings, ROOT, resource_check)
    encoded = runner.tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True,
                                                    tokenize=True, return_tensors="pt")
    attention = torch.ones_like(encoded)
    eos = runner.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    attempts = []
    for limit in (1, 40):
        with torch.inference_mode():
            generated = runner.model.generate(input_ids=encoded, attention_mask=attention,
                                               generation_config=runner.generation_config,
                                               max_new_tokens=limit, return_dict_in_generate=True, output_scores=True)
            sequence = generated.sequences[0]
            logits = runner.model(sequence[None], attention_mask=torch.ones_like(sequence[None]),
                                  use_cache=False).logits[0]
            explicit, terminal_eos = continuation_nll(sequence, logits, encoded.size(1), eos_ids)
            generated_scores = runner.model.compute_transition_scores(generated.sequences, generated.scores,
                                                                       normalize_logits=True)[0]
            reference = -generated_scores[:-1] if terminal_eos else -generated_scores
            torch.testing.assert_close(explicit, reference, atol=1e-4, rtol=1e-5)
        suffix = sequence[encoded.size(1):].tolist()
        attempts.append({"new_token_limit": limit, "reason": "eos" if terminal_eos else "token_limit",
                         "generated_ids": suffix, "content_decoded": runner.tokenizer.decode(suffix, skip_special_tokens=True),
                         "content_tokens_scored": len(explicit), "eos_excluded": terminal_eos,
                         "nll_per_content_token": explicit.tolist(), "mean_nll": explicit.mean().item(),
                         "max_absolute_difference_to_generation": (reference - explicit).abs().max().item()})
        resource_check()
    return {"scope": "synthetic token scoring check outside ByteSized32 and all study splits",
            "model": settings["checkpoint"], "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
            "chat_template_sha256": sha_text(runner.tokenizer.chat_template), "source_sha256": sha(Path(__file__)),
            "scorer_sha256": sha(ROOT / "src/models/continuation_scoring.py"),
            "platform": platform.platform(), "torch": torch.__version__, "device": settings["device"],
            "precision": settings["precision"], "seed": 1729, "greedy": True, "threads": settings["cpu_threads"],
            "prompt": MESSAGES, "prompt_ids": encoded[0].tolist(), "starting_memory": baseline,
            "ending_memory": resource_check(), "attempts": attempts, "eligible_bytesized_rows": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    save(parser.parse_args().output, verify())
