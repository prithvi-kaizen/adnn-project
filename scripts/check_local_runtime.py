"""Measure a tiny random native Qwen architecture; never download weights."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import shutil
import subprocess
import time
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve prior measurements; choose a new output file")
    seed = 1729
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=256,
                         attention_dropout=0.0)
    model = Qwen2ForCausalLM(config).eval()
    tokens = torch.arange(32).unsqueeze(0) % config.vocab_size
    results = []
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        model = model.to(device)
        inputs = tokens.to(device)
        if device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3):
                out = model(inputs, output_hidden_states=True, use_cache=False)
            if device == "mps":
                torch.mps.synchronize()
            warmup_seconds = time.perf_counter() - started
            started = time.perf_counter()
            for _ in range(20):
                out = model(inputs, output_hidden_states=True, use_cache=False)
            if device == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            generated = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                       max_new_tokens=8, do_sample=False,
                                       pad_token_id=0, eos_token_id=None)
        assert len(out.hidden_states) == 3
        assert all(tuple(h.shape) == (1, 32, 64) for h in out.hidden_states)
        assert torch.isfinite(out.logits).all()
        item = {"device": device, "precision": "float32", "warmup_three_forwards_seconds": warmup_seconds,
                "twenty_forwards_seconds": elapsed, "forward_tokens_per_second": 640 / elapsed,
                "hidden_shapes": [list(x.shape) for x in out.hidden_states],
                "greedy_output_ids": generated.cpu().tolist(),
                "greedy_output_sha256": hashlib.sha256(generated.cpu().numpy().tobytes()).hexdigest(),
                "process_max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if device == "mps":
            item.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                        mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                        mps_recommended_max_bytes=torch.mps.recommended_max_memory())
        results.append(item)
    report = {"scope": "random_tiny_architecture_runtime_diagnostic_not_pretrained_results",
              "seed": seed, "python": platform.python_version(), "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "transformers", "numpy", "scienceworld", "py4j")},
              "cpu_count": __import__("os").cpu_count(),
              "unified_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
              "available_storage_bytes": shutil.disk_usage(".").free,
              "mps_built": torch.backends.mps.is_built(),
              "mps_available": torch.backends.mps.is_available(),
              "model_config": config.to_dict(), "parameter_count": sum(p.numel() for p in model.parameters()),
              "runs": results, "weights_downloaded": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mps_available": report["mps_available"], "runs": results}, indent=2))


if __name__ == "__main__":
    main()
