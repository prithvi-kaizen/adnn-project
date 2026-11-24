"""Record native generated-continuation scores for an approved private RQ5 cohort."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_bytesized_rq5 import SYSTEM, serialize  # noqa: E402

MANIFEST = ROOT / "docs/evidence/model_storage_metadata_20261101.json"
MODELS = ("Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct")
CONTEXT = 2048
MAX_NEW = 128


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cases(path, expected):
    if sha(path) != expected:
        raise ValueError("Prepared cohort digest differs from execution receipt")
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Expected nonempty unique case IDs")
    for case in cases:
        messages = case["messages"]
        if (case["label"] not in {"reversible", "irreversible"} or len(messages) != 2
                or messages[0] != {"role": "system", "content": SYSTEM} or messages[1]["role"] != "user"
                or set(json.loads(messages[1]["content"])) != {"task", "objects", "allocator", "score", "action"}
                or set(case["gold_delta"]) != {"set", "remove", "order", "allocator", "score"}):
            raise ValueError("Prepared case failed prompt and label checks")
    return cases


def strict_prediction(text):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("Duplicate prediction key")
            output[key] = value
        return output

    try:
        value = json.loads(text, object_pairs_hook=unique)
        return serialize(value) if type(value) is dict else None
    except (ValueError, TypeError):
        return None


def execute(args, cases):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from src.models.continuation_scoring import continuation_nll
    from src.models.course_native import assert_context

    if not os.environ.get("SLURM_JOB_ID") or platform.node().startswith("dgx-login"):
        raise RuntimeError("Native execution requires an allocated Slurm compute node")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expected one allocated CUDA device with BF16 support")
    torch.manual_seed(1729)
    spec = json.loads(MANIFEST.read_text())["models"][args.model]
    if args.revision != spec["revision"]:
        raise ValueError("Use the registered immutable revision")
    for shard in spec["primary_safetensors"]:
        local = args.snapshot / shard["path"]
        if local.stat().st_size != shard["bytes"] or sha(local) != shard["lfs"]["sha256"]:
            raise ValueError(f"Cached shard identity mismatch: {shard['path']}")
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    data = []
    for row in cases:
        rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        assert_context([len(ids)], CONTEXT, MAX_NEW)
        data.append((row, rendered, ids))
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, low_cpu_mem_usage=True, device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16, attn_implementation="eager").eval()
    model_eos = model.generation_config.eos_token_id
    eos_ids = {model_eos} if isinstance(model_eos, int) else set(model_eos)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos_ids)
    config = GenerationConfig(do_sample=False, use_cache=True, eos_token_id=model_eos, pad_token_id=pad,
                              bos_token_id=tokenizer.bos_token_id)
    with args.output.open("x") as stream:
        stream.write(json.dumps({"type": "provenance", "cohort_sha256": args.sha256, "model": args.model,
            "revision": args.revision, "manifest_sha256": sha(MANIFEST), "seed": 1729, "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__, "job_id": os.environ["SLURM_JOB_ID"],
            "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16", "greedy": True, "max_new": MAX_NEW,
            "eos_ids_excluded": sorted(eos_ids), "tokenizer_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "code_sha256": sha(Path(__file__))}) + "\n")
        stream.flush()
        with torch.inference_mode():
            for row, rendered, ids in data:
                stream.write(json.dumps({**row, "type": "request", "prompt": rendered, "input_ids": ids}) + "\n")
                stream.flush()
                prompt = torch.tensor([ids], device="cuda:0")
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                sequence = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                          generation_config=config, max_new_tokens=MAX_NEW)[0]
                suffix = sequence[len(ids):].tolist()
                terminal_eos = bool(suffix and suffix[-1] in eos_ids)
                content = suffix[:-1] if terminal_eos else suffix
                text = tokenizer.decode(suffix, skip_special_tokens=True)
                stream.write(json.dumps({"type": "generation", "id": row["id"], "generated_ids": suffix,
                                         "output_text": text, "terminal_eos": terminal_eos,
                                         "hit_cap_without_eos": not terminal_eos and len(suffix) == MAX_NEW}) + "\n")
                stream.flush()
                nll = []
                if content:
                    logits = model(sequence[None], attention_mask=torch.ones_like(sequence[None]), use_cache=False).logits[0]
                    scores, _ = continuation_nll(sequence, logits, len(ids), eos_ids)
                    nll = scores.tolist()
                    del logits
                input_record = {"id": row["id"], "type": "score", "nll_per_content_token": nll,
                                "mean_nll": sum(nll) / len(nll) if nll else None,
                                "exact_delta": strict_prediction(text) == serialize(row["gold_delta"]),
                                "seconds": time.monotonic() - start, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
                stream.write(json.dumps(input_record) + "\n")
                stream.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = read_cases(args.cases, args.sha256)
    if args.check_inputs:
        print(json.dumps({"input_gate": "PASS", "cases": len(cases), "model_loaded": False}))
    else:
        if not all((args.model, args.snapshot, args.revision, args.output)):
            parser.error("Native execution requires model, snapshot, revision, and a new private output path")
        if args.output.exists():
            parser.error("Choose a new private output path")
        os.environ["HF_HUB_OFFLINE"] = "1"
        execute(args, cases)
