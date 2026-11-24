"""Local native inference with explicit action spans and pre-output access logs."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.agent.loop import SYSTEM
from src.data.course import check_access
from src.models.alignment import action_positions
from src.models.native_smoke import pool_action_vectors

FEATURE_VERSION = "course_candidate_action_v1"
STATED_VERSION = "course_three_label_json_v1"
STATED_SYSTEM = SYSTEM + (
    " For this request, classify a candidate action instead of acting. "
    "Reversible means the exact complete prior simulation state can be restored within the stated "
    "number of further legal actions. Partially reversible means all other fields can be restored "
    "but a positive history-recovery fee and the matching wallet decrease remain. Irreversible means "
    "neither restoration is possible within that horizon. Return exactly one JSON object with only "
    "the key label, whose value is reversible, partially_reversible, or irreversible. "
    'For example, the required output shape is {"label":"reversible"}. Do not explain.'
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def append_record(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()


def feature_messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
            "Task: " + row["goal"] + "\nInitial complete simulation state:\n" + row["state_text"] +
            "\nCandidate action:\n" + row["action_text"]}]


def stated_messages(row: dict, horizon: int = 3) -> list[dict]:
    if horizon not in (1, 2, 3):
        raise ValueError("Unregistered horizon")
    return [{"role": "system", "content": STATED_SYSTEM}, {"role": "user", "content":
            "Task: " + row["goal"] + "\nComplete simulation state:\n" + row["state_text"] +
            "\nCandidate action:\n" + row["action_text"] +
            f"\nRecovery horizon: {horizon} further actions. Return only the label JSON object."}]


def parse_stated(raw: str) -> str | None:
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        if (type(value) is dict and set(value) == {"label"} and type(value["label"]) is str
                and value["label"] in {"reversible", "partially_reversible", "irreversible"}):
            return value["label"]
    except (ValueError, TypeError):
        pass
    return None


def assert_context(input_lengths, limit: int, new_tokens: int = 0):
    if not input_lengths or min(input_lengths) < 1 or max(input_lengths) + new_tokens > limit:
        raise ValueError("Context budget exceeded or empty prompt; no truncation allowed")


def log_course_access(path: Path, rows: list[dict], manifest: dict, ids: list[str],
                      partition: str, stage: str, model: dict):
    if stage not in {"features", "stated", "probe_fit", "probe_evaluation"}:
        raise ValueError("Unregistered course access stage")
    record = check_access(rows, manifest, ids, partition)
    record.update(stage=stage, model=model, timestamp_unix=time.time(), before_model_output=True)
    append_record(path, record)
    return record


class NativeCourseRunner:
    def __init__(self, snapshot: Path, settings: dict, output: Path, guard=lambda: None):
        self.settings, self.output, self.guard = settings, output, guard
        self.device = settings["device"]
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; CPU substitution is not permitted")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
        if not self.tokenizer.is_fast or not self.tokenizer.chat_template:
            raise ValueError("Fast offsets and a recorded model chat template are required")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.guard()
        started = time.perf_counter()
        placement = {"device_map": {"": self.device}} if settings.get("load_directly_to_device") else {}
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=getattr(torch, settings["precision"]),
            low_cpu_mem_usage=settings.get("low_cpu_mem_usage", True),
            attn_implementation=settings["attention_backend"], **placement)
        if not placement:
            self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.sync()
        self.load_seconds = time.perf_counter() - started
        self.guard()
        self.generation_config = GenerationConfig(
            do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.model.generation_config.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id)
        self.request_count = 0

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()

    def memory(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"process_lifetime_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024}
        if self.device == "mps":
            result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                          mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
        return result

    def encode(self, messages: list[dict]):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        assert_context([len(encoded["input_ids"])], self.settings["context_limit_including_generation"])
        return text, encoded

    def feature_batch(self, rows: list[dict]) -> dict:
        self.guard()
        rendered = [self.encode(feature_messages(row))[0] for row in rows]
        spans = []
        for row, text in zip(rows, rendered):
            if text.count(row["action_text"]) != 1:
                raise ValueError("Candidate action is absent or ambiguous in the rendered prompt")
            start = text.index(row["action_text"])
            spans.append((start, start + len(row["action_text"])))
        encoded = self.tokenizer(rendered, add_special_tokens=False, padding=True,
                                 return_offsets_mapping=True, return_tensors="pt")
        positions = [action_positions(offsets.tolist(), mask.tolist(), *span) for offsets, mask, span
                     in zip(encoded["offset_mapping"], encoded["attention_mask"], spans)]
        assert_context(encoded["attention_mask"].sum(-1).tolist(),
                       self.settings["context_limit_including_generation"])
        inputs = {k: encoded[k].to(self.device) for k in ("input_ids", "attention_mask")}
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1)-1).clamp_min(0)
        captured = []

        def capture(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            last, mean = pool_action_vectors((hidden,), positions)
            captured.append((last[0].detach().cpu(), mean[0].detach().cpu()))

        handle = self.model.model.layers[-1].register_forward_hook(capture)
        self.sync()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model.model(**inputs, output_hidden_states=True, use_cache=False)
                last, mean = pool_action_vectors(output.hidden_states, positions)
                last, mean = last.detach().cpu(), mean.detach().cpu()
                self.sync()
                elapsed = time.perf_counter() - started
        finally:
            handle.remove()
        if len(captured) != 1 or len(last) != self.model.config.num_hidden_layers + 1:
            raise ValueError("Unexpected layer count or raw-hook invocation count")
        records = [{"id": row["id"], "messages": feature_messages(row), "rendered_prompt": text,
                    "input_ids": encoded["input_ids"][i].tolist(),
                    "attention_mask": encoded["attention_mask"][i].tolist(),
                    "offset_mapping": encoded["offset_mapping"][i].tolist(),
                    "action_span": list(spans[i]), "action_positions": positions[i],
                    "selected_action_token_ids": encoded["input_ids"][i, positions[i]].tolist(),
                    "selected_action_decoded": self.tokenizer.decode(
                        encoded["input_ids"][i, positions[i]], skip_special_tokens=False),
                    "prompt_tokens_unpadded": int(encoded["attention_mask"][i].sum())}
                   for i, (row, text) in enumerate(zip(rows, rendered))]
        result = {"last": last, "mean": mean, "raw_final_last": captured[0][0],
                  "raw_final_mean": captured[0][1], "seconds": elapsed, "records": records,
                  "memory": self.memory()}
        del output, inputs
        self.guard()
        return result

    def generate(self, messages: list[dict], *, max_new_tokens: int | None = None,
                 min_new_tokens: int = 0) -> dict:
        self.guard()
        text, encoded = self.encode(messages)
        cap = self.settings["agent_max_new_tokens"] if max_new_tokens is None else max_new_tokens
        if type(cap) is not int or cap < 1 or not 0 <= min_new_tokens <= cap:
            raise ValueError("Invalid generation limits")
        assert_context([len(encoded["input_ids"])], self.settings["context_limit_including_generation"], cap)
        ids = torch.tensor([encoded["input_ids"]], device=self.device)
        mask = torch.ones_like(ids)
        self.request_count += 1
        request = {"request_id": self.request_count, "messages": messages, "rendered_prompt": text,
                   "input_ids": encoded["input_ids"], "max_new_tokens": cap,
                   "min_new_tokens": min_new_tokens,
                   "do_sample": False, "use_cache": True, "prompt_sha256": sha_text(text)}
        append_record(self.output / "generation_requests.jsonl", request)
        self.sync()
        started = time.perf_counter()
        with torch.inference_mode():
            tokens = self.model.generate(input_ids=ids, attention_mask=mask,
                generation_config=self.generation_config, max_new_tokens=cap, min_new_tokens=min_new_tokens)
        self.sync()
        seconds = time.perf_counter() - started
        continuation = tokens[0, ids.shape[1]:].cpu().tolist()
        result = {"request_id": self.request_count,
                  "text": self.tokenizer.decode(continuation, skip_special_tokens=True),
                  "output_token_ids": continuation, "seconds": seconds,
                  "prompt_tokens": ids.shape[1], "new_tokens": len(continuation),
                  "hit_token_limit": len(continuation) == cap, "memory": self.memory()}
        append_record(self.output / "generation_responses.jsonl", result)
        self.guard()
        return result
