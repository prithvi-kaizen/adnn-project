"""Bounded localhost-only transport for installed exploratory models."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODELS = {
    "1b": ("llama3.2:1b-instruct-q4_K_M", "22bc6b92eb0160c4629782fca05a9032c59d306ab2058b7dacc8a4644fbafa02"),
    "8b": ("llama3.1:8b", "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"),
}
OPTIONS = {"temperature": 0, "top_k": 1, "top_p": 1, "repeat_penalty": 1,
           "seed": 13, "num_predict": 128, "num_ctx": 8192, "num_thread": 4,
           "stop": ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]}
JSON_SCHEMA = {"type": "object", "required": ["tool", "arguments"], "additionalProperties": False,
               "properties": {"tool": {"type": "string"}, "arguments": {"type": "object"}}}


def utc():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict):
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def api(endpoint: str, body=None, timeout=120):
    if endpoint not in {"/api/tags", "/api/show", "/api/ps", "/api/version", "/api/generate", "/api/chat"}:
        raise ValueError("Endpoint outside the local development allowlist")
    request = urllib.request.Request("http://127.0.0.1:11434" + endpoint,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def render(messages: list[dict]) -> str:
    result = "<|begin_of_text|>"
    for message in messages:
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError("Unsupported chat role")
        result += "<|start_header_id|>" + message["role"] + "<|end_header_id|>\n\n"
        result += message["content"] + "<|eot_id|>"
    return result + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def system_memory():
    pressure = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, check=True).stdout
    swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    free = re.search(r"free percentage:\s*(\d+)%", pressure)
    used = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap)
    if not free or not used:
        raise RuntimeError("Cannot verify Mac memory-pressure/swap guard")
    return {"memory_free_percentage": int(free.group(1)),
            "swap_used_bytes": int(float(used.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3}[used.group(2)])}


def resource_sample():
    output = subprocess.check_output(["ps", "-axo", "pid=,rss=,comm="], text=True)
    processes = []
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and "ollama" in fields[2].lower():
            processes.append({"pid": int(fields[0]), "rss_bytes": int(fields[1]) * 1024,
                              "executable_basename": Path(fields[2]).name})
    resident = [{k: v for k, v in m.items() if k in
                {"name", "digest", "size", "size_vram", "context_length"}}
                for m in api("/api/ps", timeout=3).get("models", [])]
    return {"timestamp_utc": utc(), "processes": processes, "resident_models": resident,
            "sum_process_rss_bytes": sum(p["rss_bytes"] for p in processes)}


class ResourceMonitor:
    def __init__(self, path: Path):
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                append_jsonl(self.path, resource_sample())
            except Exception as error:
                append_jsonl(self.path, {"timestamp_utc": utc(), "sample_error": type(error).__name__})
            self.stop.wait(.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


class OllamaDevelopmentClient:
    def __init__(self, model_key: str, output_dir: Path):
        self.model, self.digest = MODELS[model_key]
        self.output_dir = output_dir
        self.request_count = 0
        self.owns_residency = False
        self.baseline_memory = system_memory()
        self.deadline = time.monotonic() + (1200 if model_key == "1b" else 900)

    def preflight(self):
        models = api("/api/tags")["models"]
        selected = next((m for m in models if m["name"] == self.model), None)
        if selected is None or selected["digest"] != self.digest:
            raise RuntimeError("Installed model does not match the locked development manifest")
        if api("/api/ps").get("models"):
            raise RuntimeError("An Ollama model is already resident; do not evict it")
        details = api("/api/show", {"model": self.model})
        return {"model": selected, "ollama_version": api("/api/version"),
                "details": details.get("details"),
                "model_info": {k: v for k, v in details.get("model_info", {}).items()
                               if isinstance(v, (str, int, float, bool))},
                "tag_template_sha256": hashlib.sha256(details.get("template", "").encode()).hexdigest(),
                "tag_parameters_sha256": hashlib.sha256(details.get("parameters", "").encode()).hexdigest(),
                "license_sha256": hashlib.sha256(details.get("license", "").encode()).hexdigest(),
                "baseline_memory": self.baseline_memory, "options": OPTIONS,
                "raw_template": "explicit_Llama3_headers_v1", "raw": True,
                "precision": "GGUF_Q4_K_M", "scope": "exploratory_not_native_BF16"}

    def guard(self):
        if time.monotonic() > self.deadline:
            raise RuntimeError("Per-model engineering runtime budget exhausted")
        memory = system_memory()
        append_jsonl(self.output_dir / "memory_guards.jsonl", {
            "timestamp_utc": utc(), **memory,
            "swap_delta_bytes": memory["swap_used_bytes"] - self.baseline_memory["swap_used_bytes"]})
        if memory["memory_free_percentage"] < 10:
            raise RuntimeError("Memory-pressure guard: free percentage below 10")
        if memory["swap_used_bytes"] - self.baseline_memory["swap_used_bytes"] > 512 * 1024**2:
            raise RuntimeError("Swap growth guard: more than 512 MiB")
        return memory

    def generate(self, messages: list[dict], *, structured=False, label=None):
        memory = self.guard()
        prompt = render(messages)
        if len(prompt.encode()) + 2 + OPTIONS["num_predict"] > OPTIONS["num_ctx"]:
            raise ValueError("Conservative prompt-byte context bound exceeded; no truncation performed")
        self.request_count += 1
        request_id = self.request_count
        payload = {"model": self.model, "prompt": prompt, "raw": True, "stream": False,
                   "options": OPTIONS, "keep_alive": "5m"}
        if structured:
            payload["format"] = JSON_SCHEMA
        started = time.monotonic()
        append_jsonl(self.output_dir / "requests.jsonl", {
            "event": "request_started", "request_id": request_id, "timestamp_utc": utc(),
            "scope": "exploratory_not_final_results", "label": label, "request": payload,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "memory_before": memory})
        try:
            self.owns_residency = True
            response = api("/api/generate", payload)
            elapsed = time.monotonic() - started
            if "error" in response:
                raise RuntimeError(response["error"])
            append_jsonl(self.output_dir / "requests.jsonl", {
                "event": "request_completed", "request_id": request_id, "response": response,
                "wall_seconds": elapsed, "timestamp_utc": utc()})
            return {"text": response["response"], "request_id": request_id,
                    "wall_seconds": elapsed, "done_reason": response.get("done_reason")}
        except Exception as error:
            append_jsonl(self.output_dir / "requests.jsonl", {
                "event": "request_failed", "request_id": request_id, "timestamp_utc": utc(),
                "error_type": type(error).__name__, "error": str(error),
                "wall_seconds": time.monotonic() - started})
            raise

    def unload(self):
        if self.owns_residency and any(m["name"] == self.model for m in api("/api/ps").get("models", [])):
            api("/api/generate", {"model": self.model, "keep_alive": 0, "stream": False})
        return api("/api/ps").get("models", [])
