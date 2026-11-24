"""Collect strict Llama label JSON on the already locked retrospective test IDs."""

import argparse
import json
from pathlib import Path

from benchmark_course_mac_cached import STATED
from course_llm_audit import ROOT, load_json, save, sha
from prepare_mac_llama_features import DATA, TEMPLATE_DIGEST, rows
from run_course_llm_audit_local import call, fingerprint
from run_mac_safer_behavior import CONFIG, MacBudget

from src.models.course_native import parse_stated, stated_messages
from src.models.ollama_dev import append_jsonl


def run(probes, directory):
    locked = probes / "selection_lock.json"
    access = load_json(probes / "test_access.json")
    assert access["selection_lock_sha256"] == sha(locked)
    assert load_json(probes / "summary.json")["status"] == "PASS_REDUCED_LLAMA_PROBE_EXECUTION"
    ids = sorted({item for split in access["access"] for item in split["ids"]})
    path = DATA / "sandbox_pool.jsonl"
    data = {row["id"]: row for row in rows(path)}
    config = load_json(CONFIG)
    model = config["models"]["llama"]
    pilot = ROOT / "artifacts/exploratory/mac_cached_smoke_v3/model_1/result.json"
    observed = load_json(pilot)
    assert observed["digest"] == model["digest"]
    max_latency = max(row["elapsed_seconds"] for row in observed["requests"] if row["purpose"] == "stated")
    config["wall_seconds"] = 720
    forecast = 2 * len(ids) * max_latency
    assert forecast < config["wall_seconds"]
    assert call("version")[1]["version"] == config["ollama_version"] and not call("ps")[1]["models"]
    installed = next(m for m in call("tags")[1]["models"] if m["name"] == model["tag"])
    assert installed["digest"] == model["digest"]
    template = call("show", {"model": model["tag"]})[1]["template"]
    assert fingerprint(template) == TEMPLATE_DIGEST
    directory.mkdir(parents=True, exist_ok=False)
    plan = {"assigned_ids": ids, "test_access_sha256": sha(probes / "test_access.json"),
            "probe_selection_sha256": sha(locked), "dataset_sha256": sha(path), "cached_model": installed,
            "template_sha256": fingerprint(template), "excluded_timing_receipt_sha256": sha(pilot),
            "projected_seconds": forecast, "maximum_seconds": config["wall_seconds"],
            "scope": "retrospective quantized sandbox, already exposed benchmark"}
    save(directory / "input_lock.json", plan)
    budget = MacBudget(config, directory)
    out = {"status": "INCOMPLETE", "assigned": len(ids), "attempts": 0}
    try:
        with budget:
            for count, key in enumerate(ids, 1):
                request = {"model": model["tag"], "messages": stated_messages(data[key]), "format": STATED,
                           "keep_alive": "90s", "stream": False, "options": {**config["options"], "num_predict": 64}}
                save(directory / f"{count:03d}.request.json", request)
                status, raw = budget.chat(request)
                response = directory / f"{count:03d}.response.json"
                response.write_bytes(raw)
                budget.check()
                reply = json.loads(raw)
                if status != 200 or reply.get("model") != model["tag"] or reply.get("done") is not True:
                    raise RuntimeError("Incomplete response or model identity changed")
                label = parse_stated(reply["message"]["content"]) if reply["done_reason"] == "stop" else None
                append_jsonl(directory / "labels.jsonl", {"id": key, "request": count, "label": label,
                            "finish_reason": reply["done_reason"], "response_sha256": sha(response)})
                out["attempts"] = count
                if count % 24 == 0:
                    print(f"Stated {count}/{len(ids)}", flush=True)
        out["status"] = "COMPLETE"
    except Exception as error:
        out["error"] = str(error)
    finally:
        if any(m["name"] == model["tag"] for m in call("ps")[1]["models"]):
            call("generate", {"model": model["tag"], "keep_alive": 0, "stream": False})
        out["resident_after"] = [m["name"] for m in call("ps")[1]["models"]]
        save(directory / "result.json", out)
    if out["status"] != "COMPLETE" or out["resident_after"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.probes, args.output)
