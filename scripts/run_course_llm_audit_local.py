"""Send immutable blind packets to two pinned local judges in stateless requests."""

import argparse
import hashlib
import json
import platform
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from zipfile import ZipFile
from zoneinfo import ZoneInfo

from course_llm_audit import AUDITORS, CATEGORIES, ROOT, load_json, save, sha, validate_batch, witness_result

from src.sandbox.core import Action, File, State, ToolError

CONFIG = ROOT / "configs/course_llm_annotation_local_v1.json"
PACKETS = ROOT / "output/llm_annotation/pilot_v1"
PREP = ROOT / "docs/evidence/course_llm_annotation_v1/preparation.json"
OUTPUT_ROOT = ROOT / "data/annotation/llm_pilot_v1/private"
CODEBOOK = ROOT / "docs/LLM_ANNOTATOR_CODEBOOK.md"
LOCAL_ENDPOINT = "http://127.0.0.1:11434"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call(route, payload=None, timeout=10):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = Request(f"{LOCAL_ENDPOINT}/api/{route}", data=body,
                      headers={"Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read()
    return raw, json.loads(raw)


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def schema(count):
    tool = {
        "type": "object",
        "properties": {"tool": {"type": "string"},
                       "arguments": {"type": "object", "additionalProperties": {"type": "string"}}},
        "required": ["tool", "arguments"],
        "additionalProperties": False,
    }
    return {
        "type": "array", "minItems": count, "maxItems": count,
        "items": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "label": {"type": "string", "enum": list(CATEGORIES)},
                "reason": {"type": "string"},
                "witness": {"anyOf": [{"type": "null"}, {"type": "array", "items": tool, "maxItems": 3}]},
            },
            "required": ["case_id", "label", "reason", "witness"],
            "additionalProperties": False,
        },
    }


def read_cases(prompt):
    cases = json.loads(prompt.rsplit("```json\n", 1)[1].split("\n```", 1)[0])
    if any(set(case) != {"case_id", "before", "candidate", "horizon"} for case in cases):
        raise ValueError("Unexpected fields in the blinded case block")
    horizons = {case["case_id"]: case["horizon"] for case in cases}
    if len(cases) != len(horizons):
        raise ValueError("Duplicate cases in the blinded prompt")
    return horizons


def smoke_input():
    action = Action.make("delete_file", path="excluded_smoke.txt")
    states = [State(files={"excluded_smoke.txt": File("excluded")}, recycle_enabled=enabled)
              for enabled in (True, False)]
    cases = [{"case_id": f"excluded_{i}", "before": state.object(), "candidate": action.object(), "horizon": 3}
             for i, state in enumerate(states)]
    prompt = CODEBOOK.read_text() + (
        "\nReturn exactly one JSON array with case_id, label, reason and witness for the two cases.\n"
        "```json\n" + json.dumps(cases) + "\n```\n"
    )
    return prompt, {"state_text": states[0].text(), "action_text": action.text()}


def inspect_runtime(config, directory):
    if config["endpoint"] != LOCAL_ENDPOINT:
        raise ValueError("The reviewer endpoint must be loopback")
    _, version = call("version")
    if version["version"] != config["ollama_version"]:
        raise ValueError("Unexpected local Ollama runtime")
    _, tag_data = call("tags")
    _, active = call("ps")
    if active["models"]:
        raise RuntimeError("An Ollama model is already loaded. Preserve the existing session and stop.")
    by_name = {record["name"]: record for record in tag_data["models"]}
    found = {}
    for judge, model in config["models"].items():
        record = by_name.get(model["tag"])
        if not record or record["digest"] != model["digest"]:
            raise RuntimeError(f"Reviewer {judge} exact digest is absent")
        raw, shown = call("show", {"model": model["tag"]})
        (directory / f"{judge}_model_show.json").write_bytes(raw)
        if record["details"]["quantization_level"] != model["quantization"]:
            raise RuntimeError(f"Reviewer {judge} quantization changed")
        found[judge] = {
            "model": record,
            "model_info": {key: value for key, value in shown["model_info"].items()
                           if key.startswith("general.") or key.endswith(".context_length")},
            "template_sha256": fingerprint(shown.get("template", "")),
            "show_receipt_sha256": sha(directory / f"{judge}_model_show.json"),
        }
    return found


def one_request(prompt, judge, config, base):
    expected = read_cases(prompt)
    model = config["models"][judge]
    options = {**config["options"], "seed": model["seed"]}
    payload = {
        "model": model["tag"], "prompt": prompt, "stream": False,
        "keep_alive": config["keep_alive"], "format": schema(len(expected)), "options": options,
    }
    if "think" in model:
        payload["think"] = model["think"]
    save(base.with_suffix(".request.json"), payload)
    started = time.monotonic()
    summary = {"assigned": len(expected), "prompt_sha256": fingerprint(prompt), "status": "FAILED",
               "request_sha256": sha(base.with_suffix(".request.json"))}
    try:
        raw, reply = call("generate", payload, timeout=config["per_request_timeout_seconds"])
        base.with_suffix(".envelope.json").write_bytes(raw)
        summary["envelope_sha256"] = sha(base.with_suffix(".envelope.json"))
        for field in ("done", "done_reason", "prompt_eval_count", "eval_count",
                      "prompt_eval_duration", "eval_duration", "load_duration", "total_duration"):
            summary[field] = reply.get(field)
        base.with_suffix(".json").write_text(reply["response"])
        summary["response_sha256"] = sha(base.with_suffix(".json"))
        if reply.get("done") is not True or reply.get("done_reason") != "stop":
            raise ValueError("Ollama did not report a complete stop")
        if reply["prompt_eval_count"] + reply["eval_count"] >= options["num_ctx"]:
            raise ValueError("Token counts reach the context limit")
        checked = validate_batch(load_json(base.with_suffix(".json")), expected, expected)
        summary["status"] = "STRUCTURE_VERIFIED"
    except HTTPError as err:
        base.with_suffix(".error.txt").write_bytes(err.read())
        summary["error"] = f"Ollama HTTP {err.code}"
        checked = None
    except (OSError, URLError, ValueError, KeyError, ToolError) as err:
        summary["error"] = str(err)
        checked = None
    summary["elapsed_seconds"] = time.monotonic() - started
    save(base.with_suffix(".receipt.json"), summary)
    print(f"{judge}/{base.name}: {summary['status']} ({summary['elapsed_seconds']:.1f} s)", flush=True)
    return summary, checked


def run_smoke(directory, config):
    prompt, reversible_row = smoke_input()
    results = {}
    for judge in AUDITORS:
        summary, records = one_request(prompt, judge, config, directory / f"smoke_{judge}")
        passed = (records is not None and records["excluded_0"]["label"] == "reversible"
                  and records["excluded_1"]["label"] == "irreversible"
                  and witness_result(records["excluded_0"], reversible_row) == "exact")
        results[judge] = {**summary, "known_example_check": bool(passed)}
    return results


def run_pilot(directory, config):
    prep = load_json(PREP)
    valid_metadata, results = {}, {}
    for judge in AUDITORS:
        judge_dir = directory / judge
        judge_dir.mkdir()
        packet = PACKETS / f"blind_audit_{judge}.zip"
        if sha(packet) != prep["packet_sha256"][packet.name]:
            raise ValueError(f"Immutable packet hash mismatch for {judge}")
        batches = {}
        with ZipFile(packet) as bundle:
            for name in sorted(bundle.namelist()):
                if not re.fullmatch(r"batch_0[1-5]\.md", name):
                    raise ValueError("Unexpected packet member")
                base = judge_dir / Path(name).stem.removeprefix("batch_")
                batches[base.name], _ = one_request(bundle.read(name).decode(), judge, config, base)
        results[judge] = batches
        if len(batches) == 5 and all(item["status"] == "STRUCTURE_VERIFIED" for item in batches.values()):
            model = config["models"][judge]
            valid_metadata[judge] = {
                "provider": f"Local Ollama {config['ollama_version']}",
                "model_as_displayed": model["tag"],
                "declared_family": model["family"],
                "settings": json.dumps({"options": config["options"], "model_request": model,
                                        "schema_sha256": fingerprint(json.dumps(schema(20))),
                                        "keep_alive": config["keep_alive"]}, sort_keys=True),
                "run_date": datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat(),
                "blinded_separate_sessions": True,
            }
    if valid_metadata:
        save(directory / "metadata.json", valid_metadata)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "pilot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--smoke-receipt", type=Path)
    args = parser.parse_args()
    directory = args.output.resolve()
    if not directory.is_relative_to(OUTPUT_ROOT):
        parser.error("Write reviewer records under the private annotation path")
    config = load_json(args.config)
    locks = {"config_sha256": sha(args.config), "runner_sha256": sha(Path(__file__)), "packet_receipt_sha256": sha(PREP)}
    if args.mode == "pilot":
        if not args.smoke_receipt:
            parser.error("Supply the excluded smoke receipt")
        smoke = load_json(args.smoke_receipt)
        if smoke["status"] != "PASS_LOCAL_JUDGE_SMOKE" or smoke["locks"] != locks:
            parser.error("Both locked reviewer routes must pass smoke")
    directory.mkdir(parents=True, exist_ok=False)
    provenance = inspect_runtime(config, directory)
    if args.mode == "smoke":
        results = run_smoke(directory, config)
        passed = all(row["known_example_check"] for row in results.values())
        status = "PASS_LOCAL_JUDGE_SMOKE" if passed else "FAIL_LOCAL_JUDGE_SMOKE"
    else:
        results = run_pilot(directory, config)
        status = "PACKET_ATTEMPTS_RECORDED"
    save(directory / "run_receipt.json", {
        "status": status, "locks": locks, "scope": config["scope"], "ollama_version": config["ollama_version"],
        "machine": platform.machine(), "platform": platform.platform(), "model_identities": provenance,
        "mode": args.mode, "results": results, "external_calls": 0, "human_judgments": 0,
    })
    print(status)


if __name__ == "__main__":
    main()
