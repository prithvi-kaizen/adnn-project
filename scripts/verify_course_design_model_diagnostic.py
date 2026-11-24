"""Offline token and world replay checks for the six-case development run."""

import hashlib
import json
from pathlib import Path

from run_course_design_model_diagnostic import CONFIG, OUT, ROOT
from run_course_development_formats import read, save, sha, stated_messages
from transformers import AutoTokenizer

from src.agent.design_v1 import audit_design_episode

RECEIPT = ROOT / "docs/evidence/course_improvement_v1/i2_behavior_model/verification.json"


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def strict_label(raw):
    try:
        pairs = json.loads(raw, object_pairs_hook=tuple)
    except ValueError:
        return None
    if type(pairs) is tuple and len(pairs) == 1 and pairs[0][0] == "label":
        value = pairs[0][1]
        if value in ("reversible", "partially_reversible", "irreversible"):
            return value
    return None


def verify():
    report = read(OUT / "report.json")
    config = read(CONFIG)
    execution = read(config["model_profile"])
    assert report["status"] == "DIAGNOSTIC_OBSERVED" and report["holdout_calls"] == 0
    assert all(sha(ROOT / path) == expected for path, expected in report["source_sha256"].items())
    expected_cases = {row["id"]: row for row in read(config["design_receipt"])["tasks"]}
    assert [row["id"] for row in report["records"]] == config["case_ids"]
    snapshot = read(execution["checkpoint_receipt"])["snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    requests = {row["request_id"]: row for row in records(OUT / "generation_requests.jsonl")}
    responses = {row["request_id"]: row for row in records(OUT / "generation_responses.jsonl")}
    assert set(requests) == set(responses) == set(range(1, report["requests"] + 1))
    timing = report["timing"]["synthetic_request_id"]
    assert requests[timing]["min_new_tokens"] == requests[timing]["max_new_tokens"] == responses[timing]["new_tokens"] == 128
    observed_ids = {timing}
    verified_episodes = 0
    for row in report["records"]:
        task = expected_cases[row["id"]]
        episode_path = OUT / row["episode"]
        assert sha(episode_path) == row["episode_sha256"]
        episode = read(episode_path)
        assert audit_design_episode(task, episode) == row["score"]
        verified_episodes += 1
        stated_id = row["stated_request_id"]
        assert requests[stated_id]["messages"] == stated_messages(task)
        assert requests[stated_id]["max_new_tokens"] == 64
        assert strict_label(responses[stated_id]["text"]) == row["stated_label"]
        assert stated_id not in observed_ids
        observed_ids.add(stated_id)
        for event in episode["trace"]:
            if event["status"] == "infrastructure_failure":
                continue
            response_id = event["generation"]["request_id"]
            assert response_id not in observed_ids
            observed_ids.add(response_id)
            assert requests[response_id]["messages"] == event["request_messages"]
            assert requests[response_id]["max_new_tokens"] == 128
            assert responses[response_id] == event["generation"]
            assert responses[response_id]["text"] == event["raw_output"]
    assert set(requests) == observed_ids
    for request_id, request in requests.items():
        text = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
        assert text == request["rendered_prompt"] and request["prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert tokenizer(text, add_special_tokens=False)["input_ids"] == request["input_ids"]
        assert not request["do_sample"] and request["use_cache"]
        answer = responses[request_id]
        assert answer["text"] == tokenizer.decode(answer["output_token_ids"], skip_special_tokens=True)
    guards = records(OUT / "resource_guards.jsonl")
    return {"status": "PASS_DEVELOPMENT_REPLAY_ONLY", "audited_episodes": verified_episodes,
            "verified_requests_and_responses": len(observed_ids), "timing_only_calls": 1,
            "assigned_stated_calls": len(report["records"]),
            "agent_generation_calls": len(observed_ids) - 1 - len(report["records"]),
            "holdout_calls": 0, "minimum_free_memory_percent": min(row["memory_free_percentage"] for row in guards),
            "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in guards),
            "report_sha256": sha(OUT / "report.json"), "authorizes_inference_on_holdout": False,
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}}


if __name__ == "__main__":
    result = verify()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    save(RECEIPT, result)
    print(result)
