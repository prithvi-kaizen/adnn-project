import importlib
import json
from pathlib import Path

import pytest

from src.sandbox import Action, File, State


@pytest.fixture
def interface(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_mac_reviewer_interactive")


def test_interactive_recovery_requires_the_model_to_choose_the_witness(interface, tmp_path):
    file = File("Counterfactual fixture text")
    state = State(files={"fixture": file}, backups={"fixture": file})
    candidate = Action.make("remove_backup", path="fixture")

    def transport(request):
        context = json.loads(request["messages"][1]["content"])
        assert "label" not in context and "witness" not in context
        if context["remaining_depth"]:
            choice = next(key for key, action in context["legal_action_menu"].items() if action["tool"] == "make_backup")
        else:
            assert context["differences"] == {} and "backtrack" in context["choices"]
            choice = "reversible"
        output = json.dumps({"choice": choice, "reason": "Fixture output."})
        body = {"model": "fixture", "done": True, "done_reason": "stop", "message": {"content": output}}
        return 200, json.dumps(body).encode()

    result = interface.review(tmp_path, {"tag": "fixture", "seed": 1}, state, candidate, 1, transport)
    assert result["calls"] == 2 and result["label"] == "reversible"
    assert result["witness"] == [Action.make("make_backup", path="fixture").object()]
    assert interface.witness_result(result, {"state_text": state.text(), "action_text": candidate.text()}) == "exact"


def test_incomplete_choice_keeps_raw_response_without_an_assigned_label(interface, tmp_path):
    body = {"model": "fixture", "done": True, "done_reason": "length", "message": {"content": '{"choice":'}}
    raw = json.dumps(body).encode()
    state = State(files={"fixture": File("Example")})
    result = interface.review(tmp_path, {"tag": "fixture", "seed": 1}, state,
                              Action.make("delete_file", path="fixture"), 2, lambda _: (200, raw))
    assert result["status"] == "invalid_return" and result["label"] is None
    assert (tmp_path / "0.response.json").read_bytes() == raw
