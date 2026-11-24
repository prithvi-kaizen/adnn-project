import importlib
import socket
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_mac_safer_behavior")


def test_monitor_interrupts_an_active_request_after_swap_growth(runner, tmp_path, monkeypatch):
    initial = {"memory_free_percentage": 55, "swap_used_bytes": 0}
    breached = {"memory_free_percentage": 55, "swap_used_bytes": 400}
    readings = iter((initial, initial, breached))
    monkeypatch.setattr(runner, "system_memory", lambda: next(readings).copy())
    config = {"min_free_memory_percent": 18, "max_added_swap_bytes": 384, "min_free_disk_bytes": 0,
              "wall_seconds": 30, "sample_seconds": 0.001}
    budget = runner.MacBudget(config, tmp_path)
    interrupted = Event()
    budget.active = Mock()
    budget.active.shutdown.side_effect = lambda _: interrupted.set()
    with budget:
        assert interrupted.wait(1)
        with pytest.raises(RuntimeError, match="swap-growth"):
            budget.check()
    budget.active.shutdown.assert_called_once_with(socket.SHUT_RDWR)


def test_budget_rejects_low_memory_before_starting(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "system_memory", lambda: {"memory_free_percentage": 12, "swap_used_bytes": 0})
    config = {"min_free_memory_percent": 18, "max_added_swap_bytes": 384, "min_free_disk_bytes": 0,
              "wall_seconds": 30, "sample_seconds": 0.001}
    budget = runner.MacBudget(config, tmp_path)
    with pytest.raises(RuntimeError, match="memory-pressure"), budget:
        pytest.fail("The model request should not start")
