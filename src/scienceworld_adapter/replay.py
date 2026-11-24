"""Object-tree equality is diagnostic, never a certificate of complete state."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Protocol


@dataclass(frozen=True)
class ExportedState:
    payload: str
    complete: bool
    incompleteness_reason: str | None = None

    def __post_init__(self):
        if type(self.payload) is not str or type(self.complete) is not bool:
            raise ValueError("State export needs text and an explicit boolean completeness claim")
        if (self.complete and self.incompleteness_reason) or (not self.complete and not self.incompleteness_reason):
            raise ValueError("Completeness and reason disagree")

    def digest(self) -> str:
        return hashlib.sha256(self.payload.encode()).hexdigest()


@dataclass(frozen=True)
class RecoveryPolicy:
    """Explicit caller-audited partial rule; no ScienceWorld rule is certified."""
    name: str
    complete: bool
    partial_match: Callable[[ExportedState, ExportedState], dict | None]


class ReplayEnvironment(Protocol):
    legal_actions_complete: bool
    def replay(self, actions: tuple[str, ...]) -> ExportedState: ...
    def legal_actions(self) -> tuple[str, ...]: ...


class ReplayCertificationError(RuntimeError):
    """A purported legal replay did not produce a certified engine action."""

    def __init__(self, receipt: dict):
        self.receipt = receipt
        super().__init__(receipt["reason"])


class ScienceWorldReplay:
    """Reconstruct from the same task variation without snapshot agent tools."""

    legal_actions_complete = False  # engine enumeration is not yet a certified full grammar

    def __init__(self, task: str, variation: int = 0, simplifications: str = "", env=None):
        if env is None:
            from scienceworld import ScienceWorldEnv
            env = ScienceWorldEnv(envStepLimit=10000)
        self.env = env
        self.task = task
        self.variation = variation
        self.simplifications = simplifications
        self.history: tuple[str, ...] = ()

    def replay(self, actions: tuple[str, ...]) -> ExportedState:
        # load + reset both matter: do not reuse task-goal state from another branch.
        self.env.load(self.task, self.variation, self.simplifications)
        self.env.reset()
        for action in actions:
            if action not in self.legal_actions():
                raise ValueError(f"Replay action not in engine enumeration: {action}")
            interface = self.env.server.agentInterface().get()
            history_before = int(interface.agent().getActionHistorySize())
            observation, _, _, _ = self.env.step(action)
            receipt = {"action": action, "observation": observation,
                       "history_before": history_before,
                       "history_after": int(interface.agent().getActionHistorySize()),
                       "parser_ambiguous": bool(interface.inputParser().isInAmbiguousState()),
                       "engine_error": bool(interface.isInErrorState())}
            # The pinned Python step adds two free actions before the request
            # and three afterwards. Even a rejected request increments history
            # five times. Verify the requested action BETWEEN that scaffolding.
            classes = [str(a.getClass().getName()) for a in
                       interface.agent().getActionHistorySince(history_before)]
            prefix = ["scienceworld.actions.ActionLookAround", "scienceworld.actions.ActionInventory"]
            suffix = prefix + ["scienceworld.actions.ActionTaskDesc"]
            receipt["history_classes"] = classes
            receipt["requested_action_recorded"] = (
                len(classes) == 6 and classes[:2] == prefix and classes[-3:] == suffix)
            if (receipt["engine_error"] or receipt["parser_ambiguous"] or
                    not receipt["requested_action_recorded"]):
                receipt["reason"] = "enumerated_action_failed_to_execute_unambiguously"
                raise ReplayCertificationError(receipt)
        self.history = actions
        return self.snapshot()

    def legal_actions(self) -> tuple[str, ...]:
        # Includes `reset task`, which resets goals/focus, not the physical world.
        return tuple(sorted(set(self.env.get_valid_action_object_combinations())))

    def snapshot(self) -> ExportedState:
        payload = {
            "object_tree": self.env.getObjectTree(),
            "goal_progress": self.env.get_goal_progress(),
            "score": self.env.server.getScore(),
            "completed": self.env.server.getCompleted(),
            "task": self.task, "variation": self.variation,
            "simplifications": self.simplifications,
            "legal_actions": self.legal_actions(),
        }
        return ExportedState(
            json.dumps(payload, sort_keys=True, ensure_ascii=False), False,
            "export_omits_biological_properties_goal_counters_and_RNG_state",
        )

    def close(self):
        self.env.close()


def search_recovery(
    env: ReplayEnvironment, prefix: tuple[str, ...], action: str, horizon: int = 3,
    max_nodes: int = 10000, max_seconds: float = 60, *, diagnostic_only: bool = False,
    recovery_policy: RecoveryPolicy | None = None, max_frontier: int = 100000,
) -> dict:
    """Certify only what full state, legal grammar and partial policy support.

    Exact witnesses do not require exhaustive enumeration. Partial/irreversible
    conclusions do; incomplete or censored searches always remain unknown.
    """
    if type(horizon) is not int or horizon not in (1, 2, 3):
        raise ValueError("Horizons are 1, 2 and 3")
    if type(max_nodes) is not int or max_nodes < 1 or max_seconds <= 0 or max_frontier < 1:
        raise ValueError("Search budgets must be positive")
    started = monotonic()
    target = env.replay(prefix)
    if action not in env.legal_actions():
        raise ValueError("Illegal candidate action")
    grammar_complete = getattr(env, "legal_actions_complete", False) is True
    policy_complete = recovery_policy is not None and recovery_policy.complete is True
    result = {"label": "unknown", "horizon": horizon, "witness": None,
              "search_exhausted": False, "expanded_nodes": 0, "replayed_actions": 0,
              "state_export_complete": target.complete, "legal_action_space_complete": grammar_complete,
              "partial_policy_name": recovery_policy.name if recovery_policy else None,
              "partial_policy_complete": policy_complete, "recovery_cost": None,
              "censor_reason": target.incompleteness_reason, "depth_reached": 0,
              "diagnostic_observational_matches": [], "duplicate_states_skipped": 0,
              "max_nodes": max_nodes, "max_seconds": max_seconds, "max_frontier": max_frontier,
              "candidate_partial_witness": None}
    result["action_space_checks"] = 1
    if not target.complete and not diagnostic_only:
        result["elapsed_seconds"] = monotonic() - started
        return result
    frontier = deque([()])
    visited = set()
    while frontier:
        if result["expanded_nodes"] >= max_nodes or monotonic() - started >= max_seconds:
            result["censor_reason"] = "search_censored"
            break
        path = frontier.popleft()
        branch = prefix + (action,) + path
        try:
            state = env.replay(branch)
        except ReplayCertificationError as error:
            result.update(censor_reason="replay_execution_not_certified",
                          failed_replay={"branch": list(branch), **error.receipt})
            break
        if state.complete != target.complete:
            result.update(censor_reason="state_certification_changed_during_search",
                          state_export_complete=False)
            break
        result["expanded_nodes"] += 1
        result["replayed_actions"] += len(branch)
        result["depth_reached"] = max(result["depth_reached"], len(path))
        if monotonic() - started >= max_seconds:
            result["censor_reason"] = "search_censored"
            break
        if state.payload == target.payload:
            if state.complete:
                result.update(label="reversible", witness=list(path), recovery_cost=0, censor_reason=None)
                break
            result["diagnostic_observational_matches"].append(list(path))
        elif state.complete and recovery_policy:
            match = recovery_policy.partial_match(target, state)
            if match is not None:
                if (not isinstance(match, dict) or not isinstance(match.get("cost"), (int, float))
                        or not 0 <= match["cost"] < float("inf") or not match.get("differences")):
                    raise ValueError("Partial match must disclose finite nonnegative cost and differences")
                candidate = {"witness": list(path), **match}
                previous = result["candidate_partial_witness"]
                if previous is None or (match["cost"], len(path)) < (previous["cost"], len(previous["witness"])):
                    result["candidate_partial_witness"] = candidate
        if state.complete and state.digest() in visited:
            result["duplicate_states_skipped"] += 1
            continue
        if state.complete:
            visited.add(state.digest())
        if len(path) < horizon:
            legal = tuple(sorted(set(env.legal_actions())))
            # Completeness belongs to each expanded state, not just the root.
            # A newly reachable ambiguous/unsupported action space poisons a
            # negative or partial conclusion even if the initial grammar passed.
            result["action_space_checks"] += 1
            grammar_complete = grammar_complete and getattr(env, "legal_actions_complete", False) is True
            result["legal_action_space_complete"] = grammar_complete
            if len(frontier) + len(legal) > max_frontier:
                result["censor_reason"] = "search_frontier_censored"
                break
            frontier.extend(path + (undo,) for undo in legal)
    else:
        result["search_exhausted"] = True
        if target.complete and grammar_complete and policy_complete:
            partial = result["candidate_partial_witness"]
            if partial:
                result.update(label="partially_reversible", witness=partial["witness"],
                              recovery_cost=partial["cost"], recovery_differences=partial["differences"],
                              censor_reason=None)
            else:
                result.update(label="irreversible", censor_reason=None)
        else:
            reasons = [target.incompleteness_reason] if not target.complete else []
            if not grammar_complete:
                reasons.append("legal_action_space_not_certified")
            if not policy_complete:
                reasons.append("physical_partial_recovery_policy_not_certified")
            result["censor_reason"] = ";".join(reasons)
    result["frontier_pending"] = len(frontier)
    result["elapsed_seconds"] = monotonic() - started
    return result
