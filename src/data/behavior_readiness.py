"""Development-only safe-route certificates and conservative exposure grouping."""

import hashlib
import json
from collections import Counter, deque
from time import monotonic

from src.agent.loop import goal_met, run_episode, verify_episode
from src.data.splits import components
from src.sandbox.core import Action, legal_actions, transition
from src.sandbox.labeling import label_action


def certify_safe_route(scenario, *, max_turns=15, max_states=10000, max_seconds=30):
    """Search paths of cost-free reversible actions, reserving a turn for finish."""
    if not 2 <= max_turns <= 15 or max_states < 1 or max_seconds <= 0:
        raise ValueError("Invalid bounded-search budget")
    if goal_met(scenario, scenario.state):
        raise ValueError("Goal already met before candidate execution")
    deadline = monotonic() + max_seconds
    frontier = deque([(scenario.state, ())])
    seen = {scenario.state.digest()}
    labels = Counter()
    expanded = 0
    reached_depth = 0

    def report(status, path=None):
        return {"status": status, "world_action_limit": max_turns - 1,
                "recovery_horizon_per_step": 3, "expanded_states": expanded,
                "states_discovered": len(seen), "deepest_dequeued": reached_depth,
                "oracle_status_counts": dict(labels),
                "actions": [action.object() for action in path] if path is not None else None}

    while frontier:
        if expanded >= max_states or monotonic() >= deadline:
            return report("UNKNOWN_SEARCH_BUDGET")
        state, path = frontier.popleft()
        expanded += 1
        reached_depth = max(reached_depth, len(path))
        if goal_met(scenario, state):
            actions = iter([action.text() for action in path] + [Action.make("finish", reason="Goal met.").text()])
            trace = run_episode(scenario, lambda _: {"text": next(actions)}, max_steps=max_turns)
            verify_episode(trace)
            executed = [event for event in trace["trace"] if event["status"] == "executed"]
            if (trace["status"] != "finish" or not trace["goal_met_at_termination"] or
                    trace["final_hash"] != state.digest() or len(executed) != len(path)):
                raise ValueError("Constructive route did not reproduce in the sandbox agent loop")
            return {**report("VERIFIED_ALL_STEPS_REVERSIBLE_ROUTE", path),
                    "final_state_hash": trace["final_hash"],
                    "uses_shortcut_action_after_preparation": scenario.action in path and path[0] != scenario.action,
                    "scripted_not_model_behavior": True,
                    "trace": [{"action": event["action"], "status": event["status"],
                               "before_hash": event["before_hash"], "after_hash": event["after_hash"]}
                              for event in trace["trace"]]}
        if len(path) == max_turns - 1:
            continue
        for action in legal_actions(state):
            after, _ = transition(state, action)
            digest = after.digest()
            if digest in seen:
                continue
            remaining = deadline - monotonic()
            if remaining <= 0:
                return report("UNKNOWN_SEARCH_BUDGET")
            evidence = label_action(state, action, 3, max_seconds=remaining)
            labels[evidence.label] += 1
            if evidence.label == "reversible":
                seen.add(digest)
                frontier.append((after, (*path, action)))
    return report("UNKNOWN_STEP_LABELS" if labels["unknown"] else "NO_ALL_REVERSIBLE_ROUTE_WITHIN_LIMIT")


def shape_key(state_text):
    """Deliberately overgroup by anonymized schema shape, invariant to renamed map keys."""
    def strip(value):
        if type(value) is dict:
            return ["mapping", sorted((strip(item) for item in value.values()), key=render)]
        if type(value) is list:
            return ["ordered", [strip(item) for item in value]]
        if type(value) is str:
            return "symbolic"
        if type(value) in (int, float):
            return "zero" if value == 0 else "positive" if value > 0 else "negative"
        if type(value) is bool or value is None:
            return value
        raise ValueError("Unexpected JSON state value")

    def render(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    state = json.loads(state_text)
    return hashlib.sha256(render({name: strip(value) for name, value in state.items()}).encode()).hexdigest()


def joint_exposure_graph(history, candidates):
    """Join frozen grouping keys plus a broader quarantine key before allocating any split."""
    rows = []
    for kind, records in (("prior", history), ("candidate", candidates)):
        for index, row in enumerate(records):
            if row.get("environment", "sandbox") != "sandbox":
                raise ValueError("Only sandbox state structures are supported")
            node = {**row, "id": f"{kind}:{index}", "environment": "sandbox",
                    "scenario_id": row.get("scenario_id") or row["id"],
                    "minimal_pair_id": row.get("minimal_pair_id") or row.get("pair_id")}
            node["split_group_id"] = row.get("split_group_id")
            rows.append(node)
    original = components(rows)
    extras = [{**row, "split_group_id": shape_key(row["state_text"]), "template_id": original[row["id"]]}
              for row in rows]
    grouped = components(extras)
    prior_groups = {grouped[row["id"]] for row in rows if row["id"].startswith("prior:")}
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    output = {}
    for index, candidate in enumerate(candidates):
        group = grouped[f"candidate:{index}"]
        output[candidate["id"]] = {"group": group, "quarantined": group in prior_groups}
    return {"groups": len(set(grouped.values())), "prior_groups": len(prior_groups),
            "candidate_assignments": output, "all_historical_input_rows_quarantined": True,
            "used_model_outcomes": False, "final_split": False}
