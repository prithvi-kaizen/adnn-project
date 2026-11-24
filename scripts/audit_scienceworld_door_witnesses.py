"""Verify real door inverses with typed parser identities and portal topology."""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scienceworld_adapter.causal import (  # noqa: E402
    SOURCE_REVISION,
    canonical,
    door_target,
    object_id,
    optional_id,
    physical_record,
    scala_values,
)

JAR_SHA = "e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465"
DOOR = "scienceworld.objects.portal.Door"
SETTINGS = [(task, variation) for task in ("boil", "freeze", "chemistry-mix") for variation in range(4)]


def digest(payload):
    return hashlib.sha256(payload.encode()).hexdigest()


def engine(env, task, variation):
    env.load(task, variation, "")
    env.reset()
    return env.server.agentInterface().get()


def typed_rows(interface):
    return json.loads(interface.getValidActionObjectCombinationsJSON())["validActions"]


def matching_door_referents(env, interface, action):
    """Mirror lexical matching for the one-object open/close triggers only."""
    parser = env._gateway.jvm.scienceworld.input.InputParser
    agent = interface.agent()
    room = agent.getContainer().get()
    objects = {}
    for root in (room, agent):
        for obj in scala_values(parser.collectAccessibleObjects(root, True)):
            objects[object_id(obj)] = obj
    verb, raw_referent = action.split(" ", 1)
    if verb not in {"open", "close"}:
        raise ValueError("Not the audited one-object door trigger")
    reference = " ".join(parser.tokenize(raw_referent.lower()))
    return sorted(uid for uid, obj in objects.items() if reference in list(parser.getObjectReferents(obj, room)))


def door_objects(interface):
    universe = interface.universe()
    objects = scala_values(universe.getContainedObjectsAndPortalsRecursive(True, True))
    return {object_id(obj): obj for obj in objects if str(obj.getClass().getName()) == DOOR}


def projection(interface):
    old = json.loads(door_target(interface).payload)
    enriched = []
    for door in door_objects(interface).values():
        record = physical_record(door)
        assert not record["unsupported"] and not record["portals"]
        assert record["children"] == sorted(terminal for terminal in record["terminals"] if terminal is not None)
        assert record["properties"]["propMaterial"] is None and record["properties"]["propContainer"] is None
        assert not record["properties"]["propMoveable"]["isMovable"]
        record.pop("tickCompleted")  # cleared before each nonfree tick by AgentInterface.step
        terminals = []
        for option in (door.terminal1(), door.terminal2()):
            assert option.isDefined()
            terminal = option.get()
            assert str(terminal.getClass().getName()) == "scienceworld.objects.electricalcomponent.Terminal"
            voltage = terminal.voltage()
            terminal_record = {"uuid": object_id(terminal), "name": str(terminal.name()),
                               "parent_object": object_id(terminal.parentObject()),
                               "container": optional_id(terminal.getContainer()),
                               "hidden": bool(terminal.isHidden()), "deleted": bool(terminal.isDeleted()),
                               "voltage": float(voltage.get()) if voltage.isDefined() else None}
            assert terminal_record["parent_object"] == record["uuid"] == terminal_record["container"]
            terminals.append(terminal_record)
        record["terminal_records"] = terminals
        record["endpoint_rooms"] = [
            {"uuid": object_id(room), "parent": optional_id(room.getContainer()), "name": str(room.name()),
             "hidden": bool(room.isHidden()), "deleted": bool(room.isDeleted()),
             "contains_portal": object_id(door) in {object_id(p) for p in scala_values(room.getPortals(True))}}
            for room in (door.connectsFrom(), door.connectsTo())
        ]
        assert all(room["contains_portal"] and not room["deleted"] for room in record["endpoint_rooms"])
        enriched.append(record)
    return canonical({"old_target": old, "door_records": sorted(enriched, key=lambda row: row["uuid"])})


def step(interface, action, *, door_id=None, expected_open=None):
    agent = interface.agent()
    begin = int(agent.getActionHistorySize())
    outcome = interface.step(action)
    recorded = list(agent.getActionHistorySince(begin))
    if len(recorded) != 1 or interface.isInErrorState() or interface.inputParser().isInAmbiguousState():
        raise ValueError("Direct command did not resolve to one executed action")
    item = recorded[0]
    kind = str(item.getClass().getName())
    receipt = {"action": action, "class": kind, "score": float(outcome._2()), "completed": bool(outcome._3())}
    if door_id is not None:
        suffix = "Open" if expected_open else "Close"
        verb = "open" if expected_open else "closed"
        obj = item.assignments().apply("door")
        assert kind == f"scienceworld.actions.Action{suffix}Door" and object_id(obj) == door_id
        assert str(outcome._1()) == f"The {obj.name()} is now {verb}."
        assert bool(obj.propPortal().get().isOpen()) is expected_open
        receipt["assigned_door_uuid"] = door_id
        receipt["new_open_flag"] = expected_open
    return receipt


def witness(env, task, variation):
    interface = engine(env, task, variation)
    doors = door_objects(interface)
    # Start from a fixed natural one-turn prefix so transient tick flags have
    # already passed one complete physical update in both compared arms.
    step(interface, "wait1")
    initial = projection(interface)
    target_map = json.loads(initial)
    candidates = sorted((row for row in typed_rows(interface) if row["action"].startswith("open ")
                         and len(row["obj_ids"]) == 1 and row["obj_ids"][0] in doors),
                        key=lambda row: (row["obj_ids"], row["action"]))
    if not candidates:
        raise ValueError("No visible closed native door")
    chosen = candidates[0]
    door_id = chosen["obj_ids"][0]
    begin_referents = matching_door_referents(env, interface, chosen["action"])
    assert begin_referents == [door_id]
    before_tree = digest(canonical(env.getObjectTree()))
    open_receipt = step(interface, chosen["action"], door_id=door_id, expected_open=True)
    opened = projection(interface)
    change = json.loads(opened)
    assert initial != opened
    assert [(r["uuid"], r["properties"]["propPortal"]["isOpen"]) for r in change["door_records"]
            if r["uuid"] == door_id] == [(door_id, True)]
    assert [(r["uuid"], r["properties"]["propPortal"]["isOpen"]) for r in target_map["door_records"]
            if r["uuid"] == door_id] == [(door_id, False)]
    inverse = sorted(row["action"] for row in typed_rows(interface)
                     if row["action"].startswith("close ") and row["obj_ids"] == [door_id])
    assert len(inverse) == 1
    recovery_referents = matching_door_referents(env, interface, inverse[0])
    assert recovery_referents == [door_id]
    close_receipt = step(interface, inverse[0], door_id=door_id, expected_open=False)
    restored = projection(interface)
    after_tree = digest(canonical(env.getObjectTree()))
    return {"task": task, "variation": variation, "candidate": open_receipt, "recovery": close_receipt,
            "initial_and_inverse_referents": [begin_referents, recovery_referents],
            "restored_candidate_referents": matching_door_referents(env, interface, chosen["action"]),
            "before_sha256": digest(initial), "opened_sha256": digest(opened), "after_sha256": digest(restored),
            "exact_door_projection_match": initial == restored, "whole_tree_match": before_tree == after_tree,
            "research_label": "unknown"}


def parser_alias_fixture(env):
    """Change an injected non-door referent in otherwise identical resets."""
    arms = []
    for alias in (False, True):
        interface = engine(env, "boil", 0)
        step(interface, "wait1")
        doors = door_objects(interface)
        choices = sorted((row for row in typed_rows(interface) if row["action"].startswith("open ")
                          and len(row["obj_ids"]) == 1 and row["obj_ids"][0] in doors),
                         key=lambda row: (row["obj_ids"], row["action"]))
        action, (door_id,) = choices[0]["action"], choices[0]["obj_ids"]
        jvm = env._gateway.jvm
        distractor = jvm.scienceworld.objects.containers.furniture.Cupboard()
        distractor_name = action.removeprefix("open ") if alias else "isolated audit cupboard"
        getattr(distractor, "name_$eq")(distractor_name)
        agent = interface.agent()
        agent.getContainer().get().addObject(distractor)
        before = projection(interface)
        matching = matching_door_referents(env, interface, action)
        size = int(agent.getActionHistorySize())
        interface.step(action)
        arms.append({"aliased_distractor": alias, "action": action, "target_door_uuid": door_id,
                     "before_projection_sha256": digest(before), "door_open_after": bool(doors[door_id].propPortal().get().isOpen()),
                     "referent_matches": matching,
                     "engine_error": bool(interface.isInErrorState()),
                     "ambiguous": bool(interface.inputParser().isInAmbiguousState()),
                     "new_action_history_entries": int(agent.getActionHistorySize()) - size})
    first, second = arms
    equal = first["before_projection_sha256"] == second["before_projection_sha256"]
    diverged = any(first[key] != second[key] for key in ("door_open_after", "ambiguous", "new_action_history_entries"))
    return {"scope": "injected_alias_adversary_not_natural_case", "arms": arms,
            "equal_projected_states": equal, "continuation_diverges": equal and diverged,
            "one_command_guard_distinguishes_arms": first["referent_matches"] != second["referent_matches"]}


def run():
    from scienceworld import ScienceWorldEnv
    from scienceworld.constants import JAR_PATH

    assert hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest() == JAR_SHA
    env = ScienceWorldEnv(envStepLimit=10000)
    report = {"scope": "typed_constructive_witnesses_without_causal_closure_certificate",
              "source_revision": SOURCE_REVISION, "jar_sha256": JAR_SHA, "cases": [], "failures": [],
              "eligible_labels": 0}
    try:
        for task, variant in SETTINGS:
            try:
                first = witness(env, task, variant)
                second = witness(env, task, variant)
                report["cases"].append({**first, "repeat_identical": first == second})
            except (AssertionError, ValueError) as error:
                report["failures"].append({"task": task, "variation": variant, "type": type(error).__name__,
                                           "error": str(error), "traceback": traceback.format_exc()})
        report["parser_alias_fixture"] = parser_alias_fixture(env)
    finally:
        env.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as output:
        output.write(json.dumps(run(), indent=2) + "\n")
