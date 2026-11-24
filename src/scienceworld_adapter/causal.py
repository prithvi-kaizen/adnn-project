"""Explicit source-audit probes for a proposed amendment, never final labels.

These are named field readers, not a generic simulator serializer. A projection
can expose an omission without being a sufficient transition state. In
particular, matching projections must never be used for BFS deduplication.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .replay import ExportedState, ScienceWorldReplay, search_recovery

SOURCE_REVISION = "e8216d6044e8e39be9fcb185e3b2dfb602584b52"
SCHEMA = "scienceworld_causal_proposal_v1"
MATERIAL_FIELDS = (
    "substanceName", "color", "temperatureC", "thermalConductivity", "stateOfMatter",
    "boilingPoint", "meltingPoint", "combustionPoint", "isCombusting", "hasCombusted",
    "combustionTicks", "electricallyConductive", "frictionCoefficient",
)
PROPERTY_FIELDS = {
    "propMaterial": MATERIAL_FIELDS,
    "propContainer": ("isContainer", "isOpen", "isClosable"),
    "propMoveable": ("isMovable",),
    "propDevice": ("isActivable", "isActivated", "isUsable", "isBroken"),
    "propHeatSource": ("maxTemp",),
    "propCoolingSource": ("minTemp",),
    "propPortal": ("isOpen", "isOpenable", "isLockable", "isLocked"),
}
SUPPORTED_CLASSES = {
    "scienceworld.struct.EnvObject", "scienceworld.objects.portal.Door",
    "scienceworld.objects.substance.Water", "scienceworld.objects.substance.Ice",
    "scienceworld.objects.containers.GlassCup", "scienceworld.objects.containers.MetalPot",
    "scienceworld.objects.devices.Stove", "scienceworld.objects.devices.Freezer",
}


def canonical(value) -> str:
    """Exact binary64 values; sort mappings, retain explicit array ordering."""
    def convert(item):
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Nonfinite physical state is uncertifiable")
            return {"binary64": item.hex()}
        if isinstance(item, dict):
            if not all(isinstance(k, str) for k in item):
                raise ValueError("Canonical keys must be strings")
            return {k: convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if item is None or type(item) in (bool, int, str):
            return item
        raise ValueError(f"Unsupported state value: {type(item).__name__}")
    return json.dumps(convert(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def scala_values(collection):
    iterator = collection.iterator()
    while iterator.hasNext():
        yield iterator.next()


def object_id(obj):
    return int(obj.uuid())


def optional_id(option):
    return object_id(option.get()) if option.isDefined() else None


def property_record(obj) -> dict:
    """Read every declared base material field, including both omitted exports."""
    out = {}
    for name, fields in PROPERTY_FIELDS.items():
        option = getattr(obj, name)()
        if option.isEmpty():
            out[name] = None
            continue
        prop = option.get()
        record = {field: getattr(prop, field)() for field in fields}
        if name == "propMaterial":
            record["nameInStateOfMatter"] = {
                str(entry.productElement(0)): str(entry.productElement(1))
                for entry in scala_values(prop.nameInStateOfMatter())}
        if name in ("propHeatSource", "propCoolingSource"):
            temp = prop.curSetTemp()
            record["curSetTemp"] = float(temp.get()) if temp.isDefined() else None
        if name == "propPortal":
            record.update(connectsFrom=object_id(prop.connectsFrom()), connectsTo=object_id(prop.connectsTo()))
        out[name] = record
    return out


def physical_record(obj) -> dict:
    """A named physical record. Class-specific unsupported state is flagged."""
    name = str(obj.getClass().getName())
    unsupported = []
    if name not in SUPPORTED_CLASSES:
        unsupported.append("class_not_audited:" + name)
    for prop in ("propLife", "propChromosomePairs", "propPollination", "propEdibility"):
        if getattr(obj, prop)().isDefined():
            unsupported.append("property_not_audited:" + prop)
    electrical = obj.propElectricalConnection()
    if electrical.isDefined():
        unsupported.append("electrical_transition_closure_not_audited")
    record = {
        "uuid": object_id(obj), "class": name, "typeID": int(obj.typeID()),
        "name": str(obj.name()), "objType": str(obj.objType()),
        "deleted": bool(obj.isDeleted()), "hidden": bool(obj.isHidden()),
        "parent": optional_id(obj.getContainer()),
        "children": sorted(object_id(c) for c in scala_values(obj.getContainedObjects(True))),
        "portals": sorted(object_id(c) for c in scala_values(obj.getPortals(True))),
        "tickCompleted": bool(obj.wasTickProcessed()), "electricalRole": int(obj.electricalRole()),
        "terminals": [optional_id(obj.terminal1()), optional_id(obj.terminal2())],
        "properties": property_record(obj), "unsupported": unsupported,
    }
    if obj.propPortal().isDefined():
        # Portal has constructor endpoint vals AND mutable property endpoints.
        record["constructor_endpoints"] = [object_id(obj.connectsFrom()), object_id(obj.connectsTo())]
    record["terminal_connections"] = []
    for option in (obj.terminal1(), obj.terminal2()):
        if option.isDefined():
            term = option.get()
            prop = term.propElectricalConnection()
            record["terminal_connections"].append(
                sorted(object_id(c) for c in scala_values(prop.get().getConnections())) if prop.isDefined() else [])
    if any(record["terminal_connections"]):
        unsupported.append("connected_terminal_requires_external_closure")
    return record


def seed_counter_diagnostic(plant) -> dict:
    """A counterexample detector, expressly NOT a general plant certificate."""
    lifecycle = plant.lifecycle().get()
    stage = lifecycle.getCurStage()
    if str(stage.getClass().getName()) != "scienceworld.processes.lifestage.PlantLifeStageSeed":
        raise ValueError("Only the independently constructed seed-stage diagnostic is supported")
    return {"scope": "seed_counter_diagnostic_only", "stage_index": int(lifecycle.curStage()),
            "stage_name": str(stage.stageName()), "duration": int(stage.duration()),
            "completed": bool(stage.isCompleted()), "ticksMeetingCriteria": int(stage.ticksMeetingCriteria()),
            "stageDuration": int(stage.stageDuration())}


@dataclass(frozen=True)
class ProposedState:
    payload: str
    blockers: tuple[str, ...]

    def for_search(self) -> ExportedState:
        # No switch exists to promote this proposal into the final pipeline.
        return ExportedState(self.payload, False, ";".join(self.blockers))


def proposed_state(records: list[dict], *, family: str, context: dict | None = None) -> ProposedState:
    ids = [r["uuid"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate object identity")
    blockers = {"instructor_amendment_pending", "causal_transition_closure_not_certified",
                "full_action_grammar_not_certified", "physical_partial_policy_not_certified"}
    for record in records:
        blockers.update(record.get("unsupported", ()))
    payload = canonical({"schema": SCHEMA, "source_revision": SOURCE_REVISION, "family": family,
                         "objects": sorted(records, key=lambda r: r["uuid"]), "context": context or {}})
    return ProposedState(payload, tuple(sorted(blockers)))


def door_target(interface) -> ProposedState:
    """Proposed door-network target, not a claim about the rest of the world."""
    objects = scala_values(interface.universe().getContainedObjectsAndPortalsRecursive(True, True))
    doors = [obj for obj in objects if str(obj.getClass().getName()) == "scienceworld.objects.portal.Door"]
    records = []
    for door in doors:
        records.append({"uuid": object_id(door), "name": str(door.name()),
                        "deleted": bool(door.isDeleted()), "hidden": bool(door.isHidden()),
                        "constructor_endpoints": [object_id(door.connectsFrom()), object_id(door.connectsTo())],
                        "portal": property_record(door)["propPortal"]})
    agent = interface.agent()
    return proposed_state(records, family="door_network_target_v1", context={
        "agent_id": object_id(agent), "agent_parent": optional_id(agent.getContainer()),
        "waitingCounter": int(agent.waitingCounter()),
        "parser_ambiguous": bool(interface.inputParser().isInAmbiguousState()),
        "error_state": bool(interface.isInErrorState()),
    })


class ProposedDoorReplay(ScienceWorldReplay):
    """Diagnostic-only adapter; all returned labels remain unknown."""
    def snapshot(self) -> ExportedState:
        return door_target(self.env.server.agentInterface().get()).for_search()


def search_proposed_doors(env, prefix, action, horizon, **budgets) -> dict:
    result = search_recovery(env, prefix, action, horizon, diagnostic_only=True, **budgets)
    result["scope"] = SCHEMA + "_diagnostic_only_not_dataset"
    result["candidate_target_matches"] = result.pop("diagnostic_observational_matches")
    result["methodology_approved"] = False
    result["dataset_eligible"] = False
    assert result["label"] == "unknown" and result["duplicate_states_skipped"] == 0
    return result
