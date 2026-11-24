"""Schema, duplicate, minimal-pair and replay checks for sandbox records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from src.sandbox import Action, State, legal_actions, transition
from src.sandbox.labeling import label_action, paid_recovery_cost

LABELS = {"reversible", "partially_reversible", "irreversible"}
REQUIRED = {"id", "environment", "state_text", "action_text", "label", "horizon", "source"}


def validate_records(
    rows: list[dict],
    *,
    recompute: bool = True,
    require_final: bool = False,
    expected_source_hash: str | None = None,
) -> dict:
    errors: list[str] = []
    ids: set[str] = set()
    examples: set[tuple] = set()
    pairs: dict[str, list[dict]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    eligible = 0
    if not rows:
        errors.append("Dataset is empty")
    for index, row in enumerate(rows):
        prefix = f"row {index}"
        try:
            if type(row) is not dict:
                raise ValueError("Each JSONL record must be an object")
            if not REQUIRED <= row.keys():
                raise ValueError("Missing mandatory schema fields")
            if any(type(row[name]) is not str or not row[name] for name in REQUIRED - {"horizon"}):
                raise ValueError("Mandatory text fields must be nonempty strings")
            if type(row["horizon"]) is not int or row["horizon"] not in (1, 2, 3):
                raise ValueError("Invalid horizon")
            if row["id"] in ids:
                raise ValueError("Duplicate ID")
            ids.add(row["id"])
            if row["label"] not in LABELS | {"unknown"}:
                raise ValueError("Unknown class name")
            for name in (
                "state_hash",
                "action_hash",
                "simulator_source_sha256",
                "scenario_id",
                "split_group_id",
                "template_id",
                "goal",
                "tool_schema_version",
                "scope",
            ):
                if type(row[name]) is not str or not row[name]:
                    raise ValueError(f"Invalid provenance field: {name}")
            if row["environment"] != "sandbox":
                raise ValueError(
                    "Physical-environment validation is not implemented; do not certify this row"
                )
            if expected_source_hash is not None and row["simulator_source_sha256"] != expected_source_hash:
                raise ValueError("Simulator source hash mismatch")
            state = State.from_object(json.loads(row["state_text"]))
            action = Action.parse(row["action_text"])
            if state.digest() != row["state_hash"]:
                raise ValueError("State hash mismatch")
            if hashlib.sha256(row["action_text"].encode()).hexdigest() != row["action_hash"]:
                raise ValueError("Action hash mismatch")
            if action not in legal_actions(state):
                raise ValueError("Illegal candidate action")
            key = (row["environment"], state.digest(), action.text(), row["horizon"])
            if key in examples:
                raise ValueError("Duplicate state/action/horizon example")
            examples.add(key)
            if recompute:
                actual = label_action(state, action, row["horizon"]).object()
                if actual != row["label_evidence"] or actual["label"] != row["label"]:
                    raise ValueError("Oracle evidence mismatch")
                for h in (1, 2, 3):
                    if label_action(state, action, h).object() != row["labels_by_horizon"][str(h)]:
                        raise ValueError("Horizon evidence mismatch")
            evidence = row["label_evidence"]
            if evidence["label"] != row["label"] or evidence["horizon"] != row["horizon"]:
                raise ValueError("Label/evidence header mismatch")
            if row["label"] == "unknown" and (evidence["search_exhausted"] or not evidence["censor_reason"]):
                raise ValueError("Unknown labels require an explicit incomplete-search reason")
            if row["label"] in ("partially_reversible", "irreversible") and not evidence["search_exhausted"]:
                raise ValueError("Negative/free-recovery conclusion requires exhausted search")
            if evidence["witness"] is not None:
                restored, _ = transition(state, action)
                if len(evidence["witness"]) > row["horizon"]:
                    raise ValueError("Witness exceeds horizon")
                for raw_action in evidence["witness"]:
                    restored, _ = transition(restored, Action.parse(json.dumps(raw_action)))
                if row["label"] == "reversible" and restored.digest() != state.digest():
                    raise ValueError("Exact witness does not restore full domain state")
                if (
                    row["label"] == "partially_reversible"
                    and paid_recovery_cost(state, restored) != evidence["recovery_cost"]
                ):
                    raise ValueError("Invalid partial-recovery witness/cost")
            if row.get("minimal_pair_id") and type(row["minimal_pair_id"]) is not str:
                raise ValueError("Invalid minimal-pair identifier")
            if row.get("split"):
                if row["split"] not in {"train", "development", "test"}:
                    raise ValueError("Invalid split")
                groups[row["split_group_id"]].add(row["split"])
            if require_final and row.get("scope", "").startswith("pilot"):
                raise ValueError("Pilot examples must never enter final data")
            if require_final and row.get("scope", "").startswith("candidate"):
                raise ValueError("Candidate examples require final dual-environment review/freeze")
            if row["label"] in LABELS:
                eligible += 1
                if row.get("minimal_pair_id"):
                    pairs[row["minimal_pair_id"]].append(row)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{prefix}: {error}")
    paired_ids = set()
    for pair_id, members in pairs.items():
        if len(members) != 2:
            errors.append(f"{pair_id}: expected exactly two eligible members")
            continue
        first, second = members
        fixed = (
            "action_text",
            "goal",
            "tool_schema_version",
            "template_id",
            "split_group_id",
            "environment",
            "horizon",
        )
        if any(first.get(name) != second.get(name) for name in fixed):
            errors.append(f"{pair_id}: action, goal, tool grammar or group mismatch")
            continue
        left, right = json.loads(first["state_text"]), json.loads(second["state_text"])
        diff = [name for name in left if left[name] != right[name]]
        if len(diff) != 1 or first["label"] == second["label"]:
            errors.append(f"{pair_id}: requires one changed state property and differing labels")
            continue
        paired_ids.update(member["id"] for member in members)
    if any(len(splits) > 1 for splits in groups.values()):
        errors.append("Scenario/template group leaks across splits")
    fraction = len(paired_ids) / eligible if eligible else 0.0
    if require_final:
        if eligible < 2000:
            errors.append("Final data requires >=2000 eligible examples")
        if {
            row.get("environment")
            for row in rows
            if type(row) is dict and type(row.get("environment")) is str
        } != {"sandbox", "scienceworld"}:
            errors.append("Both compulsory environments must be represented")
        if fraction < 0.30:
            errors.append("Final data requires >=30% verified minimal-pair membership")
    return {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(rows),
        "eligible": eligible,
        "unknown": sum(row.get("label") == "unknown" for row in rows if type(row) is dict),
        "class_counts": dict(
            Counter(
                row["label"] if type(row.get("label")) is str else "<invalid>"
                for row in rows
                if type(row) is dict
            )
        ),
        "environment_counts": dict(
            Counter(
                row["environment"] if type(row.get("environment")) is str else "<invalid>"
                for row in rows
                if type(row) is dict
            )
        ),
        "independent_template_groups": len(
            {
                row["split_group_id"]
                for row in rows
                if type(row) is dict and type(row.get("split_group_id")) is str
            }
        ),
        "verified_minimal_pair_members": len(paired_ids),
        "minimal_pair_fraction": fraction,
        "scope": "final_acceptance" if require_final else "sandbox_pilot_only",
        "final_dataset_gate_passed": require_final and not errors,
        "errors": errors,
    }
