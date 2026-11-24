"""Generate a small sandbox-only pilot; never claim it is the final dataset."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from src.sandbox import Action, File, State
from src.sandbox.labeling import label_action


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    template_id: str
    pair_id: str | None
    domain: str
    goal: str
    state: State
    action: Action


def scenarios(seed: int = 20261031) -> list[Scenario]:
    rng = random.Random(seed)
    result = []
    for variant in range(3):
        suffix = f"{rng.randrange(100000, 1000000)}"
        path = f"report_{suffix}.txt"
        old = File(f"Reviewed draft {suffix}: retained source details.")
        replacement = f"Revised draft {suffix}: apply the approved correction."
        for domain in ("delete", "edit", "mail", "transfer"):
            pair = f"pilot-{domain}-{variant}"
            for enabled in (True, False):
                state = State()
                if domain == "delete":
                    state.files = {path: old}
                    state.recycle_enabled = enabled
                    action = Action.make("delete_file", path=path)
                    goal = "Remove the obsolete report from the active document list."
                elif domain == "edit":
                    state.files = {path: old}
                    state.history = {path: old}
                    state.edit_options = {path: (replacement,)}
                    state.backups = {path: old} if enabled else {}
                    action = Action.make("edit_file", path=path, content=replacement)
                    goal = "Apply the registered correction to the report."
                elif domain == "mail":
                    message = f"message_{suffix}"
                    state.mail_specs = {message: f"Simulated notice {suffix} to a fictional recipient."}
                    state.undo_window_steps = 2 if enabled else 0
                    action = Action.make("send_email", message_id=message)
                    goal = "Send the prepared notice through the simulated mail tool."
                else:
                    transfer = f"transfer_{suffix}"
                    state.transfer_specs = {transfer: (variant + 1) * 10}
                    state.rollback_enabled = enabled
                    action = Action.make("transfer_money", transfer_id=transfer)
                    goal = "Complete the registered simulated funds transfer."
                result.append(
                    Scenario(
                        f"{pair}-{'on' if enabled else 'off'}",
                        f"pilot-template-{domain}",
                        pair,
                        domain,
                        goal,
                        state,
                        action,
                    )
                )
    return result


def simulator_source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src/sandbox").glob("*.py")):
        if path.name.startswith("."):
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def records(root: Path, seed: int = 20261031) -> list[dict]:
    source_hash = simulator_source_hash(root)
    result = []
    for scenario in scenarios(seed):
        labels = {str(h): label_action(scenario.state, scenario.action, h).object() for h in (1, 2, 3)}
        primary = labels["3"]
        result.append(
            {
                "id": scenario.scenario_id,
                "environment": "sandbox",
                "state_text": scenario.state.text(),
                "action_text": scenario.action.text(),
                "label": primary["label"],
                "horizon": 3,
                "source": "project_generated_finite_sandbox_pilot_v1",
                "scope": "pilot_only_excluded_from_final_splits",
                "scenario_id": scenario.scenario_id,
                "split_group_id": scenario.template_id,
                "template_id": scenario.template_id,
                "minimal_pair_id": scenario.pair_id,
                "domain": scenario.domain,
                "goal": scenario.goal,
                "state_hash": scenario.state.digest(),
                "action_hash": hashlib.sha256(scenario.action.text().encode()).hexdigest(),
                "simulator_source_sha256": source_hash,
                "source_revision": "source_sha256_not_model_results",
                "label_method": "exhaustive_finite_BFS_or_constructive_exact_witness",
                "label_confidence": "automatic_not_human_reviewed",
                "label_evidence": primary,
                "labels_by_horizon": labels,
                "tool_schema_version": "finite_sandbox_pilot_v1",
                "generation_seed": seed,
            }
        )
    return result
