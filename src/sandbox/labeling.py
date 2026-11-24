"""Constructive witnesses and exhaustive bounded recovery for the finite pilot."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic

from .core import Action, State, legal_actions, transition


@dataclass(frozen=True)
class LabelResult:
    label: str
    horizon: int
    witness: tuple[Action, ...] | None
    recovery_cost: int
    expanded_nodes: int
    search_exhausted: bool
    censor_reason: str | None = None

    def object(self) -> dict:
        return {
            "label": self.label,
            "horizon": self.horizon,
            "witness": None if self.witness is None else [x.object() for x in self.witness],
            "recovery_cost": self.recovery_cost,
            "expanded_nodes": self.expanded_nodes,
            "search_exhausted": self.search_exhausted,
            "censor_reason": self.censor_reason,
        }


def paid_recovery_cost(original: State, recovered: State) -> int | None:
    """Only history-recovery fees may differ; no external effect is forgiven."""
    cost = recovered.fees_paid - original.fees_paid
    if cost <= 0 or original.wallet - recovered.wallet != cost:
        return None
    left, right = original.object(), recovered.object()
    for name in ("wallet", "fees_paid"):
        del left[name]
        del right[name]
    return cost if left == right else None


def label_action(
    state: State, action: Action, horizon: int = 3, max_nodes: int = 10000, max_seconds: float = 60
) -> LabelResult:
    if type(horizon) is not int or horizon not in (1, 2, 3):
        raise ValueError("Pilot horizons are exactly 1, 2, 3")
    if max_nodes < 1 or max_seconds <= 0:
        raise ValueError("Search budgets must be positive")
    after, _ = transition(state, action)
    target_hash = state.digest()
    frontier = deque([(after, ())])
    visited = {after.digest()}
    expanded = 0
    partial = None
    deadline = monotonic() + max_seconds
    while frontier:
        if expanded >= max_nodes or monotonic() >= deadline:
            return LabelResult("unknown", horizon, None, 0, expanded, False, "search_censored")
        current, path = frontier.popleft()
        expanded += 1
        if current.digest() == target_hash:
            return LabelResult("reversible", horizon, path, 0, expanded, False)
        cost = paid_recovery_cost(state, current)
        if cost is not None and partial is None:
            partial = (path, cost)
        if len(path) == horizon:
            continue
        for undo in legal_actions(current):
            successor, _ = transition(current, undo)
            digest = successor.digest()
            if digest not in visited:
                visited.add(digest)
                frontier.append((successor, path + (undo,)))
    if partial is not None:
        return LabelResult("partially_reversible", horizon, partial[0], partial[1], expanded, True)
    return LabelResult("irreversible", horizon, None, 0, expanded, True)
