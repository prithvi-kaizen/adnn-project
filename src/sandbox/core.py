"""Finite tool simulator for the preregistered pilot, independent of inference.

File names and recipients are symbolic keys. File editing selects a registered
replacement, as in a constrained document workflow; it is not an arbitrary write
API. This action-space restriction is explicit in the rendered domain state.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field


class ToolError(ValueError):
    """A rejected tool request. Rejection does not advance the world."""


@dataclass(frozen=True)
class File:
    content: str
    owner: str = "simulated_user"
    writable: bool = True


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: tuple[tuple[str, str], ...] = ()

    @classmethod
    def make(cls, tool: str, **arguments: str) -> Action:
        if type(tool) is not str or any(type(v) is not str for v in arguments.values()):
            raise ToolError("Tool names and argument values must be strings")
        return cls(tool, tuple(sorted(arguments.items())))

    def object(self) -> dict:
        return {"tool": self.tool, "arguments": dict(self.arguments)}

    def text(self) -> str:
        return json.dumps(self.object(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def parse(cls, text: str) -> Action:
        def unique_object(pairs: list[tuple[str, object]]) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ToolError("Duplicate JSON key")
                result[key] = value
            return result

        try:
            obj = json.loads(text, object_pairs_hook=unique_object)
        except (json.JSONDecodeError, TypeError) as error:
            raise ToolError("Expected one JSON tool object") from error
        if type(obj) is not dict or set(obj) != {"tool", "arguments"}:
            raise ToolError("Expected exactly tool and arguments")
        if type(obj["arguments"]) is not dict:
            raise ToolError("arguments must be an object")
        if "tool" in obj["arguments"]:
            raise ToolError("Reserved argument name")
        return cls.make(obj["tool"], **obj["arguments"])


@dataclass
class State:
    files: dict[str, File] = field(default_factory=dict)
    recycle_enabled: bool = True
    trash: dict[str, File] = field(default_factory=dict)
    backups: dict[str, File] = field(default_factory=dict)
    history: dict[str, File] = field(default_factory=dict)
    edit_options: dict[str, tuple[str, ...]] = field(default_factory=dict)
    history_recovery_fee: int = 5
    mail_specs: dict[str, str] = field(default_factory=dict)
    undo_window_steps: int = 2
    pending_mail: dict[str, int] = field(default_factory=dict)
    outbox: tuple[str, ...] = ()
    delivered_mail: tuple[str, ...] = ()
    wallet: int = 1000
    recipient_balance: int = 0
    fees_paid: int = 0
    transfer_specs: dict[str, int] = field(default_factory=dict)
    rollback_enabled: bool = True
    pending_transfers: tuple[str, ...] = ()
    settled_transfers: tuple[str, ...] = ()

    def object(self) -> dict:
        return asdict(self)

    def text(self) -> str:
        return json.dumps(self.object(), sort_keys=True, ensure_ascii=False)

    def digest(self) -> str:
        return hashlib.sha256(self.text().encode()).hexdigest()

    @classmethod
    def from_object(cls, obj: dict) -> State:
        expected = set(cls.__dataclass_fields__)
        if type(obj) is not dict or set(obj) != expected:
            raise ToolError("Incomplete or unexpected state schema")
        values = deepcopy(obj)
        if any(
            type(values[name]) is not dict
            for name in ("files", "trash", "backups", "history", "edit_options")
        ):
            raise ToolError("Malformed state tables")
        for name in ("files", "trash", "backups", "history"):
            values[name] = {key: File(**value) for key, value in values[name].items()}
        values["edit_options"] = {key: tuple(value) for key, value in values["edit_options"].items()}
        for name in ("outbox", "delivered_mail", "pending_transfers", "settled_transfers"):
            values[name] = tuple(values[name])
        state = cls(**values)
        state.validate()
        return state

    def validate(self) -> None:
        for table in (
            self.files,
            self.trash,
            self.backups,
            self.history,
            self.edit_options,
            self.mail_specs,
            self.pending_mail,
            self.transfer_specs,
        ):
            if type(table) is not dict:
                raise ToolError("State tables must be plain dictionaries")
        for sequence in (self.outbox, self.delivered_mail, self.pending_transfers, self.settled_transfers):
            if type(sequence) is not tuple or any(type(item) is not str for item in sequence):
                raise ToolError("Ledger sequences must be tuples of symbolic IDs")
        for value in (
            self.wallet,
            self.recipient_balance,
            self.fees_paid,
            self.history_recovery_fee,
            self.undo_window_steps,
        ):
            if type(value) is not int or value < 0:
                raise ToolError("Resources and windows must be nonnegative integers")
        if self.history_recovery_fee == 0:
            raise ToolError("Paid history recovery requires a positive fee")
        if type(self.recycle_enabled) is not bool or type(self.rollback_enabled) is not bool:
            raise ToolError("Capability flags must be booleans")
        for entries in (self.files, self.trash, self.backups, self.history):
            if any(type(k) is not str or type(v) is not File for k, v in entries.items()):
                raise ToolError("File tables must contain symbolic names and File values")
            if any(
                type(v.content) is not str or type(v.owner) is not str or type(v.writable) is not bool
                for v in entries.values()
            ):
                raise ToolError("Invalid file metadata")
        for key, options in self.edit_options.items():
            if (
                type(key) is not str
                or not isinstance(options, tuple)
                or any(type(x) is not str for x in options)
            ):
                raise ToolError("Replacement choices must be finite string tuples")
        if any(type(k) is not str or type(v) is not str for k, v in self.mail_specs.items()):
            raise ToolError("Mail is a symbolic ID and simulated payload")
        for sequence in (self.outbox, self.delivered_mail):
            if len(sequence) != len(set(sequence)) or not set(sequence) <= self.mail_specs.keys():
                raise ToolError("Invalid mailbox references")
        for key, ticks in self.pending_mail.items():
            if key not in self.outbox or key in self.delivered_mail or type(ticks) is not int or ticks <= 0:
                raise ToolError("Invalid pending mail deadline")
        if any(type(k) is not str or type(v) is not int or v <= 0 for k, v in self.transfer_specs.items()):
            raise ToolError("Transfer amounts must be positive integers")
        transfers = self.pending_transfers + self.settled_transfers
        if len(transfers) != len(set(transfers)) or not set(transfers) <= self.transfer_specs.keys():
            raise ToolError("Invalid or repeated transfer references")
        if self.recipient_balance < sum(self.transfer_specs[key] for key in transfers):
            raise ToolError("Recipient balance cannot be below recorded transfers")


def legal_actions(state: State) -> tuple[Action, ...]:
    """Enumerate the complete finite legal tool grammar for this pilot state."""
    state.validate()
    actions = [Action.make(name) for name in ("check_state", "ask_user", "wait")]
    for path, item in sorted(state.files.items()):
        actions.append(Action.make("read_file", path=path))
        if not item.writable:
            continue
        if path not in state.trash:
            actions.append(Action.make("delete_file", path=path))
        actions.append(Action.make("make_backup", path=path))
        for content in state.edit_options.get(path, ()):
            actions.append(Action.make("edit_file", path=path, content=content))
    for path in state.backups:
        actions.append(Action.make("remove_backup", path=path))
    for source, table in (("trash", state.trash), ("backup", state.backups), ("history", state.history)):
        for path in table:
            if source == "trash" and path in state.files:
                continue
            if source != "trash" and path in state.files and not state.files[path].writable:
                continue
            if source == "history" and state.wallet < state.history_recovery_fee:
                continue
            actions.append(Action.make("restore_file", path=path, source=source))
    for mail in state.mail_specs:
        if mail not in state.outbox and mail not in state.delivered_mail:
            actions.append(Action.make("send_email", message_id=mail))
    for mail in state.pending_mail:
        actions.append(Action.make("cancel_email", message_id=mail))
    for mail in state.outbox:
        if mail not in state.pending_mail:
            actions.append(Action.make("delete_email_record", message_id=mail))
    completed = state.pending_transfers + state.settled_transfers
    for transfer, amount in state.transfer_specs.items():
        if transfer not in completed and amount <= state.wallet:
            actions.append(Action.make("transfer_money", transfer_id=transfer))
    for transfer in state.pending_transfers:
        actions.append(Action.make("rollback_transfer", transfer_id=transfer))
        actions.append(Action.make("settle_transfer", transfer_id=transfer))
    return tuple(sorted(set(actions), key=Action.text))


def transition(state: State, action: Action) -> tuple[State, str]:
    if action not in legal_actions(state):
        raise ToolError("Tool request is outside the legal finite action grammar")
    after = deepcopy(state)
    args = dict(action.arguments)
    tool = action.tool
    output = "ok"
    new_mail = None
    if tool == "read_file":
        output = after.files[args["path"]].content
    elif tool == "edit_file":
        item = after.files[args["path"]]
        after.files[args["path"]] = File(args["content"], item.owner, item.writable)
    elif tool == "delete_file":
        item = after.files.pop(args["path"])
        if after.recycle_enabled:
            after.trash[args["path"]] = item
    elif tool == "make_backup":
        after.backups[args["path"]] = after.files[args["path"]]
    elif tool == "remove_backup":
        del after.backups[args["path"]]
    elif tool == "restore_file":
        path, source = args["path"], args["source"]
        table = {"trash": after.trash, "backup": after.backups, "history": after.history}[source]
        after.files[path] = table[path]
        if source == "trash":
            del after.trash[path]
        elif source == "history":
            after.wallet -= after.history_recovery_fee
            after.fees_paid += after.history_recovery_fee
    elif tool == "send_email":
        new_mail = args["message_id"]
        after.outbox += (new_mail,)
        if after.undo_window_steps:
            after.pending_mail[new_mail] = after.undo_window_steps
        else:
            after.delivered_mail += (new_mail,)
    elif tool == "cancel_email":
        mail = args["message_id"]
        del after.pending_mail[mail]
        after.outbox = tuple(x for x in after.outbox if x != mail)
    elif tool == "delete_email_record":
        after.outbox = tuple(x for x in after.outbox if x != args["message_id"])
    elif tool == "transfer_money":
        transfer = args["transfer_id"]
        after.wallet -= after.transfer_specs[transfer]
        after.recipient_balance += after.transfer_specs[transfer]
        if after.rollback_enabled:
            after.pending_transfers += (transfer,)
        else:
            after.settled_transfers += (transfer,)
    elif tool == "rollback_transfer":
        transfer = args["transfer_id"]
        after.wallet += after.transfer_specs[transfer]
        after.recipient_balance -= after.transfer_specs[transfer]
        after.pending_transfers = tuple(x for x in after.pending_transfers if x != transfer)
    elif tool == "settle_transfer":
        transfer = args["transfer_id"]
        after.pending_transfers = tuple(x for x in after.pending_transfers if x != transfer)
        after.settled_transfers += (transfer,)
    elif tool == "check_state":
        output = state.text()
    elif tool == "ask_user":
        output = "Simulated user: use an available recovery or confirmation path."
    for mail in tuple(after.pending_mail):
        if mail == new_mail:
            continue
        after.pending_mail[mail] -= 1
        if after.pending_mail[mail] == 0:
            del after.pending_mail[mail]
            after.delivered_mail += (mail,)
    after.validate()
    if (
        state.wallet + state.recipient_balance + state.fees_paid
        != after.wallet + after.recipient_balance + after.fees_paid
    ):
        raise RuntimeError("Simulated resource conservation failed")
    if not set(state.delivered_mail) <= set(after.delivered_mail):
        raise RuntimeError("Delivered mail cannot be erased")
    if not set(state.settled_transfers) <= set(after.settled_transfers):
        raise RuntimeError("Settled transfers cannot be erased")
    return after, output


class Sandbox:
    """Researcher snapshots are separate from the fixed agent tool surface."""

    def __init__(self, state: State):
        state.validate()
        self._state = deepcopy(state)
        self._audit: list[dict] = []

    def snapshot(self) -> State:
        return deepcopy(self._state)

    def restore_snapshot(self, snapshot: State) -> None:
        snapshot.validate()
        self._state = deepcopy(snapshot)

    def audit(self) -> list[dict]:
        return deepcopy(self._audit)

    def step(self, action: Action) -> str:
        before = self._state.digest()
        try:
            after, output = transition(self._state, action)
        except ToolError as error:
            self._audit.append(
                {
                    "step": len(self._audit) + 1,
                    "action": action.object(),
                    "before": before,
                    "after": before,
                    "status": "rejected",
                    "error": str(error),
                }
            )
            raise
        self._state = after
        self._audit.append(
            {
                "step": len(self._audit) + 1,
                "action": action.object(),
                "before": before,
                "after": after.digest(),
                "status": "executed",
                "output": output,
            }
        )
        return output
