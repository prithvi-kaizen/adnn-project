"""Replay archived ByteSized32 rows and certify source-audited game transitions."""

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import zipfile
from collections import Counter, deque
from copy import deepcopy
from pathlib import Path

REVISION = "0d5a741146c9d156f92930d5254c0d5e16facc0d"
GAME_SHA = {
    "scale-weigh": "2dce1cdb50834ed4c71f71ae0e2e5e4dd59f02a434f80231c582e380ea39c157",
    "balance-scale-weigh": "5934841a36aa070fdba4f603263c33c6674ff7969257a5e2df3ad7ac0ed17b73",
    "balance-scale-heaviest": "17ebfa749f9e897aa11a6964ce929762c1a0a123a4db65b3c7241e4204b02bcd",
    "thermometer": "882637ade152dbc5d99d2f082fa62061bf1c1f91eb6819505ccfa05ad086ca98",
    "use-bandage": "5aad117c84edb8fe5160a65c107b3f430d6f4f809c939307548f40ebd7b95a2b",
}
INITIAL_GAMES = ("scale-weigh", "balance-scale-weigh", "balance-scale-heaviest")
ARCHIVE_SHA = "c9f7adaa7c505bffa06ea60d9b8efc4206ce1c4bf8c71a374b21d24360a43137"
BOOKKEEPING = {"numSteps", "observationStr", "possibleActions"}
DISPATCH = {"look around", "inventory", "examine", "take", "put", "answer", "use", "open", "close"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical(game, uuid):
    """Retain values and object alias edges, including RNG and UUID allocation."""
    objects = {}

    def encode(value):
        kind = type(value)
        if value is None or kind in (bool, int, str):
            return value
        if kind is float:
            return ["binary64", value.hex()]
        if kind is tuple:
            return ["tuple", [encode(item) for item in value]]
        identity = id(value)
        if identity in objects:
            return ["ref", objects[identity][0]]
        index = len(objects)
        objects[identity] = (index, value)
        if kind is list:
            raw = [encode(item) for item in value]
        elif kind is dict and all(type(key) is str for key in value):
            raw = {key: encode(value[key]) for key in sorted(value)}
        elif kind is random.Random:
            raw = encode(value.getstate())
        elif kind.__module__ == type(game).__module__:
            fields = vars(value)
            raw = {key: encode(fields[key]) for key in sorted(fields) if value is not game or key not in BOOKKEEPING}
        else:
            raise ValueError(f"Unaudited value type {kind!r}")
        return [kind.__name__, index, raw]

    return serialize({"game": encode(game), "allocator": uuid})


def visible(game, last_action, uuid):
    objects = [game.rootObject, *game.rootObject.getAllContainedObjectsRecursive()]
    return {"observation": game.observationStr, "look": game.actionLook(), "inventory": game.actionInventory(),
            "taskDesc": game.getTaskDescription(), "lastAction": last_action, "max_UUID": uuid,
            "objects": [{"name": item.name, "uuid": item.uuid, "type": type(item).__name__,
                         "properties": deepcopy(item.properties), "contains": [obj.name for obj in item.contains]}
                        for item in objects]}


def score(game):
    return {name: getattr(game, name) for name in ("score", "gameOver", "gameWon")}


def load_game(directory, name):
    path = directory / "games" / f"{name}.py"
    assert sha(path) == GAME_SHA[name]
    source = ast.parse(path.read_text())
    # Neither the transition nor an object tick consumes the run's turn count.
    reads = [node.lineno for node in ast.walk(source) if isinstance(node, ast.Attribute)
             and node.attr == "numSteps" and isinstance(node.ctx, ast.Load)]
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert reads and min(reads) >= main.lineno
    game_over_writers = [(node.lineno, node.value.value) for node in ast.walk(source)
                         if isinstance(node, ast.Assign) and len(node.targets) == 1
                         and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == "gameOver"]
    assert sorted(game_over_writers)[0][1] is False
    assert all(value is True for _, value in sorted(game_over_writers)[1:])
    spec = importlib.util.spec_from_file_location(f"_adnn_bytesized_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(module, game, uuid, action):
    module.UUID = uuid
    clone = deepcopy(game)
    clone.generatePossibleActions()
    clone.step(action)
    return clone, module.UUID


def choices(game):
    actions = game.generatePossibleActions()
    assert all(assignments and assignments[0][0] in DISPATCH for assignments in actions.values())
    # The engine recognizes exactly its dictionary of generated strings. Include
    # a sentinel for all unrecognized strings, which may still advance a tick.
    return {**actions, "__unrecognized_command__": [["unrecognized"]]}


def constructive_recovery(verb, assignments, after):
    if verb not in {"take", "put", "open", "close"} or len(assignments) != 1:
        return False
    obj = assignments[0][1]
    by_id = {child.uuid: child for child in [after.rootObject, *after.rootObject.getAllContainedObjectsRecursive()]}
    found = by_id.get(obj.uuid)
    if found is None:
        return False
    if verb in {"open", "close"}:
        return obj.getProperty("isOpen") is not (verb == "open") and found.getProperty("isOpen") is (verb == "open")
    return (obj.parentContainer is not None and found.parentContainer is not None
            and found.parentContainer.uuid != obj.parentContainer.uuid)


def recover(module, start, uuid, target, *, limit=2000, max_transitions=None):
    initial = canonical(start, uuid)
    frontier = deque([(start, uuid, initial, (), True)])
    visited = {(initial, True)}
    ambiguous_match = False
    rejected_engine_steps = Counter()
    expanded = 0
    attempted = 0
    while frontier:
        game, uid, snapshot, path, constructive = frontier.popleft()
        if snapshot == target:
            if constructive:
                return {"label": "reversible", "witness": list(path), "expanded": expanded,
                        "engine_errors": dict(rejected_engine_steps)}
            ambiguous_match = True
        if len(path) == 3 or game.gameOver:
            continue
        if expanded == limit:
            return {"label": "unknown", "reason": "state_budget", "expanded": expanded,
                    "engine_errors": dict(rejected_engine_steps)}
        expanded += 1
        for action, assignments in choices(game).items():
            verb = assignments[0][0]
            # Weight answers cannot recover an unanswered state: the only
            # writer stores a number and no action restores None. The gameOver
            # flag also has no true-to-false transition outside construction.
            if verb == "answer":
                continue
            if max_transitions is not None and attempted >= max_transitions:
                return {"label": "unknown", "reason": "transition_budget", "expanded": expanded,
                        "engine_errors": dict(rejected_engine_steps)}
            attempted += 1
            try:
                after, after_uid = execute(module, game, uid, action)
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                rejected_engine_steps[type(error).__name__] += 1
                continue
            after_snapshot = canonical(after, after_uid)
            if after_snapshot == snapshot:
                continue
            witness_ok = constructive and constructive_recovery(verb, assignments, after)
            key = after_snapshot, witness_ok
            if key in visited:
                continue
            visited.add(key)
            frontier.append((after, after_uid, after_snapshot, (*path, action), witness_ok))
    reason = ("uncertified_recovery_path" if ambiguous_match else
              "engine_error_in_search" if rejected_engine_steps else "exhausted_superset")
    return {"label": "irreversible" if reason == "exhausted_superset" else "unknown", "expanded": expanded,
            "reason": reason, "engine_errors": dict(rejected_engine_steps)}


def archived_rows(directory, names):
    path = directory / "test.zip"
    assert sha(path) == ARCHIVE_SHA
    selected = {name: [] for name in names}
    with zipfile.ZipFile(path) as archive, archive.open("test.jsonl") as stream:
        for line in stream:
            row = json.loads(line)
            if row["game"] in selected:
                selected[row["game"]].append(row)
    return selected


def prefixes(directory, name, module):
    play = directory / "playthroughs" / f"{name}-playthrough.txt"
    gold = [line[1:].strip() for line in play.read_text().splitlines() if line.startswith(">")]
    gold = [action for action in gold if action not in {"", "help"}]
    game = module.TextGame(randomSeed=module.randomSeed)
    out = {}
    last = ""
    for depth, action in enumerate(gold):
        view = serialize(visible(game, last, module.UUID))
        entries = out.setdefault(view, [])
        entries.append((depth, game, module.UUID, canonical(game, module.UUID)))
        game, _ = execute(module, game, module.UUID, action)
        last = action
    return out


def audit(directory, private_cases=None, names=INITIAL_GAMES):
    rows = archived_rows(directory, names)
    games = {}
    for name, records in rows.items():
        module = load_game(directory, name)
        histories = prefixes(directory, name, module)
        stats = Counter()
        by_verb = Counter()
        stats["archived_rows"] = len(records)
        receipts = []
        for row in records:
            histories_for_row = histories.get(serialize(row["current_state"]), [])
            if not histories_for_row or len({entry[3] for entry in histories_for_row}) != 1:
                stats["unmatched_or_aliased_prefix"] += 1
                continue
            depth, game, uuid, target = histories_for_row[0]
            module.UUID = uuid
            action = row["action_state"]["lastAction"]
            clone = deepcopy(game)
            parsed = clone.generatePossibleActions().get(action, ())
            clone.step_action(action)
            action_view = visible(clone, action, module.UUID)
            clone.step_tick()
            tick_view = visible(clone, action, module.UUID)
            clone.step_calculate_score()
            if (row["current_score_state"] != score(game) or row["action_state"] != action_view
                    or row["tick_state"] != tick_view or row["next_score_state"] != score(clone)):
                stats["transition_mismatch"] += 1
                continue
            stats["exact_archived_transition_replay"] += 1
            if len(parsed) != 1 or parsed[0][0] not in {"take", "put", "answer", "open", "close"} or game.gameOver:
                stats["out_of_scope"] += 1
                continue
            verb = parsed[0][0]
            changed = canonical(clone, module.UUID) != target
            if not changed:
                stats["unchanged"] += 1
                continue
            if clone.gameOver or verb == "answer":
                label = {"label": "irreversible", "reason": "monotonic_game_over", "expanded": 0}
                assert not game.gameOver and clone.gameOver
            else:
                assert all(getattr(game, field, None) is None for field in ("answer_weight", "answer_temperature"))
                label = recover(module, clone, module.UUID, target,
                                max_transitions=None if name in INITIAL_GAMES else 8000)
            stats[label["label"]] += 1
            by_verb[(verb, label["label"])] += 1
            if "reason" in label:
                stats[f"reason_{label['reason']}"] += 1
            if "witness" in label:
                stats[f"reversible_witness_length_{len(label['witness'])}"] += 1
            stats["recovery_branch_engine_errors"] += sum(label.get("engine_errors", {}).values())
            # Store only aggregate outcomes and reproducibility hashes in the public receipt.
            receipts.append({"state_id": row["state_id"], "depth": depth, **label})
            if private_cases is not None and label["label"] != "unknown":
                assignment = [getattr(value, "uuid", value) for value in parsed[0]]
                case = {"id": f"{name}:{row['state_id']}", "game": name, "prefix_depth": depth,
                        "verb": verb, "action": action, "label": label["label"], "certificate": label,
                        "causal_state_id": hashlib.sha256(target.encode()).hexdigest(), "semantic_action": assignment,
                        "current_state": row["current_state"], "next_state": row["tick_state"],
                        "current_score": row["current_score_state"], "next_score": row["next_score_state"]}
                private_cases.write(serialize(case) + "\n")
        digest = hashlib.sha256(serialize(receipts).encode()).hexdigest()
        games[name] = {"counts": dict(sorted(stats.items())), "audit_record_sha256": digest,
                       "by_verb_and_label": {f"{verb}:{label}": count for (verb, label), count in sorted(by_verb.items())}}
    return {"scope": "private_executable_archived_transition_diagnostic",
            "upstream_revision": REVISION, "archive_sha256": ARCHIVE_SHA,
            "game_source_sha256": {name: GAME_SHA[name] for name in names},
            "horizon": 3, "games": games, "external_model_calls": 0, "inference_run": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Pinned upstream data directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", nargs="+", choices=GAME_SHA, default=INITIAL_GAMES)
    parser.add_argument("--private-cases", type=Path, help="Private research JSONL, outside published artifacts")
    args = parser.parse_args()
    with args.output.open("x") as output:
        if args.private_cases:
            with args.private_cases.open("x") as cases:
                report = audit(args.data_root, cases, args.games)
            report["private_cases_sha256"] = sha(args.private_cases)
        else:
            report = audit(args.data_root, names=args.games)
        output.write(json.dumps(report, indent=2) + "\n")
