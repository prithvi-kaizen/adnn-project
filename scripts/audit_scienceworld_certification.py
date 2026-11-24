"""Falsify incomplete-state equality and check bounded replay, without labels."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scienceworld_adapter import ScienceWorldReplay  # noqa: E402


def objects(env):
    universe = env.env.server.agentInterface().get().universe()
    iterator = universe.getContainedObjectsAndPortalsRecursive(True, True).iterator()
    out = []
    while iterator.hasNext():
        out.append(iterator.next())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('Preserve every attempt; choose a new directory')
    args.output_dir.mkdir(parents=True)
    from scienceworld.constants import JAR_PATH
    report = {'scope': 'state_certification_diagnostic_not_eligible_labels', 'complete': False,
              'python': platform.python_version(), 'scienceworld': importlib.metadata.version('scienceworld'),
              'jar_sha256': hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'replay_checks': [], 'counterexamples': [], 'eligible_physical_labels': 0}
    path = args.output_dir / 'report.json'
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    save()
    for task, variation in [('boil', 0), ('chemistry-mix', 0), ('freeze', 1)]:
        env = ScienceWorldReplay(task, variation)
        try:
            env.replay(())
            opening = next(a for a in env.legal_actions() if a.startswith('open door to '))
            closing = opening.replace('open ', 'close ', 1)
            for prefix in ((), ('wait1',), ('wait1',) * 8, (opening, closing) + ('wait1',) * 4):
                samples = []
                for _ in range(3):
                    state = env.replay(prefix)
                    before_hash = state.digest()
                    future = env.replay(prefix + ('wait1',))
                    samples.append({'prefix_hash': before_hash, 'next_hash': future.digest(),
                                    'next_engine_iteration': env.env.server.agentInterface().get().getCurIterations()})
                report['replay_checks'].append({'task': task, 'variation': variation, 'prefix': prefix,
                     'repetitions': samples, 'equal_exported_prefixes': len({s['prefix_hash'] for s in samples}) == 1,
                     'equal_exported_continuations': len({s['next_hash'] for s in samples}) == 1})
                save()
            if task != 'boil':
                continue
            # Researcher intervention inside this disposable JVM, not an agent action or data row.
            env.replay(())
            agent_room = env.env.server.agentInterface().get().agent().getContainer().get()
            portal = next(p for p in objects(env) if str(p.name()) == 'door' and
                          str(p.getCanonicalReferent(agent_room)) == opening.removeprefix('open '))
            prior = env.snapshot()
            prop = portal.propPortal().get()
            original_open = prop.isOpen()
            getattr(prop, 'isOpen_$eq')(not original_open)
            changed = env.snapshot()
            report['counterexamples'].append({'name': 'portal_open_flag_researcher_mutation',
                  'portal_uuid': portal.uuid(), 'flag_before': original_open, 'flag_after': prop.isOpen(),
                  'object_tree_equal': json.loads(prior.payload)['object_tree'] == json.loads(changed.payload)['object_tree'],
                  'entire_adapter_payload_equal': prior.payload == changed.payload,
                  'passable_after_mutation': portal.isCurrentlyPassable(),
                  'scope': 'isolated_JVM_intervention_not_legal_transition_or_dataset_label'})
            save()
            arms = []
            for shift in (0, 1000):
                initial = env.replay(())
                generator = env.env._gateway.jvm.util.UniqueIdentifier
                before_count = generator.count()
                getattr(generator, 'count_$eq')(before_count + shift)
                after_mutation = env.snapshot()
                env.env.step('wait1')
                next_state = env.snapshot()
                arms.append({'shift': shift, 'counter_before': before_count,
                             'counter_after_mutation': before_count + shift,
                             'initial_hash': initial.digest(), 'after_mutation_hash': after_mutation.digest(),
                             'next_hash': next_state.digest()})
                (args.output_dir / f'identifier-shift-{shift}-next.json').write_text(next_state.payload + '\n')
            report['counterexamples'].append({'name': 'global_identifier_allocator', 'arms': arms,
                  'same_export_before_same_action': arms[0]['after_mutation_hash'] == arms[1]['after_mutation_hash'],
                  'different_export_after_same_action': arms[0]['next_hash'] != arms[1]['next_hash'],
                  'scope': 'isolated_JVM_intervention_not_legal_transition_or_dataset_label'})
            save()
        except Exception as error:
            report.setdefault('failures', []).append({'task': task, 'type': type(error).__name__, 'error': str(error)})
            save()
        finally:
            env.close()
    save()
    print(json.dumps({k: v for k, v in report.items() if k != 'replay_checks'}, indent=2))


if __name__ == '__main__':
    main()
