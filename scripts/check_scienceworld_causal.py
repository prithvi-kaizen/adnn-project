"""Bounded real-engine adversarial diagnostics of an unapproved state proposal."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scienceworld_adapter.causal import (  # noqa: E402
    SOURCE_REVISION,
    ProposedDoorReplay,
    canonical,
    physical_record,
    property_record,
    proposed_state,
    search_proposed_doors,
    seed_counter_diagnostic,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('Preserve all attempts; choose a new directory')
    args.output_dir.mkdir(parents=True)
    started = monotonic()
    from scienceworld.constants import JAR_PATH
    report = {
        'scope': 'unapproved_causal_state_diagnostics_not_research_labels',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': SOURCE_REVISION, 'jar_sha256': sha(Path(JAR_PATH)),
        'python': platform.python_version(), 'platform': platform.platform(),
        'machine': platform.machine(), 'logical_cpus': os.cpu_count(),
        'scienceworld': importlib.metadata.version('scienceworld'),
        'py4j': importlib.metadata.version('py4j'),
        'java': subprocess.run(['java', '-version'], capture_output=True, text=True).stderr,
        'seeds': None, 'seed_note': 'No random fixtures; task variation/reset supplies engine initialization',
        'source_sha256': {p: sha(ROOT / p) for p in (
            'scripts/check_scienceworld_causal.py', 'src/scienceworld_adapter/causal.py',
            'src/scienceworld_adapter/replay.py', 'docs/SCIENCEWORLD_CAUSAL_DIAGNOSTIC_PLAN.md')},
        'fixtures': [], 'replays': [], 'door_inverses': [], 'searches': [], 'enumeration_checks': [],
        'failures': [], 'eligible_physical_labels': 0, 'amendment_applied': False,
    }
    assert report['jar_sha256'] == 'e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465'
    path = args.output_dir / 'report.json'

    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')

    def fixture(name, builder):
        try:
            row = builder()
            row.update(name=name, scope='independent_engine_class_fixture_not_sampled_task_or_label')
            report['fixtures'].append(row)
            if not row['expected_checks_pass']:
                report['failures'].append({'fixture': name, 'error': 'Expected check failed'})
        except Exception as error:
            report['failures'].append({'fixture': name, 'error_type': type(error).__name__,
                                      'error': str(error), 'traceback': traceback.format_exc()})
        save()

    save()
    env = ProposedDoorReplay('boil', 0)
    try:
        env.replay(())
        jvm = env.env._gateway.jvm
        allocator = jvm.util.UniqueIdentifier
        base = allocator.count()

        def plant():
            arms = []
            for ticks in (0, 9):
                getattr(allocator, 'count_$eq')(base)
                box = jvm.scienceworld.struct.EnvObject()
                p = jvm.scienceworld.objects.livingthing.plant.Plant()
                w = jvm.scienceworld.objects.substance.Water()
                soil = jvm.scienceworld.objects.livingthing.plant.Soil()
                for obj in (p, w, soil):
                    box.addObject(obj)
                stage = p.lifecycle().get().getCurStage()
                getattr(stage, 'ticksMeetingCriteria_$eq')(ticks)
                before = json.loads(box.toJSON())
                diagnostic = seed_counter_diagnostic(p)
                record = physical_record(p)
                rejected = proposed_state([record], family='plant_rejection').for_search()
                p.tick()
                try:
                    seed_counter_diagnostic(p)
                    post_seed_reader_rejects = False
                except ValueError:
                    post_seed_reader_rejects = True
                arms.append({'ticks': ticks, 'before': before, 'seed_diagnostic': diagnostic,
                             'unsupported': record['unsupported'], 'state_complete': rejected.complete,
                             'water_deleted': bool(w.isDeleted()), 'after': json.loads(box.toJSON()),
                             'post_seed_reader_rejects': post_seed_reader_rejects})
            return {'arms': arms, 'expected_checks_pass':
                    arms[0]['before'] == arms[1]['before'] and
                    arms[0]['seed_diagnostic'] != arms[1]['seed_diagnostic'] and
                    not arms[0]['water_deleted'] and arms[1]['water_deleted'] and
                    all(a['unsupported'] and not a['state_complete'] for a in arms) and
                    arms[1]['post_seed_reader_rejects']}

        fixture('plant_counter_alias_rejected', plant)

        def omitted_material(field):
            arms = []
            for altered in (False, True):
                getattr(allocator, 'count_$eq')(base)
                w = jvm.scienceworld.objects.substance.Water()
                prop = w.propMaterial().get()
                getattr(prop, 'temperatureC_$eq')(150.0 if field == 'combustionPoint' else -5.0)
                if altered:
                    if field == 'combustionPoint':
                        getattr(prop, 'combustionPoint_$eq')(100.0)
                    else:
                        getattr(prop, 'nameInStateOfMatter_$eq')(
                            prop.nameInStateOfMatter().updated('solid', 'alternate ice'))
                before = json.loads(w.toJSON())
                direct = property_record(w)
                if field == 'combustionPoint':
                    jvm.scienceworld.processes.Combustion.combustionTick(w)
                else:
                    jvm.scienceworld.processes.StateOfMatter.ChangeOfState(w, False)
                arms.append({'altered': altered, 'before': before, 'direct': direct,
                             'after': json.loads(w.toJSON())})
            return {'arms': arms, 'expected_checks_pass': arms[0]['before'] == arms[1]['before'] and
                    canonical(arms[0]['direct']) != canonical(arms[1]['direct']) and
                    arms[0]['after'] != arms[1]['after']}

        fixture('omitted_combustion_point_detected', lambda: omitted_material('combustionPoint'))
        fixture('omitted_phase_name_map_detected', lambda: omitted_material('nameInStateOfMatter'))

        def thermal_context():
            arms = []
            for cup_temp in (20.0, 80.0):
                getattr(allocator, 'count_$eq')(base)
                cup = jvm.scienceworld.objects.containers.GlassCup()
                w = jvm.scienceworld.objects.substance.Water()
                cup.addObject(w)
                getattr(w.propMaterial().get(), 'temperatureC_$eq')(20.0)
                getattr(cup.propMaterial().get(), 'temperatureC_$eq')(cup_temp)
                water_only = physical_record(w)
                expanded = [physical_record(w), physical_record(cup)]
                cup.clearTickProcessedRecursive(0)
                cup.tick()
                arms.append({'cup_temperature_before': cup_temp, 'water_only': water_only,
                             'with_container': expanded, 'water_temperature_after': w.propMaterial().get().temperatureC()})
            return {'arms': arms, 'expected_checks_pass': canonical(arms[0]['water_only']) ==
                    canonical(arms[1]['water_only']) and canonical(arms[0]['with_container']) !=
                    canonical(arms[1]['with_container']) and
                    arms[0]['water_temperature_after'] != arms[1]['water_temperature_after']}

        fixture('water_only_thermal_scope_fails', thermal_context)

        def external_thermal_context():
            arms = []
            for ambient in (20.0, 80.0):
                getattr(allocator, 'count_$eq')(base)
                parent = jvm.scienceworld.objects.containers.GlassCup()
                cup = jvm.scienceworld.objects.containers.GlassCup()
                w = jvm.scienceworld.objects.substance.Water()
                parent.addObject(cup)
                cup.addObject(w)
                for obj in (cup, w):
                    getattr(obj.propMaterial().get(), 'temperatureC_$eq')(20.0)
                getattr(parent.propMaterial().get(), 'temperatureC_$eq')(ambient)
                projection = [physical_record(cup), physical_record(w)]
                parent.clearTickProcessedRecursive(0)
                parent.tick()
                arms.append({'ambient_before': ambient, 'projection_before': projection,
                             'water_temperature_after': w.propMaterial().get().temperatureC()})
            return {'arms': arms, 'expected_checks_pass': canonical(arms[0]['projection_before']) ==
                    canonical(arms[1]['projection_before']) and
                    arms[0]['water_temperature_after'] != arms[1]['water_temperature_after']}

        fixture('water_plus_cup_external_thermal_alias', external_thermal_context)

        def drain_context():
            arms = []
            for opened in (False, True):
                getattr(allocator, 'count_$eq')(base)
                source = jvm.scienceworld.struct.EnvObject()
                dest = jvm.scienceworld.struct.EnvObject()
                w = jvm.scienceworld.objects.substance.Water()
                source.addObject(w)
                drain = jvm.scienceworld.objects.portal.LiquidDrain(opened, source, dest)
                source.addPortal(drain)
                before = json.loads(source.toJSON())
                direct = physical_record(drain)
                source.clearTickProcessedRecursive(0)
                source.tick()
                arms.append({'opened': opened, 'before': before, 'drain_record': direct,
                             'water_parent_after': int(w.getContainer().get().uuid())})
            return {'arms': arms, 'expected_checks_pass': arms[0]['before'] == arms[1]['before'] and
                    canonical(arms[0]['drain_record']) != canonical(arms[1]['drain_record']) and
                    arms[0]['water_parent_after'] != arms[1]['water_parent_after'] and
                    all(a['drain_record']['unsupported'] for a in arms)}

        fixture('unexported_drain_context_rejected', drain_context)
    finally:
        env.close()

    for task, variation in [('boil', 0), ('freeze', 1), ('chemistry-mix', 0)]:
        env = ProposedDoorReplay(task, variation)
        try:
            env.replay(())
            initial_strict = env.env.getObjectTree()
            opening = next(a for a in env.legal_actions() if a.startswith('open door to '))
            closing = opening.replace('open ', 'close ', 1)
            for prefix in ((), (opening,), (opening, closing)):
                samples = []
                for _ in range(3):
                    state = env.replay(prefix)
                    strict = env.env.getObjectTree()
                    future = env.replay(prefix + ('wait1',))
                    samples.append({'target_hash': state.digest(), 'strict_tree_hash': hashlib.sha256(
                        canonical(strict).encode()).hexdigest(), 'future_target_hash': future.digest(),
                        'future_strict_tree_hash': hashlib.sha256(canonical(env.env.getObjectTree()).encode()).hexdigest(),
                        'future_ticks': int(env.env.server.agentInterface().get().getCurIterations())})
                report['replays'].append({'task': task, 'variation': variation, 'prefix': prefix,
                    'samples': samples, 'all_target_and_export_repetitions_equal': all(
                        sample == samples[0] for sample in samples)})
                save()
            initial = env.replay(())
            opened = env.replay((opening,))
            closed = env.replay((opening, closing))
            final_tree = env.env.getObjectTree()
            report['door_inverses'].append({'task': task, 'variation': variation,
                'actions': [opening, closing], 'initial_target_hash': initial.digest(),
                'opened_target_hash': opened.digest(), 'closed_target_hash': closed.digest(),
                'target_inverse_matches': initial.payload == closed.payload,
                'candidate_changes_target': initial.payload != opened.payload,
                'strict_tree_inverse_matches': initial_strict == final_tree,
                'label': 'unknown', 'dataset_eligible': False})
            for horizon in (1, 2, 3):
                result = search_proposed_doors(env, (), opening, horizon,
                    max_nodes=32, max_seconds=5, max_frontier=10000)
                result.update(task=task, variation=variation, action=opening)
                report['searches'].append(result)
                save()
            env.replay(())
            registered = env.env.get_possible_actions_with_IDs()
            room = env.env.server.agentInterface().get().agent().getContainer().get()
            if str(room.name()) != 'kitchen':
                if 'open door to kitchen' in env.legal_actions():
                    env.replay(('open door to kitchen', 'go to kitchen'))
                else:
                    report['enumeration_checks'].append({'task': task, 'registered': registered,
                        'self_pour_trial': None, 'reason': 'kitchen_not_directly_accessible_from_this_prefix',
                        'full_grammar_certified': False})
                    save()
                    continue
            templates = env.env.get_valid_action_object_combinations_with_templates()
            self_pours = [t for t in templates if t['action'].startswith('pour ') and
                          len(t['obj_ids']) == 2 and t['obj_ids'][0] == t['obj_ids'][1]]
            check = {'task': task, 'registered': registered, 'kitchen_templates': templates,
                     'full_grammar_certified': False, 'self_pour_candidates': self_pours,
                     'agent_room': str(env.env.server.agentInterface().get().agent().getContainer().get().name())}
            check['self_pour_trials'] = []
            choices = [t for t in self_pours if t['action'] in ('pour sewer into sewer', 'pour metal pot into metal pot')]
            for choice in choices:
                choice = choice['action']
                # Each probe starts from the same original kitchen prefix.
                env.replay(('open door to kitchen', 'go to kitchen'))
                interface = env.env.server.agentInterface().get()
                ticks = int(interface.getCurIterations())
                history = int(interface.agent().getActionHistorySize())
                obs, score, done, info = env.env.step(choice)
                history_classes = [str(a.getClass().getName()) for a in
                                   interface.agent().getActionHistorySince(history)]
                check['self_pour_trials'].append({'action': choice, 'observation': obs,
                    'ticks_before': ticks, 'ticks_after': int(interface.getCurIterations()),
                    'history_before': history, 'history_after': int(interface.agent().getActionHistorySize()),
                    'history_classes': history_classes,
                    'parser_ambiguous': bool(interface.inputParser().isInAmbiguousState()),
                    'error_state': bool(interface.isInErrorState())})
            report['enumeration_checks'].append(check)
            save()
        except Exception as error:
            report['failures'].append({'task': task, 'error_type': type(error).__name__, 'error': str(error),
                                      'traceback': traceback.format_exc()})
            save()
        finally:
            env.close()
    report['diagnostic_checks_pass'] = (not report['failures'] and len(report['fixtures']) == 6 and
        len(report['searches']) == 9 and all(r['all_target_and_export_repetitions_equal'] for r in report['replays']) and
        all(r['label'] == 'unknown' and r['duplicate_states_skipped'] == 0 for r in report['searches']))
    report['elapsed_seconds'] = monotonic() - started
    save()
    print(json.dumps({k: report[k] for k in ('diagnostic_checks_pass', 'failures', 'eligible_physical_labels')}, indent=2))
    if not report['diagnostic_checks_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
