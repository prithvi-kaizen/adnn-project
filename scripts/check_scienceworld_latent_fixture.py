"""Construct an engine-class counterexample, not a naturally sampled data row."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scienceworld_adapter import ScienceWorldReplay  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Preserve prior evidence')
    env = ScienceWorldReplay('boil', 0)
    report = {'scope': 'independently_constructed_ScienceWorld_engine_fixture_not_dataset',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'complete_state_certified': False, 'eligible_labels': 0, 'arms': []}
    try:
        env.replay(())
        jvm = env.env._gateway.jvm
        generator = jvm.util.UniqueIdentifier
        base = generator.count()
        for ticks in (0, 9):
            getattr(generator, 'count_$eq')(base)
            container = jvm.scienceworld.struct.EnvObject()
            plant = jvm.scienceworld.objects.livingthing.plant.Plant()
            water = jvm.scienceworld.objects.substance.Water()
            soil = jvm.scienceworld.objects.livingthing.plant.Soil()
            for obj in (plant, water, soil):
                container.addObject(obj)
            lifecycle = plant.lifecycle().get()
            stage = lifecycle.getCurStage()
            getattr(stage, 'ticksMeetingCriteria_$eq')(ticks)
            before = str(container.toJSON())
            plant.tick()
            after = str(container.toJSON())
            report['arms'].append({'ticksMeetingCriteria': ticks, 'before': json.loads(before),
                'after': json.loads(after), 'stage_after': str(lifecycle.getCurStageName()),
                'water_deleted_after': water.isDeleted()})
        report['same_export_before_tick'] = report['arms'][0]['before'] == report['arms'][1]['before']
        report['different_physical_export_after_tick'] = report['arms'][0]['after'] != report['arms'][1]['after']
        report['known_outcome_checks_pass'] = (report['same_export_before_tick'] and
            report['different_physical_export_after_tick'] and
            not report['arms'][0]['water_deleted_after'] and report['arms'][1]['water_deleted_after'])
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'arms'}, indent=2))
    if not report.get('known_outcome_checks_pass'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
