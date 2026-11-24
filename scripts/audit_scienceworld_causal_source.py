"""Fingerprint a named causal dependency audit, without claiming closure."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scienceworld_adapter.causal import MATERIAL_FIELDS, PROPERTY_FIELDS, SOURCE_REVISION  # noqa: E402

FILES = (
    'scienceworld/struct/EnvObject.scala',
    'scienceworld/objects/portal/Portal.scala',
    'scienceworld/actions/ActionOpenCloseDoor.scala',
    'scienceworld/actions/ActionMoveToLocation.scala',
    'scienceworld/actions/ActionMoveObject.scala',
    'scienceworld/actions/ActionMix.scala',
    'scienceworld/actions/ActionWait.scala',
    'scienceworld/runtime/AgentInterface.scala',
    'scienceworld/runtime/pythonapi/PythonInterface.scala',
    'scienceworld/input/ActionDefinitions.scala',
    'scienceworld/input/ActionHandler.scala',
    'scienceworld/input/InputParser.scala',
    'scienceworld/objects/agent/Agent.scala',
    'scienceworld/objects/containers/Container.scala',
    'scienceworld/objects/devices/Device.scala',
    'scienceworld/objects/devices/HeatSource.scala',
    'scienceworld/objects/devices/CoolingSource.scala',
    'scienceworld/objects/substance/Substance.scala',
    'scienceworld/properties/MaterialProperties.scala',
    'scienceworld/properties/ContainerProperties.scala',
    'scienceworld/properties/MoveableProperties.scala',
    'scienceworld/properties/DeviceProperties.scala',
    'scienceworld/properties/PortalProperties.scala',
    'scienceworld/properties/HeatSourceProperties.scala',
    'scienceworld/properties/CoolingSourceProperties.scala',
    'scienceworld/properties/ElectricalConnectionProperties.scala',
    'scienceworld/processes/HeatTransfer.scala',
    'scienceworld/processes/StateOfMatter.scala',
    'scienceworld/processes/Combustion.scala',
    'scienceworld/processes/Chemistry.scala',
    'scienceworld/processes/lifestage/LifeStage.scala',
    'scienceworld/processes/lifestage/PlantLifeStage.scala',
    'scienceworld/tasks/goals/Goal.scala',
    'util/UniqueIdentifier.scala',
)


def audit(source_root: Path) -> dict:
    baseline = json.loads((ROOT / 'docs/evidence/scienceworld_certification_source_inventory.json').read_text())
    for name, expected in baseline['files'].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Cached upstream source differs from pinned inventory: ' + name)
    records = {}
    for name in FILES:
        path = source_root / name
        content = path.read_text()
        lines = content.splitlines()
        records[name] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'line_count': len(lines),
            'primary_source': f'https://github.com/allenai/ScienceWorld/blob/{SOURCE_REVISION}/simulator/src/main/scala/{name}',
            # Inventory only; local variables appear here too. No automated causal proof.
            'review_locations': [{'line': i, 'text': line.strip()} for i, line in enumerate(lines, 1)
                                 if re.search(r'\bvar\b|mutable\.|override def tick|def (runAction|isValidAction|generatePossibleValidActions|tick|step)', line)],
        }
    material = (source_root / 'scienceworld/properties/MaterialProperties.scala').read_text().split('def toJSON', 1)[0]
    declared_material = set(re.findall(r'\bvar\s+(\w+)\s*:', material))
    implemented_material = set(MATERIAL_FIELDS) | {'nameInStateOfMatter'}
    if declared_material != implemented_material:
        raise ValueError('Named material reader no longer matches audited declaration')
    properties = {}
    for prop, fields in PROPERTY_FIELDS.items():
        if prop == 'propMaterial':
            continue
        filename = {'propContainer': 'Container', 'propMoveable': 'Moveable', 'propDevice': 'Device',
                    'propHeatSource': 'HeatSource', 'propCoolingSource': 'CoolingSource', 'propPortal': 'Portal'}[prop]
        name = f'scienceworld/properties/{filename}Properties.scala'
        text = (source_root / name).read_text().split('def toJSON', 1)[0]
        declared = set(re.findall(r'\bvar\s+(\w+)\s*:', text))
        expected = set(fields)
        if prop in ('propHeatSource', 'propCoolingSource'):
            expected.add('curSetTemp')
        if prop == 'propPortal':
            expected = (expected - {'isLocked', 'isLockable'}) | {'connectsFrom', 'connectsTo'}
        if prop == 'propMoveable':
            # isMovable is an immutable constructor val; replacing propMoveable
            # can still change it, so the value remains in the named reader.
            expected = set()
        if declared != expected:
            raise ValueError('Named property reader differs from declared mutable fields: ' + prop)
        properties[prop] = sorted(declared)
    definitions = (source_root / 'scienceworld/input/ActionDefinitions.scala').read_text()
    registered = set(re.findall(r'(Action\w+)\.registerAction', definitions))
    enumerated = set(re.findall(r'(Action\w+)\.generatePossibleValidActions', definitions))
    wrapper = Path(importlib.util.find_spec('scienceworld.scienceworld').origin)
    wrapper_lines = wrapper.read_text().splitlines()
    return {
        'scope': 'named_source_field_and_handler_inventory_not_completeness_proof',
        'source_revision': SOURCE_REVISION, 'baseline_source_hashes_match': True,
        'files': records, 'material_declared_fields': sorted(declared_material),
        'python_wrapper': {'version': importlib.metadata.version('scienceworld'),
            'relative_package_file': 'scienceworld/scienceworld.py',
            'sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            'review_locations': [{'line': i, 'text': line.strip()} for i, line in enumerate(wrapper_lines, 1)
                                 if 410 <= i <= 451]},
        'material_direct_reader_covers_all_base_fields': True, 'property_mutable_fields': properties,
        'registered_handlers': sorted(registered), 'enumerated_handlers': sorted(enumerated),
        'registered_without_generator': sorted(registered - enumerated),
        'generator_without_registration': sorted(enumerated - registered),
        'handler_coverage_equal': registered == enumerated,
        'full_action_grammar_certified': False, 'causal_closure_certified': False,
        'reasons': ['handler-name parity does not prove parser/argument/precondition coverage',
                    'local field coverage does not establish transitive thermal/electrical/biological closure',
                    'scope excludes task/parser/history/RNG/allocator state without an approved independence proof'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Preserve prior evidence')
    report = audit(args.source_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('material_direct_reader_covers_all_base_fields',
        'handler_coverage_equal', 'full_action_grammar_certified', 'causal_closure_certified')}, indent=2))


if __name__ == '__main__':
    main()
