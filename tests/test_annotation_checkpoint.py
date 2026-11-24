"""All judgments here are synthetic unit fixtures, never human evidence."""

import csv
import unittest
from io import StringIO
from pathlib import Path

from src.data.annotation import LABELS, agreement_report, completed_labels, csv_text, validate_blank_package


class AnnotationCheckpointTests(unittest.TestCase):
    def test_existing_preview_is_blank_blinded_and_independently_ordered(self):
        root = Path(__file__).resolve().parents[1] / 'data/annotation/preview_v5'
        package = {p.name: p.read_text() for p in root.glob('*.csv')}
        self.assertEqual(validate_blank_package(package)['n'], 100)
        package['annotator_B.csv'] = package['annotator_A.csv']
        with self.assertRaises(ValueError):
            validate_blank_package(package)

    def test_kappa_confusion_disagreement_and_cluster_bootstrap(self):
        # Nine independent synthetic clusters, each with all three classes.
        a = {f'{g}-{i}': label for g in range(9) for i, label in enumerate(LABELS)}
        b = dict(a)
        b['0-0'] = 'irreversible'
        auto = dict(a)
        auto['1-1'] = 'reversible'
        groups = {k: k.split('-')[0] for k in a}
        report = agreement_report(a, b, auto, groups)
        self.assertEqual(report['n'], 27)
        self.assertEqual(report['cluster_count'], 9)
        self.assertAlmostEqual(report['human_human']['cohens_kappa'], 17 / 18)
        self.assertEqual(report, agreement_report(a, b, auto, groups))
        self.assertEqual(len(report['disagreements']), 2)
        self.assertEqual(sum(map(sum, report['human_human']['confusion_rows_left_columns_right'])), 27)
        self.assertEqual(report['human_human']['bootstrap_valid'], 1000)
        self.assertIsNotNone(report['human_human']['cluster_bootstrap_95_ci'])

    def test_undefined_kappa_and_incomplete_alignment_are_explicit(self):
        same = {'fixture': 'reversible'}
        report = agreement_report(same, same, same, {'fixture': 'cluster'})
        self.assertIsNone(report['human_human']['cohens_kappa'])
        self.assertIsNone(report['human_human']['cluster_bootstrap_95_ci'])
        self.assertEqual(report['human_human']['bootstrap_undefined'], 1000)
        with self.assertRaises(ValueError):
            agreement_report(same, {}, same, {'fixture': 'cluster'})

    def test_completed_form_requires_valid_confidence_and_reason(self):
        fields = ['annotation_id', 'label', 'confidence', 'reason']
        rows = [{'annotation_id': 'fixture', 'label': 'reversible',
                 'confidence': 'high', 'reason': 'Synthetic fixture rationale.'}]
        self.assertEqual(completed_labels(csv_text(rows, fields), {'fixture'}), {'fixture': 'reversible'})
        for change in ({'reason': ''}, {'confidence': ''}, {'label': 'unknown'}, {'annotation_id': 'extra'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                completed_labels(csv_text([{**rows[0], **change}], fields), {'fixture'})
        with self.assertRaises(ValueError):
            completed_labels(csv_text(rows + rows, fields), {'fixture'})
        # Parsing tests do not create any public completed annotation form.
        self.assertEqual(len(list(csv.DictReader(StringIO(csv_text(rows, fields))))), 1)

    def test_preview_cannot_be_imported_as_final_human_evidence(self):
        from unittest.mock import patch

        from scripts.import_annotations import main
        root = Path(__file__).resolve().parents[1]
        with patch('sys.argv', ['import_annotations.py', str(root / 'data/annotation/preview_v5'),
                  '--dataset', str(root / 'data/candidates/v5/sandbox_pairs.jsonl'),
                  '--annotator-a', '/unread_fixture_A', '--annotator-b', '/unread_fixture_B',
                  '--human-attestations', '/unread_fixture_attestations', '--output', '/unwritten_fixture']):
            with self.assertRaisesRegex(SystemExit, 'Only the frozen final'):
                main()
