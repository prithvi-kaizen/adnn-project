import unittest

from src.models.alignment import action_positions


class ActionAlignmentTests(unittest.TestCase):
    def test_action_overlap_excludes_special_tokens_padding_and_suffix(self):
        self.assertEqual(action_positions([(0, 0), (0, 3), (3, 5), (5, 8), (8, 10), (0, 0)],
                                          [1, 1, 1, 1, 1, 0], 4, 8), [2, 3])

    def test_unicode_and_boundary_overlap(self):
        text = '前缀 {"x":"é"} 后缀'
        start, end = text.index('{'), text.index('}') + 1
        positions = action_positions([(i, i + 1) for i in range(len(text))], [1] * len(text), start, end)
        self.assertEqual(''.join(text[i] for i in positions), '{"x":"é"}')

    def test_truncation_or_empty_span_fails_instead_of_probing_last_delimiter(self):
        for offsets, attention, start, end in [([(0, 2)], [1], 1, 5), ([(0, 0)], [1], 1, 3),
                                                ([(0, 3)], [0], 1, 3), ([(0, 3)], [1], 3, 3)]:
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                action_positions(offsets, attention, start, end)
