import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires pinned native research runtime')
class NativeHelperTests(unittest.TestCase):
    def test_tensor_and_tuple_hooks_change_only_last_position_and_clean_up(self):
        import torch

        from src.models.native_smoke import steer_last_position
        class TupleBlock(torch.nn.Module):
            def forward(self, x):
                return x, 'retained_auxiliary'
        x = torch.zeros(2, 3, 4)
        for block in (torch.nn.Identity(), TupleBlock()):
            with steer_last_position(block, torch.ones(4), 2):
                output = block(x)
            hidden = output[0] if isinstance(output, tuple) else output
            self.assertTrue(torch.equal(hidden[:, :-1], x[:, :-1]))
            self.assertTrue(torch.equal(hidden[:, -1], torch.full((2, 4), 2.)))
            self.assertTrue(torch.equal(x, torch.zeros_like(x)))
            self.assertEqual(len(block._forward_hooks), 0)
            if isinstance(output, tuple):
                self.assertEqual(output[1], 'retained_auxiliary')

    def test_tiny_native_two_family_checks_float32_and_bfloat16(self):
        from scripts.check_native_smoke import run
        for precision in ('float32', 'bfloat16'):
            records = run('cpu', precision)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r['zero_strength_greedy_ids_identical'] for r in records))
            self.assertTrue(all(r['raw_final_block_differs_from_normalized_return'] for r in records))

    def test_nonfinite_direction_and_bad_hook_shape_fail(self):
        import torch

        from src.models.native_smoke import steer_last_position
        with self.assertRaises(ValueError):
            with steer_last_position(torch.nn.Identity(), torch.tensor([float('nan')]), 1):
                pass
        block = torch.nn.Identity()
        with self.assertRaises(ValueError):
            with steer_last_position(block, torch.ones(4), 1):
                block(torch.zeros(2, 3, 5))
        self.assertEqual(len(block._forward_hooks), 0)
