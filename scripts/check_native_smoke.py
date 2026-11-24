"""Tiny random Qwen/Llama smoke test; never from_pretrained or download."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.native_smoke import pool_action_vectors, steer_last_position  # noqa: E402


def run(device='cpu', precision='float32'):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    if device == 'cuda' and (not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('dgx-login')):
        raise RuntimeError('CUDA diagnostics require an allocated compute node')
    torch.set_num_threads(1)
    dtype = getattr(torch, precision)
    records = []
    for config_cls, model_cls in ((Qwen2Config, Qwen2ForCausalLM), (LlamaConfig, LlamaForCausalLM)):
        torch.manual_seed(1729)
        config = config_cls(vocab_size=128, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=256, pad_token_id=0, eos_token_id=127)
        config._attn_implementation = 'eager'
        model = model_cls(config).to(device=device, dtype=dtype).eval()
        ids = torch.tensor([[0, 0, 1, 5, 6, 7], [1, 2, 3, 4, 5, 6]], device=device)
        mask = (ids != 0).long()
        position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
        args = {'input_ids': ids, 'attention_mask': mask, 'position_ids': position_ids,
                'output_hidden_states': True, 'use_cache': False}
        captured = []
        handle = model.model.layers[-1].register_forward_hook(
            lambda module, inputs, output: captured.append(output[0].detach().clone()))
        try:
            with torch.inference_mode():
                baseline = model(**args)
        finally:
            handle.remove()
        with torch.inference_mode():
            assert len(baseline.hidden_states) == 3
            assert torch.equal(model.model.norm(captured[0]), baseline.hidden_states[-1])
            raw_differs = not torch.equal(captured[0], baseline.hidden_states[-1])
            assert raw_differs
            last, mean = pool_action_vectors(baseline.hidden_states, [[3, 4], [3, 4]])
            assert last.shape == mean.shape == (3, 2, 64)
            unpadded = model(ids[:1, 2:], attention_mask=mask[:1, 2:],
                             position_ids=position_ids[:1, 2:], output_hidden_states=True, use_cache=False)
            tolerance = .03 if precision == 'bfloat16' else 1e-5
            for padded_h, plain_h in zip(baseline.hidden_states, unpadded.hidden_states):
                assert torch.allclose(padded_h[:1, 2:], plain_h, atol=tolerance, rtol=tolerance)
            direction = torch.randn(64, device=device, dtype=dtype)
            block = model.model.layers[-1]
            hooks_before = len(block._forward_hooks)
            with steer_last_position(block, direction, 0):
                zero = model(**args)
            assert torch.equal(baseline.logits, zero.logits)
            with steer_last_position(block, direction, 1):
                perturbed = model(**args)
            assert not torch.equal(baseline.logits[:, -1], perturbed.logits[:, -1])
            assert torch.equal(baseline.logits[:, :-1], perturbed.logits[:, :-1])
            try:
                with steer_last_position(block, direction, -1):
                    raise RuntimeError('synthetic interruption')
            except RuntimeError:
                pass
            assert len(block._forward_hooks) == hooks_before
            generation_args = {'input_ids': ids, 'attention_mask': mask, 'max_new_tokens': 4,
                               'do_sample': False, 'pad_token_id': 0, 'eos_token_id': 127}
            plain_tokens = model.generate(**generation_args)
            with steer_last_position(block, direction, 0):
                zero_tokens = model.generate(**generation_args)
            assert torch.equal(plain_tokens, zero_tokens)
            # Cached autoregression visits one-token block outputs after prefill.
            shapes = []
            observer = block.register_forward_hook(lambda m, i, o: shapes.append(list(o[0].shape)))
            try:
                with steer_last_position(block, direction, 1):
                    model.generate(**{**generation_args, 'use_cache': True, 'min_new_tokens': 3})
            finally:
                observer.remove()
            assert any(shape[1] == 1 for shape in shapes)
        records.append({'architecture': model_cls.__name__, 'precision': precision, 'device': device,
                        'seed': 1729, 'parameter_count': sum(p.numel() for p in model.parameters()),
                        'layer_count_including_embeddings': 3, 'pooled_shape': list(last.shape),
                        'raw_final_block_differs_from_normalized_return': raw_differs,
                        'zero_strength_logits_identical': True, 'zero_strength_greedy_ids_identical': True,
                        'only_last_position_changes': True, 'hook_cleanup_on_exception': True,
                        'cached_block_shapes': shapes, 'padding_atol_rtol': tolerance})
        del model
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--precision', choices=['float32', 'bfloat16'], default='float32')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new evidence path')
    records = run(args.device, args.precision)
    report = {'scope': 'random_tiny_architecture_engineering_not_pretrained_research',
              'python': platform.python_version(), 'platform': platform.platform(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers')},
              'records': records, 'weights_downloaded': False,
              'actual_pretrained_tokenizers_tested': False,
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__).resolve(), ROOT / 'src/models/native_smoke.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
