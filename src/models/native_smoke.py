"""Native hidden-state/hook helpers; no pretrained loading or network calls."""

from contextlib import contextmanager

import torch


def pool_action_vectors(hidden_states, positions):
    if not hidden_states or any(not p for p in positions):
        raise ValueError('Every example needs action positions')
    last, mean = [], []
    for hidden in hidden_states:
        if hidden.ndim != 3 or hidden.shape[0] != len(positions) or not torch.isfinite(hidden).all():
            raise ValueError('Invalid or nonfinite hidden-state tensor')
        last.append(torch.stack([hidden[i, p[-1]] for i, p in enumerate(positions)]))
        mean.append(torch.stack([hidden[i, p].mean(dim=0) for i, p in enumerate(positions)]))
    return torch.stack(last), torch.stack(mean)


@contextmanager
def steer_last_position(block, direction, strength: float):
    if direction.ndim != 1 or not torch.isfinite(direction).all() or not torch.isfinite(torch.tensor(strength)):
        raise ValueError('Finite vector and strength required')
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[-1] != direction.shape[0]:
            raise ValueError('Unexpected block output shape')
        if strength == 0:
            return None
        modified = hidden.clone()
        modified[:, -1, :] += strength * direction.to(device=hidden.device, dtype=hidden.dtype)
        return (modified,) + output[1:] if isinstance(output, tuple) else modified
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
