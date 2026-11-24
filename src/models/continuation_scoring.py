"""Causal NLL for one unpadded generated continuation, excluding terminal EOS."""

import torch


def continuation_nll(token_ids: torch.Tensor, logits: torch.Tensor, prompt_length: int, eos_ids: set[int]):
    """Score content using the logit row immediately before each generated token."""
    if token_ids.ndim != 1 or logits.ndim != 2 or logits.size(0) != token_ids.size(0):
        raise ValueError("Expected one unpadded sequence and aligned logits")
    if not 0 < prompt_length < token_ids.size(0):
        raise ValueError("Sequence must contain a prompt and a generated suffix")
    suffix = token_ids[prompt_length:].tolist()
    terminal_eos = suffix[-1] in eos_ids
    content = suffix[:-1] if terminal_eos else suffix
    if not content or any(token in eos_ids for token in content):
        raise ValueError("Need nonempty content followed by at most one EOS")
    end = prompt_length + len(content)
    shifted = logits[prompt_length - 1:end - 1].float().log_softmax(dim=-1)
    negative_logprobs = -shifted.gather(1, token_ids[prompt_length:end, None]).squeeze(1)
    if not torch.isfinite(negative_logprobs).all():
        raise ValueError("Non-finite continuation score")
    return negative_logprobs, terminal_eos
