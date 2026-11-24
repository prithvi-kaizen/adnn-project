import pytest
import torch

from src.models.continuation_scoring import continuation_nll


def test_shift_starts_at_last_prompt_row_and_excludes_eos():
    seq = torch.tensor([0, 1, 2, 3])
    logits = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    logits[1] = torch.tensor([5.0, 0.0, 7.0, 0.0])
    scores, eos = continuation_nll(seq, logits, 2, {3})
    torch.testing.assert_close(scores, (logits[1].logsumexp(0) - logits[1, 2])[None])
    assert eos


def test_length_limited_suffix_has_no_terminal_eos():
    seq = torch.tensor([0, 1, 2])
    logits = torch.zeros((3, 4))
    scores, eos = continuation_nll(seq, logits, 1, {3})
    torch.testing.assert_close(scores, torch.full((2,), 4.0).log())
    assert not eos


@pytest.mark.parametrize("suffix", [(3,), (3, 2), (2, 3, 2)])
def test_rejects_empty_content_and_tokens_after_eos(suffix):
    seq = torch.tensor((0, *suffix))
    with pytest.raises(ValueError, match="nonempty content"):
        continuation_nll(seq, torch.zeros((len(seq), 4)), 1, {3})
