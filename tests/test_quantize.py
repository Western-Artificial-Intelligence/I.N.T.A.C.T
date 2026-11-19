import pytest
import torch

from quantize import LINEARS, quantize_hf, rtn


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_rtn_error_and_levels(bits):
    """On a random [64, 32] matrix: every error <= scale / 2, and each column uses at most 2^bits - 1 levels."""
    w = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
    rounded, scale = rtn(w, bits)
    assert ((rounded - w).abs() <= scale / 2 * (1 + 1e-5)).all()  # tiny slack for float32 arithmetic
    for col in rounded.T:
        assert col.unique().numel() <= 2**bits - 1


def test_rtn_16_bits_is_near_identity():
    """16-bit rounding changes no weight by more than 0.1% of the largest weight."""
    w = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
    rounded, _ = rtn(w, 16)
    assert (rounded - w).abs().max() <= 1e-3 * w.abs().max()


def test_none_bits_is_untouched(base_hf):
    """quantize_hf(model, None) leaves every block linear weight exactly equal to the original."""
    q = quantize_hf(base_hf, None)
    for orig, new in zip(base_hf.transformer.h, q.transformer.h):
        for name in LINEARS:
            assert torch.equal(orig.get_submodule(name).weight, new.get_submodule(name).weight)


def test_only_block_linears_change(base_hf):
    """At 4 bits, scope "all": embeddings, LayerNorms and biases are exactly unchanged."""
    q = quantize_hf(base_hf, 4)
    linear_weights = {f"transformer.h.{i}.{name}.weight" for i in range(len(base_hf.transformer.h)) for name in LINEARS}
    new_params = dict(q.named_parameters())
    for name, p in base_hf.named_parameters():
        if name not in linear_weights:
            assert torch.equal(p, new_params[name]), name


def test_scopes_partition_all(base_hf):
    """circuit-only and everything-but-circuit change disjoint entries that together equal 'all'."""
    raise NotImplementedError


def test_random_heads_scope_is_size_matched(base_hf):
    """random_heads changes within 5% as many weight entries as circuit:ioi."""
    raise NotImplementedError


def test_controls_match_error_size(base_hf):
    """On one MLP layer at 4 bits: signflip error has exactly the rounding error's magnitudes; noise error has the
    same per-channel RMS within 15%."""
    raise NotImplementedError
