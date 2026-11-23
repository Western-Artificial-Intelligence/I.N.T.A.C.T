import pytest
import torch

from quantize import LINEARS, quantize_hf, rtn


def _linears(model):
    """{(layer, linear name): weight} for every block linear the rounder touches."""
    return {(i, n): b.get_submodule(n).weight for i, b in enumerate(model.transformer.h) for n in LINEARS}


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_rtn_error_and_levels(bits):
    """On a random [64, 32] matrix: every error <= scale / 2, and each column uses at most 2^bits - 1 levels."""
    w = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
    wq, scale = rtn(w, bits)
    assert ((w - wq).abs() <= scale / 2 + 1e-6).all()
    for col in range(w.shape[1]):
        assert wq[:, col].unique().numel() <= 2 ** bits - 1


def test_rtn_16_bits_is_near_identity():
    """16-bit rounding changes no weight by more than 0.1% of the largest weight."""
    w = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
    wq, _ = rtn(w, 16)
    assert (w - wq).abs().max() < 1e-3 * w.abs().max()


def test_none_bits_is_untouched(base_hf):
    """quantize_hf(model, None) leaves every block linear weight exactly equal to the original."""
    base = _linears(base_hf)
    for k, w in _linears(quantize_hf(base_hf, None)).items():
        assert torch.equal(w, base[k]), k


def test_only_block_linears_change(base_hf):
    """At 4 bits, scope "all": embeddings, LayerNorms and biases are exactly unchanged."""
    base = dict(base_hf.named_parameters())
    rounded = dict(quantize_hf(base_hf, 4, "all").named_parameters())
    for name, p in rounded.items():
        is_linear_weight = ".h." in name and any(name.endswith(f"{n}.weight") for n in LINEARS)
        if is_linear_weight:
            assert not torch.equal(p, base[name]), f"{name} should have been rounded"
        else:
            assert torch.equal(p, base[name]), f"{name} should be untouched"
    # and the caller's model is left alone
    assert all(torch.equal(p, base[n]) for n, p in base_hf.named_parameters())


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
