import pytest
import torch

from circuits import IOI_CIRCUIT, heads_of, save_induction_circuit
from quantize import D_HEAD, LINEARS, quantize_hf, rtn, scope_heads


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


@pytest.mark.parametrize("circuit", ["ioi", "induction"])
def test_scopes_partition_all(base_hf, tmp_path, circuit):
    """circuit-only and everything-but-circuit change disjoint entries that together equal 'all'."""
    groups = IOI_CIRCUIT
    if circuit == "induction":
        groups = {"induction": [(1, 2)], "previous token": [(0, 0)]}
        save_induction_circuit(groups, tmp_path)
    selected = set(heads_of(groups))
    inside = _linears(quantize_hf(base_hf, 4, f"circuit:{circuit}", results_dir=tmp_path))
    outside = _linears(quantize_hf(base_hf, 4, f"rest:{circuit}", results_dir=tmp_path))
    for key, original in _linears(base_hf).items():
        layer, name = key
        rounded, _ = rtn(original, 4)
        # Check head ownership independently, including each of the three fused Q/K/V slices.
        head_columns = torch.tensor([(layer, h) in selected for h in range(base_hf.config.n_head)])
        channels = head_columns.repeat_interleave(D_HEAD)
        if name == "attn.c_attn":
            mask = channels.repeat(3)[None, :]
        elif name == "attn.c_proj":
            mask = channels[:, None]
        else:
            mask = torch.tensor(False)
        assert torch.equal(inside[key], torch.where(mask, rounded, original)), key
        assert torch.equal(outside[key], torch.where(mask, original, rounded)), key


def test_random_heads_scope_is_size_matched(base_hf):
    """random_heads changes within 5% as many weight entries as circuit:ioi."""
    chosen, invert = scope_heads("random_heads", "", seed=7)
    assert not invert
    assert len(set(chosen)) == 26
    assert set(chosen).isdisjoint(heads_of(IOI_CIRCUIT))
    assert chosen == scope_heads("random_heads", "", seed=7)[0]

    original = _linears(base_hf)

    def changed_count(scope):
        rounded = _linears(quantize_hf(base_hf, 4, scope, seed=7))
        return sum((w != original[key]).sum().item() for key, w in rounded.items())

    assert changed_count("random_heads") == pytest.approx(changed_count("circuit:ioi"), rel=0.05)


def test_controls_match_error_size(base_hf):
    """On one MLP layer at 4 bits: signflip error has exactly the rounding error's magnitudes; noise error has the
    same per-channel RMS within 15%."""
    weight = base_hf.transformer.h[0].mlp.c_fc.weight
    rounded, _ = rtn(weight, 4)
    error = rounded - weight

    def control_error(control):
        model = quantize_hf(base_hf, 4, control=control, seed=3)
        return model.transformer.h[0].mlp.c_fc.weight - weight

    flipped = control_error("signflip")
    torch.testing.assert_close(flipped.abs(), error.abs(), rtol=0, atol=1e-7)
    assert (flipped * error < 0).any()
    noise = control_error("noise")
    torch.testing.assert_close(noise.square().mean(0).sqrt(), error.square().mean(0).sqrt(), rtol=0.15, atol=0)
    assert not torch.allclose(noise.abs(), error.abs())
