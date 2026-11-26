import pytest
import torch

from circuits import IOI_CIRCUIT, heads_of, save_induction_circuit
from quantize import LINEARS, head_masks, quantize_hf, rtn, scope_heads


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


def test_head_masks_select_each_heads_slices():
    """A head owns 64 columns in each of the Q, K, V thirds of c_attn and 64 rows of c_proj; MLPs are never masked;
    heads from other layers are ignored."""
    shapes = {"attn.c_attn": (768, 2304), "attn.c_proj": (768, 768), "mlp.c_fc": (768, 3072),
              "mlp.c_proj": (3072, 768)}
    m = head_masks(3, [(3, 2), (5, 0)], shapes)
    qkv_cols = m["attn.c_attn"].all(dim=0).nonzero().flatten().tolist()
    assert qkv_cols == [c + third * 768 for third in range(3) for c in range(128, 192)]
    assert m["attn.c_attn"].sum() == 768 * 64 * 3
    assert m["attn.c_proj"].all(dim=1).nonzero().flatten().tolist() == list(range(128, 192))
    assert m["attn.c_proj"].sum() == 64 * 768
    assert not m["mlp.c_fc"].any() and not m["mlp.c_proj"].any()
    assert not any(mask.any() for mask in head_masks(4, [(3, 2), (5, 0)], shapes).values())


def _induction_dir(tmp_path):
    """A results dir holding a frozen induction circuit (the pilot's heads), as #8 writes it."""
    save_induction_circuit({"previous token": [(4, 11)], "induction": [(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)]},
                           str(tmp_path))
    return str(tmp_path)


@pytest.mark.parametrize("circuit", ["ioi", "induction"])
def test_scopes_partition_all(base_hf, tmp_path, circuit):
    """circuit-only and everything-but-circuit change disjoint entries that together equal 'all'."""
    results = _induction_dir(tmp_path)
    base = _linears(base_hf)
    full = _linears(quantize_hf(base_hf, 4, "all", results_dir=results))
    inside = _linears(quantize_hf(base_hf, 4, f"circuit:{circuit}", results_dir=results))
    outside = _linears(quantize_hf(base_hf, 4, f"rest:{circuit}", results_dir=results))
    heads = set(scope_heads(f"circuit:{circuit}", results, 0)[0])
    for k, w in base.items():
        d_in, d_out = inside[k] != w, outside[k] != w
        assert not (d_in & d_out).any(), k  # disjoint
        assert torch.equal(torch.where(d_in, inside[k], outside[k]), full[k]), k  # together they are exactly "all"
        if k[1].startswith("mlp") or not any(layer == k[0] for layer, _ in heads):
            assert not d_in.any(), k  # circuit-only never touches MLPs or layers without circuit heads


def test_random_heads_scope_is_size_matched(base_hf):
    """random_heads changes within 5% as many weight entries as circuit:ioi."""
    heads, invert = scope_heads("random_heads", "results", seed=0)
    ioi = heads_of(IOI_CIRCUIT)
    assert not invert and len(heads) == len(ioi) == 26 and not set(heads) & set(ioi)
    assert scope_heads("random_heads", "results", seed=0)[0] == heads  # same seed, same pick

    def n_changed(scope):
        q = _linears(quantize_hf(base_hf, 4, scope))
        return sum(int((q[k] != w).sum()) for k, w in _linears(base_hf).items())

    n_random, n_ioi = n_changed("random_heads"), n_changed("circuit:ioi")
    assert n_ioi > 0 and abs(n_random - n_ioi) <= 0.05 * n_ioi


def test_controls_match_error_size(base_hf):
    """On one MLP layer at 4 bits: signflip error has exactly the rounding error's magnitudes; noise error has the
    same per-channel RMS within 15%."""
    k = (6, "mlp.c_fc")
    w = _linears(base_hf)[k]
    err = {c: _linears(quantize_hf(base_hf, 4, control=c))[k] - w for c in ("rtn", "signflip", "noise")}

    torch.testing.assert_close(err["signflip"].abs(), err["rtn"].abs(), rtol=0, atol=1e-6)
    flipped = (err["signflip"] * err["rtn"] < 0).float().mean()
    assert 0.4 < flipped < 0.6  # about half the signs flip

    def rms(e):
        return e.pow(2).mean(dim=0).sqrt()

    ratio = rms(err["noise"]) / rms(err["rtn"])
    assert ((ratio > 0.85) & (ratio < 1.15)).all()
    assert not torch.equal(err["noise"], err["rtn"])

    again = _linears(quantize_hf(base_hf, 4, control="noise"))[k] - w
    assert torch.equal(again, err["noise"])  # seeded: same noise every time
