import torch

from tasks import BABA_TEMPLATES, BOS, make_ioi

N_PER = 4  # x 15 templates = 60 prompts, enough to cover both orders in every template


def test_seeded_reproducible(tok):
    """Same seed -> identical tokens, flipped and abc tensors; a different seed -> different tokens."""
    a, b, c = make_ioi(tok, N_PER, seed=0), make_ioi(tok, N_PER, seed=0), make_ioi(tok, N_PER, seed=1)
    assert a.prompts == b.prompts
    assert torch.equal(a.tokens, b.tokens)
    assert torch.equal(a.flipped, b.flipped)
    assert torch.equal(a.abc, b.abc)
    assert not torch.equal(a.tokens, c.tokens)


def test_flipped_twin_differs_only_at_s2(tok):
    """Each flipped prompt differs from its clean prompt at exactly one token: S2, which now holds IO."""
    d = make_ioi(tok, N_PER, seed=0)
    idx = torch.arange(len(d))
    diff = d.tokens != d.flipped
    assert (diff.sum(dim=1) == 1).all()
    assert diff[idx, d.pos["S2"]].all()
    assert torch.equal(d.flipped[idx, d.pos["S2"]], d.io)


def test_abc_twin_differs_only_at_name_slots(tok):
    """Clean and abc prompts are identical everywhere except the IO, S1 and S2 positions."""
    d = make_ioi(tok, N_PER, seed=0)
    idx = torch.arange(len(d))
    slots = torch.zeros_like(d.tokens, dtype=torch.bool)
    for key in ("IO", "S1", "S2"):
        slots[idx, d.pos[key]] = True
    assert torch.equal(d.tokens != d.abc, slots)  # all three slots change, nothing else does
    # three different fresh names, so the abc twin carries no IOI signal
    abc_names = torch.stack([d.abc[idx, d.pos[key]] for key in ("IO", "S1", "S2")], dim=1)
    assert all(len(set(row.tolist())) == 3 for row in abc_names)


def test_positions_and_lengths(tok):
    """pos["IO"] holds IO, pos["S1"] and pos["S2"] hold S, every prompt starts with BOS, every prompt of one template
    has the same length, and both ABBA and BABA prompts exist."""
    d = make_ioi(tok, N_PER, seed=0)
    idx = torch.arange(len(d))
    assert len(d) == N_PER * len(BABA_TEMPLATES) == len(d.tokens)
    assert torch.equal(d.tokens[idx, d.pos["IO"]], d.io)
    assert torch.equal(d.tokens[idx, d.pos["S1"]], d.s)
    assert torch.equal(d.tokens[idx, d.pos["S2"]], d.s)
    assert torch.equal(d.pos["S1+1"], d.pos["S1"] + 1)
    assert (d.pos["S1"] < d.pos["S2"]).all() and (d.pos["S2"] < d.pos["end"]).all()
    assert (d.tokens[:, 0] == BOS).all()
    assert (d.tokens[idx, d.pos["end"]] != BOS).all()  # "end" is the last real token ...
    for i in idx:
        assert (d.tokens[i, d.pos["end"][i] + 1:] == BOS).all()  # ... and only padding comes after it
    for t in d.template.unique():
        assert len(d.pos["end"][d.template == t].unique()) == 1
    assert d.abba.any() and (~d.abba).any()
    # ABBA puts IO first, BABA puts S first
    assert (d.pos["IO"] < d.pos["S1"])[d.abba].all() and (d.pos["S1"] < d.pos["IO"])[~d.abba].all()
    assert not any("  " in p or "[" in p for p in d.prompts)  # no double spaces, no unfilled blanks


def test_names_are_single_tokens(tok):
    """Every IO name decodes and re-encodes to exactly one token."""
    d = make_ioi(tok, N_PER, seed=0)
    for name_id in torch.cat([d.io, d.s]).unique().tolist():
        assert tok.encode(tok.decode([name_id])) == [name_id]


def test_induction_repeats():
    """tokens = [BOS, A, A]; corrupt matches tokens up to the end of the first A, then differs."""
    raise NotImplementedError
