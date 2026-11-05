import pytest


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_rtn_error_and_levels(bits):
    """On a random [64, 32] matrix: every error <= scale / 2, and each column uses at most 2^bits - 1 levels."""
    raise NotImplementedError


def test_rtn_16_bits_is_near_identity():
    """16-bit rounding changes no weight by more than 0.1% of the largest weight."""
    raise NotImplementedError


def test_none_bits_is_untouched(base_hf):
    """quantize_hf(model, None) leaves every block linear weight exactly equal to the original."""
    raise NotImplementedError


def test_only_block_linears_change(base_hf):
    """At 4 bits, scope "all": embeddings, LayerNorms and biases are exactly unchanged."""
    raise NotImplementedError


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
