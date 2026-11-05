"""The causal measurement: copy a head's output from the clean run into the corrupted run, see how much the answer
comes back. Not "does this part look normal" but "is this part still causing the answer".

All patching here is within one model (clean and corrupted runs of the same quantized model), so it asks whether
the circuit still carries the behaviour in *that* model.
"""

import torch


def run_with_heads_patched(model, corrupt_tokens, clean_cache, heads: list):
    """Corrupted run with the given heads' outputs (hook_z, all positions) copied from the clean cache. Returns logits."""
    raise NotImplementedError


def normalized_recovery(patched, clean, corrupt) -> float:
    """(patched - corrupt) / (clean - corrupt). 0 = no better than corrupted, 1 = fully clean."""
    raise NotImplementedError


def end_logit_diff(model, final_resid, data) -> torch.Tensor:
    """logit(IO) - logit(S) from the final residual stream, unembedding only the last position (ln_final, W_U, b_U)."""
    raise NotImplementedError


def head_patch_map(model, data, clean_cache, corrupt_cache, ld_clean: float, ld_corrupt: float) -> torch.Tensor:
    """[L, H]: recovery of the IOI logit difference from patching each single head (clean -> flipped).

    Must give the same numbers as transformer_lens.patching.get_act_patch_attn_head_out_all_pos (checked in
    tests/test_patching.py). That reference is slow (~130 s per model); to speed it up, note a head in layer l can
    only change layers l and later, so restart from the corrupted resid_pre at layer l (start_at_layer=l) and
    unembed only the last position with end_logit_diff.
    """
    raise NotImplementedError
