import torch
from transformer_lens import patching as tl_patching

import metrics as M
from circuits import ALL_HEADS, IOI_CIRCUIT, KEEP_POSITION
from metrics import logit_diff
from patching import end_logit_diff, head_patch_map, normalized_recovery, run_with_heads_patched


def test_patching_clean_into_clean_is_identity(fp_model, data):
    """Patching every head from the clean cache into the clean run gives the clean logits (atol 1e-4)."""
    clean_logits, clean_cache = fp_model.run_with_cache(data.tokens)

    # pasting the clean run's own values back in should change nothing
    patched = run_with_heads_patched(fp_model, data.tokens, clean_cache, ALL_HEADS)

    assert torch.allclose(patched, clean_logits, atol=1e-4)


def test_patching_nothing_changes_nothing(fp_model, data):
    """Patching an empty head list into the flipped run gives exactly the flipped logits."""
    _, clean_cache = fp_model.run_with_cache(data.tokens)
    flipped_logits = fp_model(data.flipped)

    # no heads -> no hooks -> a plain flipped run
    patched = run_with_heads_patched(fp_model, data.flipped, clean_cache, [])

    assert torch.equal(patched, flipped_logits)


def test_end_logit_diff_matches_full_forward(fp_model, data):
    """end_logit_diff on the final residual stream equals metrics.logit_diff on full logits (atol 1e-4)."""
    logits, cache = fp_model.run_with_cache(data.tokens)

    # residual stream after the last block, i.e. right before ln_final + unembed
    final_resid = cache[f"blocks.{fp_model.cfg.n_layers - 1}.hook_resid_post"]

    assert torch.allclose(end_logit_diff(fp_model, final_resid, data), logit_diff(logits, data), atol=1e-4)


def test_fast_head_map_matches_transformer_lens(fp_model, data):
    """head_patch_map equals transformer_lens.patching.get_act_patch_attn_head_out_all_pos with the same
    normalized-logit-diff metric (atol 1e-4)."""
    clean_logits, clean_cache = fp_model.run_with_cache(data.tokens)
    corrupt_logits, corrupt_cache = fp_model.run_with_cache(data.flipped)
    ld_clean = logit_diff(clean_logits, data).mean().item()
    ld_corrupt = logit_diff(corrupt_logits, data).mean().item()

    # same metric for both: mean logit diff, normalized between flipped (0) and clean (1)
    def metric(logits):
        return normalized_recovery(logit_diff(logits, data).mean(), ld_clean, ld_corrupt)

    ours = head_patch_map(fp_model, data, clean_cache, corrupt_cache, ld_clean, ld_corrupt)

    # the slow reference (~130 s): full forward pass for every head
    reference = tl_patching.get_act_patch_attn_head_out_all_pos(fp_model, data.flipped, clean_cache, metric)

    assert ours.shape == (fp_model.cfg.n_layers, fp_model.cfg.n_heads)
    assert torch.allclose(ours, reference.to(ours.dtype), atol=1e-4)


def test_mean_ablation_keep_everything_is_identity(fp_model, data):
    """Mean-ablation hooks with a keep-everything mask leave the logits unchanged (atol 1e-4)."""
    seq_len = data.tokens.shape[1]
    keep = M.ablate_heads_mask(data, [], seq_len)  # ablate nothing
    means = M.template_means(fp_model, data, data.abc)

    clean = fp_model(data.tokens)
    hooked = fp_model.run_with_hooks(data.tokens, fwd_hooks=M.mean_ablation_hooks(keep, means))

    assert torch.allclose(hooked, clean, atol=1e-4)


def test_keep_mask_marks_one_position_per_circuit_head(data):
    """keep_mask with IOI_CIRCUIT and KEEP_POSITION marks exactly (26 heads x N prompts) entries True."""
    n, seq_len = data.tokens.shape
    keep = M.keep_mask(data, IOI_CIRCUIT, KEEP_POSITION, seq_len)

    assert sum(int(k.sum()) for k in keep.values()) == 26 * n
