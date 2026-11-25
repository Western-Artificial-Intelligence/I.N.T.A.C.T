import torch

import metrics as M
from circuits import IOI_CIRCUIT, KEEP_POSITION


def test_patching_clean_into_clean_is_identity(fp_model, data):
    """Patching every head from the clean cache into the clean run gives the clean logits (atol 1e-4)."""
    raise NotImplementedError


def test_patching_nothing_changes_nothing(fp_model, data):
    """Patching an empty head list into the flipped run gives exactly the flipped logits."""
    raise NotImplementedError


def test_end_logit_diff_matches_full_forward(fp_model, data):
    """end_logit_diff on the final residual stream equals metrics.logit_diff on full logits (atol 1e-4)."""
    raise NotImplementedError


def test_fast_head_map_matches_transformer_lens(fp_model, data):
    """head_patch_map equals transformer_lens.patching.get_act_patch_attn_head_out_all_pos with the same
    normalized-logit-diff metric (atol 1e-4)."""
    raise NotImplementedError


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
