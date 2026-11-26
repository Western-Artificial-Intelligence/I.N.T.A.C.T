"""The causal measurement: copy a head's output from the clean run into the corrupted run, see how much the answer
comes back. Not "does this part look normal" but "is this part still causing the answer".

All patching here is within one model (clean and corrupted runs of the same quantized model), so it asks whether
the circuit still carries the behaviour in *that* model.
"""
import collections

import torch


def run_with_heads_patched(model, corrupt_tokens, clean_cache, heads: list):
    """Corrupted run with the given heads' outputs (hook_z, all positions) copied from the clean cache. Returns logits."""
    # creates a dict to store heads in layer keys
    by_layer = collections.defaultdict(list)

    # layer is the key and head is stored with layer
    for layer, head in heads:
        by_layer[layer].append(head)

    # list to store each of the hooks
    hooks = []

    for layer, layer_heads in by_layer.items():
        def hook_fn(z, hook, layer=layer, layer_heads=layer_heads):
            # this layer's z from the clean run, [N, T, H, d_head]
            clean_z = clean_cache["z", layer]

            # swaps each patched head's slice (every prompt, every position) for the clean one
            for h in layer_heads:
                z[:, :, h, :] = clean_z[:, :, h, :]

            return z

        hooks.append((f"blocks.{layer}.attn.hook_z", hook_fn))  # stores hooks in list

    # gathers logits with hooks
    logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)

    return logits


def normalized_recovery(patched, clean, corrupt) -> float:
    """(patched - corrupt) / (clean - corrupt). 0 = no better than corrupted, 1 = fully clean."""
    return (patched - corrupt) / (clean - corrupt)


def end_logit_diff(model, final_resid, data) -> torch.Tensor:
    """logit(IO) - logit(S) from the final residual stream, unembedding only the last position (ln_final, W_U, b_U)."""
    # picks out each prompt's end position from final_resid
    N = final_resid.size(0)
    end_resid = final_resid[torch.arange(N), data.pos["end"]]

    normed = model.ln_final(end_resid)  # Applying the final LayerNorm
    end_logits = normed @ model.W_U + model.b_U  # gives the end position logits

    res = end_logits[torch.arange(N), data.io] - end_logits[torch.arange(N), data.s]  # logit(IO) - logit(S)

    return res


def head_patch_map(model, data, clean_cache, corrupt_cache, ld_clean: float, ld_corrupt: float) -> torch.Tensor:
    """[L, H]: recovery of the IOI logit difference from patching each single head (clean -> flipped).

    Must give the same numbers as transformer_lens.patching.get_act_patch_attn_head_out_all_pos (checked in
    tests/test_patching.py). That reference is slow (~130 s per model); to speed it up, note a head in layer l can
    only change layers l and later, so restart from the corrupted resid_pre at layer l (start_at_layer=l) and
    unembed only the last position with end_logit_diff.
    """
    # creating empty result tensor
    res = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    with torch.no_grad():
        # iterates over each layer, head pair
        for layer in range(model.cfg.n_layers):
            # starting point
            resid = corrupt_cache["resid_pre", layer]

            for head in range(model.cfg.n_heads):
                def hook_fn(z, hook, layer=layer, head=head):
                    # this layer's z from the clean run, [N, T, H, d_head]
                    clean_z = clean_cache["z", layer]

                    # swaps just this head's slice (every prompt, every position) for the clean one
                    z[:, :, head, :] = clean_z[:, :, head, :]

                    return z

                # gets the final residual stream
                final_resid = model.run_with_hooks(resid,
                                              start_at_layer=layer,  # tells input is residual stream
                                              stop_at_layer=model.cfg.n_layers,  # hands back to final residual stream
                                              fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)])

                # gets the logit difference
                logit_diff = end_logit_diff(model, final_resid, data).mean()

                # stores the normalized recovery in res
                res[layer, head] = normalized_recovery(logit_diff, ld_clean, ld_corrupt)

    return res
