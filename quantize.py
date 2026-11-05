"""The rounder: our one independent variable.

Weight-only, symmetric, per-output-channel round-to-nearest (RTN) on the transformer-block linear layers, matching
Duan (arXiv 2606.03002). Embeddings / unembedding (tied in GPT-2), LayerNorms and biases stay FP32.

We round the original Hugging Face weights and only then load them into TransformerLens. TransformerLens folds
LayerNorm into the weights and centres them on load, so rounding after loading would round a weight distribution
that never ships.

Scales always come from the full weight matrix; a scope mask then decides which entries take the rounded value.
That way "circuit-only" + "everything-but-circuit" partition exactly the same rounded numbers as "all".
"""

import torch
from transformer_lens import HookedTransformer

D_HEAD = 64
LINEARS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


def rtn(w: torch.Tensor, bits: int):
    """Round each output channel to a symmetric grid of 2^bits - 1 levels. Returns (rounded w, scale).

    HF Conv1D weights are [in, out], so a channel is a column: qmax = 2^(bits-1) - 1, scale = max|column| / qmax.
    """
    raise NotImplementedError


def head_masks(layer: int, heads: list, shapes: dict) -> dict:
    """Boolean mask per linear selecting the Q/K/V/O weight entries that belong to `heads` in this layer.

    c_attn is fused QKV [in, 3 * d_model]: head h owns columns h*64..(h+1)*64 inside each of the Q, K and V thirds.
    c_proj (W_O) is [d_model, d_model]: head h owns rows h*64..(h+1)*64. MLP masks stay all False.
    """
    raise NotImplementedError


def scope_heads(scope: str, results_dir: str, seed: int):
    """Returns (heads, invert): quantize those heads' weights, or everything except them if invert.

    "all" -> ([], True); "none" -> ([], False); "circuit:ioi" / "circuit:induction" -> (circuit heads, False);
    "rest:ioi" / "rest:induction" -> (circuit heads, True); "random_heads" -> (26 random non-IOI heads, False).
    """
    raise NotImplementedError


@torch.no_grad()
def quantize_hf(hf_model, bits, scope="all", control="rtn", results_dir="results", seed=0):
    """Return a rounded deep copy of a HF GPT2LMHeadModel. bits=None returns an untouched copy.

    control: "rtn" = real rounding; "noise" = Gaussian error with the same per-channel RMS as rounding;
    "signflip" = the rounding error with random signs. Only masked entries change (see scope_heads).
    """
    raise NotImplementedError


def load_tl(hf_model, tokenizer, device="cpu") -> HookedTransformer:
    """Load a (possibly rounded) HF model into TransformerLens as "gpt2", in eval mode."""
    raise NotImplementedError


def build_model(base_hf, tokenizer, bits, scope="all", control="rtn", cfg=None) -> HookedTransformer:
    """A fresh model for one condition of the sweep: quantize_hf, then load_tl."""
    raise NotImplementedError
