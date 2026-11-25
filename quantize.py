"""The rounder: our one independent variable.

Weight-only, symmetric, per-output-channel round-to-nearest (RTN) on the transformer-block linear layers, matching
Duan (arXiv 2606.03002). Embeddings / unembedding (tied in GPT-2), LayerNorms and biases stay FP32.

We round the original Hugging Face weights and only then load them into TransformerLens. TransformerLens folds
LayerNorm into the weights and centres them on load, so rounding after loading would round a weight distribution
that never ships.

Scales always come from the full weight matrix; a scope mask then decides which entries take the rounded value.
That way "circuit-only" + "everything-but-circuit" partition exactly the same rounded numbers as "all".
"""

import copy

import torch
from transformer_lens import HookedTransformer

from circuits import IOI_CIRCUIT, heads_of, load_induction_circuit, random_heads
from config import Config

D_HEAD = 64
LINEARS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


def rtn(w: torch.Tensor, bits: int):
    """Round each output channel to a symmetric grid of 2^bits - 1 levels. Returns (rounded w, scale).

    HF Conv1D weights are [in, out], so a channel is a column: qmax = 2^(bits-1) - 1, scale = max|column| / qmax.
    """
    qmax = 2 ** (bits - 1) - 1
    # one scale per column; clamp so an all-zero column doesn't divide by zero
    scale = (w.abs().amax(dim=0, keepdim=True) / qmax).clamp(min=1e-12)
    rounded = (w / scale).round().clamp(-qmax, qmax) * scale
    return rounded, scale


def head_masks(layer: int, heads: list, shapes: dict) -> dict:
    """Boolean mask per linear selecting the Q/K/V/O weight entries that belong to `heads` in this layer.

    c_attn is fused QKV [in, 3 * d_model]: head h owns columns h*64..(h+1)*64 inside each of the Q, K and V thirds.
    c_proj (W_O) is [d_model, d_model]: head h owns rows h*64..(h+1)*64. MLP masks stay all False.
    """
    masks = {name: torch.zeros(shape, dtype=torch.bool) for name, shape in shapes.items()}
    qkv = masks["attn.c_attn"]
    width = qkv.shape[1] // 3
    for head_layer, head in heads:
        if head_layer == layer:
            start = head * D_HEAD
            for offset in (0, width, 2 * width):
                qkv[:, offset + start:offset + start + D_HEAD] = True
            masks["attn.c_proj"][start:start + D_HEAD, :] = True
    return masks


def scope_heads(scope: str, results_dir: str, seed: int):
    """Returns (heads, invert): quantize those heads' weights, or everything except them if invert.

    "all" -> ([], True); "none" -> ([], False); "circuit:ioi" / "circuit:induction" -> (circuit heads, False);
    "rest:ioi" / "rest:induction" -> (circuit heads, True); "random_heads" -> (26 random non-IOI heads, False).
    """
    if scope in ("all", "none"):
        return [], scope == "all"
    if scope == "random_heads":
        ioi = heads_of(IOI_CIRCUIT)
        return random_heads(len(ioi), ioi, seed), False

    selection, _, circuit = scope.partition(":")
    if selection not in ("rest", "circuit") or circuit not in ("ioi", "induction"):
        raise ValueError(f"Unknown rounding scope: {scope}")
    groups = IOI_CIRCUIT if circuit == "ioi" else load_induction_circuit(results_dir)
    return heads_of(groups), selection == "rest"


@torch.no_grad()
def quantize_hf(hf_model, bits, scope="all", control="rtn", results_dir="results", seed=0):
    """Return a rounded deep copy of a HF GPT2LMHeadModel. bits=None returns an untouched copy.

    control: "rtn" = real rounding; "noise" = Gaussian error with the same per-channel RMS as rounding;
    "signflip" = the rounding error with random signs. Only masked entries change (see scope_heads).
    """
    model = copy.deepcopy(hf_model)  # never touch the caller's model
    if bits is None:
        return model
    if control not in ("rtn", "noise", "signflip"):
        raise ValueError(f"Unknown rounding control: {control}")
    heads, invert = scope_heads(scope, results_dir, seed)
    if scope == "none":
        return model

    rng = torch.Generator().manual_seed(seed)
    for layer, block in enumerate(model.transformer.h):
        weights = {name: block.get_submodule(name).weight for name in LINEARS}
        if scope != "all":
            masks = head_masks(layer, heads, {name: w.shape for name, w in weights.items()})
        for name, weight in weights.items():
            rounded, _ = rtn(weight, bits)
            if control == "noise":
                rms = (rounded - weight).square().mean(dim=0).sqrt()
                noise = torch.randn(weight.shape, generator=rng, dtype=weight.dtype).to(weight.device)
                rounded = weight + noise * (rms / noise.square().mean(dim=0).sqrt().clamp_min(1e-12))
            elif control == "signflip":
                signs = torch.randint(2, weight.shape, generator=rng, dtype=torch.int8).to(weight.device)
                rounded = weight + (rounded - weight) * (2 * signs - 1)
            if scope != "all":
                mask = masks[name].to(weight.device)
                rounded = torch.where(~mask if invert else mask, rounded, weight)
            weight.copy_(rounded)
    return model


def load_tl(hf_model, tokenizer, device="cpu") -> HookedTransformer:
    """Load a (possibly rounded) HF model into TransformerLens as "gpt2", in eval mode."""
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf_model, tokenizer=tokenizer, device=device)
    model.eval()
    return model


def build_model(base_hf, tokenizer, bits, scope="all", control="rtn", cfg=None) -> HookedTransformer:
    """A fresh model for one condition of the sweep: quantize_hf, then load_tl."""
    cfg = cfg or Config()
    hf = quantize_hf(base_hf, bits, scope, control, cfg.results_dir, cfg.seed)
    model = load_tl(hf, tokenizer, cfg.device)
    del hf  # TransformerLens made its own copy of the weights
    return model
