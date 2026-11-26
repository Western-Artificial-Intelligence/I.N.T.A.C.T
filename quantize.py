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

    `heads` is a list of (layer, head) tuples; only those in `layer` count. `shapes` is {linear name: weight shape}.
    c_attn is fused QKV [in, 3 * d_model]: head h owns columns h*64..(h+1)*64 inside each of the Q, K and V thirds.
    c_proj (W_O) is [d_model, d_model]: head h owns rows h*64..(h+1)*64. MLP masks stay all False.
    """
    masks = {name: torch.zeros(shape, dtype=torch.bool) for name, shape in shapes.items()}
    for l, h in heads:
        if l != layer:
            continue
        cols = slice(h * D_HEAD, (h + 1) * D_HEAD)
        if "attn.c_attn" in masks:
            d_model = shapes["attn.c_attn"][1] // 3
            for third in range(3):  # Q, K, V
                masks["attn.c_attn"][:, third * d_model + cols.start: third * d_model + cols.stop] = True
        if "attn.c_proj" in masks:
            masks["attn.c_proj"][cols, :] = True
    return masks


def scope_heads(scope: str, results_dir: str, seed: int):
    """Returns (heads, invert): quantize those heads' weights, or everything except them if invert.

    "all" -> ([], True); "none" -> ([], False); "circuit:ioi" / "circuit:induction" -> (circuit heads, False);
    "rest:ioi" / "rest:induction" -> (circuit heads, True); "random_heads" -> (26 random non-IOI heads, False).
    The induction circuit is the frozen list in <results_dir>/induction_heads.json (see circuits.py).
    """
    from circuits import IOI_CIRCUIT, heads_of, load_induction_circuit, random_heads

    if scope == "all":
        return [], True
    if scope == "none":
        return [], False
    if scope == "random_heads":
        ioi = heads_of(IOI_CIRCUIT)
        return random_heads(len(ioi), exclude=ioi, seed=seed), False

    kind, _, circuit = scope.partition(":")
    if kind not in ("circuit", "rest") or circuit not in ("ioi", "induction"):
        raise ValueError(f"unknown scope {scope!r}")
    if circuit == "ioi":
        heads = heads_of(IOI_CIRCUIT)
    else:
        try:
            heads = heads_of(load_induction_circuit(results_dir))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"scope {scope!r} needs the frozen induction circuit in {results_dir}/; "
                                    "select and save it on the FP model first (circuits.py)") from e
    return heads, kind == "rest"


CONTROLS = ("rtn", "noise", "signflip")


def _perturb(w: torch.Tensor, bits: int, control: str, gen: torch.Generator) -> torch.Tensor:
    """The whole matrix after rounding (or its control). Scales always come from the full matrix."""
    rounded, _ = rtn(w, bits)
    if control == "rtn":
        return rounded  # exactly on the grid (w + err can be off by float rounding)
    err = rounded - w
    if control == "signflip":
        signs = torch.randint(0, 2, w.shape, generator=gen, dtype=w.dtype) * 2 - 1
        return w + err * signs
    # noise: Gaussian with the rounding error's RMS in each output channel (column)
    rms = err.pow(2).mean(dim=0, keepdim=True).sqrt()
    return w + torch.randn(w.shape, generator=gen, dtype=w.dtype) * rms


@torch.no_grad()
def quantize_hf(hf_model, bits, scope="all", control="rtn", results_dir="results", seed=0):
    """Return a rounded deep copy of a HF GPT2LMHeadModel. bits=None returns an untouched copy.

    control: "rtn" = real rounding; "noise" = Gaussian error with the same per-channel RMS as rounding;
    "signflip" = the rounding error with random signs. Only masked entries change (see scope_heads).

    Every matrix is perturbed in full (full-matrix scales; noise and signs drawn for every entry from a generator
    seeded by (seed, layer, linear)), and the scope mask then picks which entries take the new value. So any scope's
    changed entries hold exactly the values scope "all" gives them, whatever the scope or loop order.
    """
    if control not in CONTROLS:
        raise ValueError(f"unknown control {control!r}; expected one of {CONTROLS}")
    model = copy.deepcopy(hf_model)  # never touch the caller's model
    if bits is None:
        return model
    heads, invert = scope_heads(scope, results_dir, seed)
    for layer, block in enumerate(model.transformer.h):
        mods = {name: block.get_submodule(name) for name in LINEARS}
        masks = head_masks(layer, heads, {name: mod.weight.shape for name, mod in mods.items()})
        for j, (name, mod) in enumerate(mods.items()):
            mask = ~masks[name] if invert else masks[name]
            if not mask.any():
                continue
            w = mod.weight.data
            gen = torch.Generator().manual_seed(seed * 1000 + layer * len(LINEARS) + j)
            mod.weight.data = torch.where(mask, _perturb(w, bits, control, gen), w)  # weights only; biases stay FP32
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
