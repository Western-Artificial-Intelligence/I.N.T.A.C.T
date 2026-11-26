"""The measurements: the outside view (benchmark, behaviour) and per-head readouts (DLA, attention, geometry).

Every IOI function returns one value per prompt so analyze.py can bootstrap over prompts.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from circuits import N_HEADS, N_LAYERS


# ---------- benchmark: perplexity ----------

def load_ppl_windows(tokenizer, cfg) -> torch.Tensor:
    """First cfg.ppl_tokens tokens of WikiText-2 test (Salesforce/wikitext, wikitext-2-raw-v1), joined with "\\n\\n",
    as non-overlapping [n, cfg.ppl_window] windows. Cached to <data_dir>/ so the download happens once."""
    # cached windows, named by size so --quick (4096 tokens) and full runs don't share a file
    cache_path = Path(cfg.data_dir) / f"wikitext2_test_{cfg.ppl_tokens}_{cfg.ppl_window}.pt"

    # already downloaded before, just load it
    if cache_path.exists():
        return torch.load(cache_path)

    # first run: download the WikiText-2 test split, ["text"] is a list of lines
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    # one long string, lines separated by "\n\n" (standard HF perplexity recipe)
    text = "\n\n".join(dataset["text"])

    # tokenizes the text into one long 1-D tensor of token ids (no truncation, step 5 cuts it)
    # tokenizer gives back a dict, only need "input_ids", [0] drops the batch dim of 1
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    # keep the first ppl_tokens, cut into non-overlapping rows of ppl_window -> [n, window]
    windows = tokens[: cfg.ppl_tokens].reshape(-1, cfg.ppl_window)

    # saves so the next run skips the download + tokenizing
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(windows, cache_path)

    return windows


def window_losses(model, windows, batch_size=4) -> torch.Tensor:
    """Mean next-token loss per window. Perplexity = exp(mean of these)."""
    losses = []

    with torch.no_grad():
        # batches of windows, all 16 at once would be 3+ GB of logits
        for i in range(0, len(windows), batch_size):
            batch = windows[i: i + batch_size].to(model.cfg.device)

            # loss at every position -> [batch, window - 1] (last token has nothing to predict)
            per_token = model(batch, return_type="loss", loss_per_token=True)

            # avg over positions, one loss per window -> [batch]
            losses.append(per_token.mean(dim=1))

    # all windows -> [n]
    return torch.cat(losses)


# ---------- behaviour: IOI ----------

def end_logits(logits, data) -> torch.Tensor:
    """[N, vocab]: each prompt's logits at its own last real token (data.pos["end"]), skipping padding."""
    return logits[torch.arange(len(data)), data.pos["end"]]


def logit_diff(logits, data) -> torch.Tensor:
    """logit(IO) - logit(S) at the last position. Positive = model prefers the right name."""
    end = end_logits(logits, data)
    return end.gather(1, data.io[:, None])[:, 0] - end.gather(1, data.s[:, None])[:, 0]


def ioi_behaviour(logits, data) -> dict:
    """Per prompt: {"logit_diff", "correct" (logit_diff > 0), "top1" (argmax == IO), "io_prob"}."""
    end = end_logits(logits, data)
    ld = logit_diff(logits, data)
    return {
        "logit_diff": ld,
        "correct": (ld > 0).float(),
        "top1": (end.argmax(-1) == data.io).float(),
        "io_prob": end.softmax(-1).gather(1, data.io[:, None])[:, 0],
    }


# ---------- behaviour: induction ----------

def next_token_logprobs(logits, tokens) -> torch.Tensor:
    """[N, 2T]: log-prob that position p assigns to token p+1."""
    return logits[:, :-1].log_softmax(-1).gather(-1, tokens[:, 1:, None])[..., 0]


def induction_logprob(logits, ind) -> dict:
    """Mean log-prob of the clean next token on the repeated half ("lp_second") and the first half ("lp_first")."""
    T = ind.half_len
    lp = next_token_logprobs(logits, ind.tokens)
    # Sequence is [BOS, A (1..T), A (T+1..2T)]. Position T+1 onward can copy; position T can't (the repeat
    # hasn't started yet), and position 0 (BOS -> first token) is a blind guess, so both are left out.
    return {"lp_second": lp[:, T + 1: 2 * T].mean(1), "lp_first": lp[:, 1:T].mean(1)}


# ---------- per-head readouts ----------

def head_outputs_at(model, cache, positions) -> torch.Tensor:
    """What each head writes into the residual stream at one position per prompt: [L, H, N, d_model] (z @ W_O)."""
    outputs = []
    for layer, block in enumerate(model.blocks):
        z = cache["z", layer]
        rows = torch.arange(len(positions), device=z.device)
        end_z = z[rows, positions.to(z.device)]
        outputs.append(torch.einsum("nhd,hdm->hnm", end_z, block.attn.W_O))
    return torch.stack(outputs)


def direct_logit_attribution(model, cache, data, head_out=None) -> torch.Tensor:
    """[L, H, N]: how much each head pushes logit(IO) - logit(S).

    Uses the final LayerNorm scale at each prompt's end and the IO-minus-S unembedding direction.
    """
    if head_out is None:
        head_out = head_outputs_at(model, cache, data.pos["end"])

    scale = cache["ln_final.hook_scale"]
    rows = torch.arange(len(data), device=scale.device)
    end_scale = scale[rows, data.pos["end"].to(scale.device), 0]

    unembed = model.W_U
    direction = unembed[:, data.io.to(unembed.device)] - unembed[:, data.s.to(unembed.device)]
    # Centering the direction accounts for LayerNorm's mean subtraction on each head output.
    direction = direction - direction.mean(dim=0, keepdim=True)
    return torch.einsum("lhnm,mn->lhn", head_out, direction) / end_scale


def attention_to(cache, data, key: str) -> torch.Tensor:
    """[L, H]: mean attention from the last position to data.pos[key] (e.g. name movers -> IO)."""
    attention = []
    for layer in range(cache.model.cfg.n_layers):
        pattern = cache["pattern", layer].permute(0, 2, 3, 1)  # [prompt, query, key, head]
        rows = torch.arange(len(data), device=pattern.device)
        query = data.pos["end"].to(pattern.device)
        target = data.pos[key].to(pattern.device)
        attention.append(pattern[rows, query, target].mean(dim=0))
    return torch.stack(attention)


def class_sums(dla: torch.Tensor, circuit: dict) -> dict:
    """Per-prompt total DLA of each class of the circuit: {class: [N]}."""
    totals = {}
    for name, heads in circuit.items():
        indices = torch.tensor(heads, dtype=torch.long, device=dla.device).reshape(-1, 2)
        totals[name] = dla[indices[:, 0], indices[:, 1]].sum(dim=0)
    return totals


def cosine_per_head(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[L, H]: mean over prompts of cos(head output in model a, head output in model b)."""
    return F.cosine_similarity(a, b, dim=-1).mean(dim=-1)


# ---------- circuit-level: mean ablation and faithfulness ----------

def template_means(model, data, tokens) -> dict:
    """Per-layer hook_z averaged over prompts of the same template: {layer: [N, T, H, d_head]}.

    Call it on data.abc so the mean carries no IOI signal.
    """
    with torch.no_grad():
        # only keep hook_z (each head's output before W_O), the rest of the cache isn't needed
        _, cache = model.run_with_cache(tokens.to(model.cfg.device),
                                        names_filter=lambda name: name.endswith("attn.hook_z"))

    means = {}
    for layer in range(model.cfg.n_layers):
        z = cache["z", layer]  # [N, T, H, d_head]
        mean = torch.empty_like(z)

        # prompts of one template share every position, so averaging over them lines up token for token
        for t in data.template.unique():
            rows = (data.template == t).to(z.device)
            mean[rows] = z[rows].mean(dim=0, keepdim=True)  # one average, copied back to every prompt of t

        means[layer] = mean

    return means


def keep_mask(data, circuit: dict, keep_position: dict, seq_len: int) -> dict:
    """{layer: [N, T, H] bool}: True where a head keeps its real output; everything else gets the ABC mean.

    A circuit head is kept only at its class's KEEP_POSITION; every non-circuit head is ablated everywhere.
    """
    n = len(data)
    rows = torch.arange(n)

    # start with every head ablated at every position
    keep = {layer: torch.zeros(n, seq_len, N_HEADS, dtype=torch.bool) for layer in range(N_LAYERS)}

    # turn each circuit head back on, but only at the one position where its class does its job
    for cls, heads in circuit.items():
        positions = data.pos[keep_position[cls]]  # [N], one position per prompt
        for layer, head in heads:
            keep[layer][rows, positions, head] = True

    return keep


def mean_ablation_hooks(keep: dict, means: dict) -> list:
    """TransformerLens fwd_hooks on blocks.{l}.attn.hook_z: z where keep is True, the template mean elsewhere."""

    def make_hook(layer):
        def hook(z, hook):
            # [N, T, H] -> [N, T, H, 1] so it broadcasts over d_head
            k = keep[layer].to(z.device)[..., None]
            return torch.where(k, z, means[layer].to(z.device))

        return hook

    return [(f"blocks.{layer}.attn.hook_z", make_hook(layer)) for layer in keep]


def ablate_heads_mask(data, heads: list, seq_len: int) -> dict:
    """Keep everything except `heads`, which are mean-ablated at every position."""
    n = len(data)

    # start with every head kept at every position
    keep = {layer: torch.ones(n, seq_len, N_HEADS, dtype=torch.bool) for layer in range(N_LAYERS)}

    # switch off the given heads everywhere
    for layer, head in heads:
        keep[layer][:, :, head] = False

    return keep
