"""The measurements: the outside view (benchmark, behaviour) and per-head readouts (DLA, attention, geometry).

Every IOI function returns one value per prompt so analyze.py can bootstrap over prompts.
"""
from pathlib import Path

import torch
from datasets import load_dataset


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
    raise NotImplementedError


def logit_diff(logits, data) -> torch.Tensor:
    """logit(IO) - logit(S) at the last position. Positive = model prefers the right name."""
    raise NotImplementedError


def ioi_behaviour(logits, data) -> dict:
    """Per prompt: {"logit_diff", "correct" (logit_diff > 0), "top1" (argmax == IO), "io_prob"}."""
    raise NotImplementedError


# ---------- behaviour: induction ----------

def next_token_logprobs(logits, tokens) -> torch.Tensor:
    """[N, 2T]: log-prob that position p assigns to token p+1."""
    raise NotImplementedError


def induction_logprob(logits, ind) -> dict:
    """Mean log-prob of the clean next token on the repeated half ("lp_second") and the first half ("lp_first")."""
    raise NotImplementedError


# ---------- per-head readouts ----------

def head_outputs_at(model, cache, positions) -> torch.Tensor:
    """What each head writes into the residual stream at one position per prompt: [L, H, N, d_model] (z @ W_O)."""
    raise NotImplementedError


def direct_logit_attribution(model, cache, data, head_out=None) -> torch.Tensor:
    """[L, H, N]: how much each head pushes logit(IO) - logit(S).

    = head output / ln_final scale at the end position, dotted with W_U[:, IO] - W_U[:, S].
    """
    raise NotImplementedError


def attention_to(cache, data, key: str) -> torch.Tensor:
    """[L, H]: mean attention from the last position to data.pos[key] (e.g. name movers -> IO)."""
    raise NotImplementedError


def class_sums(dla: torch.Tensor, circuit: dict) -> dict:
    """Per-prompt total DLA of each class of the circuit: {class: [N]}."""
    raise NotImplementedError


def cosine_per_head(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[L, H]: mean over prompts of cos(head output in model a, head output in model b)."""
    raise NotImplementedError


# ---------- circuit-level: mean ablation and faithfulness ----------

def template_means(model, data, tokens) -> dict:
    """Per-layer hook_z averaged over prompts of the same template: {layer: [N, T, H, d_head]}.

    Call it on data.abc so the mean carries no IOI signal.
    """
    raise NotImplementedError


def keep_mask(data, circuit: dict, keep_position: dict, seq_len: int) -> dict:
    """{layer: [N, T, H] bool}: True where a head keeps its real output; everything else gets the ABC mean.

    A circuit head is kept only at its class's KEEP_POSITION; every non-circuit head is ablated everywhere.
    """
    raise NotImplementedError


def mean_ablation_hooks(keep: dict, means: dict) -> list:
    """TransformerLens fwd_hooks on blocks.{l}.attn.hook_z: z where keep is True, the template mean elsewhere."""
    raise NotImplementedError


def ablate_heads_mask(data, heads: list, seq_len: int) -> dict:
    """Keep everything except `heads`, which are mean-ablated at every position."""
    raise NotImplementedError
