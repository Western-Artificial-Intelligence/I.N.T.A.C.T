"""The watch list: which heads belong to which circuit, and which job each one does.

Head numbers live here and nowhere else. A head is a (layer, head) tuple, e.g. (9, 6) = "9.6".
"""

import json
from pathlib import Path

import torch

# TODO: fill from Wang et al. 2022, "Interpretability in the Wild" (26 heads in 7 classes; ARENA lists them).
IOI_CIRCUIT = {
    "name mover": [9.9, 9.6, 10.0],  # 3 heads
    "backup name mover": [10.10, 10.6, 10.2, 10.1, 11.2, 9.7, 9.0, 11.9],  # 8 heads
    "negative name mover": [10.7, 11.10],  # 2 heads
    "s2 inhibition": [7.3, 7.9, 8.6, 8.10],  # 4 heads
    "induction": [5.5, 5.8, 5.9, 6.9],  # 4 heads
    "duplicate token": [0.1, 0.10, 3.0],  # 3 heads
    "previous token": [2.2, 4.11],  # 2 heads
}

# TODO: where each class does its job ("end", "S2" or "S1+1"; ARENA / Wang et al. convention). When we mean-ablate
# everything outside the circuit, a circuit head keeps its real output only at this position.
KEEP_POSITION = {cls: None for cls in IOI_CIRCUIT}

N_LAYERS, N_HEADS = 12, 12
ALL_HEADS = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]


def heads_of(circuit: dict) -> list:
    """Flat, sorted, de-duplicated list of every (layer, head) in a circuit dict."""
    res = []  # sorted list of every (layer, head)

    for heads in circuit.values():  # grabs the heads from the circuit dict
        for layer, head in heads:
            if (layer, head) not in res:  # checks for duplicates
                res.append((layer, head))

    res.sort()

    return res


def induction_scores(model, tokens: torch.Tensor, half_len: int) -> (torch.Tensor, torch.Tensor):
    """Per-head induction and previous-token scores on [BOS, A, A] repeated random tokens. Returns two [L, H].

    induction score = mean attention from each position in the second A to the token right after its earlier copy.
    previous-token score = mean attention from each position to the position just before it.
    """

    with torch.no_grad():
        # runs model with cache
        _, cache = model.run_with_cache(  # logits is discarded
            tokens,
            names_filter=lambda name: name.endswith("hook_pattern")
        )  # keeps only the attention patterns

    induction_score = []
    prev_token_score = []

    for layer in range(N_LAYERS):
        pattern = cache["pattern", layer]  # [N, H, Q, K] for layer

        # induction: second A looks (T - 1) back, at the token after its earlier copy
        diag = torch.diagonal(pattern, offset=-(half_len - 1), dim1=-2, dim2=-1)
        # first 2 entries are first-half queries (junk), keep last T, avg over prompts + positions -> [H]
        induction_score.append(diag[..., -half_len:].mean(dim=(0, 2)))

        # prev token: every position looks 1 back, avg over prompts + positions -> [H]
        prev_token_score.append(
            torch.diagonal(pattern, offset=-1, dim1=-2, dim2=-1).mean(dim=(0, 2))
        )

    # stacks the scores into [L, H] tensors
    induction_score_res = torch.stack(induction_score)
    prev_token_score_res = torch.stack(prev_token_score)

    return induction_score_res, prev_token_score_res


def select_induction_circuit(model, tokens, cfg) -> dict:
    """{"previous token": [...], "induction": [...]} chosen by score on the FP model.

    Induction heads: induction score > cfg.induction_score_threshold. Previous-token heads: prev-token score >
    cfg.prev_token_score_threshold, in a layer before the deepest induction head, and not already an induction head.
    """
    # gets scores
    ind_score_res, prev_token_score_res = induction_scores(model, tokens, cfg.induction_half_len)

    # stores layers and heads which are greater than the threshold
    ind_heads = []

    # filters inductions scores through the threshold and saves the layer, head in ind_heads
    for layer, head in ALL_HEADS:
        if ind_score_res[layer, head] > cfg.induction_score_threshold:
            ind_heads.append((layer, head))

    # grabs the deepest layer if ind heads is not empty
    if not ind_heads:
        raise ValueError("no head beat induction_score_threshold, lower it in config.py")
    max_layer = max(layer for layer, _ in ind_heads)

    # stores previous-token heads that pass all three checks
    prev_token_heads = []

    for layer, head in ALL_HEADS:
        # checks if it matches three conditions before appending to list
        if (prev_token_score_res[layer, head] > cfg.prev_token_score_threshold
                and layer < max_layer
                and (layer, head) not in ind_heads):
            prev_token_heads.append((layer, head))

    # stores result in dict
    res = {
        "previous token": prev_token_heads,
        "induction": ind_heads
    }

    return res


def save_induction_circuit(circuit: dict, results_dir: str):
    """Write the circuit to <results_dir>/induction_heads.json so every condition watches the same heads."""
    raise NotImplementedError


def load_induction_circuit(results_dir: str) -> dict:
    """Read <results_dir>/induction_heads.json back into {class: [(layer, head), ...]}."""
    raise NotImplementedError


def random_heads(n: int, exclude: list, seed: int) -> list:
    """Size-matched control: n random heads drawn (seeded) from outside `exclude`."""
    raise NotImplementedError
