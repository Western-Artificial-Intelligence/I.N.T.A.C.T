"""The watch list: which heads belong to which circuit, and which job each one does.

Head numbers live here and nowhere else. A head is a (layer, head) tuple, e.g. (9, 6) = "9.6".
"""

import json
from pathlib import Path

import torch

# TODO: fill from Wang et al. 2022, "Interpretability in the Wild" (26 heads in 7 classes; ARENA lists them).
IOI_CIRCUIT = {
    "name mover": [],  # 3 heads
    "backup name mover": [],  # 8 heads
    "negative name mover": [],  # 2 heads
    "s2 inhibition": [],  # 4 heads
    "induction": [],  # 4 heads
    "duplicate token": [],  # 3 heads
    "previous token": [],  # 2 heads
}

# TODO: where each class does its job ("end", "S2" or "S1+1"; ARENA / Wang et al. convention). When we mean-ablate
# everything outside the circuit, a circuit head keeps its real output only at this position.
KEEP_POSITION = {cls: None for cls in IOI_CIRCUIT}

N_LAYERS, N_HEADS = 12, 12
ALL_HEADS = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]


def heads_of(circuit: dict) -> list:
    """Flat, sorted, de-duplicated list of every (layer, head) in a circuit dict."""
    raise NotImplementedError


def induction_scores(model, tokens: torch.Tensor, half_len: int):
    """Per-head induction and previous-token scores on [BOS, A, A] repeated random tokens. Returns two [L, H].

    induction score = mean attention from each position in the second A to the token right after its earlier copy.
    previous-token score = mean attention from each position to the position just before it.
    """
    raise NotImplementedError


def select_induction_circuit(model, tokens, cfg) -> dict:
    """{"previous token": [...], "induction": [...]} chosen by score on the FP model.

    Induction heads: induction score > cfg.induction_score_threshold. Previous-token heads: prev-token score >
    cfg.prev_token_score_threshold, in a layer before the deepest induction head, and not already an induction head.
    """
    raise NotImplementedError


def save_induction_circuit(circuit: dict, results_dir: str):
    """Write the circuit to <results_dir>/induction_heads.json so every condition watches the same heads."""
    raise NotImplementedError


def load_induction_circuit(results_dir: str) -> dict:
    """Read <results_dir>/induction_heads.json back into {class: [(layer, head), ...]}."""
    raise NotImplementedError


def random_heads(n: int, exclude: list, seed: int) -> list:
    """Size-matched control: n random heads drawn (seeded) from outside `exclude`."""
    raise NotImplementedError
