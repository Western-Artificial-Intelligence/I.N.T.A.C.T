"""The watch list: which heads belong to which circuit, and which job each one does.

Head numbers live here and nowhere else. A head is a (layer, head) tuple, e.g. (9, 6) = "9.6".
"""

import json
import random
from pathlib import Path

import torch

# Wang et al. 2022, "Interpretability in the Wild" (arXiv 2211.00593): 26 heads in 7 classes. Second check done
# for #35: every head and class matches the paper's Appendix K table (Figure 20) and ARENA 3.0's IOI `CIRCUIT`.
# Order within a class doesn't matter (heads_of sorts). Induction 5.8 and 5.9 are the paper's "fuzzy" induction
# heads (Appendix C), and duplicate token 0.10 is its "fuzzy" duplicate token head; both are in the 26.
IOI_CIRCUIT = {
    "name mover": [(9, 9), (9, 6), (10, 0)],  # 3 heads
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],  # 8 heads
    "negative name mover": [(10, 7), (11, 10)],  # 2 heads
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],  # 4 heads
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],  # 4 heads
    "duplicate token": [(0, 1), (0, 10), (3, 0)],  # 3 heads
    "previous token": [(2, 2), (4, 11)],  # 2 heads
}

# Where each class does its job (matches ARENA 3.0's IOI `SEQ_POS_TO_KEEP`; checked in #35). When we mean-ablate
# everything outside the circuit, a circuit head keeps its real output only at this position.
KEEP_POSITION = {
    "name mover": "end",
    "backup name mover": "end",
    "negative name mover": "end",
    "s2 inhibition": "end",
    "induction": "S2",
    "duplicate token": "S2",
    "previous token": "S1+1",
}

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


def induction_scores(model, tokens: torch.Tensor, half_len: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def _pick_heads(ind_score_res, prev_token_score_res, ind_cutoff: float, prev_cutoff: float) -> dict:
    """The selection rules on already-computed [L, H] scores. Both lists empty if no head beats ind_cutoff.

    Shared by select_induction_circuit and threshold_sensitivity so both pick heads the exact same way.
    """
    # stores layers and heads which are greater than the threshold
    ind_heads = []

    # filters inductions scores through the threshold and saves the layer, head in ind_heads
    for layer, head in ALL_HEADS:
        if ind_score_res[layer, head] > ind_cutoff:
            ind_heads.append((layer, head))

    # no induction heads means no deepest layer, so no prev-token heads either
    if not ind_heads:
        return {"previous token": [], "induction": []}
    max_layer = max(layer for layer, _ in ind_heads)

    # stores previous-token heads that pass all three checks
    prev_token_heads = []

    for layer, head in ALL_HEADS:
        # checks if it matches three conditions before appending to list
        if (prev_token_score_res[layer, head] > prev_cutoff
                and layer < max_layer
                and (layer, head) not in ind_heads):
            prev_token_heads.append((layer, head))

    return {"previous token": prev_token_heads, "induction": ind_heads}


def select_induction_circuit(model, tokens, cfg) -> dict:
    """{"previous token": [...], "induction": [...]} chosen by score on the FP model.

    Induction heads: induction score > cfg.induction_score_threshold. Previous-token heads: prev-token score >
    cfg.prev_token_score_threshold, in a layer before the deepest induction head, and not already an induction head.
    """
    # gets scores
    ind_score_res, prev_token_score_res = induction_scores(model, tokens, cfg.induction_half_len)

    res = _pick_heads(ind_score_res, prev_token_score_res,
                      cfg.induction_score_threshold, cfg.prev_token_score_threshold)

    if not res["induction"]:
        raise ValueError("no head beat induction_score_threshold, lower it in config.py")

    return res


def _head_name(head) -> str:
    """(7, 10) -> "7.10", for printing."""
    return f"{head[0]}.{head[1]}"


def threshold_sensitivity(model, tokens, cfg, delta: float = 0.1) -> dict:
    """Which heads get picked with each cutoff moved delta below / above its chosen value (ticket #8).

    Induction (copying) cutoff and previous-token cutoff are moved one at a time; the other stays at its chosen value.
    Borderline heads = picked at some cutoffs in the range but not all of them, i.e. the ones a copying result should
    be re-checked with and without. Prints the table and returns it as a JSON-ready dict.
    """
    # same scores the real pick uses
    ind_score_res, prev_token_score_res = induction_scores(model, tokens, cfg.induction_half_len)

    sweeps = [
        ("induction", ind_score_res, cfg.induction_score_threshold),
        ("previous token", prev_token_score_res, cfg.prev_token_score_threshold),
    ]

    report = {"delta": delta}

    for cls, scores, chosen in sweeps:
        picks = {}

        for cutoff in (chosen - delta, chosen, chosen + delta):
            cutoff = round(cutoff, 4)  # 0.7 - 0.1 would otherwise be 0.6000000000000001
            # only move this class's cutoff, keep the other one at its chosen value
            ind_cutoff = cutoff if cls == "induction" else cfg.induction_score_threshold
            prev_cutoff = cutoff if cls == "previous token" else cfg.prev_token_score_threshold
            picks[cutoff] = _pick_heads(ind_score_res, prev_token_score_res, ind_cutoff, prev_cutoff)[cls]

        # in at some cutoff but not at all of them -> borderline
        picked_ever = set().union(*picks.values())
        picked_always = set.intersection(*(set(heads) for heads in picks.values()))
        borderline = sorted(picked_ever - picked_always)

        report[cls] = {
            "chosen_cutoff": chosen,
            "picks": {f"{c:.2f}": [list(h) for h in heads] for c, heads in picks.items()},
            "borderline": [{"head": list(h), "score": round(float(scores[h[0], h[1]]), 4)} for h in borderline],
            "stable": not borderline,
        }

        # prints the table
        print(f"{cls} cutoff (chosen {chosen:.2f}):")
        for c, heads in picks.items():
            print(f"  {c:.2f}: {' '.join(_head_name(h) for h in heads) or '(none)'}")
        if borderline:
            names = ", ".join(f"{_head_name(h)} ({float(scores[h[0], h[1]]):.2f})" for h in borderline)
            print(f"  borderline: {names}")
        else:
            print("  stable: same heads across the whole range")

    return report


def save_induction_circuit(circuit: dict, results_dir: str):
    """Write the circuit to <results_dir>/induction_heads.json so every condition watches the same heads."""
    # creates folder if it does not exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # writes induction heads into the file
    with open(Path(results_dir) / "induction_heads.json", "w", encoding="utf-8") as file:
        json.dump(circuit, file)


def load_induction_circuit(results_dir: str) -> dict:
    """Read <results_dir>/induction_heads.json back into {class: [(layer, head), ...]}."""
    with open(Path(results_dir) / "induction_heads.json", "r", encoding="utf-8") as file:
        circuit = json.load(file)

    # stores the circuit in res
    res = {}
    for cls, heads in circuit.items():
        res[cls] = []
        for h in heads:
            res[cls].append(tuple(h))

    return res


def save_threshold_sensitivity(report: dict, results_dir: str):
    """Write threshold_sensitivity's report to <results_dir>/induction_thresholds.json, next to induction_heads.json."""
    # creates folder if it does not exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(results_dir) / "induction_thresholds.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def random_heads(n: int, exclude: list, seed: int) -> list:
    """Size-matched control: n random heads drawn (seeded) from outside `exclude`."""
    # every head that isn't in the circuit we're comparing against
    excluded = set(exclude)
    candidates = [h for h in ALL_HEADS if h not in excluded]

    # own seeded generator so the pick is the same every run and nothing else can shift it
    rng = random.Random(seed)

    # n distinct heads (no repeats), sorted to match heads_of
    return sorted(rng.sample(candidates, n))
