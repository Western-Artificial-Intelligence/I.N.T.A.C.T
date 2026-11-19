"""Turns the raw sweep rows into curves, tables and a hypothesis-by-hypothesis summary.

    python analyze.py [--quick]

Reads only <results_dir>/raw, so the analysis can be redone any number of times without re-running the sweep.
Every number is normalized to FP32 (1.0 = unchanged). CIs are paired bootstraps over prompts (the same resampled
prompts for the condition and for FP32).
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

B = 1000  # bootstrap resamples
THRESHOLDS = (0.95, 0.9, 0.8)  # "broken" = normalized metric below 0.9 (0.95 and 0.8 as sensitivity)

# Reference palette (dataviz skill), light mode. Slots are used in order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ef8a7e", "#b8302f"]  # for heatmaps, centred on 0

# Which level of the story each metric belongs to.
LEVEL = {"perplexity": "benchmark", "IOI logit diff": "behaviour", "IOI accuracy": "behaviour",
         "induction gain": "behaviour", "IOI faithfulness": "circuit", "IOI circuit patching": "circuit",
         "name-mover direct effect": "circuit", "induction circuit patching": "circuit"}

# One function per metric name in LEVEL, f(P, I) -> value. P = per-prompt arrays from one condition;
# I = None for the point estimate, or {task: [B, n] resample indices} for the bootstrap. Formulas:
#   perplexity                  1 / ppl, so the ratio to FP32 reads "fraction of FP32 quality"
#   IOI logit diff / accuracy   mean of ("ioi", "logit_diff") / ("ioi", "correct")
#   IOI faithfulness            mean ld_faith / mean logit_diff
#   IOI circuit patching        (ld_circuit_patch - ld_corrupt) / (logit_diff - ld_corrupt)
#   name-mover direct effect    mean ("ioi", "dla:name mover")
#   induction gain              lp_second - lp_first
#   induction circuit patching  (lp_circuit_patch - lp_corrupt) / (lp_second - lp_corrupt)
def _mean(P, I, task, metric):
    """Mean over prompts of one per-prompt array: a float, or [B] means (one per resample) when I is given."""
    x = P[(task, metric)]
    return x.mean() if I is None else x[I[task]].mean(axis=1)


def _recovery(P, I, task, patched, clean, corrupt):
    """Share of the clean-vs-corrupt gap that patching the circuit restores."""
    floor = _mean(P, I, task, corrupt)
    return (_mean(P, I, task, patched) - floor) / (_mean(P, I, task, clean) - floor)


METRICS: dict = {
    "perplexity": lambda P, I: 1 / np.exp(_mean(P, I, "ppl", "window_loss")),
    "IOI logit diff": lambda P, I: _mean(P, I, "ioi", "logit_diff"),
    "IOI accuracy": lambda P, I: _mean(P, I, "ioi", "correct"),
    "induction gain": lambda P, I: _mean(P, I, "induction", "lp_second") - _mean(P, I, "induction", "lp_first"),
    "IOI faithfulness": lambda P, I: _mean(P, I, "ioi", "ld_faith") / _mean(P, I, "ioi", "logit_diff"),
    "IOI circuit patching": lambda P, I: _recovery(P, I, "ioi", "ld_circuit_patch", "logit_diff", "ld_corrupt"),
    "name-mover direct effect": lambda P, I: _mean(P, I, "ioi", "dla:name mover"),
    "induction circuit patching": lambda P, I: _recovery(P, I, "induction", "lp_circuit_patch", "lp_second",
                                                         "lp_corrupt"),
}

FP32 = "fp32"  # the reference condition's folder name


# ---------- loading ----------

def load(results_dir) -> dict:
    """{condition: {"meta": summary.json, "P": {(task, metric): array}, "H": {metric: [L, H] array}}} for every
    finished condition under <results_dir>/raw."""
    conds = {}
    for summary in sorted((Path(results_dir) / "raw").glob("*/summary.json")):  # no summary.json = not finished
        d = summary.parent
        prompts = pd.read_csv(d / "prompts.csv").sort_values("idx")
        P = {key: g["value"].to_numpy(dtype=float) for key, g in prompts.groupby(["task", "metric"])}
        heads = pd.read_csv(d / "heads.csv")
        H = {m: g.pivot(index="layer", columns="head", values="value").to_numpy(dtype=float)
             for m, g in heads.groupby("metric")}
        conds[d.name] = {"meta": json.loads(summary.read_text()), "P": P, "H": H}
    return conds


def _setting(name, meta):
    """(bits, scope, control) of a condition: from its meta if run_sweep saved them, otherwise from its name
    ("b4_circuit-ioi_rtn"). FP32 counts as 32 bits."""
    if name == FP32:
        return 32, "all", "rtn"
    b, *scope, control = name.split("_")  # the scope itself may contain "_" (random_heads)
    return (int(meta.get("bits") or b[1:]), meta.get("scope") or "_".join(scope).replace("-", ":"),
            meta.get("control") or control)


# ---------- numbers ----------

def normalized_table(conds):
    """Long DataFrame: condition, bits (FP32 = 32), scope, control, metric, level, value, lo, hi.

    value = metric / FP32 metric; lo, hi = 2.5 / 97.5 percentiles of the paired bootstrap (same resampled prompts
    for the condition and FP32; fixed rng seed).
    """
    fp = conds[FP32]["P"]
    rng = np.random.default_rng(0)
    # one set of resampled prompt indices per task, shared by every condition and by FP32 (that is the pairing)
    I = {}
    for task, metric in sorted(fp):
        if task not in I:
            n = len(fp[(task, metric)])
            I[task] = rng.integers(0, n, size=(B, n))

    rows = []
    for name, cond in conds.items():
        bits, scope, control = _setting(name, cond["meta"])
        for metric, f in METRICS.items():
            value = f(cond["P"], None) / f(fp, None)
            lo, hi = np.percentile(f(cond["P"], I) / f(fp, I), [2.5, 97.5])
            rows.append({"condition": name, "bits": bits, "scope": scope, "control": control, "metric": metric,
                         "level": LEVEL[metric], "value": value, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    return df.sort_values(["bits", "condition"], ascending=[False, True], kind="stable", ignore_index=True)


def series(df, metric, scope="all", control="rtn"):
    """One metric across precisions, FP32 first, then falling bits."""
    keep = (df["metric"] == metric) & ((df["condition"] == FP32) | ((df["scope"] == scope) & (df["control"] == control)))
    return df[keep].sort_values("bits", ascending=False, ignore_index=True)


def crossing(s, thr, use_ci=False):
    """First bits where the value (or the whole CI, if use_ci) drops below thr; None if never."""
    below = s[s["hi" if use_ci else "value"] < thr]  # the whole CI is below thr when its upper end is
    return int(below["bits"].iloc[0]) if len(below) else None


def save_table(df, results_dir) -> Path:
    """Write the normalized table to <results_dir>/normalized.csv and return that path."""
    path = Path(results_dir) / "normalized.csv"
    df.to_csv(path, index=False)
    return path


def bit_label(b) -> str:
    """32 -> "FP32", otherwise "INT<b>"."""
    return "FP32" if int(b) == 32 else f"INT{int(b)}"


# ---------- figures (all saved to <results_dir>/figures/) ----------

def fig_levels(df, out):
    """fig1_three_levels.png: benchmark vs behaviour vs circuit metrics against bits, IOI and induction panels,
    with a dashed 90% line."""
    raise NotImplementedError


def fig_compensation(conds, out):
    """fig2_compensation.png: direct effect of name movers, backups, negative name movers and total logit diff
    against bits. Question: do the backups take over as precision falls?"""
    raise NotImplementedError


def fig_scopes(df, out):
    """fig3_scopes.png: IOI logit diff and induction gain for all / circuit only / everything except circuit /
    random heads. Question: is the damage local to the circuit or inherited from elsewhere?"""
    raise NotImplementedError


def fig_controls(df, out):
    """fig4_controls.png: rounding vs matched noise vs sign-flipped rounding error, for perplexity, IOI logit diff
    and faithfulness. Question: is it rounding specifically, or any change of the same size?"""
    raise NotImplementedError


def geometry_frame(conds, cond, min_dla=0.1):
    """One row per head with |FP32 DLA| >= min_dla: head, group (name mover / backup / other), cos to FP32,
    DLA ratio quantized / FP32, FP32 DLA."""
    raise NotImplementedError


def fig_geometry(conds, cond, out):
    """fig5_geometry_vs_causality_<cond>.png: scatter of cosine (looks the same?) vs DLA ratio (works the same?).
    Returns the geometry_frame."""
    raise NotImplementedError


def fig_where(conds, cond, out):
    """fig6_where_it_went_<cond>.png: 144-head patching maps for FP32, the condition and their difference, with the
    26 circuit heads boxed. Returns the 5 biggest gains outside the circuit."""
    raise NotImplementedError


# ---------- text summary ----------

def summarize(df, conds, out):
    """Write and print summary.txt:
    - normalized metric table (scope all, RTN) with CIs
    - where each metric first crosses each threshold
    - H1: precisions where a circuit metric < 0.9 while perplexity >= 0.9
    - H2: which breaks first, induction gain or IOI logit diff
    - compensation: direct effect by class at each bits
    - H3: IOI logit-diff loss for all / circuit-only / rest-only / random heads, and all - circuit - rest
    - controls: rtn vs noise vs signflip
    """
    raise NotImplementedError


def main():
    """load -> normalized_table (save normalized.csv) -> fig1-4 -> summarize. Then for each bits <= 5: fig5 plus a
    Spearman(cos, |DLA ratio - 1|) line and the heads that look the same but act differently, and fig6."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
