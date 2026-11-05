"""Turns the raw sweep rows into curves, tables and a hypothesis-by-hypothesis summary.

    python analyze.py [--quick]

Reads only <results_dir>/raw, so the analysis can be redone any number of times without re-running the sweep.
Every number is normalized to FP32 (1.0 = unchanged). CIs are paired bootstraps over prompts (the same resampled
prompts for the condition and for FP32).
"""

import matplotlib

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

# TODO: one function per metric name in LEVEL, f(P, I) -> value. P = per-prompt arrays from one condition;
# I = None for the point estimate, or {task: [B, n] resample indices} for the bootstrap. Formulas:
#   perplexity                  1 / ppl, so the ratio to FP32 reads "fraction of FP32 quality"
#   IOI logit diff / accuracy   mean of ("ioi", "logit_diff") / ("ioi", "correct")
#   IOI faithfulness            mean ld_faith / mean logit_diff
#   IOI circuit patching        (ld_circuit_patch - ld_corrupt) / (logit_diff - ld_corrupt)
#   name-mover direct effect    mean ("ioi", "dla:name mover")
#   induction gain              lp_second - lp_first
#   induction circuit patching  (lp_circuit_patch - lp_corrupt) / (lp_second - lp_corrupt)
METRICS: dict = {}


# ---------- loading ----------

def load(results_dir) -> dict:
    """{condition: {"meta": summary.json, "P": {(task, metric): array}, "H": {metric: [L, H] array}}} for every
    finished condition under <results_dir>/raw."""
    raise NotImplementedError


# ---------- numbers ----------

def normalized_table(conds):
    """Long DataFrame: condition, bits (FP32 = 32), scope, control, metric, level, value, lo, hi.

    value = metric / FP32 metric; lo, hi = 2.5 / 97.5 percentiles of the paired bootstrap (same resampled prompts
    for the condition and FP32; fixed rng seed).
    """
    raise NotImplementedError


def series(df, metric, scope="all", control="rtn"):
    """One metric across precisions, FP32 first, then falling bits."""
    raise NotImplementedError


def crossing(s, thr, use_ci=False):
    """First bits where the value (or the whole CI, if use_ci) drops below thr; None if never."""
    raise NotImplementedError


def bit_label(b) -> str:
    """32 -> "FP32", otherwise "INT<b>"."""
    raise NotImplementedError


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
