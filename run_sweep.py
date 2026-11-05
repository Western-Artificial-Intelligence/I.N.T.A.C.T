"""The loop: every condition run identically, on a freshly rounded model, one row per measurement.

    python run_sweep.py            # pilot sweep (~49 conditions)
    python run_sweep.py --quick    # tiny smoke test

Raw results go to <results_dir>/raw/<condition>/ (summary.json, prompts.csv, heads.csv). Finished conditions are
skipped on rerun, so a crash never costs more than one condition.
"""

import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")


def setup(cfg):
    """Seed + deterministic torch + no grad; load GPT-2 tokenizer and HF model; build IOI data, induction data and
    perplexity windows. Returns (tokenizer, base_hf, ioi_data, induction_data, ppl_windows)."""
    raise NotImplementedError


def induction_circuit(fp_model, ind, cfg) -> dict:
    """Selected once on the FP model, then frozen to disk so every condition watches the same heads."""
    raise NotImplementedError


def measure_condition(model, data, ind, windows, ind_circuit, fp_head_out=None, head_map=False):
    """Everything we measure on one model. Returns (summary, prompts, heads, head_out).

    prompts = {(task, metric): [n] per-prompt values}; heads = {metric: [L, H]}. analyze.py reads these exact keys:
      1. Benchmark       ("ppl", "window_loss")
      2. IOI behaviour   ("ioi", "logit_diff" | "correct" | "top1" | "io_prob"), ("ioi", "dla:<class>") per class;
                         heads "dla", "att_io", "att_s2", "out_norm", and "cos_fp" (vs fp_head_out) when given
      3. Patching        clean -> flipped twin: ("ioi", "ld_corrupt" | "ld_circuit_patch" | "ld_rest_patch");
                         heads "patch" (144-head map) only if head_map
      4. Faithfulness    circuit alone, everything else mean-ablated over ABC twins: ("ioi", "ld_faith")
      5. Self-repair     mean-ablate the 3 name movers: ("ioi", "ld_nm_ablated"), ("ioi", "dla_nm_ablated:<class>")
      6. Induction       ("induction", "lp_second" | "lp_first" | "lp_corrupt" | "lp_circuit_patch");
                         heads "induction_score", "prev_token_score"
    summary = {"ppl", "logit_diff", "accuracy", "faithfulness" (ld_faith / logit_diff),
               "circuit_patch_recovery", "induction_lp_second"}. head_out = [L, H, N, d_model] at the end position.
    """
    raise NotImplementedError


def write_condition(out_dir: Path, meta: dict, summary: dict, prompts: dict, heads: dict):
    """Write prompts.csv (task, metric, idx, value), heads.csv (metric, layer, head, value), then summary.json
    (meta + summary). summary.json is written last: its existence is the "this condition is done" marker."""
    raise NotImplementedError


def conditions(cfg):
    """Yield (name, bits, scope, control): "fp32" first; then for each bits, scope "all" x every control, and every
    other scope with "rtn". Names look like "b4_all_noise", "b4_circuit-ioi_rtn" (":" -> "-")."""
    raise NotImplementedError


def main():
    """1. setup; write <results_dir>/config.json.
    2. Build FP32, select/freeze the induction circuit, measure FP32 once to get fp_head_out.
    3. For each condition not already done: build a fresh model, measure (head_map only for scope "all" + rtn),
       write_condition, print one progress line with ppl / LD / faithfulness / patch / induction and seconds.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
