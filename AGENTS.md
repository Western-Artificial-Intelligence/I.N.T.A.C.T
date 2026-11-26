# AGENTS.md — I.N.T.A.C.T.

Notes for anyone (human or AI agent) picking up work in this repo. The README covers setup and the run order; this
file covers what the project is, where things live, the rules that are easy to break, and what's left.

## What the project is

Do the circuits that interpretability research found in GPT-2 small still do their job after the weights are
rounded to fewer bits? We round GPT-2's weights (weight-only, symmetric, per-output-channel round-to-nearest, at
16/8/6/5/4/3 bits) and measure at three levels:

1. **Benchmark**: perplexity on WikiText-2 (normal GPT-2 ≈ 30.8).
2. **Behaviour**: the indirect-object (IOI) name puzzle ("When Mary and John went to the store, John gave a drink
   to ___") and induction (copying a repeated random sequence).
3. **Mechanism**: is the known circuit still causing the answer? Measured with per-head direct effect, attention,
   mean ablation (circuit alone) and activation patching.

The three hypotheses (in `PREDICTIONS.md`, written before any compressed result was seen):
- **H1 Decoupling**: some precision breaks the mechanism while the benchmark still looks fine.
- **H2 Structure predicts fragility**: the induction circuit degrades before the redundant 26-head IOI circuit.
- **H3 Local vs inherited damage**: damage comes from rounding the circuit's own weights, not from noise elsewhere.

"Broken" = normalized metric below 0.9 of FP32 (0.8 and 0.95 also reported).

## Where things are

| Path | What it is |
|---|---|
| `I.N.T.A.C.T (1).pdf` | The full project plan. |
| `PREDICTIONS.md` | The hypotheses and the fixed settings. **Never edit it after seeing results.** |
| `README.md` | Setup (Python 3.12, pinned `requirements.txt`), the run order, and a table of files in build order. |
| `config.py` | Every setting that changes between runs. No other file hardcodes one. `QUICK` = the tiny smoke-test config. |
| `tasks.py` | IOI prompts plus their `flipped` and `abc` twins; induction sequences `[BOS, A, A]` / `[BOS, A, B]`. All seeded. |
| `circuits.py` | `IOI_CIRCUIT` (26 heads in 7 classes), `KEEP_POSITION`, `random_heads`, induction-head scoring, selection and save/load. Head numbers live here and nowhere else. |
| `quantize.py` | The rounder: `rtn`, `quantize_hf`, scopes, noise/signflip controls, `load_tl`, `build_model`. |
| `metrics.py` | Perplexity, logit diff, induction log-prob, per-head direct logit attribution (DLA), attention, cosine, mean ablation. |
| `patching.py` | Activation patching: circuit sets and the 144-head map (a faster version of TransformerLens's). |
| `run_sweep.py` | The main loop: one fresh model per condition, results go to `results/raw/<condition>/`. |
| `checks.py` | The gate: 15 PASS/FAIL checks that must all pass before the full sweep. |
| `analyze.py` | Normalizes to FP32, paired bootstrap error bars, threshold crossings, figures, `summary.txt`. |
| `tests/` | pytest. `conftest.py` holds the shared fixtures (`tok`, `base_hf`, `fp_model`, `data`, `ind`). `test_analyze.py` runs on made-up results and needs no model. |
| `data/` (git-ignored) | Downloaded WikiText-2 windows, cached on first use. |
| `results/`, `results_quick/` | Sweep output (`raw/`), `normalized.csv`, `figures/`, `summary.txt`. |
| `../I.N.T.A.C.T_pilot_backup_2026-11-05/` | A complete working version from the 2026-10-22 pilot, with its results. Use it as a reference for expected numbers, not as code to copy wholesale. |
| `tickets.tsv` (untracked, local) | The ticket sheet the GitHub issues were made from (acceptance criteria, notes, dependencies). |
| GitHub issues #1–#20 | **The source of truth for tickets.** Each has acceptance criteria and a dependency checklist. |

**Each function's docstring is its spec**, and the tests in `tests/` define "done". Unbuilt functions
`raise NotImplementedError`.

## Commands

```bash
.venv/Scripts/python -m pytest -q          # Windows (Mac/Linux: .venv/bin/python)
.venv/Scripts/python checks.py [--quick]   # must print 15/15 PASS before the full sweep
.venv/Scripts/python run_sweep.py [--quick]
.venv/Scripts/python analyze.py [--quick]
```

## Rules that are easy to break

**Rounding**
- Round the **Hugging Face** weights first, *then* load them into TransformerLens (`load_tl`). TransformerLens
  folds LayerNorm into the weights and centres them on load, so rounding after loading rounds the wrong numbers.
- HF Conv1D weights are `[in, out]`, so one channel = one column.
- Only the four block linears are rounded (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`). Embeddings,
  LayerNorms and biases are never touched.
- For partial scopes, compute the scale from the **whole matrix first**, then use a mask to choose which entries
  take the rounded value. That way "circuit only" + "everything but the circuit" split exactly the same numbers as
  "all".
- `quantize_hf` never modifies the caller's model (it deep-copies). Write weights with `.data` or under
  `torch.no_grad()`: an in-place `copy_` on a parameter fails when gradients are on.

**Data**
- Names, places and objects must be a single GPT-2 token *with a leading space*. Template blanks already have a
  space before them, so fill them with bare words.
- Twins are edited at the token level, so they line up with the clean prompt token for token.
- Prompts are right-padded with BOS (50256). Read each prompt at `data.pos["end"]`, **never the last column**.
- Induction tokens never include BOS mid-sequence.

**Circuits**
- A head is a `(layer, head)` tuple, e.g. `(9, 6)` = "9.6".
- The induction circuit is picked once on the FP32 model, saved to `results/induction_heads.json`, and every
  condition reuses that frozen list. The pilot picked 5.1, 5.5, 6.9, 7.2 and 7.10 at a cutoff of 0.7. The IOI name
  movers also score about 0.5 on copying, so 0.4 is too low.

**Pipeline contract**
- `run_sweep.measure_condition`'s docstring lists the exact `(task, metric)` result names that `analyze.load`
  reads. Change one side, change the other.
- Condition folders are named like `fp32`, `b4_all_noise`, `b4_circuit-ioi_rtn` (`:` becomes `-`).
- `summary.json` is written **last**. Its existence marks the condition as finished, and reruns skip finished
  conditions.
- Real (non-`--quick`) rounding runs only happen after `checks.py` passes 15/15. Don't loosen a check's threshold
  without writing down why.
- CPU only and seeded throughout, so runs are deterministic.

**Environment**
- Keep the version pins (`numpy<2`, `pyarrow<21`, `datasets<4`, `transformer-lens 2.16.1`). Newer versions clash.

## How work is done

- One ticket per branch and PR, with "Closes #N" in the PR body. Name branches after the ticket (e.g.
  `ticket-7-behaviour-scores`).
- A PR fills in the functions its ticket names and the tests its ticket lists. If something outside the ticket
  is needed, say so in the PR.
- Say in the PR what was checked against real GPT-2 numbers and what wasn't.
- Merge commits (not squash) are the house style.

## Status (as of 2026-11-26; check the GitHub issues for the current picture)

**Done and merged:** #1 IOI sentences, #2 induction sequences, #3 IOI circuit (*still needs a second person to
check all 26 heads against the paper*), #4 rounding, #6 perplexity, #7 behaviour scores, #14 normalized table.

**Open PRs:**
- #29 (Conrad): #8 cutoff-sensitivity report, plus fast head patching for #11.
- #30 (N-Look): #9 per-head DLA, attention and cosine.
- #31 (leoliCYL): #10 mean ablation.

**Not started:**
- #5: partial scopes and the noise/signflip controls (`head_masks`, `scope_heads`).
- #11: the rest of `patching.py`.
- `config.from_args` (part of #12).
- #12: `run_sweep.py`. Its acceptance now includes: 16-bit reads within 0.5% of FP32 in `normalized_table` on
  real output.
- #13: `checks.py`.
- #15: the `analyze.py` figures and `summarize`.

## Next steps

Critical path: **#5 and #11 → #12 main loop (`measure_condition` first) → #13 checks (15/15 PASS) → full 49-condition
sweep (~30 min on an M4 laptop) → #15 charts and summary → answer H1–H3.**

- #15 can be built now against made-up results, the same way `tests/test_analyze.py` works.
- After the first real sweep, confirm that 16-bit reads within 0.5% of FP32. That is the final check on #14.

P1 experiments, once the P0 pipeline works:
- **#16 Round several times, not once**: 5 runs per setting with a seeded nudge to the rounding grid, plus 5 fresh
  sentence sets. This tests whether "circuit-only scores better at 3 bits than at 4" survives.
- **#17 Find which layer causes the damage**: at 4 bits, protect one layer at a time (12 MLP + 12 attention).
- **#18 Harder name puzzles**: more names, names further apart, extra sentences in between.
- **#19 GPTQ**: the real-world rounding method, on the same per-channel grid, tuned on WikiText-2 *train* text
  only, at scope "all" only.
- **#20 Cheap repair**: keep only the guilty layers from #17 at 8 bits and report size against recovered skill.
