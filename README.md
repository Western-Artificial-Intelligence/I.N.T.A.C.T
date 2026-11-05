# I.N.T.A.C.T. — skeleton

Do the circuits interpretability found in GPT-2 small (IOI, induction) still do their job after the weights are
rounded to fewer bits? Full plan: `I.N.T.A.C.T (1).pdf`.

This is a **skeleton**: every file, function name, input/output and spec is in place, and every body is
`raise NotImplementedError`. Each function's docstring is its ticket spec. The tests in `tests/` say what "done"
means. A working version of everything (the 2026-10-22 pilot, with its results) is kept outside this folder in
`../I.N.T.A.C.T_pilot_backup_2026-11-05/` for reference.

## Setup (once)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run, in this order (once built)

```bash
.venv/bin/python -m pytest -q        # 21 unit tests: all fail until their functions are built
.venv/bin/python checks.py           # the gate: must print 15/15 PASS before anything else
.venv/bin/python run_sweep.py        # 49 conditions -> results/raw/ (resumable)
.venv/bin/python analyze.py          # tables + figures -> results/summary.txt, results/figures/
```

Add `--quick` to any of the last three for a tiny smoke test (writes to `results_quick/`).

## Files, in build order

| File | Job | Done when |
|---|---|---|
| `config.py` | every setting that changes between runs | `from_args` works |
| `tasks.py` | IOI prompts + flipped / ABC twins, repeated-token sequences, all from a seed | `tests/test_tasks.py` passes |
| `circuits.py` | the watch list: 26 IOI heads by class; induction heads selected by score on FP32 | used by quantize / patching tests |
| `quantize.py` | the rounder: weight-only per-channel RTN, scopes, noise controls | `tests/test_quantize.py` passes |
| `metrics.py` | perplexity, logit diff, induction log-prob, per-head direct effect, attention, faithfulness, geometry | patching tests pass |
| `patching.py` | activation patching: circuit sets and the 144-head map | `tests/test_patching.py` passes |
| `run_sweep.py` | the loop, one fresh model per condition | `--quick` writes results_quick/raw/ |
| `checks.py` | reproduce published numbers, prove the pipeline invents no drift | 15/15 PASS |
| `analyze.py` | normalize to FP32, bootstrap CIs, threshold crossings, figures, hypothesis summary | figures + summary.txt |
| `PREDICTIONS.md` | hypotheses, dated before the first compressed run | already written |
