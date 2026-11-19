"""analyze.py on made-up sweep output, written in the format run_sweep.write_condition promises. No model needed."""

import json

import numpy as np
import pandas as pd
import pytest

import analyze

N = {"ppl": 16, "ioi": 120, "induction": 32}  # prompts per task, as in the real sweep
CENTRE = {("ppl", "window_loss"): 3.4, ("ioi", "logit_diff"): 3.0, ("ioi", "correct"): 1.0, ("ioi", "ld_faith"): 2.5,
          ("ioi", "ld_corrupt"): -3.0, ("ioi", "ld_circuit_patch"): 2.0, ("ioi", "dla:name mover"): 2.0,
          ("induction", "lp_second"): -0.5, ("induction", "lp_first"): -11.0, ("induction", "lp_corrupt"): -10.0,
          ("induction", "lp_circuit_patch"): -3.0}


def write(results, name, prompts, meta=None, shuffle=False, finished=True):
    """One condition folder: prompts.csv (task, metric, idx, value), heads.csv, and summary.json if finished."""
    d = results / "raw" / name
    d.mkdir(parents=True)
    rows = pd.DataFrame([(t, m, i, v) for (t, m), x in prompts.items() for i, v in enumerate(x)],
                        columns=["task", "metric", "idx", "value"])
    if shuffle:
        rows = rows.sample(frac=1, random_state=0)
    rows.to_csv(d / "prompts.csv", index=False)
    pd.DataFrame([("dla", layer, head, layer + head / 100) for layer in range(12) for head in range(12)],
                 columns=["metric", "layer", "head", "value"]).to_csv(d / "heads.csv", index=False)
    if finished:
        (d / "summary.json").write_text(json.dumps(meta or {}))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """fp32, a 16-bit run that is FP32 plus tiny noise, weaker 8- and 4-bit runs, two extra 4-bit conditions, and
    one unfinished folder."""
    out = tmp_path_factory.mktemp("results")
    rng = np.random.default_rng(0)
    fp = {k: c + rng.normal(0, 0.3, N[k[0]]) for k, c in CENTRE.items()}
    fp[("ioi", "correct")] = (rng.random(N["ioi"]) < 0.95).astype(float)

    def ioi_scaled(f):  # every IOI number shrinks by f; perplexity and induction stay as they are
        return {k: v * f if k[0] == "ioi" else v for k, v in fp.items()}

    write(out, "fp32", fp)
    write(out, "b16_all_rtn", {k: v + rng.normal(0, 1e-4, len(v)) for k, v in fp.items()},
          meta={"bits": 16, "scope": "all", "control": "rtn"})
    write(out, "b8_all_rtn", ioi_scaled(0.93), shuffle=True)
    write(out, "b4_all_rtn", ioi_scaled(0.5))
    write(out, "b4_circuit-ioi_rtn", ioi_scaled(0.8))
    write(out, "b4_random_heads_rtn", fp)
    write(out, "b3_all_rtn", fp, finished=False)
    return out


@pytest.fixture(scope="module")
def conds(results):
    return analyze.load(results)


@pytest.fixture(scope="module")
def table(conds):
    return analyze.normalized_table(conds)


def test_load_reads_finished_conditions_only(conds):
    """A folder without summary.json is skipped; per-prompt arrays come back in idx order whatever the row order."""
    assert "b3_all_rtn" not in conds and len(conds) == 6
    assert conds["fp32"]["H"]["dla"].shape == (12, 12) and conds["fp32"]["H"]["dla"][3, 5] == 3.05
    np.testing.assert_allclose(conds["b8_all_rtn"]["P"][("ioi", "logit_diff")],
                               conds["fp32"]["P"][("ioi", "logit_diff")] * 0.93)


def test_fp32_reads_one_everywhere(table):
    """The normal model is exactly 1.00 on every metric, error bars included."""
    fp = table[table["condition"] == "fp32"]
    assert set(fp["metric"]) == set(analyze.LEVEL) == set(analyze.METRICS)
    assert (fp[["value", "lo", "hi"]] == 1.0).all().all()


def test_near_identical_model_reads_within_half_a_percent(table):
    """A run that matches FP32 up to tiny noise (what 16 bits should be) reads within 0.5% of 1."""
    b16 = table[table["condition"] == "b16_all_rtn"]
    assert ((b16[["value", "lo", "hi"]] - 1).abs() < 0.005).all().all()


def test_bootstrap_is_paired_and_seeded(conds, table):
    """Halving every prompt's value reads 0.5 with no error bar, which only holds if the condition and FP32 are
    resampled with the same prompts. lo <= value <= hi everywhere, and a rerun gives the same table."""
    ld = table[(table["condition"] == "b4_all_rtn") & (table["metric"] == "IOI logit diff")].iloc[0]
    np.testing.assert_allclose([ld["value"], ld["lo"], ld["hi"]], 0.5)
    assert (table["lo"] <= table["value"] + 1e-12).all() and (table["value"] <= table["hi"] + 1e-12).all()
    pd.testing.assert_frame_equal(table, analyze.normalized_table(conds))


def test_condition_names_become_bits_scope_control(table):
    """Settings come from summary.json when saved there, otherwise from the folder name."""
    got = {r.condition: (r.bits, r.scope, r.control)
           for r in table.drop_duplicates("condition").itertuples()}
    assert got == {"fp32": (32, "all", "rtn"), "b16_all_rtn": (16, "all", "rtn"), "b8_all_rtn": (8, "all", "rtn"),
                   "b4_all_rtn": (4, "all", "rtn"), "b4_circuit-ioi_rtn": (4, "circuit:ioi", "rtn"),
                   "b4_random_heads_rtn": (4, "random_heads", "rtn")}


def test_series_and_crossing(table):
    """series is FP32 first then falling bits for one scope; crossing is the first bits below the threshold."""
    s = analyze.series(table, "IOI logit diff")
    assert list(s["condition"]) == ["fp32", "b16_all_rtn", "b8_all_rtn", "b4_all_rtn"]
    assert list(analyze.series(table, "IOI logit diff", scope="circuit:ioi")["condition"]) == ["fp32",
                                                                                               "b4_circuit-ioi_rtn"]
    assert analyze.crossing(s, 0.95) == 8 and analyze.crossing(s, 0.9) == 4 and analyze.crossing(s, 0.1) is None
    assert analyze.crossing(s, 0.9, use_ci=True) == 4
    assert analyze.crossing(analyze.series(table, "perplexity"), 0.9) is None


def test_table_is_saved_and_read_back(table, results):
    """save_table writes normalized.csv with every row and column."""
    path = analyze.save_table(table, results)
    assert path == results / "normalized.csv"
    pd.testing.assert_frame_equal(pd.read_csv(path), table)
    assert analyze.bit_label(32) == "FP32" and analyze.bit_label(4) == "INT4"
