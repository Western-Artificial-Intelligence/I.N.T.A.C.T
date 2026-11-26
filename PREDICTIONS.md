# Predictions — pre-registered before the full sweep

**Date:** 2026-11-26
**Status:** FINAL. Pre-registered before the full 49-setting run (#12). Do not edit anything above "Deviations" after
this merges. Later changes go in the dated **Deviations** section at the bottom, with the reason. The original pilot
record (2026-10-22) is kept unchanged at the end of this file.

## What has been seen before this was written

Its own header requires the final version to say what the pilot showed. These are all the compressed-model results
anyone on the team has seen so far. (The full pilot results live in `../I.N.T.A.C.T_pilot_backup_2026-11-05/`; whoever
holds them should check this list before merging and add anything missing.)

- **Pilot perplexity, WikiText-2, 16,384 tokens, scope "all", RTN** (reproduced exactly in #22): FP32 30.81,
  16-bit 30.81, 8-bit 30.80, 4-bit 48.12, 3-bit 2131.94. Normalized (FP32 / ppl): 1.00, 1.00, 1.00, **0.64**, 0.01.
  So the benchmark is already below 0.9 at 4 bits. 6- and 5-bit were not reported here.
- **Pilot, normal GPT-2:** IOI name gap 3.57, 99% right; the circuit alone keeps 85% of the name gap (89% in #31).
- **Pilot (per ticket #16):** rounding only the circuit scored better at 3 bits than at 4. This is unexplained, and
  #16 exists to test whether it survives.
- **Quick check in #33** (4 bits, 4,096 tokens, perplexity only): FP32 28.2; all 43.0; noise 47.3; signflip 41.3;
  circuit:ioi 28.8; rest:ioi 40.6; random_heads 28.7; circuit:induction 28.5; rest:induction 41.1.
- **No IOI, induction or circuit-level metric has been seen on any rounded model** in this codebase.

## What changed since the pilot

- **Copying circuit fixed** (#8, #34): copying heads 5.1, 5.5, 6.9, 7.2, 7.10 (cutoff 0.7, the same 5 at 0.6–0.8);
  previous-token heads 2.2, 3.2, 3.7, 4.11, 5.6, 6.8 (cutoff 0.4). Selected once on FP32 and frozen to
  `results/induction_heads.json`.
- **The IOI circuit list was checked against the paper** (#35): no changes.
- **49 settings, not the outline's 42.** FP32, plus 6 bit sizes × (3 controls at scope "all" + 5 other scopes).
- **The baseline is FP32, not FP16** as the outline's vocabulary table says. Everything is normalized to FP32.
- The outline describes the induction circuit as "2 heads". What we actually measure is the 11 heads above (already
  noted in the pilot record).

## Definitions used by every test

- **Normalized metric** = metric / FP32 metric (1.0 = unchanged), with a 95% paired-bootstrap CI over prompts
  (`analyze.normalized_table`). Perplexity is normalized as FP32 ppl / ppl, so lower = worse like everything else.
- **Levels** (`analyze.LEVEL`): *benchmark* = perplexity. *Behaviour* = IOI logit diff, IOI accuracy, induction gain.
  *Circuit* = IOI faithfulness, IOI circuit patching, name-mover direct effect, induction circuit patching.
- **Broken at threshold t:** the normalized value is below t. **Clearly broken:** the whole CI is below t (upper end
  below t). **Intact:** the value is at least t. The main threshold is **t = 0.9**. Results at 0.8 and 0.95 are
  reported as sensitivity checks, but the conclusions are drawn at 0.9.
- **Crossing bits:** going down the bit sizes 16, 8, 6, 5, 4, 3, the first one where a metric is broken
  (`analyze.crossing`).
- Unless a test says otherwise, it uses scope "all" with real rounding (RTN).

## Pre-registered tests

### H1: Decoupling
*There exists a precision where the mechanism is measurably broken and the benchmark is not.*

- **Supported** if at some bit size, at least one circuit-level metric is *clearly broken* at 0.9 while perplexity
  is *intact* at 0.9.
- **Weakly supported** if this holds only on point estimates (a circuit metric below 0.9 but its CI reaching 0.9).
- **Not supported** otherwise.
- Also reported, not used for the verdict: the same test with behaviour-level metrics in place of circuit-level
  ones, and every bit size where the condition holds.
- Note: the pilot puts perplexity at 0.64 by 4 bits, so H1 will be decided at 8, 6 or 5 bits.

### H2: Structure predicts fragility
*The tightly composed induction circuit degrades before the IOI circuit.*

- **Main comparison:** induction gain vs IOI logit diff, as `analyze.summarize` reports it.
  - **Supported** if induction gain crosses 0.9 at a strictly higher bit size than IOI logit diff. If both cross
    first at the same bit size, it is supported only if induction gain is lower there and their CIs don't overlap.
  - **Contradicted** by the mirror image: IOI crosses first, or at the same bit size IOI is lower with
    non-overlapping CIs.
  - **Inconclusive** otherwise, including when neither crosses.
- **Second comparison, reported alongside:** the same test with induction circuit patching vs IOI circuit patching.
- With only 6 bit sizes, ties are likely; the tie-break above is fixed now so it can't be chosen after seeing the
  results.

### H3: Local vs inherited damage
*Damage comes from rounding the circuit's own weights, not from noise accumulated elsewhere.*

The main test is at **4 bits**, with 3 bits reported as a second test. For each scope, the loss L = 1 − normalized
IOI logit diff, with its CI.

- **Local damage supported** if both hold:
  - L(circuit:ioi) > L(random_heads), with non-overlapping CIs (the circuit matters more than 26 random heads), and
  - L(circuit:ioi) > L(rest:ioi).
- **Inherited damage supported** if L(rest:ioi) > L(circuit:ioi), with non-overlapping CIs.
- **Mixed** otherwise.
- Also reported:
  - the interaction L(all) − L(circuit:ioi) − L(rest:ioi);
  - the same comparison for induction gain with circuit:induction and rest:induction. There is no size-matched
    random control for the induction circuit; random_heads is matched to the 26 IOI heads.

## Sensitivity checks fixed now

- **Copying circuit with 4.11 as the only previous-token head** (cutoff 0.7, the only clear gap in the scores; see
  #34). The induction scopes are rerun with it. If any H2 or H3 conclusion about the copying circuit changes, both
  are reported and the 0.4 result stays the main one.
- **Thresholds 0.8 and 0.95** for every crossing, reported next to 0.9.
- **Repeated rounding** (#16, P1): a conclusion counts as robust only if it holds in at least 4 of the 5 rounding
  runs, and in at least 4 of the 5 fresh sentence sets. Otherwise it is reported as fragile.
- **Controls** (noise and signflip at scope "all") are described in the results, not tested: is the damage specific
  to rounding, or does any change of the same size do it?

## Not predicted, only described

Backup name-mover compensation (direct effect by class at each bit size), geometry vs causality (cosine vs direct
effect ratio per head), and where the patching effect moves outside the circuit.

## Deviations

*(None yet. Add dated entries here with the reason; never edit the sections above.)*

---

# Original pilot record (2026-10-22), unchanged

*Original title: "Predictions — written before any compressed result was seen"*

**Date:** 2026-10-22
**Status:** PILOT. These are the team's hypotheses from the project outline, copied word for word, recorded before
the first pilot sweep. Only the INT16 null run (which must match FP32) had been run at this point. The pilot results
will be seen before the real pre-registration, so the real one must say what the pilot showed.

## H1: Decoupling
There exists a precision where the mechanism is measurably broken and the benchmark is not. Measured at three
levels: (1) general task loss (perplexity), (2) task behaviour (IOI accuracy / logit difference), (3) circuit-level
causal effect (patching the circuit's heads, compared with the full-precision model).

## H2: Structure predicts fragility
The tightly composed induction circuit (2 heads, no known redundancy) degrades before the IOI circuit (26 heads,
7 functional classes, documented backup components).

*Note recorded at the same time:* in GPT-2 small the induction circuit we actually measure is 5 induction heads
(5.1, 5.5, 6.9, 7.2, 7.10) plus 6 previous-token heads, selected by score on the FP model. It is not 2 heads.

## H3: Local vs inherited damage
Damage comes from quantization error in the circuit's own weight matrices, not from accumulated noise elsewhere in
the network.

## Settings fixed before the pilot
- Weight-only, symmetric, per-output-channel RTN on block linears (matches Duan 2606.03002).
- Bits: 16 (null), 8, 6, 5, 4, 3. Scopes: all, circuit:ioi, rest:ioi, random_heads (26, size-matched),
  circuit:induction, rest:induction. Controls at scope "all": rtn, noise, signflip.
- "Broken" = normalized metric below 0.9 of FP32 (0.8 and 0.95 reported as sensitivity).
