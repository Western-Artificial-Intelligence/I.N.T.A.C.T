# Predictions — written before any compressed result was seen

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
