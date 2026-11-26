"""Every setting that changes between runs lives here, so no other file hardcodes one."""

import argparse
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    device: str = "cpu"  # CPU keeps runs deterministic; GPT-2 small is fast enough
    seed: int = 0

    # Test questions
    n_ioi_per_template: int = 8  # x 15 templates = 120 IOI prompts
    n_induction: int = 32  # repeated random-token sequences
    induction_half_len: int = 50
    ppl_tokens: int = 16_384  # WikiText-2 test tokens for perplexity
    ppl_window: int = 1024

    # The sweep. FP32 is always run once as the reference; 16 bits is the null (should match FP32).
    bits: tuple = (16, 8, 6, 5, 4, 3)
    controls: tuple = ("rtn", "noise", "signflip")  # run at scope "all" only
    scopes: tuple = ("circuit:ioi", "rest:ioi", "random_heads", "circuit:induction", "rest:induction")

    # Induction circuit selection on the FP model. Decided in #34 from the scores on FP32 GPT-2 (32 sequences,
    # half_len 50):
    #   induction:      5.5 .94, 6.9 .92, 5.1 .91, 7.10 .91, 7.2 .83 | 10.1 .52, 9.6 .52, 9.9 .49, 10.7 .47, ...
    #                   0.7 sits in the clear gap (the late hits are IOI name movers); the same 5 heads are picked
    #                   anywhere in 0.6-0.8.
    #   previous token: 4.11 .99 | 3.7 .54, 6.8 .51, 2.2 .49, 5.6 .42, 3.2 .41 | 2.9 .37, 3.8 .36, 2.5 .35, ...
    #                   No clear gap below 4.11. We keep 0.4 (6 heads: 2.2, 3.2, 3.7, 4.11, 5.6, 6.8) because it
    #                   gives the 6 previous-token heads PREDICTIONS.md recorded before the pilot; moving the cutoff
    #                   after seeing the scores would be tuning the circuit H2 is about. The pick changes inside
    #                   0.3-0.5, so PREDICTIONS.md also reports the induction results with 4.11 alone (0.7, the
    #                   only clear gap) as a sensitivity check.
    induction_score_threshold: float = 0.7
    prev_token_score_threshold: float = 0.4

    results_dir: str = "results"
    data_dir: str = "data"


QUICK = Config(n_ioi_per_template=2, n_induction=8, ppl_tokens=4096, bits=(8, 4))


def from_args(description: str) -> Config:
    """Parse --quick (use QUICK, results to results_quick/) and --results-dir. Returns the Config to run with."""
    raise NotImplementedError
