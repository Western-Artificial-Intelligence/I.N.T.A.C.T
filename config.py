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

    # Induction circuit selection on the FP model.
    # TODO: pick these by printing every head's score on FP32 and putting the cut in the clear gap between real
    # induction heads and late copying heads (e.g. the IOI name movers also score on repeated text).
    induction_score_threshold: float = 0.7
    prev_token_score_threshold: float = 0.4

    results_dir: str = "results"
    data_dir: str = "data"


QUICK = Config(n_ioi_per_template=2, n_induction=8, ppl_tokens=4096, bits=(8, 4))


def from_args(description: str) -> Config:
    """Parse --quick (use QUICK, results to results_quick/) and --results-dir. Returns the Config to run with."""
    raise NotImplementedError
