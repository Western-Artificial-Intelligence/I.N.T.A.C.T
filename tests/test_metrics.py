import torch

import metrics as M
from tasks import make_induction, make_ioi


def test_end_logits_reads_each_prompts_own_last_token(data):
    """Prompts are padded to one length, so the answer must be read at pos["end"], not at the last column."""
    n, t = data.tokens.shape
    logits = torch.zeros(n, t, 50257)
    rows = torch.arange(n)
    logits[rows, data.pos["end"], data.io] = 2.0
    logits[rows, data.pos["end"], data.s] = 0.5
    padded = data.pos["end"] < t - 1
    assert padded.any()
    logits[rows[padded], -1, data.s[padded]] = 9.0  # what reading the padded last column would see
    assert torch.allclose(M.logit_diff(logits, data), torch.full((n,), 1.5))
    assert (M.ioi_behaviour(logits, data)["top1"] == 1).all()


def test_full_precision_behaviour(fp_model, tok):
    """Ticket #7 on normal GPT-2: name gap in [3.0, 4.2], right > 90% of the time, repeated half > 5 nats easier."""
    data = make_ioi(tok, 8, seed=0)  # the sweep's 120 prompts
    ioi = M.ioi_behaviour(fp_model(data.tokens), data)
    assert 3.0 <= ioi["logit_diff"].mean() <= 4.2
    assert ioi["correct"].mean() > 0.9

    ind = make_induction(32, 50, seed=0)
    lp = M.induction_logprob(fp_model(ind.tokens), ind)
    assert (lp["lp_second"] - lp["lp_first"]).mean() > 5
