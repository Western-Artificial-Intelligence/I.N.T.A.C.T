from types import SimpleNamespace

import torch
from transformer_lens import ActivationCache

import metrics as M
from circuits import IOI_CIRCUIT
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


def test_per_head_readouts_use_each_prompts_positions_and_scale():
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1),
        blocks=[SimpleNamespace(attn=SimpleNamespace(W_O=torch.tensor([[[2.0, 0.0]], [[0.0, 3.0]]])))],
        W_U=torch.tensor([[3.0, 1.0, 0.0], [1.0, 1.0, 2.0]]),
    )
    class Prompts:
        pos = {"end": torch.tensor([0, 1]), "IO": torch.tensor([0, 0])}
        io = torch.tensor([0, 2])
        s = torch.tensor([1, 0])

        def __len__(self):
            return len(self.io)

    data = Prompts()
    pattern = torch.tensor([
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        [[[1.0, 0.0], [0.2, 0.8]], [[1.0, 0.0], [0.6, 0.4]]],
    ])
    cache = ActivationCache({
        "blocks.0.attn.hook_z": torch.tensor([[[[1.0], [2.0]], [[9.0], [9.0]]],
                                              [[[9.0], [9.0]], [[3.0], [4.0]]]]),
        "blocks.0.attn.hook_pattern": pattern,
        "ln_final.hook_scale": torch.tensor([[[2.0], [9.0]], [[9.0], [4.0]]]),
    }, model)

    outputs = M.head_outputs_at(model, cache, data.pos["end"])
    torch.testing.assert_close(outputs, torch.tensor([[[[2.0, 0.0], [6.0, 0.0]], [[0.0, 6.0], [0.0, 12.0]]]]))
    dla = M.direct_logit_attribution(model, cache, data, outputs)
    torch.testing.assert_close(dla, torch.tensor([[[1.0, -3.0], [-3.0, 6.0]]]))
    torch.testing.assert_close(M.direct_logit_attribution(model, cache, data), dla)
    torch.testing.assert_close(M.attention_to(cache, data, "IO"), torch.tensor([[0.6, 0.8]]))
    torch.testing.assert_close(
        M.class_sums(dla, {"both": [(0, 0), (0, 1)], "empty": []}),
        {"both": torch.tensor([-2.0, 3.0]), "empty": torch.zeros(2)},
    )

    changed = outputs.clone()
    changed[0, 0, 1] *= -1
    changed[0, 1] *= 2
    torch.testing.assert_close(M.cosine_per_head(outputs, changed), torch.tensor([[0.0, 1.0]]))


def test_full_precision_head_rankings(fp_model, tok):
    data = make_ioi(tok, 8, seed=0)
    _, cache = fp_model.run_with_cache(
        data.tokens,
        return_type=None,
        names_filter=lambda name: name.endswith(("hook_z", "hook_pattern")) or name == "ln_final.hook_scale",
    )
    dla = M.direct_logit_attribution(fp_model, cache, data)
    assert dla.shape == (12, 12, len(data))
    assert M.attention_to(cache, data, "IO").shape == (12, 12)
    assert M.attention_to(cache, data, "S2").shape == (12, 12)

    scores = dla.mean(dim=-1).flatten()
    top = {divmod(index, 12) for index in scores.topk(3).indices.tolist()}
    bottom = {divmod(index, 12) for index in scores.topk(3, largest=False).indices.tolist()}
    assert len(top.intersection(IOI_CIRCUIT["name mover"])) >= 2, top
    assert bottom.intersection(IOI_CIRCUIT["negative name mover"]), bottom
