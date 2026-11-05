"""The gate (outline steps 4 and 5). Nothing gets compressed until every row says PASS.

    python checks.py [--quick]

Part A: on the uncompressed model, do our measurements reproduce the published IOI results?
Part B: does the pipeline invent drift when nothing changed?
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")

rows = []


def check(name, passed, detail):
    """Record one row in `rows` and print "PASS"/"FAIL", the check name and its detail on one line."""
    raise NotImplementedError


def top(t, k, largest=True):
    """The k largest (or smallest) entries of an [L, H] tensor as [((layer, head), value), ...]."""
    raise NotImplementedError


def max_abs_diff(a: dict, b: dict) -> float:
    """Largest absolute difference across every tensor of two same-keyed dicts."""
    raise NotImplementedError


def main():
    """Run every check, write <results_dir>/checks.json, exit 1 if any failed.

    A. Reproduce the published results on FP32
       - twin prompts line up token for token (flipped differs only at S2; one length per template)
       - IOI logit diff in [3.0, 4.2] (Wang: 3.56); IOI accuracy > 0.9
       - >= 2 of the 3 name movers in the top-3 positive DLA; 10.7 or 11.10 in the top-3 negative DLA
       - faithfulness in [0.70, 1.05] (Wang: ~0.87)
       - self-repair: backup DLA > 1.5x when name movers are ablated; LD drop < half the name movers' direct effect
       - induction: repeated half > 5 nats easier than first half; induction circuit patch recovers > 80%
    B. No invented drift
       - our HF -> TL load == plain HookedTransformer.from_pretrained("gpt2"), max |dlogit| < 1e-4
       - FP vs FP through the whole pipeline: every value within 1e-6; normalized metrics == 1; head cosines == 1
       - INT16 rounding changes every summary metric < 0.5%
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
