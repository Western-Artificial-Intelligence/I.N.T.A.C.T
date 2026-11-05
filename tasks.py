"""The test questions: IOI sentences (with their twins) and repeated-token sequences, all generated from a seed.

Every IOI prompt comes with two near-identical twins:
  flipped: the second mention of S is replaced by IO, so the right answer flips. Used for activation patching.
  abc:     all three name slots get three fresh distinct names, so there is no IOI signal. Used for mean-ablation.
Names, places and objects are single GPT-2 tokens, so a prompt and its twins line up token for token.
"""

from dataclasses import dataclass

import torch

BOS = 50256  # GPT-2's BOS, EOS and our right-padding token

# TODO: Wang et al. 2022's 15 BABA templates. [B] is the subject S (mentioned twice), [A] the indirect object IO.
# The prompt stops just before the final [A]; ABBA versions swap the first two names. Format example:
#   "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to"
BABA_TEMPLATES: list = []

# TODO: candidate words. Keep only those that are ONE GPT-2 token with a leading space (see single_token_words).
NAMES: list = []
PLACES: list = []
OBJECTS: list = []


@dataclass
class IOIData:
    prompts: list  # clean strings, for eyeballing
    tokens: torch.Tensor  # [N, T] clean, right-padded
    flipped: torch.Tensor  # [N, T]
    abc: torch.Tensor  # [N, T]
    io: torch.Tensor  # [N] token id of the right answer
    s: torch.Tensor  # [N] token id of the subject (wrong answer)
    pos: dict  # "IO", "S1", "S1+1", "S2", "end" -> [N] positions
    template: torch.Tensor  # [N] template index (prompts of one template share every position)
    abba: torch.Tensor  # [N] bool

    def __len__(self):
        raise NotImplementedError


@dataclass
class InductionData:
    tokens: torch.Tensor  # [N, 1 + 2T]: BOS, A, A
    corrupt: torch.Tensor  # [N, 1 + 2T]: BOS, A, B  (second half no longer repeats)
    half_len: int


def single_token_words(tokenizer, words) -> list:
    """Keep only words where " " + word encodes to exactly one GPT-2 token."""
    raise NotImplementedError


def make_ioi(tokenizer, n_per_template: int, seed: int) -> IOIData:
    """n_per_template prompts per template (alternating ABBA / BABA), each with its flipped and abc twins.

    Tokens start with BOS and are right-padded with BOS to the longest prompt. Positions are found by token id, so
    assert IO appears once and S twice. Every random choice comes from one torch.Generator seeded with `seed`.
    Watch out: placeholders already have a space before them, so fill them with bare words (no double spaces).
    """
    raise NotImplementedError


def make_induction(n: int, half_len: int, seed: int, d_vocab: int = 50257) -> InductionData:
    """n sequences of [BOS, A, A] with A = half_len random tokens, plus a corrupt [BOS, A, B] twin (B fresh)."""
    raise NotImplementedError
