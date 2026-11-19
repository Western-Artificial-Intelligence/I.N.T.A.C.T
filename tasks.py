"""The test questions: IOI sentences (with their twins) and repeated-token sequences, all generated from a seed.

Every IOI prompt comes with two near-identical twins:
  flipped: the second mention of S is replaced by IO, so the right answer flips. Used for activation patching.
  abc:     all three name slots get three fresh distinct names, so there is no IOI signal. Used for mean-ablation.
Names, places and objects are single GPT-2 tokens, so a prompt and its twins line up token for token.
"""

from dataclasses import dataclass

import torch

BOS = 50256  # GPT-2's BOS, EOS and our right-padding token

# Wang et al. 2022's 15 BABA templates. [B] is the subject S (mentioned twice), [A] the indirect object IO.
# The prompt stops just before the final [A]; ABBA versions swap the first two names.
BABA_TEMPLATES: list = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to",
]

# Candidate words (Wang et al.'s lists). make_ioi keeps only those that are ONE GPT-2 token with a leading space
# (see single_token_words).
NAMES: list = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua", "Amanda", "Daniel", "David",
    "James", "Robert", "John", "Joseph", "Andrew", "Ryan", "Brandon", "Jason", "Justin", "Sarah", "William",
    "Jonathan", "Stephanie", "Brian", "Nicole", "Nicholas", "Anthony", "Heather", "Eric", "Elizabeth", "Adam",
    "Megan", "Melissa", "Kevin", "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel", "Laura", "Lauren",
    "Amber", "Brittany", "Danielle", "Richard", "Kimberly", "Jeffrey", "Amy", "Crystal", "Michelle", "Tiffany",
    "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles", "Rebecca", "Jacob", "Stephen", "Patrick", "Sean",
    "Erin", "Jamie", "Kelly", "Samantha", "Nathan", "Sara", "Dustin", "Paul", "Angela", "Tyler", "Scott",
    "Katherine", "Andrea", "Gregory", "Erica", "Mary", "Travis", "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen",
    "Jose", "Alexander", "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney", "Christine", "Alicia",
    "Cody", "Allison", "Bradley", "Samuel",
]
PLACES: list = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station"]
OBJECTS: list = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack"]


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
        return len(self.prompts)


@dataclass
class InductionData:
    tokens: torch.Tensor  # [N, 1 + 2T]: BOS, A, A
    corrupt: torch.Tensor  # [N, 1 + 2T]: BOS, A, B  (second half no longer repeats)
    half_len: int


def single_token_words(tokenizer, words) -> list:
    """Keep only words where " " + word encodes to exactly one GPT-2 token."""
    return [w for w in words if len(tokenizer.encode(" " + w)) == 1]


def _fill(template: str, io: str, s: str, place: str, obj: str, abba: bool) -> str:
    """Fill a BABA template with bare words. ABBA swaps the first two name slots (IO first, then S)."""
    if abba:
        template = template.replace("[B]", "[X]", 1).replace("[A]", "[B]", 1).replace("[X]", "[A]", 1)
    return template.replace("[A]", io).replace("[B]", s).replace("[PLACE]", place).replace("[OBJECT]", obj)


def make_ioi(tokenizer, n_per_template: int, seed: int) -> IOIData:
    """n_per_template prompts per template (alternating ABBA / BABA), each with its flipped and abc twins.

    Tokens start with BOS and are right-padded with BOS to the longest prompt. Positions are found by token id, so
    assert IO appears once and S twice. Every random choice comes from one torch.Generator seeded with `seed`.
    Watch out: placeholders already have a space before them, so fill them with bare words (no double spaces).
    """
    g = torch.Generator().manual_seed(seed)
    names = single_token_words(tokenizer, NAMES)
    places = single_token_words(tokenizer, PLACES)
    objects = single_token_words(tokenizer, OBJECTS)
    name_ids = torch.tensor([tokenizer.encode(" " + n)[0] for n in names])

    def pick(n_options: int) -> int:
        return int(torch.randint(n_options, (1,), generator=g))

    prompts, rows, pos = [], [], {k: [] for k in ("IO", "S1", "S1+1", "S2", "end")}
    io_ids, s_ids, abc_ids, template_idx, abba_flags = [], [], [], [], []
    for t, template in enumerate(BABA_TEMPLATES):
        for i in range(n_per_template):
            # five distinct names: IO, S, and three fresh ones for the abc twin
            io, s, *fresh = torch.randperm(len(names), generator=g)[:5].tolist()
            # alternate over the running count, so both orders show up even with one prompt per template
            abba = (t * n_per_template + i) % 2 == 0
            prompt = _fill(template, names[io], names[s], places[pick(len(places))], objects[pick(len(objects))], abba)
            toks = torch.tensor([BOS] + tokenizer.encode(prompt))

            io_pos = (toks == name_ids[io]).nonzero().flatten()
            s_pos = (toks == name_ids[s]).nonzero().flatten()
            assert len(io_pos) == 1 and len(s_pos) == 2, prompt

            prompts.append(prompt)
            rows.append(toks)
            pos["IO"].append(io_pos[0])
            pos["S1"].append(s_pos[0])
            pos["S1+1"].append(s_pos[0] + 1)
            pos["S2"].append(s_pos[1])
            pos["end"].append(len(toks) - 1)
            io_ids.append(name_ids[io])
            s_ids.append(name_ids[s])
            abc_ids.append(name_ids[fresh])
            template_idx.append(t)
            abba_flags.append(abba)

    n = len(rows)
    idx = torch.arange(n)
    pos = {k: torch.tensor(v) for k, v in pos.items()}
    io_ids, s_ids, abc_ids = torch.stack(io_ids), torch.stack(s_ids), torch.stack(abc_ids)

    tokens = torch.full((n, max(len(r) for r in rows)), BOS)
    for j, r in enumerate(rows):
        tokens[j, : len(r)] = r

    # twins are edited at the token level, so they line up with the clean prompt token for token
    flipped = tokens.clone()
    flipped[idx, pos["S2"]] = io_ids

    abc = tokens.clone()
    abc[idx, pos["IO"]] = abc_ids[:, 0]
    abc[idx, pos["S1"]] = abc_ids[:, 1]
    abc[idx, pos["S2"]] = abc_ids[:, 2]

    return IOIData(prompts=prompts, tokens=tokens, flipped=flipped, abc=abc, io=io_ids, s=s_ids, pos=pos,
                   template=torch.tensor(template_idx), abba=torch.tensor(abba_flags))


def make_induction(n: int, half_len: int, seed: int, d_vocab: int = 50257) -> InductionData:
    """n sequences of [BOS, A, A] with A = half_len random tokens, plus a corrupt [BOS, A, B] twin (B fresh)."""
    g = torch.Generator().manual_seed(seed)
    # Every random token is uniform over IDs 0..d_vocab-2, i.e. the whole vocab except BOS (the last ID).
    # A BOS mid-sequence would act as a fresh start and break the copying the induction task relies on.
    vocab = d_vocab - 1
    a = torch.randint(0, vocab, (n, half_len), generator=g)
    b = torch.randint(0, vocab, (n, half_len), generator=g)
    bos = torch.full((n, 1), BOS, dtype=a.dtype)
    return InductionData(
        tokens=torch.cat([bos, a, a], dim=1),
        corrupt=torch.cat([bos, a, b], dim=1),
        half_len=half_len,
    )
