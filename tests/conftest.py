import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

torch.set_grad_enabled(False)


@pytest.fixture(scope="session")
def tok():
    """GPT2TokenizerFast for "gpt2"."""
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def base_hf():
    """GPT2LMHeadModel "gpt2" in eval mode (unrounded)."""
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained("gpt2").eval()


@pytest.fixture(scope="session")
def fp_model(base_hf, tok):
    """base_hf loaded into TransformerLens with quantize.load_tl."""
    from quantize import load_tl
    return load_tl(base_hf, tok)


@pytest.fixture(scope="session")
def data(tok):
    """A small IOI set: make_ioi(tok, n_per_template=1, seed=0)."""
    raise NotImplementedError


@pytest.fixture(scope="session")
def ind():
    """A small induction set: make_induction(4, 20, seed=0)."""
    raise NotImplementedError
