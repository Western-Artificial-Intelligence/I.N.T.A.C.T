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
    raise NotImplementedError


@pytest.fixture(scope="session")
def base_hf():
    """GPT2LMHeadModel "gpt2" in eval mode (unrounded)."""
    raise NotImplementedError


@pytest.fixture(scope="session")
def fp_model(base_hf, tok):
    """base_hf loaded into TransformerLens with quantize.load_tl."""
    raise NotImplementedError


@pytest.fixture(scope="session")
def data(tok):
    """A small IOI set: make_ioi(tok, n_per_template=1, seed=0)."""
    raise NotImplementedError


@pytest.fixture(scope="session")
def ind():
    """A small induction set: make_induction(4, 20, seed=0)."""
    raise NotImplementedError
