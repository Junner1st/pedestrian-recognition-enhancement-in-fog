import os
import random
from hashlib import sha256
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def hash_file(path: str, algo: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Return hex digest for a file; default sha256."""
    if algo != "sha256":
        raise ValueError("Only sha256 supported in this helper")
    h = sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def snapshot_env(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Capture minimal environment info for manifests/runs."""
    info = {
        "cwd": os.getcwd(),
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if extra:
        info.update(extra)
    return info
