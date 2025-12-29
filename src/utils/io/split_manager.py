import json
import random
from pathlib import Path
from typing import List, Tuple


def fixed_split(items: List[str], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    """Return train/val/test lists with fixed seed."""
    rng = random.Random(seed)
    perm = items.copy()
    rng.shuffle(perm)
    n = len(perm)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val = perm[:n_val]
    test = perm[n_val : n_val + n_test]
    train = perm[n_val + n_test :]
    return train, val, test


def save_split(train: List[str], val: List[str], test: List[str], out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "train.txt").write_text("\n".join(train))
    (Path(out_dir) / "val.txt").write_text("\n".join(val))
    (Path(out_dir) / "test.txt").write_text("\n".join(test))
    # JSON for reproducibility
    (Path(out_dir) / "split.json").write_text(json.dumps({"train": train, "val": val, "test": test}, indent=2))
