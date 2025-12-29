from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.repro.seed import set_seed


def load_midas_model(model_type: str = "DPT_Large"):
    """Load MiDaS depth model and matching transform."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("isl-org/MiDaS", model_type)
    model.to(device)
    model.eval()

    transforms = torch.hub.load("isl-org/MiDaS", "transforms")
    transform = transforms.dpt_transform
    return model, transform, device


def compute_depth_map(img_bgr, midas_model, midas_transform, device):
    """Return relative depth map from single RGB image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = midas_transform(img_rgb).to(device)

    with torch.no_grad():
        pred = midas_model(inp)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        depth = pred.squeeze().cpu().numpy()
    return depth


def normalize_depth_map(depth_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize arbitrary depth map to [0, 1] range."""
    depth = depth_map.astype(np.float32)
    min_val = float(depth.min())
    max_val = float(depth.max())
    spread = max(max_val - min_val, eps)
    return (depth - min_val) / spread

def infer_and_cache(
    image_paths: Iterable[str],
    output_dir: str,
    model_type: str = "DPT_Large",
    save_dtype: str = "float16",
    seed: int | None = None,
    skip_existing: bool = True,
) -> List[str]:
    """Run MiDaS on images and cache normalized depth (inverse depth notion)."""
    if seed is not None:
        set_seed(seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = list(image_paths)
    out_paths = [str(output_root / f"{Path(p).stem}.npy") for p in paths]
    pending = []
    for p in paths:
        out_path = output_root / f"{Path(p).stem}.npy"
        if skip_existing and out_path.exists():
            continue
        pending.append((p, out_path))

    if pending:
        model, transform, torch_device = load_midas_model(model_type=model_type)
        for img_path, out_path in tqdm(pending, desc="MiDaS"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            depth = compute_depth_map(img, model, transform, torch_device)
            depth_norm = normalize_depth_map(depth)
            np.save(out_path, depth_norm.astype(save_dtype))

    return out_paths

