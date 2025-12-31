import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

from src.utils.io.manifest import SampleManifest, write_manifest
from src.utils.repro.seed import set_seed


def apply_fog(rgb: np.ndarray, depth_norm: np.ndarray, beta: float, airlight: float, blur_sigma: float = 0.0, jpeg_quality: int = 100) -> np.ndarray:
    """Apply atmospheric scattering model to an RGB image using normalized depth."""
    depth = depth_norm.astype(np.float32)
    t = np.exp(-beta * depth)
    t = np.clip(t, 0.0, 1.0)
    air_vec = np.ones_like(rgb, dtype=np.float32) * airlight
    fogged = rgb.astype(np.float32) / 255.0
    fogged = fogged * t[..., None] + air_vec * (1.0 - t[..., None])

    if blur_sigma > 0:
        k = max(1, int(blur_sigma * 3) | 1)
        fogged = cv2.GaussianBlur(fogged, (k, k), blur_sigma)

    fogged = np.clip(fogged, 0.0, 1.0)
    fogged = (fogged * 255).astype(np.uint8)

    if jpeg_quality < 100:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        success, enc = cv2.imencode('.jpg', fogged, encode_param)
        if success:
            fogged = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return fogged


def mean_transmission(depth_norm: np.ndarray, beta: float) -> float:
    t = np.exp(-beta * depth_norm.astype(np.float32))
    return float(np.mean(t))


def search_beta_for_target(depth_norm: np.ndarray, target_tbar: float, beta_min: float, beta_max: float, steps: int) -> float:
    """Binary-search beta to match target mean transmission."""
    best_beta = beta_min
    best_gap = float("inf")
    lo, hi = beta_min, beta_max
    for _ in range(max(1, steps)):
        mid = 0.5 * (lo + hi)
        t_mean = mean_transmission(depth_norm, mid)
        gap = abs(t_mean - target_tbar)
        if gap < best_gap:
            best_gap = gap
            best_beta = mid
        # Transmission decreases as beta grows, so adjust bounds accordingly.
        if t_mean > target_tbar:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-6:
            break
    return float(best_beta)


def synthesize_tiers(
    images: Iterable[Tuple[str, str, str]],
    tiers: Dict[str, float],
    airlight_cfg: Dict[str, float],
    beta_search: Dict[str, float],
    out_root: str,
    blur_sigma: float = 0.0,
    jpeg_quality: int = 100,
    seed: int = 42,
    manifest_path: str | None = None,
) -> Dict[str, List[str]]:
    """
    Apply fog tiers to a list of (image_path, depth_path, subset) tuples.
    tiers: mapping tier_name -> target mean transmission.
    airlight_cfg: mean/std/min/max.
    Returns mapping tier -> list of fog image paths.
    """
    set_seed(seed)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)
    results: Dict[str, List[str]] = {k: [] for k in tiers.keys()}
    records: List[SampleManifest] = []

    beta_min = beta_search["beta_min"]
    beta_max = beta_search["beta_max"]
    steps = beta_search["steps"]

    rng = np.random.default_rng(seed)

    image_list = list(images)
    total_steps = len(image_list) * max(len(tiers), 1)
    with tqdm(total=total_steps, desc="Fog synthesis") as pbar:
        for img_path, depth_path, subset in image_list:
            depth = np.load(depth_path)
            rgb = cv2.imread(img_path)
            if rgb is None:
                continue

            for tier, t_target in tiers.items():
                beta = search_beta_for_target(depth, target_tbar=t_target, beta_min=beta_min, beta_max=beta_max, steps=steps)
                airlight = float(
                    np.clip(
                        rng.normal(loc=airlight_cfg["mean"], scale=airlight_cfg["std"]),
                        airlight_cfg["min"],
                        airlight_cfg["max"],
                    )
                )
                fogged = apply_fog(rgb, depth, beta=beta, airlight=airlight, blur_sigma=blur_sigma, jpeg_quality=jpeg_quality)
                tier_dir = out_root_path / tier
                tier_dir.mkdir(parents=True, exist_ok=True)
                out_img_path = tier_dir / f"{Path(img_path).stem}.jpg"
                cv2.imwrite(str(out_img_path), fogged)
                results[tier].append(str(out_img_path))
                records.append(
                    SampleManifest(
                        image=img_path,
                        label="",  # unchanged bbox is shared; optionally filled by caller
                        depth=depth_path,
                        fog_image=str(out_img_path),
                        subset=subset,
                        source="synthetic_fog",
                        beta=beta,
                        airlight=airlight,
                        tbar=t_target,
                        hash=None,
                    )
                )
                pbar.update(1)

    if manifest_path:
        meta = {"tiers": tiers, "beta_search": beta_search, "airlight": airlight_cfg, "blur_sigma": blur_sigma, "jpeg_quality": jpeg_quality}
        write_manifest(records, manifest_path, meta=meta)
    return results
