import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm

from src.models.depth.midas_infer import infer_and_cache
from src.models.fog.fog_synthesizer import synthesize_tiers
from src.utils.io.dataset_builder import Sample, build_from_yolo_splits, build_unified_from_coco
from src.utils.io.manifest import SampleManifest, write_manifest
from src.utils.repro.seed import set_seed


def load_config(path: str) -> Dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_dataset_spec(root: str, name: str) -> Dict[str, str]:
    base = Path(root) / name
    images_root = base / "images"
    labels_root = base / "labels"
    if labels_root.exists() and images_root.exists():
        return {"format": "yolo", "images_root": str(images_root), "labels_root": str(labels_root)}
    raise FileNotFoundError(
        f"Dataset {name} under {base} must contain annotations/instances_train.json or images+/labels+ folders."
    )


def stage_yolo_subset(subset_name: str, image_paths: List[str], labels_dir: str, out_root: str) -> Path:
    """Create YOLO-friendly folder with images/ and labels/ symlinks."""
    subset_root = Path(out_root) / subset_name
    images_out = subset_root / "images"
    labels_out = subset_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(image_paths, desc=f"Stage {subset_name}", leave=False):
        stem = Path(img_path).stem
        label_src = Path(labels_dir) / f"{stem}.txt"
        if not label_src.exists():
            continue
        img_dst = images_out / f"{stem}{Path(img_path).suffix}"
        lbl_dst = labels_out / f"{stem}.txt"
        if not img_dst.exists():
            img_dst.symlink_to(Path(img_path).resolve())
        if not lbl_dst.exists():
            lbl_dst.symlink_to(label_src.resolve())
    return subset_root


def write_data_yaml(name: str, subset_root: Path, out_path: Path) -> None:
    data = {
        "path": str(subset_root),
        "train": "images",
        "val": "images",
        "nc": 1,
        "names": ["person"],
    }
    out_path.write_text(yaml.safe_dump(data))


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["repro"]["seed"])

    datasets_cfg = cfg["datasets"]
    unified_classes = cfg["datasets"]["unified_classes"]
    labels_dir = datasets_cfg["output_labels"]
    splits_dir = datasets_cfg["splits_dir"]

    primary_name = datasets_cfg["primary"]
    dataset_spec = resolve_dataset_spec(datasets_cfg["root"], primary_name)
    if dataset_spec["format"] == "coco":
        samples: List[Sample] = build_unified_from_coco(
            name=primary_name,
            coco_json=dataset_spec["coco_json"],
            images_root=dataset_spec["images_root"],
            output_labels=labels_dir,
            splits_dir=splits_dir,
            unified_classes=unified_classes,
            val_ratio=datasets_cfg["val_ratio"],
            test_ratio=datasets_cfg["test_ratio"],
            seed=datasets_cfg["split_seed"],
        )
    else:
        samples = build_from_yolo_splits(
            name=primary_name,
            images_root=dataset_spec["images_root"],
            labels_root=dataset_spec["labels_root"],
            output_labels=labels_dir,
            splits_dir=splits_dir,
            seed=datasets_cfg["split_seed"],
        )

    depth_cfg = cfg["midas"]
    depth_out = depth_cfg["output_dir"]
    image_paths = [s.image_path for s in samples]
    depth_cache = infer_and_cache(
        image_paths=image_paths,
        output_dir=depth_out,
        model_type=depth_cfg["model_type"],
        save_dtype=depth_cfg["save_dtype"],
        seed=cfg["repro"]["seed"],
    )

    depth_map = {Path(p).stem: p for p in depth_cache}
    fog_input: List[Tuple[str, str, str]] = []
    for s in samples:
        stem = Path(s.image_path).stem
        depth_path = depth_map.get(stem)
        if depth_path:
            fog_input.append((s.image_path, depth_path, s.subset))

    fog_cfg = cfg["fog"]
    tiers = fog_cfg["tbar_targets"]
    beta_search = fog_cfg["beta_search"]
    fog_output_dir = fog_cfg["output_dir"]
    manifest_path = Path("data/manifests/fog_build.json")
    synthesize_tiers(
        images=fog_input,
        tiers=tiers,
        airlight_cfg=fog_cfg["airlight"],
        beta_search=beta_search,
        out_root=fog_output_dir,
        blur_sigma=fog_cfg["blur_sigma"],
        jpeg_quality=fog_cfg["jpeg_quality"],
        seed=cfg["repro"]["seed"],
        manifest_path=str(manifest_path),
    )

    # Stage YOLO-ready subset folders and manifests
    subsets = {"clear": [s for s in samples if s.subset in {"train", "val"}]}
    for tier in tiers.keys():
        fog_dir = Path(fog_output_dir) / tier
        images = [str(p) for p in fog_dir.glob("*.jpg")]
        subsets[tier] = []
        for img_path in images:
            stem = Path(img_path).stem
            # use same subset as clear
            subset = next((s.subset for s in samples if Path(s.image_path).stem == stem), "train")
            subsets[tier].append(Sample(image_path=img_path, label_path=str(Path(labels_dir) / f"{stem}.txt"), subset=subset, source=tier))

    yolo_root = Path("data/yolo")
    yolo_root.mkdir(parents=True, exist_ok=True)
    data_yaml_paths = {}
    manifest_records: List[SampleManifest] = []
    for subset_name, records in subsets.items():
        imgs = [r.image_path for r in records]
        staged = stage_yolo_subset(subset_name, imgs, labels_dir=labels_dir, out_root=str(yolo_root))
        yaml_path = yolo_root / f"{subset_name}.yaml"
        write_data_yaml(subset_name, staged, yaml_path)
        data_yaml_paths[subset_name] = str(yaml_path)
        for rec in records:
            manifest_records.append(
                SampleManifest(
                    image=rec.image_path,
                    label=rec.label_path,
                    depth=depth_map.get(Path(rec.image_path).stem, None),
                    fog_image=rec.image_path if subset_name != "clear" else None,
                    subset=rec.subset,
                    source=rec.source,
                    beta=None,
                    airlight=None,
                    tbar=tiers.get(subset_name) if subset_name != "clear" else None,
                    hash=None,
                )
            )

    write_manifest(
        manifest_records,
        out_path="data/manifests/dataset_manifest.json",
        meta={"config": config_path, "data_yamls": data_yaml_paths},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    args = parser.parse_args()
    main(args.config)

