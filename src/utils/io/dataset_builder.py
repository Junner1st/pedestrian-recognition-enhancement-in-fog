import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from .split_manager import fixed_split, save_split
from .yolo_format import bbox_to_yolo


@dataclass
class Sample:
    image_path: str
    label_path: str
    subset: str
    source: str


def _index_coco(coco: Dict) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]]]:
    images = {img["id"]: img for img in coco["images"]}
    anns: Dict[int, List[Dict]] = {img_id: [] for img_id in images.keys()}
    for ann in coco["annotations"]:
        anns.setdefault(ann["image_id"], []).append(ann)
    return images, anns


def _load_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return w, h


def convert_coco_to_yolo(
    coco_json: str,
    images_root: str,
    out_label_dir: str,
    unified_classes: List[str],
    person_name: str = "person",
) -> List[str]:
    """Convert COCO-style person boxes to YOLO format. Returns list of image paths."""
    coco = json.loads(Path(coco_json).read_text())
    images, anns = _index_coco(coco)
    images_root_path = Path(images_root)
    out_dir = Path(out_label_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_id = unified_classes.index(person_name)
    used_images: List[str] = []

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        img_path = images_root_path / file_name
        if not img_path.exists():
            continue
        w = img_info.get("width")
        h = img_info.get("height")
        if w is None or h is None:
            w, h = _load_image_size(img_path)

        labels = []
        for ann in anns.get(img_id, []):
            cat_id = ann.get("category_id")
            cat_name = next((c["name"] for c in coco.get("categories", []) if c["id"] == cat_id), None)
            if cat_name != person_name:
                continue
            x, y, bw, bh = ann["bbox"]  # coco bbox in xywh
            bbox_xyxy = (x, y, x + bw, y + bh)
            labels.append((cls_id, bbox_xyxy, (w, h)))

        if not labels:
            continue
        label_path = out_dir / f"{Path(file_name).stem}.txt"
        out_lines = []
        for cls, bbox, size in labels:
            cx, cy, bw_n, bh_n = bbox_to_yolo(bbox, size[0], size[1])
            out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
        label_path.write_text("\n".join(out_lines))
        used_images.append(str(img_path))
    return used_images


def build_unified_from_coco(
    name: str,
    coco_json: str,
    images_root: str,
    output_labels: str,
    splits_dir: str,
    unified_classes: List[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[Sample]:
    """Build YOLO labels and a fixed split from a COCO-style dataset."""
    used_images = convert_coco_to_yolo(coco_json, images_root, output_labels, unified_classes)
    stems = [Path(p).stem for p in used_images]
    train, val, test = fixed_split(stems, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    save_split(train, val, test, out_dir=splits_dir)

    samples: List[Sample] = []
    for stem in stems:
        subset = "train" if stem in train else "val" if stem in val else "test"
        samples.append(
            Sample(
                image_path=str(Path(images_root) / f"{stem}.jpg"),
                label_path=str(Path(output_labels) / f"{stem}.txt"),
                subset=subset,
                source=name,
            )
        )
    return samples


IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp")


def _match_image_path(images_root: Path, subset: str, rel_label_path: Path) -> Path | None:
    base = images_root / subset / rel_label_path
    base_no_suffix = base.with_suffix("")
    for ext in IMG_EXTS:
        candidate = Path(str(base_no_suffix) + ext)
        if candidate.exists():
            return candidate
    return None


def build_from_yolo_splits(
    name: str,
    images_root: str,
    labels_root: str,
    output_labels: str,
    splits_dir: str,
    seed: int,
) -> List[Sample]:
    """Reuse existing train/val/test folder splits where labels already exist."""
    images_root_path = Path(images_root)
    labels_root_path = Path(labels_root)
    out_dir = Path(output_labels)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["train", "val", "test"]
    split_lists: Dict[str, List[str]] = {k: [] for k in subsets}
    samples: List[Sample] = []

    for subset in subsets:
        subset_label_dir = labels_root_path / subset
        if not subset_label_dir.exists():
            continue
        for label_path in subset_label_dir.rglob("*.txt"):
            rel = label_path.relative_to(subset_label_dir)
            image_path = _match_image_path(images_root_path, subset, rel)
            if not image_path:
                continue
            stem = image_path.stem
            dest_label = out_dir / f"{stem}.txt"
            shutil.copy2(label_path, dest_label)
            samples.append(
                Sample(
                    image_path=str(image_path),
                    label_path=str(dest_label),
                    subset=subset,
                    source=name,
                )
            )
            split_lists[subset].append(stem)

    # ensure split files exist even if some subsets missing
    save_split(split_lists.get("train", []), split_lists.get("val", []), split_lists.get("test", []), out_dir=splits_dir)
    return samples

