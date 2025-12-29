from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# YOLO txt format lines: class x_center y_center width height (normalized)


def bbox_to_yolo(bbox: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert bbox (xmin, ymin, xmax, ymax) to YOLO normalized format."""
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return (cx, cy, w, h)


def save_yolo_labels(labels: Iterable[Tuple[int, Tuple[float, float, float, float], Tuple[int, int]]], out_path: str) -> None:
    """Save iterable of (class_id, bbox_xyxy, (img_w,img_h)) to file."""
    out_lines: List[str] = []
    for cls_id, bbox, size in labels:
        x, y, w, h = bbox_to_yolo(bbox, size[0], size[1])
        out_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out_lines))


def filter_person_only(boxes: np.ndarray, classes: List[str], person_name: str = "person") -> np.ndarray:
    """Return boxes whose class name equals person_name. boxes columns: (cls_name,xmin,ymin,xmax,ymax)."""
    keep = []
    for row in boxes:
        if len(row) < 5:
            continue
        cls_name = str(row[0])
        if cls_name.lower() == person_name:
            keep.append(row[1:5].astype(float))
    return np.array(keep)
