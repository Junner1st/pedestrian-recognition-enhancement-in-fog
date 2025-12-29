import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
from ultralytics import YOLO

from src.utils.repro.seed import set_seed


def load_config(path: str) -> Dict:
    return yaml.safe_load(Path(path).read_text())


def list_images(subset: str) -> List[str]:
    root = Path("data/yolo") / subset / "images"
    # Accept both jpg and png assets since the staged dataset uses png
    candidates = list(root.glob("*.jpg")) + list(root.glob("*.png"))
    return [str(p) for p in sorted(candidates)]


def mix_images(clear: List[str], fog: List[str], fog_ratio: float) -> List[str]:
    if not fog:
        return clear
    fog_count = int(len(clear) * fog_ratio)
    return clear + fog[:fog_count]


def write_phase_yaml(name: str, train_images: List[str], val_images: List[str], out_dir: Path) -> Path:
    lists_dir = out_dir / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    train_list = lists_dir / f"{name}_train.txt"
    val_list = lists_dir / f"{name}_val.txt"
    train_list.write_text("\n".join(train_images))
    val_list.write_text("\n".join(val_images))
    data_yaml = out_dir / f"{name}.yaml"
    data_yaml.write_text(
        yaml.safe_dump({
            "path": ".",
            "train": str(train_list),
            "val": str(val_list),
            "nc": 1,
            "names": ["person"],
        })
    )
    return data_yaml


def evaluate_model_on_subsets(
    model: YOLO,
    subsets: List[str],
    imgsz: int,
    eval_save_dir: Path,
    run_prefix: str,
    plots: bool,
    save_conf: bool,
    workers: int,
) -> Dict[str, Dict]:
    """Run eval with plots for the requested subsets and save JSON + figure outputs."""
    eval_save_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    for subset in subsets:
        data_yaml = Path("data/yolo") / f"{subset}.yaml"
        if not data_yaml.exists():
            print(f"[eval] Missing data yaml for subset {subset}: {data_yaml}")
            continue
        run_name = f"{run_prefix}_{subset}"
        res = model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            split="val",
            plots=plots,
            save_conf=save_conf,
            workers=workers,
            verbose=False,
            project=str(eval_save_dir),
            name=run_name,
        )
        metrics = res.results_dict
        plot_dir = Path(res.save_dir)
        payload = {**metrics, "_plots_dir": str(plot_dir)}
        (eval_save_dir / f"{run_name}.json").write_text(json.dumps(payload, indent=2))
        results[subset] = payload
        print(f"[eval] {run_name}: {metrics} (plots: {plot_dir})")
    return results


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["repro"]["seed"])
    train_cfg = cfg["training"]
    eval_cfg = cfg.get("eval", {})
    phases = train_cfg["curriculum"]
    eval_subsets = eval_cfg.get("subsets", [])
    eval_save_dir = Path(eval_cfg.get("save_dir", "runs/eval"))
    eval_plots = eval_cfg.get("plots", True)
    eval_save_conf = eval_cfg.get("save_conf", True)
    eval_workers = eval_cfg.get("workers", 0)  # use single-worker to avoid MP shutdown warnings

    clear_imgs = list_images("clear")
    val_imgs = [img for img in clear_imgs if "val" in Path(img).parts or "val" in Path(img).name]
    base_val = val_imgs if val_imgs else clear_imgs[-max(1, len(clear_imgs)//10):]

    model = YOLO(train_cfg["base_weights"])
    out_root = Path("runs/train")
    out_root.mkdir(parents=True, exist_ok=True)

    fog_cache = {tier: list_images(tier) for tier in ["thin", "mid", "dense"]}

    for phase_name, phase_cfg in phases.items():
        subsets = phase_cfg["subsets"]
        fog_ratio = phase_cfg["fog_ratio"]
        train_images = clear_imgs
        for sub in subsets:
            if sub == "clear":
                continue
            train_images = mix_images(train_images, fog_cache.get(sub, []), fog_ratio)
        data_yaml = write_phase_yaml(phase_name, train_images, base_val, out_root)

        mosaic_final_pct = float(train_cfg.get("mosaic_final_pct", 0.0))
        close_mosaic_epochs = int(phase_cfg["epochs"] * mosaic_final_pct) if mosaic_final_pct > 0 else 0

        print(f"Starting phase {phase_name} with {len(train_images)} train images")
        model.train(
            data=str(data_yaml),
            epochs=phase_cfg["epochs"],
            imgsz=train_cfg["imgsz"],
            device=train_cfg["device"],
            workers=train_cfg["workers"],
            lr0=train_cfg["lr0"],
            lrf=train_cfg["lrf"],
            weight_decay=train_cfg["weight_decay"],
            warmup_epochs=train_cfg["warmup_epochs"],
            close_mosaic=close_mosaic_epochs,
            resume=False,
            verbose=True,
            project=str(out_root),
            name=phase_name,
            exist_ok=True,
        )
        trainer = getattr(model, "trainer", None)
        save_dir = Path(getattr(trainer, "save_dir", out_root / phase_name))
        last_weights = save_dir / "weights" / "last.pt"
        if not last_weights.exists():
            raise FileNotFoundError(f"Missing last weights for phase {phase_name}: {last_weights}")
        # Use last weights as start for next phase
        model = YOLO(str(last_weights))

        if eval_subsets:
            phase_prefix = f"train_{phase_name}"
            phase_results = evaluate_model_on_subsets(
                model=model,
                subsets=eval_subsets,
                imgsz=train_cfg["imgsz"],
                eval_save_dir=eval_save_dir,
                run_prefix=phase_prefix,
                plots=eval_plots,
                save_conf=eval_save_conf,
                workers=eval_workers,
            )
            summary_path = eval_save_dir / f"{phase_prefix}_summary.json"
            summary_path.write_text(json.dumps(phase_results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    args = parser.parse_args()
    main(args.config)
