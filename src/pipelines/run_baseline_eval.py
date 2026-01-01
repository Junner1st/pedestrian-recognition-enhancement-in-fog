import argparse
import json
from pathlib import Path
from typing import Dict

import yaml
from ultralytics import YOLO

from src.utils.repro.seed import set_seed


def load_config(path: str) -> Dict:
    return yaml.safe_load(Path(path).read_text())


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["repro"]["seed"])
    eval_cfg = cfg["eval"]
    subsets = eval_cfg["subsets"]
    eval_save_dir = Path(eval_cfg.get("save_dir", "runs/eval"))
    plots_enabled = eval_cfg.get("plots", True)
    save_conf = eval_cfg.get("save_conf", True)
    eval_save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(cfg["training"]["base_weights"])

    results_all: Dict[str, Dict] = {}
    for subset in subsets:
        data_yaml = Path("data/yolo") / f"{subset}.yaml"
        if not data_yaml.exists():
            print(f"Missing data yaml for subset {subset}: {data_yaml}")
            continue
        run_name = f"baseline_{subset}"
        res = model.val(
            data=str(data_yaml),
            imgsz=cfg["training"]["imgsz"],
            split="val",
            save=False,
            plots=plots_enabled,
            save_conf=save_conf,
            verbose=False,
            project=str(eval_save_dir),
            name=run_name,
            classes=[0],
        )
        metrics = res.results_dict
        plot_dir = Path(res.save_dir)
        subset_payload = {**metrics, "_plots_dir": str(plot_dir)}
        results_all[subset] = subset_payload
        out_json = eval_save_dir / f"{run_name}.json"
        out_json.write_text(json.dumps(subset_payload, indent=2))
        print(f"subset {subset}: {metrics} (plots: {plot_dir})")

    summary_path = eval_save_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(results_all, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    args = parser.parse_args()
    main(args.config)
