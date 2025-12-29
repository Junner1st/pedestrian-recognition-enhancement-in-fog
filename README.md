# Fog-Robust Pedestrian Detection

This repository turns pedestrian datasets into fog-aware training and evaluation splits by combining MiDaS depth caching, atmospheric scattering synthesis, and YOLO11n fine-tuning.

## Disclaimer

This project skeleton was bootstrapped with the help of GPT-5.1-Codex and should be reviewed before production use.

## Environment Setup

Create a virtual environment, activate it, and install dependencies before running anything else.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Staging

Download CityPersons (and any optional datasets) yourself from the official portals, extract them into `data/raw/<dataset_name>/`, keep images under `images/`, keep COCO annotations such as `instances_train.json` under `annotations/`, and only modify `configs/project.yaml` once the files are present locally.

## Build Derived Assets

Run the build pipeline to unify labels, cache MiDaS depth, generate fog tiers, and stage YOLO-ready folders plus manifests.

```
python -m src.pipelines.build_dataset --config configs/project.yaml
```

## Baseline Evaluation

Evaluate the untouched `yolo11n.pt` weights on the clear and fogged validation splits defined in `eval.subsets` to produce JSON metrics under `runs/eval/`.

```
python -m src.pipelines.run_baseline_eval --config configs/project.yaml
```

## Curriculum Training

Fine-tune from `yolo11n.pt` through the staged clear→thin→mid→dense curriculum while mixing clear data and tapering strong augmentations so each phase inherits the previous weights.

```
python -m src.pipelines.run_training --config configs/project.yaml
```

## Reproducibility and Gates

All scripts respect the seed declared in `repro.seed`, so bump it for new fog airlight draws or shuffles, and rely on manual gating by comparing `runs/eval/baseline_*.json` with the latest evaluation outputs before deciding whether fog parameters or curriculum ratios need another pass.

Every configurable knob—dataset selection, MiDaS model choice, fog strengths, YOLO hyperparameters—is documented inline in `configs/project.yaml` for quick reference.
