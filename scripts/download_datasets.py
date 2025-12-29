from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

import requests
import yaml
from tqdm import tqdm


def load_config(path: str) -> Dict:
    return yaml.safe_load(Path(path).read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, headers: Dict[str, str] | None = None, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        print(f"[skip] {dest.name} already exists; reuse")
        return dest

    headers = headers or {}
    with requests.get(url, stream=True, headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=f"downloading {dest.name}")
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 15):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(len(chunk))
        progress.close()
    return dest


def extract_archive(archive_path: Path, target_dir: Path, archive_type: str) -> None:
    archive_type = archive_type or "auto"
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_type in {"zip", "auto"} and archive_path.suffix == ".zip":
        archive_type = "zip"
    elif archive_type in {"tar", "tar.gz", "tgz", "auto"} and archive_path.suffix in {".tar", ".gz", ".tgz", ".tar.gz"}:
        archive_type = "tar"

    if archive_type == "zip":
        import zipfile

        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
    elif archive_type == "tar":
        import tarfile

        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(target_dir)
    elif archive_type in {"file", "none"}:
        shutil.copy2(archive_path, target_dir / archive_path.name)
    else:
        raise ValueError(f"Unsupported archive type '{archive_type}' for {archive_path}")


def dataset_already_prepared(target: Path) -> bool:
    return target.exists() and any(target.iterdir())


def process_dataset(
    name: str,
    meta: Dict,
    raw_root: Path,
    tmp_root: Path,
    force: bool,
    redownload: bool,
) -> None:
    if not meta.get("enabled", True):
        print(f"[skip] dataset '{name}' disabled in config")
        return
    artifacts = meta.get("artifacts", [])
    if not artifacts:
        print(f"[warn] No artifacts configured for dataset '{name}'")
        return

    target_root = raw_root / meta.get("output_dir", name)
    if dataset_already_prepared(target_root) and not force:
        print(f"[skip] {name} already present at {target_root}")
        return
    if force and target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Processing dataset '{name}' -> {target_root}")
    for idx, artifact in enumerate(artifacts):
        url = artifact.get("url")
        if not url:
            print(f"  [warn] artifact {idx} missing url; update configs")
            continue
        headers = artifact.get("headers") or {}
        filename = artifact.get("filename") or Path(url).name or f"{name}_{idx}"
        archive_path = tmp_root / filename
        download_file(url, archive_path, headers=headers, overwrite=redownload)

        checksum = artifact.get("checksum")
        if checksum:
            digest = sha256(archive_path)
            if digest.lower() != checksum.lower():
                raise ValueError(f"Checksum mismatch for {filename}: {digest} != {checksum}")

        extract_flag = artifact.get("extract", True)
        if not extract_flag:
            dest_file = target_root / filename
            shutil.move(str(archive_path), dest_file)
            continue

        extract_rel = artifact.get("extract_to", ".")
        extract_dir = target_root if extract_rel in {".", ""} else target_root / extract_rel
        if extract_dir.exists() and artifact.get("clean_extract", False):
            shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [info] extracting {filename} -> {extract_dir}")
        extract_archive(archive_path, extract_dir, artifact.get("archive_type", "auto"))


def main():
    parser = argparse.ArgumentParser(description="Download datasets defined in configs/project.yaml")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--datasets", nargs="*", help="Datasets to download (default: all configured)")
    parser.add_argument("--force", action="store_true", help="Delete existing dataset folders before extracting")
    parser.add_argument("--redownload", action="store_true", help="Overwrite existing artifact files inside data/raw/.downloads")
    parser.add_argument("--show-config", action="store_true", help="Print dataset download section and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets_cfg = cfg.get("datasets", {})
    downloads_cfg = datasets_cfg.get("downloads", {})
    if args.show_config:
        print(json.dumps(downloads_cfg, indent=2))
        return

    if not downloads_cfg:
        raise SystemExit("No datasets.downloads entries found in config")

    targets: Iterable[str]
    if args.datasets:
        targets = args.datasets
    else:
        targets = downloads_cfg.keys()

    raw_root = Path(datasets_cfg.get("root", "data/raw"))
    tmp_root = raw_root / ".downloads"

    for name in targets:
        meta = downloads_cfg.get(name)
        if not meta:
            print(f"[warn] Dataset '{name}' not defined; skipping")
            continue
        process_dataset(name, meta, raw_root=raw_root, tmp_root=tmp_root, force=args.force, redownload=args.redownload)


if __name__ == "__main__":
    main()
