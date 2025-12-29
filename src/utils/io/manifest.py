import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from ..repro.seed import snapshot_env


@dataclass
class SampleManifest:
    image: str
    label: str
    depth: str | None
    fog_image: str | None
    subset: str
    source: str
    beta: float | None
    airlight: float | None
    tbar: float | None
    hash: str | None


def write_manifest(records: List[SampleManifest], out_path: str, meta: Dict[str, Any]) -> None:
    """Write manifest to JSON with metadata and env snapshot."""
    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta,
        "env": snapshot_env(),
        "records": [asdict(r) for r in records],
    }
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2))


def load_manifest(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
