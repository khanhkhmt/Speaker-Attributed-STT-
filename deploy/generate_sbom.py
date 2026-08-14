"""Generate a lightweight SPDX-inspired dependency/model inventory (spec 11.2, 20)."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import yaml


def build_sbom(manifest_dir: Path) -> dict[str, Any]:
    packages: list[dict[str, str]] = []
    for item in importlib.metadata.distributions():
        name = item.metadata["Name"]
        if name:
            packages.append({"name": name, "version": item.version})
    packages.sort(key=lambda item: (item["name"].lower(), item["version"]))
    models: list[dict[str, Any]] = []
    for path in sorted(manifest_dir.glob("*.yaml")):
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            models.append(
                {
                    "backend": raw.get("backend"),
                    "revision": raw.get("revision"),
                    "sha256": raw.get("sha256"),
                    "weight_license": raw.get("weight_license"),
                    "production_action": raw.get("production_action"),
                }
            )
    return {"schema_version": "1.0", "packages": packages, "model_manifests": models}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("model-manifests"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(
        json.dumps(build_sbom(args.manifest_dir), indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
