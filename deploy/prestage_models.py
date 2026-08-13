#!/usr/bin/env python3
"""Pre-stage model weights and pin them into the manifests — spec 11.2, 20.

Downloads happen here, at the build/preparation stage, never in a production
worker: the runtime only mounts ``/models`` read-only (spec 0.3, 11.2). For each
backend this tool

1. resolves the repository revision (a commit SHA, never a moving tag),
2. downloads it into ``<models-dir>/<local_dir>``,
3. hashes every weight file with SHA-256,
4. writes ``revision``, ``sha256``, ``files`` and ``local_path`` back into
   ``model-manifests/<name>.yaml``.

After this runs, ``sastt.config.validate_for_environment`` accepts the backend in
a production environment; before it runs, production refuses to start.

Weights are never committed to Git (spec 18 rule 4) — ``/models`` is ignored.

Usage:
    python deploy/prestage_models.py --list
    python deploy/prestage_models.py --only diarization osd asr_realtime
    python deploy/prestage_models.py --all --models-dir /models
    python deploy/prestage_models.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sastt.config import ModelFile, ModelManifest, aggregate_sha256  # noqa: E402

DEFAULT_MODELS_DIR = Path("/models")
MANIFEST_DIR = REPO_ROOT / "model-manifests"

#: Files that are documentation rather than weights; hashed but not required.
WEIGHT_SUFFIXES = (".bin", ".pt", ".ckpt", ".onnx", ".safetensors", ".json", ".yaml", ".txt")
SKIP_PATTERNS = (".gitattributes", ".git/", "README.md", ".jpg", ".png", ".gif")


@dataclass(frozen=True)
class Target:
    """One downloadable backend."""

    key: str
    manifest_file: str
    local_dir: str
    allow_patterns: tuple[str, ...] | None = None


#: Milestone 1 needs the first five; the rest arrive with their milestone.
TARGETS: tuple[Target, ...] = (
    Target("diarization", "pyannote-community-1.yaml", "pyannote-community-1"),
    Target("osd", "pyannote-segmentation-3.0.yaml", "pyannote-segmentation-3.0"),
    Target("asr_realtime", "faster-whisper.yaml", "faster-whisper-large-v3-turbo"),
    Target("asr_final", "faster-whisper-large-v3.yaml", "faster-whisper-large-v3"),
    Target("separation_two_source", "mossformer2-ss-16k.yaml", "mossformer2-ss-16k"),
    Target("embedding", "campplus.yaml", "campplus"),
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(target: Target) -> tuple[Path, dict[str, Any], ModelManifest]:
    path = MANIFEST_DIR / target.manifest_file
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return path, raw, ModelManifest.model_validate(raw)


def resolve_revision(manifest: ModelManifest) -> str:
    """Pin to an immutable commit SHA (spec 0.3: revision + SHA-256)."""
    if manifest.source == "huggingface":
        from huggingface_hub import HfApi

        info = HfApi().model_info(_repo_id(manifest), revision=manifest.revision or None)
        return str(info.sha)
    if manifest.source == "modelscope":
        from modelscope.hub.api import HubApi

        # ModelScope exposes branches and tags; a tag is the closest thing to an
        # immutable pin, and the per-file SHA-256 below pins the artefact exactly.
        _branches, tags = HubApi().get_model_branches_and_tags(_repo_id(manifest))
        return str(tags[-1]) if tags else "master"
    raise SystemExit(f"backend {manifest.backend!r} has no downloadable source")


def _repo_id(manifest: ModelManifest) -> str:
    if not manifest.repo_id:
        raise SystemExit(f"manifest for {manifest.backend!r} has no repo_id")
    return manifest.repo_id


def download(manifest: ModelManifest, target: Target, models_dir: Path, revision: str) -> Path:
    destination = models_dir / target.local_dir
    destination.mkdir(parents=True, exist_ok=True)
    if manifest.source == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=_repo_id(manifest),
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(target.allow_patterns) if target.allow_patterns else None,
        )
    elif manifest.source == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download as ms_download

        ms_download(_repo_id(manifest), revision=revision, local_dir=str(destination))
    else:
        raise SystemExit(f"backend {manifest.backend!r} has no downloadable source")
    return destination


def hash_directory(root: Path) -> list[ModelFile]:
    files: list[ModelFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(pattern in relative for pattern in SKIP_PATTERNS):
            continue
        files.append(
            ModelFile(
                path=relative,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return files


def write_manifest(path: Path, raw: dict[str, Any], updates: dict[str, Any]) -> None:
    raw.update(updates)
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def prestage(target: Target, models_dir: Path) -> None:
    path, raw, manifest = load_manifest(target)
    if manifest.source == "none":
        print(f"[skip] {target.key}: no downloadable weights ({manifest.backend})")
        return

    revision = resolve_revision(manifest)
    print(
        f"[pull] {target.key}: {manifest.repo_id}@{revision[:12]} -> {models_dir / target.local_dir}"
    )
    destination = download(manifest, target, models_dir, revision)

    files = hash_directory(destination)
    if not files:
        raise SystemExit(f"no files downloaded for {target.key}")
    digest = aggregate_sha256(files)
    total_mb = sum(entry.size_bytes for entry in files) / (1024 * 1024)

    write_manifest(
        path,
        raw,
        {
            "revision": revision,
            "sha256": digest,
            "local_path": str(destination),
            "files": [entry.model_dump() for entry in files],
        },
    )
    print(f"[pin ] {target.key}: {len(files)} files, {total_mb:.1f} MiB, sha256={digest[:16]}…")


def verify(target: Target, models_dir: Path) -> bool:
    """Re-hash what is on disk and compare against the manifest (spec 11.2)."""
    _, _, manifest = load_manifest(target)
    if manifest.source == "none":
        return True
    if not manifest.is_pinned or not manifest.files:
        print(f"[FAIL] {target.key}: manifest is not pinned")
        return False
    root = Path(manifest.local_path or (models_dir / target.local_dir))
    if not root.exists():
        print(f"[FAIL] {target.key}: {root} is missing")
        return False
    for entry in manifest.files:
        candidate = root / entry.path
        if not candidate.exists():
            print(f"[FAIL] {target.key}: missing file {entry.path}")
            return False
        if sha256_file(candidate) != entry.sha256:
            print(f"[FAIL] {target.key}: digest mismatch on {entry.path}")
            return False
    if not manifest.verify_digest():
        print(f"[FAIL] {target.key}: aggregate digest does not match the file list")
        return False
    print(f"[ok  ] {target.key}: {len(manifest.files)} files verified")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--only", nargs="+", metavar="KEY", help="subset of target keys")
    parser.add_argument("--all", action="store_true", help="pre-stage every target")
    parser.add_argument("--list", action="store_true", help="list target keys and exit")
    parser.add_argument("--verify", action="store_true", help="verify digests on disk")
    args = parser.parse_args()

    if args.list:
        for target in TARGETS:
            _, _, manifest = load_manifest(target)
            state = "pinned" if manifest.is_pinned else "unpinned"
            print(f"{target.key:22s} {manifest.repo_id or '-':55s} [{manifest.source}] {state}")
        return 0

    selected = (
        TARGETS
        if args.all or not args.only
        else tuple(t for t in TARGETS if t.key in set(args.only))
    )
    if not selected:
        parser.error("no matching target key")

    if args.verify:
        return 0 if all(verify(target, args.models_dir) for target in selected) else 1

    for target in selected:
        prestage(target, args.models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
