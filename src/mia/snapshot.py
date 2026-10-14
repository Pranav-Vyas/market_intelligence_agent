"""Frozen data snapshot (task B7): one zip with everything `mia run --offline` needs.

A snapshot holds the disk cache (pages, API responses, search results, LLM answers, embeddings)
and the saved search indexes. Model weights are left out unless asked for: fastembed downloads
them on first use, and they're the same for everyone.

LLM answers are cached per model ID and thinking level, so an offline run only finds them when
its MIA_MODEL_* and MIA_THINKING_* settings match the snapshot's. The manifest records them, and
`restore` reports any difference.

Scraped content never goes in git (SPEC.md §11.9): share the zip privately, e.g. on Google Drive.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath

from mia import runtime
from mia.config import Settings

MANIFEST = "manifest.json"
SKIP_CACHE = ("google_play",)  # namespaces of sources we no longer use
TOP_DIRS = ("cache", "indexes", "models")  # the only folders a snapshot may write into


class SnapshotError(Exception):
    pass


def settings_to_match(s: Settings) -> dict[str, str]:
    """The settings an offline run needs to match to find the snapshot's LLM answers."""
    return {
        "MIA_MODEL_BULK": s.model_bulk,
        "MIA_MODEL_SMART": s.model_smart,
        "MIA_THINKING_BULK": s.thinking_bulk,
        "MIA_THINKING_SMART": s.thinking_smart,
        "MIA_EMBED_MODEL": s.embed_model,
    }


def create(name: str = "snapshot", *, include_models: bool = False) -> Path:
    s = runtime.settings()
    out = s.snapshots_dir / f"{name}-{datetime.now():%Y%m%d-%H%M}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    roots = [s.cache_dir, s.indexes_dir] + ([s.models_dir] if include_models else [])
    files = sorted(f for root in roots for f in _files(root, s.data_dir))
    if not files:
        raise SnapshotError(f"Nothing to snapshot in {s.data_dir}. Run `mia run` or `mia index`.")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings_to_match(s),
        "indexes": sorted(p.name for p in s.indexes_dir.glob("*") if p.is_dir()),
        "files": len(files),
        "includes_models": include_models,
    }
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST, json.dumps(manifest, indent=2))
        for f in files:
            zf.write(f, f.relative_to(s.data_dir).as_posix())
    return out


def restore(zip_path: Path) -> dict:
    """Unpack a snapshot into the data directory and return its manifest. Existing files with
    the same names are replaced; everything else in the data directory is kept."""
    s = runtime.settings()
    with zipfile.ZipFile(zip_path) as zf:
        if MANIFEST not in zf.namelist():
            raise SnapshotError(f"{zip_path} is not a snapshot (no {MANIFEST}).")
        manifest = json.loads(zf.read(MANIFEST))
        for member in zf.infolist():
            if member.filename == MANIFEST or member.is_dir():
                continue
            target = s.data_dir / _safe(member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))
    return manifest


def mismatches(manifest: dict, s: Settings) -> dict[str, tuple[str, str]]:
    """Setting → (snapshot value, current value), for every setting that differs."""
    current = settings_to_match(s)
    return {
        name: (value, current.get(name, ""))
        for name, value in manifest.get("settings", {}).items()
        if value != current.get(name, "")
    }


def _files(root: Path, data_dir: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for f in root.rglob("*"):
        rel = f.relative_to(data_dir).parts
        if f.is_file() and not (rel[0] == "cache" and len(rel) > 1 and rel[1] in SKIP_CACHE):
            yield f


def _safe(name: str) -> Path:
    """A zip member's path, refused if it would land outside the data directory."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.parts[0] not in TOP_DIRS:
        raise SnapshotError(f"Refusing to extract {name!r}: unexpected path.")
    return Path(*path.parts)
