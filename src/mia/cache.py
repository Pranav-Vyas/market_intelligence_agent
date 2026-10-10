"""Disk cache for everything fetched from the network: pages, API responses, search results.

Values must be JSON-serializable. In offline mode a cache miss raises `CacheMiss` instead of
touching the network, which is what makes `mia run --offline` and the frozen snapshot work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mia import runtime
from mia.schemas import utcnow


class CacheMiss(Exception):
    """Offline mode and the item isn't cached."""


def _path(namespace: str, key: Any) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return runtime.settings().cache_dir / namespace / digest[:2] / f"{digest}.json"


def get(namespace: str, key: Any) -> Any | None:
    path = _path(namespace, key)
    if not path.exists():
        return None
    return json.loads(path.read_text())["value"]


def put(namespace: str, key: Any, value: Any) -> None:
    path = _path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "stored_at": utcnow().isoformat(), "value": value}
    path.write_text(json.dumps(record))


def get_or_fetch(namespace: str, key: Any, fetch: Callable[[], Any]) -> Any:
    """Return the cached value, or call `fetch()` and cache its result (unless it is None)."""
    cached = get(namespace, key)
    if cached is not None:
        return cached
    if runtime.is_offline():
        raise CacheMiss(f"{namespace}: {key} is not cached (offline mode)")
    value = fetch()
    if value is not None:
        put(namespace, key, value)
    return value
