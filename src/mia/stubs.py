"""Marks functions that still return fake data, so a run can report what isn't real yet.

When you replace a stub with a real implementation, delete its `@stub` decorator.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

log = logging.getLogger(__name__)
_used: set[str] = set()


def stub[F: Callable](func: F) -> F:
    name = f"{func.__module__.removeprefix('mia.')}.{func.__qualname__}"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if name not in _used:
            log.info("Using stub: %s (returns fake data)", name)
            _used.add(name)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def used() -> list[str]:
    return sorted(_used)


def reset() -> None:
    _used.clear()
