"""Shared artifact id validation and repair helpers."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any


ARTIFACT_ID_RE = re.compile(r"[0-9a-f]{32}")


def is_valid_artifact_id(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return ARTIFACT_ID_RE.fullmatch(text) is not None


def coerce_artifact_id(value: Any, *, seed: str | None = None) -> str:
    """Return a valid 32-hex artifact id, preserving already valid ids."""
    text = str(value or "").strip().lower()
    if ARTIFACT_ID_RE.fullmatch(text):
        return text
    if seed:
        return hashlib.md5(seed.encode("utf-8")).hexdigest()
    return uuid.uuid4().hex
