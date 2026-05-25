"""
Resolve file:// and http(s):// URLs to local filesystem paths.

- file://  → direct path extraction (zero-copy, same-machine)
- http(s):// → download to temporary directory, return local path
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

_TEMP_PREFIX = "mvp_async_"


def resolve_url(url: str, *, temp_dir: str | None = None) -> str:
    """Resolve *url* to a local filesystem path.

    For ``file://`` URLs the local path is extracted directly.
    For ``http(s)://`` URLs the content is downloaded into *temp_dir*
    (or a newly created temporary directory) and the local path returned.

    The caller is responsible for cleaning up *temp_dir* after processing.
    """
    parsed = urlparse(url)

    if parsed.scheme == "file":
        return _resolve_file(parsed)

    if parsed.scheme in ("http", "https"):
        return _download(url, temp_dir=temp_dir)

    raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r} (supported: file, http, https)")


def _resolve_file(parsed: urlparse) -> str:
    """Extract local path from a ``file://`` URL.

    Handles both POSIX (``file:///home/user/img.jpg``) and Windows
    (``file:///D:/datahub/img.jpg``) forms.
    """
    path = parsed.path
    # On Windows, urlparse gives /D:/path — strip leading /
    if path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path.lstrip("/")
    if not os.path.exists(path):
        raise FileNotFoundError(f"file:// path does not exist: {path}")
    return path


def _download(url: str, *, temp_dir: str | None) -> str:
    """Download *url* to a local temp directory and return the local path."""
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix=_TEMP_PREFIX)

    parsed = urlparse(url)
    # Derive a sensible filename from the URL path
    filename = os.path.basename(parsed.path.rstrip("/"))
    if not filename:
        filename = "download"

    dest = os.path.join(temp_dir, filename)

    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)

    return dest


def make_temp_dir() -> str:
    """Create a temporary directory for downloaded assets."""
    return tempfile.mkdtemp(prefix=_TEMP_PREFIX)


def clean_temp_dir(temp_dir: str) -> None:
    """Remove a temporary directory and all its contents."""
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
