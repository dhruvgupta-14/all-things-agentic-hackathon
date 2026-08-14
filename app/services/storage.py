"""Object storage behind a two-method seam.

Local development writes to a directory; deployment swaps in Cloud Storage
without the ingestion pipeline noticing. Only the backend knows how a URI maps
to bytes — callers pass the URI around and never build a filesystem path from
user input (`papers.original_filename` is display-only, per ARCHITECTURE 4.3).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol

from app.config import get_settings


class ObjectNotFoundError(Exception):
    """The stored original is gone. Ingestion treats this as permanent."""


class Storage(Protocol):
    def put(self, data: bytes, *, content_hash: str) -> str:
        """Store bytes and return the URI they can be fetched back with."""
        ...

    def get(self, uri: str) -> bytes:
        ...


class LocalStorage:
    """Filesystem backend for local development."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, name: str) -> Path:
        # Resolve and re-check containment: the name is derived from a hash we
        # computed, but this class must not become a path-traversal primitive
        # if a caller ever passes something else.
        candidate = (self._root / name).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"refusing to address {name!r} outside the storage root")
        return candidate

    def put(self, data: bytes, *, content_hash: str) -> str:
        name = f"{content_hash}.pdf"
        self._path_for(name).write_bytes(data)
        return f"file://{name}"

    def get(self, uri: str) -> bytes:
        if not uri.startswith("file://"):
            raise ObjectNotFoundError(f"not a local storage uri: {uri!r}")
        path = self._path_for(uri.removeprefix("file://"))
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(uri) from exc


class GCSStorage:
    """Cloud Storage backend. Private objects, uniform bucket-level access."""

    def __init__(self, bucket: str) -> None:
        self._bucket_name = bucket

    def _bucket(self):
        from google.cloud import storage  # imported lazily: local dev has no creds

        return storage.Client().bucket(self._bucket_name)

    def put(self, data: bytes, *, content_hash: str) -> str:
        name = f"papers/{content_hash}.pdf"
        self._bucket().blob(name).upload_from_string(data, content_type="application/pdf")
        return f"gs://{self._bucket_name}/{name}"

    def get(self, uri: str) -> bytes:
        prefix = f"gs://{self._bucket_name}/"
        if not uri.startswith(prefix):
            raise ObjectNotFoundError(f"uri does not belong to this bucket: {uri!r}")
        blob = self._bucket().blob(uri.removeprefix(prefix))
        if not blob.exists():
            raise ObjectNotFoundError(uri)
        return blob.download_as_bytes()


def get_storage() -> Storage:
    settings = get_settings()
    if settings.storage_bucket:
        return GCSStorage(settings.storage_bucket)
    return LocalStorage(settings.local_storage_dir)


def new_object_name() -> str:
    return uuid.uuid4().hex
