"""Object storage: one private Cloud Storage bucket.

The `Storage` protocol stays because the ingestion pipeline takes a backend
rather than reaching for a global, which is what makes it testable. What is
gone is the *filesystem* implementation that used to stand in when no bucket
was configured: on Cloud Run that wrote uploaded PDFs to a container disk which
is discarded on every restart, so papers ingested successfully and then could
not be re-read. An unset bucket is now a startup failure, not a quiet downgrade.

Only the backend knows how a URI maps to bytes — callers pass the URI around
and never build a filesystem path from user input (`papers.original_filename`
is display-only, per ARCHITECTURE 4.3).
"""

from __future__ import annotations

import uuid
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


class GCSStorage:
    """Cloud Storage backend. Private objects, uniform bucket-level access."""

    def __init__(self, bucket: str) -> None:
        self._bucket_name = bucket

    def _bucket(self):
        from google.cloud import storage  # lazily: constructing a client resolves credentials

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
    return GCSStorage(get_settings().storage_bucket)


def new_object_name() -> str:
    return uuid.uuid4().hex
