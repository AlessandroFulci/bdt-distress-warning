"""
MinIO storage client for the Bronze layer.

Wraps raw uploads with two important Bronze-layer concepts:
  1. Immutability: files are written once, never overwritten in place.
  2. Lineage: every uploaded object gets a sidecar .meta.json with
     source URL, fetch timestamp, HTTP status, content hash, and size.

The caller passes raw bytes; this client handles the storage layout.
"""
import io
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class BronzeStore:
    """Bronze-layer object store backed by MinIO."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.bucket = bucket
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info(f"Created bucket: {self.bucket}")

    def object_exists(self, object_name: str) -> bool:
        """Check whether an object already exists (for resumable ingestion)."""
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False

    def put_with_metadata(
        self,
        object_name: str,
        data: bytes,
        source_url: str,
        http_status: int,
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Upload raw bytes to the bronze bucket and write a sidecar .meta.json
        next to it. Returns the metadata dict that was stored.
        """
        size = len(data)
        sha256 = hashlib.sha256(data).hexdigest()
        fetched_at = datetime.now(timezone.utc).isoformat()

        metadata = {
            "object_name": object_name,
            "source_url": source_url,
            "http_status": http_status,
            "fetched_at_utc": fetched_at,
            "size_bytes": size,
            "sha256": sha256,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        # Upload the main object
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            length=size,
            content_type="application/json",
        )

        # Upload the sidecar metadata
        meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
        meta_object_name = object_name + ".meta.json"
        self.client.put_object(
            self.bucket,
            meta_object_name,
            io.BytesIO(meta_bytes),
            length=len(meta_bytes),
            content_type="application/json",
        )

        return metadata

