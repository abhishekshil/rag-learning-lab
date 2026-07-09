# minio_blob.py
# Object storage for raw files (PDFs, images). Vectors store POINTERS, not blobs.

from __future__ import annotations

import os

from .config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    PAPER_DOC_ID,
)


class MinioBlobStore:
    """Upload and fetch raw files via MinIO (S3-compatible API)."""

    def __init__(
        self,
        *,
        endpoint: str = MINIO_ENDPOINT,
        access_key: str = MINIO_ACCESS_KEY,
        secret_key: str = MINIO_SECRET_KEY,
        bucket: str = MINIO_BUCKET,
    ):
        import boto3
        from botocore.client import Config

        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:
            self._client.create_bucket(Bucket=self.bucket)

    def upload_file(self, local_path: str, object_key: str | None = None) -> str:
        """Upload a local file; return an s3:// URI pointer."""
        key = object_key or f"{PAPER_DOC_ID}/{os.path.basename(local_path)}"
        self._client.upload_file(local_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download_text(self, media_uri: str) -> str:
        """Fetch text content from an s3://bucket/key pointer."""
        if not media_uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got {media_uri!r}")
        _, _, rest = media_uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def name(self) -> str:
        return f"MinIO(bucket={self.bucket})"
