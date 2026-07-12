from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import Settings, get_settings


class ObjectStorage:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path" if self.settings.s3_force_path_style else "auto"},
        )
        self.client = self._make_client(self.settings.s3_endpoint_url)
        public_endpoint = self.settings.s3_public_endpoint_url
        self.signing_client = (
            self._make_client(public_endpoint) if public_endpoint else self.client
        )
        self.bucket = self.settings.s3_bucket

    def _make_client(self, endpoint_url: str | None):
        return boto3.client(
            "s3",
            region_name=self.settings.s3_region,
            endpoint_url=endpoint_url,
            aws_access_key_id=self.settings.s3_access_key_id,
            aws_secret_access_key=self.settings.s3_secret_access_key,
            config=self.config,
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code not in {"404", "NoSuchBucket", "NotFound"}:
                raise

        kwargs: dict[str, Any] = {"Bucket": self.bucket}
        if self.settings.s3_region != "us-east-1" and not self.settings.s3_endpoint_url:
            kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": self.settings.s3_region
            }
        self.client.create_bucket(**kwargs)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def upload_file(self, source: str | Path, key: str, sha256: str) -> None:
        self.client.upload_file(
            str(source),
            self.bucket,
            key,
            ExtraArgs={
                "ContentType": "application/x-ndjson",
                "Metadata": {"sha256": sha256},
            },
        )

    def get_object(self, key: str, byte_range: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if byte_range:
            kwargs["Range"] = byte_range
        return self.client.get_object(**kwargs)

    def iter_body(self, body: Any, chunk_size: int) -> Iterator[bytes]:
        try:
            for chunk in body.iter_chunks(chunk_size=chunk_size):
                if chunk:
                    yield chunk
        finally:
            body.close()

    def presigned_get_url(self, key: str, expires_in: int) -> str:
        return self.signing_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )
