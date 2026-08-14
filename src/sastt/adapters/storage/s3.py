"""Tenant-scoped S3-compatible object storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sastt.domain.errors import SasttError, TenantAccessDeniedError


def _safe_key(key: str) -> str:
    normalized = key.strip("/")
    if not normalized or ".." in normalized.split("/"):
        raise SasttError("invalid object key")
    return normalized


@dataclass
class S3ObjectStore:
    """``ObjectStore`` backed by a private S3-compatible bucket.

    Callers use logical object keys. The adapter prefixes every key with the
    tenant ID and requests server-side encryption on writes.
    """

    client: Any
    bucket: str
    prefix: str = "sastt"

    @classmethod
    def from_environment(cls) -> S3ObjectStore:
        """Build an S3 client lazily so unit tests need no cloud SDK."""
        import os

        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise SasttError("S3 object storage needs boto3") from exc
        client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        return cls(
            client=client,
            bucket=os.environ.get("S3_BUCKET", "sastt-audio"),
            prefix=os.environ.get("S3_PREFIX", "sastt"),
        )

    def _key(self, tenant_id: str, key: str) -> str:
        if not tenant_id:
            raise TenantAccessDeniedError("tenant is required for object storage")
        return f"{_safe_key(self.prefix)}/{_safe_key(tenant_id)}/{_safe_key(key)}"

    def put(self, tenant_id: str, key: str, payload: bytes) -> str:
        object_key = self._key(tenant_id, key)
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=payload,
                ServerSideEncryption="AES256",
            )
        except Exception as exc:  # noqa: BLE001 - SDK errors are implementation detail
            raise SasttError("could not store audio object") from exc
        return f"s3://{self.bucket}/{object_key}"

    def get(self, tenant_id: str, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(tenant_id, key))
            return bytes(response["Body"].read())
        except Exception as exc:  # noqa: BLE001 - never reveal another tenant's object
            raise TenantAccessDeniedError("object not found for this tenant") from exc

    def delete(self, tenant_id: str, key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self._key(tenant_id, key))
        except Exception as exc:  # noqa: BLE001
            raise TenantAccessDeniedError("object not found for this tenant") from exc
        return True
