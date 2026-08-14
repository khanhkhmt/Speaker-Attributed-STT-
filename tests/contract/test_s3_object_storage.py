"""S3/MinIO object-store contract — spec 10.1, 10.3."""

from __future__ import annotations

import io

import pytest

from sastt.adapters.storage import S3ObjectStore
from sastt.domain.errors import SasttError, TenantAccessDeniedError


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> None:
        self.put_calls.append(kwargs)
        self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = bytes(kwargs["Body"])

    def get_object(self, **kwargs: object) -> dict[str, object]:
        return {"Body": io.BytesIO(self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))])}

    def delete_object(self, **kwargs: object) -> None:
        self.objects.pop((str(kwargs["Bucket"]), str(kwargs["Key"])))


def test_s3_store_scopes_keys_and_requests_encryption() -> None:
    client = FakeS3Client()
    store = S3ObjectStore(client=client, bucket="audio", prefix="private")

    assert store.put("tenant-a", "jobs/job-1/input", b"audio") == (
        "s3://audio/private/tenant-a/jobs/job-1/input"
    )
    assert client.put_calls[0]["ServerSideEncryption"] == "AES256"
    assert store.get("tenant-a", "jobs/job-1/input") == b"audio"
    with pytest.raises(TenantAccessDeniedError):
        store.get("tenant-b", "jobs/job-1/input")


def test_s3_store_refuses_path_traversal() -> None:
    store = S3ObjectStore(client=FakeS3Client(), bucket="audio")
    with pytest.raises(SasttError):
        store.put("tenant-a", "../other/input", b"audio")
