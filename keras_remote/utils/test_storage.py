"""Tests for keras_remote.utils.storage — Cloud Storage operations."""

import os
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from keras_remote.utils.storage import (
  _get_project,
  cleanup_artifacts,
  download_result,
  upload_artifacts,
)


class _GcsTestBase(absltest.TestCase):
  """Base class that provides a mocked GCS client."""

  def setUp(self):
    super().setUp()
    self.mock_gcs = MagicMock()
    patcher = mock.patch(
      "keras_remote.utils.storage.storage.Client",
      return_value=self.mock_gcs,
    )
    patcher.start()
    self.addCleanup(patcher.stop)


class TestUploadArtifacts(_GcsTestBase):
  def test_uploads_payload_and_context(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    upload_artifacts(
      bucket_name="my-bucket",
      job_id="job-abc123",
      payload_path="/tmp/payload.pkl",
      context_path="/tmp/context.zip",
      project="test-project",
    )

    mock_bucket.blob.assert_any_call("job-abc123/payload.pkl")
    mock_bucket.blob.assert_any_call("job-abc123/context.zip")
    self.assertEqual(mock_blob.upload_from_filename.call_count, 2)

  def test_uses_correct_bucket(self):
    upload_artifacts(
      bucket_name="my-custom-bucket",
      job_id="job-123",
      payload_path="/tmp/p.pkl",
      context_path="/tmp/c.zip",
      project="proj",
    )
    self.mock_gcs.bucket.assert_called_with("my-custom-bucket")


class TestDownloadResult(_GcsTestBase):
  def test_downloads_result_blob(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    download_result("my-bucket", "job-abc", project="proj")

    mock_bucket.blob.assert_called_once_with("job-abc/result.pkl")
    mock_blob.download_to_filename.assert_called_once()

  def test_returns_path_with_job_id(self):
    result = download_result("my-bucket", "job-xyz", project="proj")
    self.assertIn("result-job-xyz.pkl", result)


class TestCleanupArtifacts(_GcsTestBase):
  def test_deletes_all_blobs(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    blob1 = MagicMock()
    blob2 = MagicMock()
    blob3 = MagicMock()
    mock_bucket.list_blobs.return_value = [blob1, blob2, blob3]

    cleanup_artifacts("my-bucket", "job-abc", project="proj")

    mock_bucket.list_blobs.assert_called_once_with(prefix="job-abc/")
    blob1.delete.assert_called_once()
    blob2.delete.assert_called_once()
    blob3.delete.assert_called_once()

  def test_no_blobs_no_error(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_bucket.list_blobs.return_value = []

    cleanup_artifacts("my-bucket", "job-abc", project="proj")

    mock_bucket.list_blobs.assert_called_once_with(prefix="job-abc/")


class TestGetProject(parameterized.TestCase):
  @parameterized.parameters(
    # Only KERAS_REMOTE_PROJECT set: use it directly
    ("kr-proj", None, "kr-proj"),
    # Only GOOGLE_CLOUD_PROJECT set: fall back to it
    (None, "gc-proj", "gc-proj"),
    # Neither set: no project resolved
    (None, None, None),
    # Both set: KERAS_REMOTE_PROJECT takes precedence
    ("kr-proj", "gc-proj", "kr-proj"),
  )
  def test_resolves_project(self, kr_project, gc_project, expected):
    env = {
      k: v
      for k, v in os.environ.items()
      if k not in ("KERAS_REMOTE_PROJECT", "GOOGLE_CLOUD_PROJECT")
    }
    if kr_project:
      env["KERAS_REMOTE_PROJECT"] = kr_project
    if gc_project:
      env["GOOGLE_CLOUD_PROJECT"] = gc_project
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(_get_project(), expected)


if __name__ == "__main__":
  absltest.main()
