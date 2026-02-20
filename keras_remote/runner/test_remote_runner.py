"""Tests for keras_remote.runner.remote_runner — GCS helpers and execution."""

import os
import pathlib
import shutil
import sys
import tempfile
import zipfile
from unittest import mock
from unittest.mock import MagicMock

import cloudpickle
from absl.testing import absltest

from keras_remote.runner.remote_runner import (
  _download_from_gcs,
  _upload_to_gcs,
  main,
  run_gcs_mode,
)


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class TestDownloadFromGcs(absltest.TestCase):
  def test_parses_gcs_path(self):
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    _download_from_gcs(
      mock_client, "gs://my-bucket/path/to/file.pkl", "/tmp/local.pkl"
    )

    mock_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.blob.assert_called_once_with("path/to/file.pkl")
    mock_blob.download_to_filename.assert_called_once_with("/tmp/local.pkl")

  def test_handles_nested_path(self):
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    _download_from_gcs(
      mock_client,
      "gs://bucket/a/b/c/deep/file.zip",
      "/tmp/out.zip",
    )

    mock_client.bucket.assert_called_once_with("bucket")
    mock_bucket.blob.assert_called_once_with("a/b/c/deep/file.zip")


class TestUploadToGcs(absltest.TestCase):
  def test_parses_gcs_path(self):
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    _upload_to_gcs(
      mock_client, "/tmp/result.pkl", "gs://my-bucket/results/result.pkl"
    )

    mock_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.blob.assert_called_once_with("results/result.pkl")
    mock_blob.upload_from_filename.assert_called_once_with("/tmp/result.pkl")


class TestRunGcsMode(absltest.TestCase):
  def setUp(self):
    super().setUp()
    # Prevent run_gcs_mode's sys.path.insert from leaking across tests.
    original_path = sys.path[:]
    self.addCleanup(setattr, sys, "path", original_path)

  def _setup_gcs_test(self, tmp_path, func, env_vars=None):
    """Set up common GCS test fixtures: context zip, payload pkl, patches."""
    if env_vars is None:
      env_vars = {}

    src_dir = tmp_path / "src"
    work_dir = tmp_path / "work"
    src_dir.mkdir()
    work_dir.mkdir()

    context_zip = src_dir / "context.zip"
    with zipfile.ZipFile(context_zip, "w") as zf:
      zf.writestr("dummy.py", "x = 1")

    payload_pkl = src_dir / "payload.pkl"
    with open(payload_pkl, "wb") as f:
      cloudpickle.dump(
        {
          "func": func,
          "args": getattr(func, "_test_args", ()),
          "kwargs": {},
          "env_vars": env_vars,
        },
        f,
      )

    mock_client = MagicMock()

    def fake_download(client, gcs_path, local_path):
      if "context.zip" in gcs_path:
        shutil.copy(str(context_zip), local_path)
      elif "payload.pkl" in gcs_path:
        shutil.copy(str(payload_pkl), local_path)

    return work_dir, mock_client, fake_download

  def test_success_flow(self):
    """Verify successful function execution: download, execute, upload result."""
    tmp_path = _make_temp_path(self)
    src_dir = tmp_path / "src"
    work_dir = tmp_path / "work"
    src_dir.mkdir()
    work_dir.mkdir()

    context_zip = src_dir / "context.zip"
    with zipfile.ZipFile(context_zip, "w") as zf:
      zf.writestr("dummy.py", "x = 1")

    payload_pkl = src_dir / "payload.pkl"

    def add(a, b):
      return a + b

    with open(payload_pkl, "wb") as f:
      cloudpickle.dump(
        {"func": add, "args": (2, 3), "kwargs": {}, "env_vars": {}},
        f,
      )

    mock_client = MagicMock()

    def fake_download(client, gcs_path, local_path):
      if "context.zip" in gcs_path:
        shutil.copy(str(context_zip), local_path)
      elif "payload.pkl" in gcs_path:
        shutil.copy(str(payload_pkl), local_path)

    with (
      mock.patch(
        "sys.argv",
        [
          "remote_runner.py",
          "gs://bucket/context.zip",
          "gs://bucket/payload.pkl",
          "gs://bucket/result.pkl",
        ],
      ),
      mock.patch(
        "keras_remote.runner.remote_runner.TEMP_DIR",
        str(work_dir),
      ),
      mock.patch(
        "keras_remote.runner.remote_runner._download_from_gcs",
        side_effect=fake_download,
      ),
      mock.patch(
        "keras_remote.runner.remote_runner._upload_to_gcs",
      ) as mock_upload,
      mock.patch(
        "keras_remote.runner.remote_runner.storage.Client",
        return_value=mock_client,
      ),
    ):
      with self.assertRaises(SystemExit) as cm:
        run_gcs_mode()
      self.assertEqual(cm.exception.code, 0)

      mock_upload.assert_called_once()
      upload_args = mock_upload.call_args[0]
      self.assertEqual(upload_args[2], "gs://bucket/result.pkl")

      result_path = upload_args[1]
      with open(result_path, "rb") as f:
        result_payload = cloudpickle.load(f)
      self.assertTrue(result_payload["success"])
      self.assertEqual(result_payload["result"], 5)

  def test_function_exception(self):
    """When the user function raises, result has success=False."""
    tmp_path = _make_temp_path(self)
    src_dir = tmp_path / "src"
    work_dir = tmp_path / "work"
    src_dir.mkdir()
    work_dir.mkdir()

    context_zip = src_dir / "context.zip"
    with zipfile.ZipFile(context_zip, "w") as zf:
      zf.writestr("dummy.py", "x = 1")

    def bad_func():
      raise ValueError("test error")

    payload_pkl = src_dir / "payload.pkl"
    with open(payload_pkl, "wb") as f:
      cloudpickle.dump(
        {"func": bad_func, "args": (), "kwargs": {}, "env_vars": {}},
        f,
      )

    mock_client = MagicMock()

    def fake_download(client, gcs_path, local_path):
      if "context.zip" in gcs_path:
        shutil.copy(str(context_zip), local_path)
      elif "payload.pkl" in gcs_path:
        shutil.copy(str(payload_pkl), local_path)

    with (
      mock.patch(
        "sys.argv",
        [
          "remote_runner.py",
          "gs://bucket/context.zip",
          "gs://bucket/payload.pkl",
          "gs://bucket/result.pkl",
        ],
      ),
      mock.patch(
        "keras_remote.runner.remote_runner.TEMP_DIR",
        str(work_dir),
      ),
      mock.patch(
        "keras_remote.runner.remote_runner._download_from_gcs",
        side_effect=fake_download,
      ),
      mock.patch(
        "keras_remote.runner.remote_runner._upload_to_gcs",
      ) as mock_upload,
      mock.patch(
        "keras_remote.runner.remote_runner.storage.Client",
        return_value=mock_client,
      ),
    ):
      with self.assertRaises(SystemExit) as cm:
        run_gcs_mode()
      self.assertEqual(cm.exception.code, 1)

      result_path = mock_upload.call_args[0][1]
      with open(result_path, "rb") as f:
        result_payload = cloudpickle.load(f)
      self.assertFalse(result_payload["success"])
      self.assertIsInstance(result_payload["exception"], ValueError)
      self.assertIn("test error", str(result_payload["exception"]))

  def test_env_vars_applied(self):
    """Verify env_vars from payload are set in os.environ."""
    tmp_path = _make_temp_path(self)
    src_dir = tmp_path / "src"
    work_dir = tmp_path / "work"
    src_dir.mkdir()
    work_dir.mkdir()

    context_zip = src_dir / "context.zip"
    with zipfile.ZipFile(context_zip, "w") as zf:
      zf.writestr("dummy.py", "x = 1")

    def read_env():
      return os.environ.get("TEST_REMOTE_VAR")

    payload_pkl = src_dir / "payload.pkl"
    with open(payload_pkl, "wb") as f:
      cloudpickle.dump(
        {
          "func": read_env,
          "args": (),
          "kwargs": {},
          "env_vars": {"TEST_REMOTE_VAR": "hello"},
        },
        f,
      )

    mock_client = MagicMock()

    def fake_download(client, gcs_path, local_path):
      if "context.zip" in gcs_path:
        shutil.copy(str(context_zip), local_path)
      elif "payload.pkl" in gcs_path:
        shutil.copy(str(payload_pkl), local_path)

    with (
      mock.patch(
        "sys.argv",
        [
          "remote_runner.py",
          "gs://bucket/context.zip",
          "gs://bucket/payload.pkl",
          "gs://bucket/result.pkl",
        ],
      ),
      mock.patch(
        "keras_remote.runner.remote_runner.TEMP_DIR",
        str(work_dir),
      ),
      mock.patch(
        "keras_remote.runner.remote_runner._download_from_gcs",
        side_effect=fake_download,
      ),
      mock.patch(
        "keras_remote.runner.remote_runner._upload_to_gcs",
      ) as mock_upload,
      mock.patch(
        "keras_remote.runner.remote_runner.storage.Client",
        return_value=mock_client,
      ),
    ):
      with self.assertRaises(SystemExit) as cm:
        run_gcs_mode()
      self.assertEqual(cm.exception.code, 0)

      result_path = mock_upload.call_args[0][1]
      with open(result_path, "rb") as f:
        result_payload = cloudpickle.load(f)
      self.assertTrue(result_payload["success"])
      self.assertEqual(result_payload["result"], "hello")


class TestMain(absltest.TestCase):
  def test_too_few_args(self):
    with mock.patch("sys.argv", ["remote_runner.py"]):
      with self.assertRaises(SystemExit) as cm:
        main()
      self.assertEqual(cm.exception.code, 1)

  def test_too_few_args_two(self):
    with mock.patch(
      "sys.argv", ["remote_runner.py", "gs://bucket/context.zip"]
    ):
      with self.assertRaises(SystemExit) as cm:
        main()
      self.assertEqual(cm.exception.code, 1)

  def test_correct_args_calls_run_gcs_mode(self):
    with (
      mock.patch(
        "sys.argv",
        [
          "remote_runner.py",
          "gs://bucket/context.zip",
          "gs://bucket/payload.pkl",
          "gs://bucket/result.pkl",
        ],
      ),
      mock.patch("keras_remote.runner.remote_runner.run_gcs_mode") as mock_run,
    ):
      main()
      mock_run.assert_called_once()


if __name__ == "__main__":
  absltest.main()
