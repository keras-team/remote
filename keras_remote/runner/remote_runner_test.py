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
  _download_data,
  _download_from_gcs,
  _upload_to_gcs,
  main,
  resolve_data_refs,
  resolve_volumes,
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


class TestDownloadData(absltest.TestCase):
  def test_downloads_files_skips_marker(self):
    tmp = _make_temp_path(self)
    target = tmp / "output"

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    blob_data = MagicMock()
    blob_data.name = "prefix/hash/train.csv"
    blob_data.download_to_filename = MagicMock(
      side_effect=lambda p: pathlib.Path(p).write_text("train")
    )

    blob_marker = MagicMock()
    blob_marker.name = "prefix/hash/.cache_marker"

    blob_dir = MagicMock()
    blob_dir.name = "prefix/hash/"

    mock_bucket.list_blobs.return_value = [
      blob_dir,
      blob_data,
      blob_marker,
    ]

    ref = {
      "__data_ref__": True,
      "gcs_uri": "gs://bucket/prefix/hash",
      "is_dir": True,
    }

    _download_data(ref, str(target), mock_client)

    blob_data.download_to_filename.assert_called_once()
    blob_marker.download_to_filename.assert_not_called()
    blob_dir.download_to_filename.assert_not_called()

  def test_creates_subdirectories(self):
    tmp = _make_temp_path(self)
    target = tmp / "output"

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    blob = MagicMock()
    blob.name = "prefix/hash/sub/deep.csv"
    blob.download_to_filename = MagicMock(
      side_effect=lambda p: (
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
        or pathlib.Path(p).write_text("data")
      )
    )
    mock_bucket.list_blobs.return_value = [blob]

    ref = {
      "__data_ref__": True,
      "gcs_uri": "gs://bucket/prefix/hash",
      "is_dir": True,
    }

    _download_data(ref, str(target), mock_client)

    # The call should include the nested path
    call_path = blob.download_to_filename.call_args[0][0]
    self.assertIn("sub", call_path)
    self.assertTrue(call_path.endswith("deep.csv"))


class TestResolveDataRefs(absltest.TestCase):
  def test_replaces_ref_with_path(self):
    tmp = _make_temp_path(self)
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = []

    ref = {
      "__data_ref__": True,
      "gcs_uri": "gs://b/p",
      "is_dir": True,
      "mount_path": None,
    }

    with mock.patch(
      "keras_remote.runner.remote_runner.DATA_DIR",
      str(tmp / "data"),
    ):
      args, kwargs = resolve_data_refs((ref, 42), {}, mock_client)

    self.assertIsInstance(args[0], str)
    self.assertEqual(args[1], 42)

  def test_nested_refs_in_list(self):
    tmp = _make_temp_path(self)
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = []

    ref = {
      "__data_ref__": True,
      "gcs_uri": "gs://b/p",
      "is_dir": True,
      "mount_path": None,
    }

    with mock.patch(
      "keras_remote.runner.remote_runner.DATA_DIR",
      str(tmp / "data"),
    ):
      args, _ = resolve_data_refs(([ref, "other"],), {}, mock_client)

    self.assertIsInstance(args[0][0], str)
    self.assertEqual(args[0][1], "other")

  def test_single_file_returns_file_path(self):
    tmp = _make_temp_path(self)
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    blob = MagicMock()
    blob.name = "prefix/hash/config.json"
    blob.download_to_filename = MagicMock(
      side_effect=lambda p: pathlib.Path(p).write_text("{}")
    )
    mock_bucket.list_blobs.return_value = [blob]

    ref = {
      "__data_ref__": True,
      "gcs_uri": "gs://b/prefix/hash",
      "is_dir": False,
      "mount_path": None,
    }

    with mock.patch(
      "keras_remote.runner.remote_runner.DATA_DIR",
      str(tmp / "data"),
    ):
      args, _ = resolve_data_refs((ref,), {}, mock_client)

    self.assertTrue(args[0].endswith("config.json"))

  def test_non_ref_dict_preserved(self):
    mock_client = MagicMock()
    args, kwargs = resolve_data_refs(({"key": "value"},), {"x": 1}, mock_client)
    self.assertEqual(args[0], {"key": "value"})
    self.assertEqual(kwargs["x"], 1)

  def test_kwargs_refs_resolved(self):
    tmp = _make_temp_path(self)
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = []

    ref = {
      "__data_ref__": True,
      "gcs_uri": "gs://b/p",
      "is_dir": True,
      "mount_path": None,
    }

    with mock.patch(
      "keras_remote.runner.remote_runner.DATA_DIR",
      str(tmp / "data"),
    ):
      _, kwargs = resolve_data_refs((), {"data": ref, "lr": 0.01}, mock_client)

    self.assertIsInstance(kwargs["data"], str)
    self.assertEqual(kwargs["lr"], 0.01)


class TestResolveVolumes(absltest.TestCase):
  def test_downloads_to_mount_path(self):
    tmp = _make_temp_path(self)
    mount_path = str(tmp / "data")

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = []

    refs = [
      {
        "__data_ref__": True,
        "gcs_uri": "gs://b/cache/hash",
        "is_dir": True,
        "mount_path": mount_path,
      }
    ]

    resolve_volumes(refs, mock_client)

    self.assertTrue(os.path.isdir(mount_path))

  def test_multiple_volumes(self):
    tmp = _make_temp_path(self)
    path1 = str(tmp / "data1")
    path2 = str(tmp / "data2")

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = []

    refs = [
      {
        "__data_ref__": True,
        "gcs_uri": "gs://b/h1",
        "is_dir": True,
        "mount_path": path1,
      },
      {
        "__data_ref__": True,
        "gcs_uri": "gs://b/h2",
        "is_dir": True,
        "mount_path": path2,
      },
    ]

    resolve_volumes(refs, mock_client)

    self.assertTrue(os.path.isdir(path1))
    self.assertTrue(os.path.isdir(path2))


class TestRunGcsMode(absltest.TestCase):
  def setUp(self):
    super().setUp()
    original_path = sys.path[:]
    self.addCleanup(setattr, sys, "path", original_path)

  def _setup_gcs_test(
    self, tmp_path, func, args=(), env_vars=None, volumes=None
  ):
    """Set up common GCS test fixtures."""
    if env_vars is None:
      env_vars = {}

    src_dir = tmp_path / "src"
    src_dir.mkdir()

    context_zip = src_dir / "context.zip"
    with zipfile.ZipFile(context_zip, "w") as zf:
      zf.writestr("dummy.py", "x = 1")

    payload = {
      "func": func,
      "args": args,
      "kwargs": {},
      "env_vars": env_vars,
    }
    if volumes:
      payload["volumes"] = volumes

    payload_pkl = src_dir / "payload.pkl"
    with open(payload_pkl, "wb") as f:
      cloudpickle.dump(payload, f)

    mock_client = MagicMock()

    def fake_download(client, gcs_path, local_path):
      if "context.zip" in gcs_path:
        shutil.copy(str(context_zip), local_path)
      elif "payload.pkl" in gcs_path:
        shutil.copy(str(payload_pkl), local_path)

    return mock_client, fake_download

  def _run_gcs_mode(self, func, args=(), env_vars=None, volumes=None):
    """Set up fixtures, run run_gcs_mode(), return (exit_code, result)."""
    tmp_path = _make_temp_path(self)
    mock_client, fake_download = self._setup_gcs_test(
      tmp_path,
      func,
      args=args,
      env_vars=env_vars,
      volumes=volumes,
    )

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

      result_path = mock_upload.call_args[0][1]
      with open(result_path, "rb") as f:
        result_payload = cloudpickle.load(f)

    return cm.exception.code, result_payload

  def test_success_flow(self):
    def add(a, b):
      return a + b

    exit_code, result = self._run_gcs_mode(add, args=(2, 3))

    self.assertEqual(exit_code, 0)
    self.assertTrue(result["success"])
    self.assertEqual(result["result"], 5)

  def test_function_exception(self):
    def bad_func():
      raise ValueError("test error")

    exit_code, result = self._run_gcs_mode(bad_func)

    self.assertEqual(exit_code, 1)
    self.assertFalse(result["success"])
    self.assertIsInstance(result["exception"], ValueError)
    self.assertIn("test error", str(result["exception"]))

  def test_env_vars_applied(self):
    def read_env():
      return os.environ.get("TEST_REMOTE_VAR")

    exit_code, result = self._run_gcs_mode(
      read_env, env_vars={"TEST_REMOTE_VAR": "hello"}
    )

    self.assertEqual(exit_code, 0)
    self.assertTrue(result["success"])
    self.assertEqual(result["result"], "hello")

  def test_data_ref_resolved_before_execution(self):
    """Data refs in args are resolved to local paths."""

    def check_is_string(data_path):
      assert isinstance(data_path, str), f"Expected str, got {type(data_path)}"
      return "resolved"

    ref = {
      "__data_ref__": True,
      "gcs_uri": "gs://b/cache/hash",
      "is_dir": True,
      "mount_path": None,
    }

    with mock.patch(
      "keras_remote.runner.remote_runner._download_data"
    ) as mock_dl:

      def fake_dl(ref, target_dir, client):
        os.makedirs(target_dir, exist_ok=True)

      mock_dl.side_effect = fake_dl

      exit_code, result = self._run_gcs_mode(check_is_string, args=(ref,))

    self.assertEqual(exit_code, 0)
    self.assertTrue(result["success"])
    self.assertEqual(result["result"], "resolved")

  def test_volumes_resolved_before_execution(self):
    """Volumes are downloaded to mount paths before function execution."""
    tmp = _make_temp_path(self)
    mount_path = str(tmp / "mounted_data")

    def check_mount(expected_path):
      assert os.path.isdir(expected_path), (
        f"Mount path should exist: {expected_path}"
      )
      return "mounted"

    volume_refs = [
      {
        "__data_ref__": True,
        "gcs_uri": "gs://b/cache/hash",
        "is_dir": True,
        "mount_path": mount_path,
      }
    ]

    with mock.patch(
      "keras_remote.runner.remote_runner._download_data"
    ) as mock_dl:

      def fake_dl(ref, target_dir, client):
        os.makedirs(target_dir, exist_ok=True)

      mock_dl.side_effect = fake_dl

      exit_code, result = self._run_gcs_mode(
        check_mount,
        args=(mount_path,),
        volumes=volume_refs,
      )

    self.assertEqual(exit_code, 0)
    self.assertTrue(result["success"])
    self.assertEqual(result["result"], "mounted")

  def test_no_data_no_volumes_unchanged(self):
    """Original behavior preserved when no Data args or volumes."""

    def identity(x):
      return x

    exit_code, result = self._run_gcs_mode(identity, args=(42,))

    self.assertEqual(exit_code, 0)
    self.assertTrue(result["success"])
    self.assertEqual(result["result"], 42)


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
