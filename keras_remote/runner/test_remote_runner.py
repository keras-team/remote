"""Tests for keras_remote.runner.remote_runner — GCS helpers and execution."""

import os
import sys

import cloudpickle
import pytest

from keras_remote.runner.remote_runner import (
  _download_from_gcs,
  _upload_to_gcs,
  main,
  run_gcs_mode,
)


class TestDownloadFromGcs:
  def test_parses_gcs_path(self, mocker):
    mock_client = mocker.MagicMock()
    mock_bucket = mocker.MagicMock()
    mock_blob = mocker.MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    _download_from_gcs(
      mock_client, "gs://my-bucket/path/to/file.pkl", "/tmp/local.pkl"
    )

    mock_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.blob.assert_called_once_with("path/to/file.pkl")
    mock_blob.download_to_filename.assert_called_once_with("/tmp/local.pkl")

  def test_handles_nested_path(self, mocker):
    mock_client = mocker.MagicMock()
    mock_bucket = mocker.MagicMock()
    mock_blob = mocker.MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    _download_from_gcs(
      mock_client,
      "gs://bucket/a/b/c/deep/file.zip",
      "/tmp/out.zip",
    )

    mock_client.bucket.assert_called_once_with("bucket")
    mock_bucket.blob.assert_called_once_with("a/b/c/deep/file.zip")


class TestUploadToGcs:
  def test_parses_gcs_path(self, mocker):
    mock_client = mocker.MagicMock()
    mock_bucket = mocker.MagicMock()
    mock_blob = mocker.MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    _upload_to_gcs(
      mock_client, "/tmp/result.pkl", "gs://my-bucket/results/result.pkl"
    )

    mock_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.blob.assert_called_once_with("results/result.pkl")
    mock_blob.upload_from_filename.assert_called_once_with("/tmp/result.pkl")


class TestRunGcsMode:
  @pytest.fixture(autouse=True)
  def _protect_sys_path(self, monkeypatch):
    """Prevent run_gcs_mode's sys.path.insert from leaking across tests."""
    monkeypatch.setattr(sys, "path", sys.path[:])

  def test_success_flow(self, mocker, tmp_path):
    """Verify successful function execution: download, execute, upload result."""
    src_dir = tmp_path / "src"
    work_dir = tmp_path / "work"
    src_dir.mkdir()
    work_dir.mkdir()

    mocker.patch(
      "sys.argv",
      [
        "remote_runner.py",
        "gs://bucket/context.zip",
        "gs://bucket/payload.pkl",
        "gs://bucket/result.pkl",
      ],
    )
    mocker.patch(
      "keras_remote.runner.remote_runner.TEMP_DIR",
      str(work_dir),
    )

    import zipfile

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

    mock_client = mocker.MagicMock()

    def fake_download(client, gcs_path, local_path):
      import shutil

      if "context.zip" in gcs_path:
        shutil.copy(str(context_zip), local_path)
      elif "payload.pkl" in gcs_path:
        shutil.copy(str(payload_pkl), local_path)

    mocker.patch(
      "keras_remote.runner.remote_runner._download_from_gcs",
      side_effect=fake_download,
    )
    mock_upload = mocker.patch(
      "keras_remote.runner.remote_runner._upload_to_gcs",
    )
    mocker.patch(
      "keras_remote.runner.remote_runner.storage.Client",
      return_value=mock_client,
    )

    # run_gcs_mode calls sys.exit, so catch it
    with pytest.raises(SystemExit) as exc_info:
      run_gcs_mode()

    assert exc_info.value.code == 0

    # Verify upload was called
    mock_upload.assert_called_once()
    upload_args = mock_upload.call_args[0]
    assert upload_args[2] == "gs://bucket/result.pkl"

    # Verify result payload
    result_path = upload_args[1]
    with open(result_path, "rb") as f:
      result_payload = cloudpickle.load(f)
    assert result_payload["success"] is True
    assert result_payload["result"] == 5

  def test_function_exception(self, mocker, tmp_path):
    """When the user function raises, result has success=False."""
    src_dir = tmp_path / "src"
    work_dir = tmp_path / "work"
    src_dir.mkdir()
    work_dir.mkdir()

    mocker.patch(
      "sys.argv",
      [
        "remote_runner.py",
        "gs://bucket/context.zip",
        "gs://bucket/payload.pkl",
        "gs://bucket/result.pkl",
      ],
    )
    mocker.patch(
      "keras_remote.runner.remote_runner.TEMP_DIR",
      str(work_dir),
    )

    import zipfile

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

    mock_client = mocker.MagicMock()

    def fake_download(client, gcs_path, local_path):
      import shutil

      if "context.zip" in gcs_path:
        shutil.copy(str(context_zip), local_path)
      elif "payload.pkl" in gcs_path:
        shutil.copy(str(payload_pkl), local_path)

    mocker.patch(
      "keras_remote.runner.remote_runner._download_from_gcs",
      side_effect=fake_download,
    )
    mock_upload = mocker.patch(
      "keras_remote.runner.remote_runner._upload_to_gcs",
    )
    mocker.patch(
      "keras_remote.runner.remote_runner.storage.Client",
      return_value=mock_client,
    )

    with pytest.raises(SystemExit) as exc_info:
      run_gcs_mode()

    assert exc_info.value.code == 1

    # Verify result payload has the exception
    result_path = mock_upload.call_args[0][1]
    with open(result_path, "rb") as f:
      result_payload = cloudpickle.load(f)
    assert result_payload["success"] is False
    assert isinstance(result_payload["exception"], ValueError)
    assert "test error" in str(result_payload["exception"])

  def test_env_vars_applied(self, mocker, tmp_path):
    """Verify env_vars from payload are set in os.environ."""
    src_dir = tmp_path / "src"
    work_dir = tmp_path / "work"
    src_dir.mkdir()
    work_dir.mkdir()

    mocker.patch(
      "sys.argv",
      [
        "remote_runner.py",
        "gs://bucket/context.zip",
        "gs://bucket/payload.pkl",
        "gs://bucket/result.pkl",
      ],
    )
    mocker.patch(
      "keras_remote.runner.remote_runner.TEMP_DIR",
      str(work_dir),
    )

    import zipfile

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

    mock_client = mocker.MagicMock()

    def fake_download(client, gcs_path, local_path):
      import shutil

      if "context.zip" in gcs_path:
        shutil.copy(str(context_zip), local_path)
      elif "payload.pkl" in gcs_path:
        shutil.copy(str(payload_pkl), local_path)

    mocker.patch(
      "keras_remote.runner.remote_runner._download_from_gcs",
      side_effect=fake_download,
    )
    mock_upload = mocker.patch(
      "keras_remote.runner.remote_runner._upload_to_gcs",
    )
    mocker.patch(
      "keras_remote.runner.remote_runner.storage.Client",
      return_value=mock_client,
    )

    with pytest.raises(SystemExit) as exc_info:
      run_gcs_mode()

    assert exc_info.value.code == 0

    result_path = mock_upload.call_args[0][1]
    with open(result_path, "rb") as f:
      result_payload = cloudpickle.load(f)
    assert result_payload["success"] is True
    assert result_payload["result"] == "hello"


class TestMain:
  def test_too_few_args(self, mocker):
    mocker.patch("sys.argv", ["remote_runner.py"])
    with pytest.raises(SystemExit) as exc_info:
      main()
    assert exc_info.value.code == 1

  def test_too_few_args_two(self, mocker):
    mocker.patch("sys.argv", ["remote_runner.py", "gs://bucket/context.zip"])
    with pytest.raises(SystemExit) as exc_info:
      main()
    assert exc_info.value.code == 1

  def test_correct_args_calls_run_gcs_mode(self, mocker):
    mocker.patch(
      "sys.argv",
      [
        "remote_runner.py",
        "gs://bucket/context.zip",
        "gs://bucket/payload.pkl",
        "gs://bucket/result.pkl",
      ],
    )
    mock_run = mocker.patch("keras_remote.runner.remote_runner.run_gcs_mode")
    main()
    mock_run.assert_called_once()
