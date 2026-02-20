"""Tests for keras_remote.backend.execution — JobContext and execute_remote."""

import re
from unittest.mock import MagicMock

import pytest

from keras_remote.backend.execution import (
  JobContext,
  _find_requirements,
  execute_remote,
)


class TestJobContext:
  def _make_func(self):
    def my_train():
      return 42

    return my_train

  def test_post_init_derived_fields(self):
    ctx = JobContext(
      func=self._make_func(),
      args=(),
      kwargs={},
      env_vars={},
      accelerator="cpu",
      container_image=None,
      zone="europe-west4-b",
      project="my-proj",
    )
    assert ctx.bucket_name == "my-proj-keras-remote-jobs"
    assert ctx.region == "europe-west4"
    assert ctx.display_name.startswith("keras-remote-my_train-")
    assert re.fullmatch(r"job-[0-9a-f]{8}", ctx.job_id)

  def test_from_params_explicit(self):
    ctx = JobContext.from_params(
      func=self._make_func(),
      args=(1, 2),
      kwargs={"k": "v"},
      accelerator="l4",
      container_image="my-image:latest",
      zone="us-west1-a",
      project="explicit-proj",
      env_vars={"X": "Y"},
    )
    assert ctx.zone == "us-west1-a"
    assert ctx.project == "explicit-proj"
    assert ctx.accelerator == "l4"
    assert ctx.container_image == "my-image:latest"
    assert ctx.args == (1, 2)
    assert ctx.kwargs == {"k": "v"}
    assert ctx.env_vars == {"X": "Y"}

  def test_from_params_resolves_zone_from_env(self, monkeypatch):
    monkeypatch.setenv("KERAS_REMOTE_ZONE", "asia-east1-c")
    monkeypatch.setenv("KERAS_REMOTE_PROJECT", "env-proj")

    ctx = JobContext.from_params(
      func=self._make_func(),
      args=(),
      kwargs={},
      accelerator="cpu",
      container_image=None,
      zone=None,
      project=None,
      env_vars={},
    )
    assert ctx.zone == "asia-east1-c"
    assert ctx.project == "env-proj"

  def test_from_params_no_project_raises(self, monkeypatch):
    monkeypatch.delenv("KERAS_REMOTE_PROJECT", raising=False)

    with pytest.raises(ValueError, match="project must be specified"):
      JobContext.from_params(
        func=self._make_func(),
        args=(),
        kwargs={},
        accelerator="cpu",
        container_image=None,
        zone="us-central1-a",
        project=None,
        env_vars={},
      )


class TestFindRequirements:
  def test_finds_in_start_dir(self, tmp_path):
    """Returns the path when requirements.txt exists in the start directory."""
    (tmp_path / "requirements.txt").write_text("numpy\n")
    assert _find_requirements(str(tmp_path)) == str(
      tmp_path / "requirements.txt"
    )

  def test_finds_in_parent_dir(self, tmp_path):
    """Walks up the directory tree to find requirements.txt in a parent."""
    (tmp_path / "requirements.txt").write_text("numpy\n")
    child = tmp_path / "subdir"
    child.mkdir()
    assert _find_requirements(str(child)) == str(tmp_path / "requirements.txt")

  def test_returns_none_when_not_found(self, tmp_path):
    """Returns None when no requirements.txt exists in any ancestor."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _find_requirements(str(empty)) is None


class TestExecuteRemote:
  def _make_func(self):
    def my_train():
      return 42

    return my_train

  def _make_ctx(self, container_image=None):
    return JobContext(
      func=self._make_func(),
      args=(),
      kwargs={},
      env_vars={},
      accelerator="cpu",
      container_image=container_image,
      zone="us-central1-a",
      project="proj",
    )

  def test_success_flow(self, mocker):
    mocker.patch("keras_remote.backend.execution._build_container")
    mocker.patch("keras_remote.backend.execution._upload_artifacts")
    mocker.patch(
      "keras_remote.backend.execution._download_result",
      return_value={"success": True, "result": 42},
    )
    mocker.patch(
      "keras_remote.backend.execution._cleanup_and_return",
      return_value=42,
    )

    ctx = self._make_ctx()
    backend = MagicMock()

    result = execute_remote(ctx, backend)

    backend.submit_job.assert_called_once_with(ctx)
    backend.wait_for_job.assert_called_once()
    backend.cleanup_job.assert_called_once()
    assert result == 42

  def test_cleanup_on_wait_failure(self, mocker):
    mocker.patch("keras_remote.backend.execution._build_container")
    mocker.patch("keras_remote.backend.execution._upload_artifacts")

    ctx = self._make_ctx()
    backend = MagicMock()
    backend.wait_for_job.side_effect = RuntimeError("job failed")

    with pytest.raises(RuntimeError, match="job failed"):
      execute_remote(ctx, backend)

    # cleanup_job is called in finally block even when wait fails
    backend.cleanup_job.assert_called_once()
