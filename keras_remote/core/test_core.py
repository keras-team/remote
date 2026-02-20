"""Tests for keras_remote.core.core — run decorator and env var capture."""

from unittest.mock import MagicMock

from keras_remote.core.core import run


class TestRunDecorator:
  def test_decorator_preserves_wrapped_function(self):
    @run(accelerator="cpu")
    def my_func():
      """My docstring."""

    assert callable(my_func)
    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."


class TestEnvVarCapture:
  def test_exact_match(self, monkeypatch, mocker):
    monkeypatch.setenv("MY_VAR", "my_val")
    mock_exec = mocker.patch("keras_remote.core.core._execute_on_gke")

    @run(accelerator="cpu", capture_env_vars=["MY_VAR"])
    def func():
      pass

    func()
    call_args = mock_exec.call_args
    env_vars = call_args[0][-1]  # last positional arg
    assert env_vars == {"MY_VAR": "my_val"}

  def test_wildcard_pattern(self, monkeypatch, mocker):
    monkeypatch.setenv("PREFIX_A", "1")
    monkeypatch.setenv("PREFIX_B", "2")
    monkeypatch.setenv("OTHER", "3")
    mock_exec = mocker.patch("keras_remote.core.core._execute_on_gke")

    @run(accelerator="cpu", capture_env_vars=["PREFIX_*"])
    def func():
      pass

    func()
    env_vars = mock_exec.call_args[0][-1]
    assert "PREFIX_A" in env_vars
    assert "PREFIX_B" in env_vars
    assert "OTHER" not in env_vars

  def test_missing_var_skipped(self, monkeypatch, mocker):
    monkeypatch.delenv("NONEXISTENT", raising=False)
    mock_exec = mocker.patch("keras_remote.core.core._execute_on_gke")

    @run(accelerator="cpu", capture_env_vars=["NONEXISTENT"])
    def func():
      pass

    func()
    env_vars = mock_exec.call_args[0][-1]
    assert env_vars == {}

  def test_none_capture(self, mocker):
    mock_exec = mocker.patch("keras_remote.core.core._execute_on_gke")

    @run(accelerator="cpu", capture_env_vars=None)
    def func():
      pass

    func()
    env_vars = mock_exec.call_args[0][-1]
    assert env_vars == {}

  def test_mixed_exact_and_wildcard(self, monkeypatch, mocker):
    monkeypatch.setenv("EXACT_VAR", "exact")
    monkeypatch.setenv("WILD_A", "a")
    monkeypatch.setenv("WILD_B", "b")
    mock_exec = mocker.patch("keras_remote.core.core._execute_on_gke")

    @run(
      accelerator="cpu",
      capture_env_vars=["EXACT_VAR", "WILD_*"],
    )
    def func():
      pass

    func()
    env_vars = mock_exec.call_args[0][-1]
    assert env_vars == {"EXACT_VAR": "exact", "WILD_A": "a", "WILD_B": "b"}


class TestExecuteOnGkeDefaults:
  def test_cluster_from_env(self, monkeypatch, mocker):
    """When cluster=None, falls back to KERAS_REMOTE_GKE_CLUSTER env var."""
    monkeypatch.setenv("KERAS_REMOTE_GKE_CLUSTER", "env-cluster")
    monkeypatch.setenv("KERAS_REMOTE_PROJECT", "proj")
    mock_exec = mocker.patch(
      "keras_remote.core.core.execute_remote",
      return_value=42,
    )
    mocker.patch(
      "keras_remote.core.core.JobContext.from_params",
      return_value=MagicMock(),
    )

    @run(accelerator="cpu", cluster=None)
    def func():
      pass

    func()

    call_args = mock_exec.call_args
    backend = call_args[0][1]
    assert backend.cluster == "env-cluster"

  def test_namespace_from_env(self, monkeypatch, mocker):
    """When namespace=None, falls back to KERAS_REMOTE_GKE_NAMESPACE env var."""
    monkeypatch.setenv("KERAS_REMOTE_GKE_NAMESPACE", "custom-ns")
    monkeypatch.setenv("KERAS_REMOTE_PROJECT", "proj")
    mock_exec = mocker.patch(
      "keras_remote.core.core.execute_remote",
      return_value=42,
    )
    mocker.patch(
      "keras_remote.core.core.JobContext.from_params",
      return_value=MagicMock(),
    )

    @run(accelerator="cpu", namespace=None)
    def func():
      pass

    func()

    backend = mock_exec.call_args[0][1]
    assert backend.namespace == "custom-ns"
