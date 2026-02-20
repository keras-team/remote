"""E2E tests for remote execution with CPU accelerator.

These tests require a real GCP project with:
- A GKE cluster with a CPU node pool
- Cloud Storage, Cloud Build, and Artifact Registry APIs enabled
- Proper IAM permissions

Set E2E_TESTS=1 to enable.
"""

import os

import pytest

import keras_remote


@pytest.mark.e2e
@pytest.mark.timeout(600)
class TestCpuExecution:
  def test_simple_function(self, gcp_project):
    """Execute a simple add function remotely and verify the result."""

    @keras_remote.run(accelerator="cpu")
    def add(a, b):
      return a + b

    result = add(2, 3)
    assert result == 5

  def test_complex_return_type(self, gcp_project):
    """Verify complex return types survive serialization roundtrip."""

    @keras_remote.run(accelerator="cpu")
    def complex_return():
      return {
        "key": [1, 2, 3],
        "nested": {"a": True, "b": None},
        "tuple": (4, 5),
      }

    result = complex_return()
    assert result["key"] == [1, 2, 3]
    assert result["nested"]["a"] is True
    assert result["nested"]["b"] is None
    assert result["tuple"] == (4, 5)

  def test_function_that_raises(self, gcp_project):
    """Verify remote exceptions are re-raised locally."""

    @keras_remote.run(accelerator="cpu")
    def bad_func():
      raise ValueError("intentional test error")

    with pytest.raises(ValueError, match="intentional test error"):
      bad_func()

  def test_env_var_propagation(self, gcp_project, monkeypatch):
    """Verify captured env vars are available in the remote environment."""
    monkeypatch.setenv("E2E_TEST_VAR", "hello_from_local")

    @keras_remote.run(
      accelerator="cpu",
      capture_env_vars=["E2E_TEST_VAR"],
    )
    def read_env():
      return os.environ.get("E2E_TEST_VAR")

    result = read_env()
    assert result == "hello_from_local"

  def test_function_with_args_and_kwargs(self, gcp_project):
    """Verify positional and keyword arguments are passed correctly."""

    @keras_remote.run(accelerator="cpu")
    def compute(x, y, scale=1.0, offset=0.0):
      return (x + y) * scale + offset

    result = compute(3, 4, scale=2.0, offset=1.0)
    assert result == 15.0
