"""Shared fixtures for integration and e2e tests."""

import pytest


@pytest.fixture
def sample_function():
  """A simple function suitable for serialization tests."""

  def add(a, b):
    return a + b

  return add


@pytest.fixture
def gcp_env(monkeypatch):
  """Set standard GCP env vars for tests."""
  monkeypatch.setenv("KERAS_REMOTE_PROJECT", "test-project")
  monkeypatch.setenv("KERAS_REMOTE_ZONE", "us-central1-a")
  monkeypatch.setenv("KERAS_REMOTE_GKE_CLUSTER", "test-cluster")
