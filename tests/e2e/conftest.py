"""E2E test configuration — skip unless E2E_TESTS is set."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
  """Skip e2e tests unless E2E_TESTS env var is set."""
  if os.environ.get("E2E_TESTS"):
    return
  skip = pytest.mark.skip(reason="E2E_TESTS not set")
  for item in items:
    if "e2e" in str(item.fspath):
      item.add_marker(skip)


@pytest.fixture
def gcp_project():
  """Return GCP project from env, skip if not set."""
  project = os.environ.get("KERAS_REMOTE_PROJECT")
  if not project:
    pytest.skip("KERAS_REMOTE_PROJECT not set")
  return project
