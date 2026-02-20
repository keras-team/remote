"""Shared fixtures for colocated unit tests."""

import pytest


@pytest.fixture
def mock_storage_client(mocker):
  """Mock google.cloud.storage.Client."""
  mock_client = mocker.MagicMock()
  mocker.patch("google.cloud.storage.Client", return_value=mock_client)
  return mock_client


@pytest.fixture
def mock_kube_config(mocker):
  """Mock kubernetes config loading."""
  mocker.patch("keras_remote.backend.gke_client._load_kube_config")


@pytest.fixture
def mock_batch_v1(mocker):
  """Mock kubernetes BatchV1Api."""
  mock_api = mocker.MagicMock()
  mocker.patch("kubernetes.client.BatchV1Api", return_value=mock_api)
  return mock_api
