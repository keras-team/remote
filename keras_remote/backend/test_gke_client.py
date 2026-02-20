"""Tests for keras_remote.backend.gke_client — K8s job submission and monitoring."""

from unittest.mock import MagicMock

import pytest
from kubernetes.config import ConfigException

from keras_remote.backend.gke_client import (
  _check_pod_scheduling,
  _create_job_spec,
  _load_kube_config,
  _parse_accelerator,
  wait_for_job,
)


class TestParseAccelerator:
  def test_cpu(self):
    result = _parse_accelerator("cpu")
    assert result["node_selector"] == {}
    assert result["resource_limits"] == {}
    assert result["resource_requests"] == {}
    assert result["tolerations"] == []
    assert result["jax_platform"] == "cpu"

  def test_gpu_l4(self):
    result = _parse_accelerator("l4")
    assert result["node_selector"] == {
      "cloud.google.com/gke-accelerator": "nvidia-l4"
    }
    assert result["resource_limits"] == {"nvidia.com/gpu": "1"}
    assert result["resource_requests"] == {"nvidia.com/gpu": "1"}
    assert result["jax_platform"] == "gpu"
    assert len(result["tolerations"]) == 1
    assert result["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert result["tolerations"][0]["operator"] == "Exists"
    assert result["tolerations"][0]["effect"] == "NoSchedule"

  def test_gpu_a100x4(self):
    result = _parse_accelerator("a100x4")
    assert result["resource_limits"] == {"nvidia.com/gpu": "4"}
    assert result["resource_requests"] == {"nvidia.com/gpu": "4"}

  def test_tpu_v3_8(self):
    result = _parse_accelerator("v3-8")
    assert "cloud.google.com/gke-tpu-accelerator" in result["node_selector"]
    assert "cloud.google.com/gke-tpu-topology" in result["node_selector"]
    assert result["resource_limits"] == {"google.com/tpu": "8"}
    assert result["resource_requests"] == {"google.com/tpu": "8"}
    assert result["jax_platform"] == "tpu"
    assert len(result["tolerations"]) == 1
    assert result["tolerations"][0]["key"] == "google.com/tpu"

  def test_tpu_v5litepod_4(self):
    result = _parse_accelerator("v5litepod-4")
    assert result["node_selector"] == {
      "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
      "cloud.google.com/gke-tpu-topology": "2x2",
    }
    assert result["resource_limits"] == {"google.com/tpu": "4"}


class TestCreateJobSpec:
  def _make_gpu_config(self):
    return {
      "node_selector": {"cloud.google.com/gke-accelerator": "nvidia-l4"},
      "resource_limits": {"nvidia.com/gpu": "1"},
      "resource_requests": {"nvidia.com/gpu": "1"},
      "tolerations": [
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
      ],
      "jax_platform": "gpu",
    }

  def _make_cpu_config(self):
    return {
      "node_selector": {},
      "resource_limits": {},
      "resource_requests": {},
      "tolerations": [],
      "jax_platform": "cpu",
    }

  def test_gpu_job_structure(self):
    job = _create_job_spec(
      job_name="keras-remote-job-abc",
      container_uri="us-docker.pkg.dev/proj/repo/img:tag",
      accel_config=self._make_gpu_config(),
      job_id="job-abc",
      bucket_name="proj-keras-remote-jobs",
      namespace="default",
    )
    assert job.metadata.name == "keras-remote-job-abc"
    assert job.metadata.namespace == "default"

    container = job.spec.template.spec.containers[0]
    assert container.image == "us-docker.pkg.dev/proj/repo/img:tag"
    assert container.command == ["python3", "-u", "/app/remote_runner.py"]

    # Check env vars
    env_names = {e.name: e.value for e in container.env}
    assert env_names["KERAS_BACKEND"] == "jax"
    assert env_names["JAX_PLATFORMS"] == "gpu"
    assert env_names["JOB_ID"] == "job-abc"
    assert env_names["GCS_BUCKET"] == "proj-keras-remote-jobs"

    # Check GCS args
    assert "gs://proj-keras-remote-jobs/job-abc/context.zip" in container.args
    assert "gs://proj-keras-remote-jobs/job-abc/payload.pkl" in container.args
    assert "gs://proj-keras-remote-jobs/job-abc/result.pkl" in container.args

    # Job spec fields
    assert job.spec.backoff_limit == 0
    assert job.spec.ttl_seconds_after_finished == 600
    assert job.spec.template.spec.restart_policy == "Never"

    # Labels on both job and pod template
    assert job.metadata.labels["app"] == "keras-remote"
    assert job.metadata.labels["job-id"] == "job-abc"
    assert job.spec.template.metadata.labels["app"] == "keras-remote"
    assert job.spec.template.metadata.labels["job-id"] == "job-abc"

    # GPU-specific: node selector and tolerations
    assert (
      "cloud.google.com/gke-accelerator" in job.spec.template.spec.node_selector
    )
    tolerations = job.spec.template.spec.tolerations
    assert len(tolerations) == 1
    assert tolerations[0].key == "nvidia.com/gpu"

  def test_cpu_job_structure(self):
    job = _create_job_spec(
      job_name="cpu-job",
      container_uri="img",
      accel_config=self._make_cpu_config(),
      job_id="j",
      bucket_name="b",
      namespace="ns",
    )
    assert job.spec.template.spec.node_selector is None


class TestWaitForJob:
  @pytest.fixture(autouse=True)
  def _mock_kube(self, mocker):
    mocker.patch("keras_remote.backend.gke_client._load_kube_config")

  def _make_mock_job(self):
    job = MagicMock()
    job.metadata.name = "keras-remote-job-abc"
    return job

  @pytest.mark.parametrize(
    "succeeded, failed, error_match",
    [
      (1, None, None),
      (None, 1, "failed"),
    ],
    ids=["success", "failure"],
  )
  def test_first_poll_outcome(self, mocker, succeeded, failed, error_match):
    mock_batch = MagicMock()
    mock_status = MagicMock()
    mock_status.status.succeeded = succeeded
    mock_status.status.failed = failed
    mock_batch.read_namespaced_job_status.return_value = mock_status
    mocker.patch(
      "keras_remote.backend.gke_client.client.BatchV1Api",
      return_value=mock_batch,
    )

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []
    mocker.patch(
      "keras_remote.backend.gke_client.client.CoreV1Api", return_value=mock_core
    )

    if error_match:
      with pytest.raises(RuntimeError, match=error_match):
        wait_for_job(self._make_mock_job())
    else:
      result = wait_for_job(self._make_mock_job())
      assert result == "success"

  def test_timeout_raises(self, mocker):
    mock_batch = MagicMock()
    mock_status = MagicMock()
    mock_status.status.succeeded = None
    mock_status.status.failed = None
    mock_batch.read_namespaced_job_status.return_value = mock_status
    mocker.patch(
      "keras_remote.backend.gke_client.client.BatchV1Api",
      return_value=mock_batch,
    )
    mocker.patch("keras_remote.backend.gke_client.client.CoreV1Api")
    mocker.patch("keras_remote.backend.gke_client.time.sleep")

    with pytest.raises(RuntimeError, match="timed out"):
      wait_for_job(self._make_mock_job(), timeout=0)

  def test_polls_until_success(self, mocker):
    mock_batch = MagicMock()
    running = MagicMock()
    running.status.succeeded = None
    running.status.failed = None
    succeeded = MagicMock()
    succeeded.status.succeeded = 1
    succeeded.status.failed = None
    mock_batch.read_namespaced_job_status.side_effect = [running, succeeded]
    mocker.patch(
      "keras_remote.backend.gke_client.client.BatchV1Api",
      return_value=mock_batch,
    )

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []
    mocker.patch(
      "keras_remote.backend.gke_client.client.CoreV1Api", return_value=mock_core
    )

    mock_sleep = mocker.patch("keras_remote.backend.gke_client.time.sleep")

    result = wait_for_job(self._make_mock_job(), poll_interval=5)
    assert result == "success"
    mock_sleep.assert_called_with(5)


class TestLoadKubeConfig:
  def test_kubeconfig_fallback(self, mocker):
    """Falls back to local kubeconfig when in-cluster config is unavailable."""
    mocker.patch(
      "keras_remote.backend.gke_client.config.load_incluster_config",
      side_effect=ConfigException("not in cluster"),
    )
    mocker.patch("keras_remote.backend.gke_client.config.load_kube_config")
    _load_kube_config()


class TestCheckPodScheduling:
  def _make_pending_pod(self, message):
    pod = MagicMock()
    pod.status.phase = "Pending"
    condition = MagicMock()
    condition.type = "PodScheduled"
    condition.status = "False"
    condition.message = message
    pod.status.conditions = [condition]
    return pod

  @pytest.mark.parametrize(
    "condition_message, error_match",
    [
      ("Insufficient nvidia.com/gpu", "No GPU nodes available"),
      ("didn't match Pod's node affinity/selector", "No nodes match"),
    ],
    ids=["insufficient_gpu", "node_selector_mismatch"],
  )
  def test_scheduling_failure_raises(self, condition_message, error_match):
    mock_core = MagicMock()
    pod = self._make_pending_pod(condition_message)
    mock_core.list_namespaced_pod.return_value.items = [pod]

    with pytest.raises(RuntimeError, match=error_match):
      _check_pod_scheduling(mock_core, "job-1", "default")

  def test_running_pod_no_error(self, mocker):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Running"
    pod.status.conditions = []
    mock_core.list_namespaced_pod.return_value.items = [pod]

    _check_pod_scheduling(mock_core, "job-1", "default")  # should not raise

  def test_pending_no_conditions(self, mocker):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Pending"
    pod.status.conditions = None
    mock_core.list_namespaced_pod.return_value.items = [pod]

    _check_pod_scheduling(mock_core, "job-1", "default")  # should not raise
