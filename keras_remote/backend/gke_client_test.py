"""Tests for keras_remote.backend.gke_client — K8s job submission and monitoring."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized
from kubernetes.config import ConfigException

from keras_remote.backend.gke_client import (
  _check_pod_scheduling,
  _create_job_spec,
  _load_kube_config,
  _parse_accelerator,
  wait_for_job,
)


class TestParseAccelerator(absltest.TestCase):
  def test_cpu(self):
    result = _parse_accelerator("cpu")
    self.assertEqual(result["node_selector"], {})
    self.assertEqual(result["resource_limits"], {})
    self.assertEqual(result["resource_requests"], {})
    self.assertEqual(result["tolerations"], [])
    self.assertEqual(result["jax_platform"], "cpu")

  def test_gpu_l4(self):
    result = _parse_accelerator("l4")
    self.assertEqual(
      result["node_selector"],
      {"cloud.google.com/gke-accelerator": "nvidia-l4"},
    )
    self.assertEqual(result["resource_limits"], {"nvidia.com/gpu": "1"})
    self.assertEqual(result["resource_requests"], {"nvidia.com/gpu": "1"})
    self.assertEqual(result["jax_platform"], "gpu")
    self.assertLen(result["tolerations"], 1)
    self.assertEqual(result["tolerations"][0]["key"], "nvidia.com/gpu")
    self.assertEqual(result["tolerations"][0]["operator"], "Exists")
    self.assertEqual(result["tolerations"][0]["effect"], "NoSchedule")

  def test_gpu_a100x4(self):
    result = _parse_accelerator("a100x4")
    self.assertEqual(result["resource_limits"], {"nvidia.com/gpu": "4"})
    self.assertEqual(result["resource_requests"], {"nvidia.com/gpu": "4"})

  def test_tpu_v3_8(self):
    result = _parse_accelerator("v3-4")
    self.assertIn(
      "cloud.google.com/gke-tpu-accelerator", result["node_selector"]
    )
    self.assertIn("cloud.google.com/gke-tpu-topology", result["node_selector"])
    self.assertEqual(result["resource_limits"], {"google.com/tpu": "4"})
    self.assertEqual(result["resource_requests"], {"google.com/tpu": "4"})
    self.assertEqual(result["jax_platform"], "tpu")
    self.assertLen(result["tolerations"], 1)
    self.assertEqual(result["tolerations"][0]["key"], "google.com/tpu")

  def test_tpu_v5litepod_4(self):
    result = _parse_accelerator("v5litepod-4")
    self.assertEqual(
      result["node_selector"],
      {
        "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
        "cloud.google.com/gke-tpu-topology": "2x2",
      },
    )
    self.assertEqual(result["resource_limits"], {"google.com/tpu": "4"})


class TestCreateJobSpec(absltest.TestCase):
  def _make_gpu_config(self):
    return {
      "node_selector": {"cloud.google.com/gke-accelerator": "nvidia-l4"},
      "resource_limits": {"nvidia.com/gpu": "1"},
      "resource_requests": {"nvidia.com/gpu": "1"},
      "tolerations": [
        {
          "key": "nvidia.com/gpu",
          "operator": "Exists",
          "effect": "NoSchedule",
        }
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
    self.assertEqual(job.metadata.name, "keras-remote-job-abc")
    self.assertEqual(job.metadata.namespace, "default")

    container = job.spec.template.spec.containers[0]
    self.assertEqual(container.image, "us-docker.pkg.dev/proj/repo/img:tag")
    self.assertEqual(
      container.command, ["python3", "-u", "/app/remote_runner.py"]
    )

    # Check env vars
    env_names = {e.name: e.value for e in container.env}
    self.assertEqual(env_names["KERAS_BACKEND"], "jax")
    self.assertEqual(env_names["JAX_PLATFORMS"], "gpu")
    self.assertEqual(env_names["JOB_ID"], "job-abc")
    self.assertEqual(env_names["GCS_BUCKET"], "proj-keras-remote-jobs")

    # Check GCS args
    self.assertIn(
      "gs://proj-keras-remote-jobs/job-abc/context.zip", container.args
    )
    self.assertIn(
      "gs://proj-keras-remote-jobs/job-abc/payload.pkl", container.args
    )
    self.assertIn(
      "gs://proj-keras-remote-jobs/job-abc/result.pkl", container.args
    )

    # Job spec fields
    self.assertEqual(job.spec.backoff_limit, 0)
    self.assertEqual(job.spec.ttl_seconds_after_finished, 600)
    self.assertEqual(job.spec.template.spec.restart_policy, "Never")

    # Labels on both job and pod template
    self.assertEqual(job.metadata.labels["app"], "keras-remote")
    self.assertEqual(job.metadata.labels["job-id"], "job-abc")
    self.assertEqual(job.spec.template.metadata.labels["app"], "keras-remote")
    self.assertEqual(job.spec.template.metadata.labels["job-id"], "job-abc")

    # GPU-specific: node selector and tolerations
    self.assertIn(
      "cloud.google.com/gke-accelerator",
      job.spec.template.spec.node_selector,
    )
    tolerations = job.spec.template.spec.tolerations
    self.assertLen(tolerations, 1)
    self.assertEqual(tolerations[0].key, "nvidia.com/gpu")

  def test_cpu_job_structure(self):
    job = _create_job_spec(
      job_name="cpu-job",
      container_uri="img",
      accel_config=self._make_cpu_config(),
      job_id="j",
      bucket_name="b",
      namespace="ns",
    )
    self.assertIsNone(job.spec.template.spec.node_selector)


class TestWaitForJob(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(
      mock.patch("keras_remote.backend.gke_client._load_kube_config")
    )

  def _make_mock_job(self):
    job = MagicMock()
    job.metadata.name = "keras-remote-job-abc"
    return job

  def test_first_poll_success(self):
    mock_batch = MagicMock()
    mock_status = MagicMock()
    mock_status.status.succeeded = 1
    mock_status.status.failed = None
    mock_batch.read_namespaced_job_status.return_value = mock_status

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []

    with (
      mock.patch(
        "keras_remote.backend.gke_client.client.BatchV1Api",
        return_value=mock_batch,
      ),
      mock.patch(
        "keras_remote.backend.gke_client.client.CoreV1Api",
        return_value=mock_core,
      ),
    ):
      result = wait_for_job(self._make_mock_job())
    self.assertEqual(result, "success")

  def test_first_poll_failure(self):
    mock_batch = MagicMock()
    mock_status = MagicMock()
    mock_status.status.succeeded = None
    mock_status.status.failed = 1
    mock_batch.read_namespaced_job_status.return_value = mock_status

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []

    with (
      mock.patch(
        "keras_remote.backend.gke_client.client.BatchV1Api",
        return_value=mock_batch,
      ),
      mock.patch(
        "keras_remote.backend.gke_client.client.CoreV1Api",
        return_value=mock_core,
      ),
      self.assertRaisesRegex(RuntimeError, "failed"),
    ):
      wait_for_job(self._make_mock_job())

  def test_timeout_raises(self):
    mock_batch = MagicMock()
    mock_status = MagicMock()
    mock_status.status.succeeded = None
    mock_status.status.failed = None
    mock_batch.read_namespaced_job_status.return_value = mock_status

    with (
      mock.patch(
        "keras_remote.backend.gke_client.client.BatchV1Api",
        return_value=mock_batch,
      ),
      mock.patch("keras_remote.backend.gke_client.client.CoreV1Api"),
      mock.patch("keras_remote.backend.gke_client.time.sleep"),
      self.assertRaisesRegex(RuntimeError, "timed out"),
    ):
      wait_for_job(self._make_mock_job(), timeout=0)

  def test_polls_until_success(self):
    mock_batch = MagicMock()
    running = MagicMock()
    running.status.succeeded = None
    running.status.failed = None
    succeeded = MagicMock()
    succeeded.status.succeeded = 1
    succeeded.status.failed = None
    mock_batch.read_namespaced_job_status.side_effect = [running, succeeded]

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []

    with (
      mock.patch(
        "keras_remote.backend.gke_client.client.BatchV1Api",
        return_value=mock_batch,
      ),
      mock.patch(
        "keras_remote.backend.gke_client.client.CoreV1Api",
        return_value=mock_core,
      ),
      mock.patch("keras_remote.backend.gke_client.time.sleep") as mock_sleep,
    ):
      result = wait_for_job(self._make_mock_job(), poll_interval=5)
    self.assertEqual(result, "success")
    mock_sleep.assert_called_with(5)


class TestLoadKubeConfig(absltest.TestCase):
  def test_kubeconfig_fallback(self):
    """Falls back to local kubeconfig when in-cluster config is unavailable."""
    with (
      mock.patch(
        "keras_remote.backend.gke_client.config.load_incluster_config",
        side_effect=ConfigException("not in cluster"),
      ),
      mock.patch("keras_remote.backend.gke_client.config.load_kube_config"),
    ):
      _load_kube_config()


class TestCheckPodScheduling(parameterized.TestCase):
  def _make_pending_pod(self, message):
    pod = MagicMock()
    pod.status.phase = "Pending"
    condition = MagicMock()
    condition.type = "PodScheduled"
    condition.status = "False"
    condition.message = message
    pod.status.conditions = [condition]
    return pod

  @parameterized.named_parameters(
    dict(
      testcase_name="insufficient_gpu",
      condition_message="Insufficient nvidia.com/gpu",
      error_match="No GPU nodes available",
    ),
    dict(
      testcase_name="node_selector_mismatch",
      condition_message="didn't match Pod's node affinity/selector",
      error_match="No nodes match",
    ),
  )
  def test_scheduling_failure_raises(self, condition_message, error_match):
    mock_core = MagicMock()
    pod = self._make_pending_pod(condition_message)
    mock_core.list_namespaced_pod.return_value.items = [pod]

    with self.assertRaisesRegex(RuntimeError, error_match):
      _check_pod_scheduling(mock_core, "job-1", "default")

  def test_running_pod_no_error(self):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Running"
    pod.status.conditions = []
    mock_core.list_namespaced_pod.return_value.items = [pod]

    _check_pod_scheduling(mock_core, "job-1", "default")  # should not raise

  def test_pending_no_conditions(self):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Pending"
    pod.status.conditions = None
    mock_core.list_namespaced_pod.return_value.items = [pod]

    _check_pod_scheduling(mock_core, "job-1", "default")  # should not raise


if __name__ == "__main__":
  absltest.main()
