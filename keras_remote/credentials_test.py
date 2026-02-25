"""Tests for keras_remote.credentials — shared credential checks."""

import json
import os
import subprocess
from unittest import mock

from absl.testing import absltest
from kubernetes.config import ConfigException

from keras_remote import credentials

_MODULE = "keras_remote.credentials"


class TestEnsureGcloud(absltest.TestCase):
  def test_present(self):
    with mock.patch("shutil.which", return_value="/usr/bin/gcloud"):
      credentials.ensure_gcloud()

  def test_missing(self):
    with (
      mock.patch("shutil.which", return_value=None),
      self.assertRaisesRegex(RuntimeError, "gcloud CLI not found"),
    ):
      credentials.ensure_gcloud()


class TestEnsureGkeAuthPlugin(absltest.TestCase):
  def test_present(self):
    with mock.patch(
      "shutil.which", return_value="/usr/bin/gke-gcloud-auth-plugin"
    ):
      credentials.ensure_gke_auth_plugin()

  def test_missing_install_succeeds(self):
    with (
      mock.patch("shutil.which", return_value=None),
      mock.patch(f"{_MODULE}.subprocess.run") as mock_run,
    ):
      credentials.ensure_gke_auth_plugin()
      mock_run.assert_called_once()
      args = mock_run.call_args[0][0]
      self.assertIn("gke-gcloud-auth-plugin", args)
      self.assertIn("--quiet", args)

  def test_missing_install_fails(self):
    with (
      mock.patch("shutil.which", return_value=None),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "gcloud"),
      ),
      self.assertRaisesRegex(
        RuntimeError, "Failed to install gke-gcloud-auth-plugin"
      ),
    ):
      credentials.ensure_gke_auth_plugin()


class TestEnsureAdc(absltest.TestCase):
  def test_token_success(self):
    with mock.patch(f"{_MODULE}.subprocess.run") as mock_run:
      mock_run.return_value.returncode = 0
      credentials.ensure_adc()
      self.assertEqual(mock_run.call_count, 1)

  def test_token_failure_triggers_login(self):
    with mock.patch(f"{_MODULE}.subprocess.run") as mock_run:
      token_result = mock.MagicMock()
      token_result.returncode = 1
      # First call returns failure, second call (login) succeeds.
      mock_run.side_effect = [token_result, mock.DEFAULT]

      credentials.ensure_adc()

      self.assertEqual(mock_run.call_count, 2)
      login_call = mock_run.call_args_list[1]
      self.assertIn("login", login_call[0][0])

  def test_login_failure_raises(self):
    with mock.patch(f"{_MODULE}.subprocess.run") as mock_run:
      token_result = mock.MagicMock()
      token_result.returncode = 1
      mock_run.side_effect = [
        token_result,
        subprocess.CalledProcessError(1, "gcloud"),
      ]

      with self.assertRaisesRegex(
        RuntimeError, "Failed to configure Application Default"
      ):
        credentials.ensure_adc()


class TestEnsureKubeconfig(absltest.TestCase):
  def _mock_active_context(self, cluster_name):
    """Return mock values for list_kube_config_contexts."""
    ctx = {"context": {"cluster": cluster_name}, "name": cluster_name}
    return [ctx], ctx

  def test_correct_cluster_context(self):
    """When the active context matches the expected cluster, no reconfigure."""
    with (
      mock.patch(f"{_MODULE}.config.load_kube_config"),
      mock.patch(
        f"{_MODULE}.config.list_kube_config_contexts",
        return_value=self._mock_active_context(
          "gke_my-proj_us-central1-a_my-cluster"
        ),
      ),
      mock.patch(f"{_MODULE}._configure_kubeconfig") as mock_configure,
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", "my-cluster")
      mock_configure.assert_not_called()

  def test_wrong_cluster_context_triggers_reconfigure(self):
    """When the active context doesn't match, reconfigure."""
    with (
      mock.patch(f"{_MODULE}.config.load_kube_config"),
      mock.patch(
        f"{_MODULE}.config.list_kube_config_contexts",
        return_value=self._mock_active_context(
          "gke_other-proj_us-west1-b_other-cluster"
        ),
      ),
      mock.patch(f"{_MODULE}._configure_kubeconfig") as mock_configure,
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", "my-cluster")
      mock_configure.assert_called_once_with(
        "my-cluster", "us-central1-a", "my-proj"
      )

  def test_no_kubeconfig_triggers_configure(self):
    """When no kubeconfig exists, configure from scratch."""
    with (
      mock.patch(
        f"{_MODULE}.config.load_kube_config",
        side_effect=ConfigException("no config"),
      ),
      mock.patch(f"{_MODULE}._configure_kubeconfig") as mock_configure,
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", "my-cluster")
      mock_configure.assert_called_once_with(
        "my-cluster", "us-central1-a", "my-proj"
      )

  def test_cluster_none_accepts_any_valid_context(self):
    """When cluster is None, any valid kubeconfig is accepted."""
    with (
      mock.patch(f"{_MODULE}.config.load_kube_config"),
      mock.patch(
        f"{_MODULE}.config.list_kube_config_contexts",
        return_value=self._mock_active_context(
          "gke_some-proj_us-east1-b_some-cluster"
        ),
      ),
      mock.patch(f"{_MODULE}._configure_kubeconfig") as mock_configure,
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", cluster=None)
      mock_configure.assert_not_called()

  def test_cluster_from_env_var(self):
    """When cluster is None but env var is set, uses env var for validation."""
    with (
      mock.patch.dict(os.environ, {"KERAS_REMOTE_CLUSTER": "env-cluster"}),
      mock.patch(
        f"{_MODULE}.config.load_kube_config",
        side_effect=ConfigException("no config"),
      ),
      mock.patch(f"{_MODULE}._configure_kubeconfig") as mock_configure,
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", cluster=None)
      mock_configure.assert_called_once_with(
        "env-cluster", "us-central1-a", "my-proj"
      )

  def test_configure_failure_raises(self):
    with (
      mock.patch(
        f"{_MODULE}.config.load_kube_config",
        side_effect=ConfigException("no config"),
      ),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "gcloud"),
      ),
      self.assertRaisesRegex(RuntimeError, "Failed to configure kubeconfig"),
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", "my-cluster")


class TestEnsureDockerAuth(absltest.TestCase):
  def test_already_configured(self):
    """When credHelpers already has the AR host, no-op."""
    docker_cfg = {"credHelpers": {"us-docker.pkg.dev": "gcloud"}}
    with (
      mock.patch(
        "builtins.open",
        mock.mock_open(read_data=json.dumps(docker_cfg)),
      ),
      mock.patch("os.path.exists", return_value=True),
      mock.patch(f"{_MODULE}.subprocess.run") as mock_run,
    ):
      credentials.ensure_docker_auth("us-central1-a")
      mock_run.assert_not_called()

  def test_not_configured_triggers_setup(self):
    """When Docker config missing or no matching host, configure."""
    with (
      mock.patch("os.path.exists", return_value=False),
      mock.patch(f"{_MODULE}.subprocess.run") as mock_run,
    ):
      credentials.ensure_docker_auth("us-central1-a")
      mock_run.assert_called_once()
      args = mock_run.call_args[0][0]
      self.assertIn("us-docker.pkg.dev", args)

  def test_failure_is_nonfatal(self):
    """Docker auth failure logs a warning but does not raise."""
    with (
      mock.patch("os.path.exists", return_value=False),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "gcloud"),
      ),
    ):
      # Should not raise
      credentials.ensure_docker_auth("us-central1-a")


class TestEnsureCredentials(absltest.TestCase):
  def test_calls_all_checks(self):
    with (
      mock.patch(f"{_MODULE}.ensure_gcloud") as m_gcloud,
      mock.patch(f"{_MODULE}.ensure_gke_auth_plugin") as m_plugin,
      mock.patch(f"{_MODULE}.ensure_adc") as m_adc,
      mock.patch(f"{_MODULE}.ensure_kubeconfig") as m_kube,
      mock.patch(f"{_MODULE}.ensure_docker_auth") as m_docker,
    ):
      credentials.ensure_credentials("proj", "us-central1-a", "cluster")

      m_gcloud.assert_called_once()
      m_plugin.assert_called_once()
      m_adc.assert_called_once()
      m_kube.assert_called_once_with("proj", "us-central1-a", "cluster")
      m_docker.assert_called_once_with("us-central1-a")

  def test_early_failure_short_circuits(self):
    """If gcloud check fails, later checks are not run."""
    with (
      mock.patch(
        f"{_MODULE}.ensure_gcloud",
        side_effect=RuntimeError("no gcloud"),
      ),
      mock.patch(f"{_MODULE}.ensure_adc") as m_adc,
      self.assertRaises(RuntimeError),
    ):
      credentials.ensure_credentials("proj", "us-central1-a")

    m_adc.assert_not_called()


if __name__ == "__main__":
  absltest.main()
