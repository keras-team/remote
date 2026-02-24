"""Tests for keras_remote.cli.commands.up — resilience to partial failures."""

import subprocess
from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner
from pulumi.automation import errors as pulumi_errors

from keras_remote.cli.commands.up import up

# Shared CLI args that skip interactive prompts.
_CLI_ARGS = [
  "--project",
  "test-project",
  "--zone",
  "us-central2-b",
  "--accelerator",
  "cpu",
  "--yes",
]

# Patches applied to every test to bypass prerequisites and infrastructure.
_BASE_PATCHES = {
  "check_all": mock.patch("keras_remote.cli.commands.up.check_all"),
  "create_program": mock.patch(
    "keras_remote.cli.commands.up.create_program",
  ),
  "get_stack": mock.patch("keras_remote.cli.commands.up.get_stack"),
  "configure_docker_auth": mock.patch(
    "keras_remote.cli.commands.up.configure_docker_auth",
  ),
  "configure_kubectl": mock.patch(
    "keras_remote.cli.commands.up.configure_kubectl",
  ),
  "install_lws": mock.patch("keras_remote.cli.commands.up.install_lws"),
  "install_gpu_drivers": mock.patch(
    "keras_remote.cli.commands.up.install_gpu_drivers",
  ),
}


def _start_patches(test_case):
  """Start all base patches and return a dict of mock objects."""
  mocks = {}
  for name, patcher in _BASE_PATCHES.items():
    mocks[name] = test_case.enterContext(patcher)
  return mocks


class UpCommandResilienceTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)

    # Default: stack.up() succeeds.
    mock_stack = mock.MagicMock()
    mock_stack.up.return_value.summary.resource_changes = {"create": 5}
    self.mocks["get_stack"].return_value = mock_stack
    self.mock_stack = mock_stack

  def test_full_success(self):
    """All steps succeed — exit code 0, 'Setup Complete' shown."""
    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Setup Complete", result.output)
    self.assertNotIn("Warnings", result.output)
    self.mocks["install_lws"].assert_called_once()
    self.mocks["configure_docker_auth"].assert_called_once()
    self.mocks["configure_kubectl"].assert_called_once()
    self.mocks[
      "install_gpu_drivers"
    ].assert_not_called()  # CPU accelerator, no GPU drivers.

  def test_pulumi_failure_still_runs_post_deploy(self):
    """stack.up() raises CommandError — post-deploy steps still execute."""
    self.mock_stack.up.side_effect = pulumi_errors.CommandError(
      "resource already exists"
    )

    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.mocks["configure_docker_auth"].assert_called_once()
    self.mocks["configure_kubectl"].assert_called_once()
    self.mocks["install_lws"].assert_called_once()
    self.assertIn("Setup Completed With Warnings", result.output)
    self.assertIn("Pulumi provisioning encountered errors", result.output)

  def test_post_deploy_failure_does_not_block_others(self):
    """One post-deploy step failing doesn't prevent the others from running."""
    self.mocks[
      "configure_docker_auth"
    ].side_effect = subprocess.CalledProcessError(1, "gcloud")

    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    # Subsequent steps still called despite Docker auth failure.
    self.mocks["configure_kubectl"].assert_called_once()
    self.mocks["install_lws"].assert_called_once()
    self.assertIn("Setup Completed With Warnings", result.output)
    self.assertIn("Docker authentication", result.output)

  def test_multiple_post_deploy_failures(self):
    """Multiple post-deploy failures are all reported."""
    self.mocks["configure_kubectl"].side_effect = subprocess.CalledProcessError(
      1, "gcloud"
    )
    self.mocks["install_lws"].side_effect = subprocess.CalledProcessError(
      1, "kubectl"
    )

    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("kubectl configuration", result.output)
    self.assertIn("LWS CRD installation", result.output)

  def test_pulumi_and_post_deploy_failures_combined(self):
    """Both Pulumi and post-deploy failures are reported together."""
    self.mock_stack.up.side_effect = pulumi_errors.CommandError("conflict")
    self.mocks["install_lws"].side_effect = subprocess.CalledProcessError(
      1, "kubectl"
    )

    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pulumi provisioning encountered errors", result.output)
    self.assertIn("LWS CRD installation", result.output)
    self.assertIn("re-run", result.output)


class UpCommandGpuDriverTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)

    mock_stack = mock.MagicMock()
    mock_stack.up.return_value.summary.resource_changes = {"create": 5}
    self.mocks["get_stack"].return_value = mock_stack

  def test_gpu_driver_failure_reported(self):
    """GPU driver installation failure is caught and reported."""
    self.mocks[
      "install_gpu_drivers"
    ].side_effect = subprocess.CalledProcessError(1, "kubectl")
    args = [
      "--project",
      "test-project",
      "--zone",
      "us-central1-a",
      "--accelerator",
      "t4",
      "--yes",
    ]

    result = self.runner.invoke(up, args)

    self.assertEqual(result.exit_code, 0, result.output)
    self.mocks["install_gpu_drivers"].assert_called_once()
    self.assertIn("GPU driver installation", result.output)


if __name__ == "__main__":
  absltest.main()
