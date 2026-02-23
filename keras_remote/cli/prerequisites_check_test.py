"""Tests for keras_remote.cli.prerequisites_check — tool availability checks."""

from unittest import mock

import click
from absl.testing import absltest, parameterized

from keras_remote.cli.prerequisites_check import (
  check_docker,
  check_gcloud,
  check_gcloud_auth,
  check_kubectl,
  check_pulumi,
)


class TestToolChecks(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(
      testcase_name="gcloud",
      check_fn=check_gcloud,
      error_match="gcloud CLI not found",
    ),
    dict(
      testcase_name="pulumi",
      check_fn=check_pulumi,
      error_match="Pulumi CLI not found",
    ),
    dict(
      testcase_name="kubectl",
      check_fn=check_kubectl,
      error_match="kubectl not found",
    ),
    dict(
      testcase_name="docker",
      check_fn=check_docker,
      error_match="Docker not found",
    ),
  )
  def test_present(self, check_fn, error_match):
    with mock.patch("shutil.which", return_value="/usr/bin/tool"):
      check_fn()

  @parameterized.named_parameters(
    dict(
      testcase_name="gcloud",
      check_fn=check_gcloud,
      error_match="gcloud CLI not found",
    ),
    dict(
      testcase_name="pulumi",
      check_fn=check_pulumi,
      error_match="Pulumi CLI not found",
    ),
    dict(
      testcase_name="kubectl",
      check_fn=check_kubectl,
      error_match="kubectl not found",
    ),
    dict(
      testcase_name="docker",
      check_fn=check_docker,
      error_match="Docker not found",
    ),
  )
  def test_missing(self, check_fn, error_match):
    with (
      mock.patch("shutil.which", return_value=None),
      self.assertRaisesRegex(click.ClickException, error_match),
    ):
      check_fn()


class TestCheckGcloudAuth(absltest.TestCase):
  def test_token_success(self):
    """When print-access-token succeeds, no login is triggered."""
    with mock.patch(
      "keras_remote.cli.prerequisites_check.subprocess.run",
    ) as mock_run:
      mock_run.return_value.returncode = 0
      check_gcloud_auth()
      # Only called once (the token check), not a second time for login
      self.assertEqual(mock_run.call_count, 1)

  def test_token_failure_triggers_login(self):
    """When print-access-token fails, gcloud auth login is run."""
    with (
      mock.patch(
        "keras_remote.cli.prerequisites_check.subprocess.run",
      ) as mock_run,
      mock.patch("keras_remote.cli.prerequisites_check.warning"),
      mock.patch("click.echo"),
    ):
      token_result = mock.MagicMock()
      token_result.returncode = 1
      mock_run.return_value = token_result

      check_gcloud_auth()

      self.assertEqual(mock_run.call_count, 2)
      # Second call should be the login command
      login_call = mock_run.call_args_list[1]
      self.assertIn("login", login_call[0][0])


if __name__ == "__main__":
  absltest.main()
