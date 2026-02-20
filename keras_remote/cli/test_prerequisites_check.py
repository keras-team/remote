"""Tests for keras_remote.cli.prerequisites_check — tool availability checks."""

import click
import pytest

from keras_remote.cli.prerequisites_check import (
  check_docker,
  check_gcloud,
  check_gcloud_auth,
  check_kubectl,
  check_pulumi,
)


@pytest.mark.parametrize(
  ("check_fn", "error_match"),
  [
    (check_gcloud, "gcloud CLI not found"),
    (check_pulumi, "Pulumi CLI not found"),
    (check_kubectl, "kubectl not found"),
    (check_docker, "Docker not found"),
  ],
  ids=["gcloud", "pulumi", "kubectl", "docker"],
)
class TestToolChecks:
  def test_present(self, mocker, check_fn, error_match):
    mocker.patch("shutil.which", return_value="/usr/bin/tool")
    check_fn()

  def test_missing(self, mocker, check_fn, error_match):
    mocker.patch("shutil.which", return_value=None)
    with pytest.raises(click.ClickException, match=error_match):
      check_fn()


class TestCheckGcloudAuth:
  def test_token_success(self, mocker):
    """When print-access-token succeeds, no login is triggered."""
    mock_run = mocker.patch(
      "keras_remote.cli.prerequisites_check.subprocess.run",
    )
    mock_run.return_value.returncode = 0

    check_gcloud_auth()

    # Only called once (the token check), not a second time for login
    assert mock_run.call_count == 1

  def test_token_failure_triggers_login(self, mocker):
    """When print-access-token fails, gcloud auth login is run."""
    mock_run = mocker.patch(
      "keras_remote.cli.prerequisites_check.subprocess.run",
    )
    # First call = token check (fails), second call = login (succeeds)
    token_result = mocker.MagicMock()
    token_result.returncode = 1
    mock_run.return_value = token_result

    mocker.patch("keras_remote.cli.prerequisites_check.warning")
    mocker.patch("click.echo")

    check_gcloud_auth()

    assert mock_run.call_count == 2
    # Second call should be the login command
    login_call = mock_run.call_args_list[1]
    assert "login" in login_call[0][0]
