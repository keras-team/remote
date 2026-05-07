"""Tests for kinetic.cli.commands.config — show."""

import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

from kinetic.cli.commands.profile import profile as profile_cmd
from kinetic.cli.main import cli


def _tmp(testcase):
  td = tempfile.TemporaryDirectory()
  testcase.addCleanup(td.cleanup)
  return Path(td.name)


def _make_runner_env(tmp_path):
  return {"KINETIC_PROFILES_FILE": str(tmp_path / "profiles.json")}


class ConfigShowTest(absltest.TestCase):
  def test_show_with_no_profile(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("No active profile", result.output)
    # Pulumi state row hides until project is known.
    self.assertNotIn("Pulumi State", result.output)

  def test_show_with_active_profile_includes_state_url(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    runner.invoke(
      profile_cmd,
      [
        "create",
        "dev",
        "--project",
        "super-proj",
        "--zone",
        "us-west4-b",
        "--cluster",
        "my-cluster",
        "--namespace",
        "ns",
      ],
      env=env,
    )
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("Pulumi State", result.output)
    self.assertIn("gs://super-proj-kinetic-state", result.output)

  def test_show_uses_env_project_for_state_url(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = dict(_make_runner_env(tmp))
    env["KINETIC_PROJECT"] = "from-env"
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("gs://from-env-kinetic-state", result.output)


if __name__ == "__main__":
  absltest.main()
