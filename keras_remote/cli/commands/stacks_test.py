"""Tests for keras_remote.cli.commands.stacks — list, set, and delete."""

from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

from keras_remote.cli.commands.stacks import stacks
from keras_remote.cli.infra.stack_manager import StackInfo


class StacksListTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mock_list = self.enterContext(
      mock.patch("keras_remote.cli.commands.stacks.list_stacks")
    )
    self.mock_resolve = self.enterContext(
      mock.patch("keras_remote.cli.commands.stacks.resolve_stack_info")
    )
    self.mock_active = self.enterContext(
      mock.patch("keras_remote.cli.commands.stacks.get_active_stack")
    )

  def test_empty_list(self):
    self.mock_list.return_value = []

    result = self.runner.invoke(stacks, ["list"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("No stacks found", result.output)

  def test_shows_stacks_table(self):
    summary = mock.MagicMock()
    summary.name = "proj-my-cluster"
    summary.last_update = "2025-01-01"
    self.mock_list.return_value = [summary]
    self.mock_resolve.return_value = StackInfo(
      "proj", "us-central1-a", "my-cluster"
    )
    self.mock_active.return_value = "proj-my-cluster"

    result = self.runner.invoke(stacks, ["list"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("proj-my-cluster", result.output)
    self.assertIn("my-cluster", result.output)

  def test_default_subcommand_is_list(self):
    self.mock_list.return_value = []

    result = self.runner.invoke(stacks)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("No stacks found", result.output)


class StacksSetTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mock_exists = self.enterContext(
      mock.patch("keras_remote.cli.commands.stacks.stack_exists")
    )
    self.mock_set = self.enterContext(
      mock.patch("keras_remote.cli.commands.stacks.set_active_stack")
    )

  def test_set_valid_stack(self):
    self.mock_exists.return_value = True

    result = self.runner.invoke(stacks, ["set", "proj-my-cluster"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.mock_set.assert_called_once_with("proj-my-cluster")
    self.assertIn("Active stack set", result.output)

  def test_set_nonexistent_stack_fails(self):
    self.mock_exists.return_value = False

    result = self.runner.invoke(stacks, ["set", "nope"])

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("not found", result.output)
    self.mock_set.assert_not_called()


class StacksDeleteTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mock_exists = self.enterContext(
      mock.patch("keras_remote.cli.commands.stacks.stack_exists")
    )
    self.mock_delete = self.enterContext(
      mock.patch("keras_remote.cli.commands.stacks.delete_stack")
    )

  def test_delete_valid_stack(self):
    self.mock_exists.return_value = True

    result = self.runner.invoke(stacks, ["delete", "proj-cluster", "--yes"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.mock_delete.assert_called_once_with("proj-cluster")
    self.assertIn("deleted", result.output)

  def test_delete_nonexistent_stack_fails(self):
    self.mock_exists.return_value = False

    result = self.runner.invoke(stacks, ["delete", "nope", "--yes"])

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("not found", result.output)
    self.mock_delete.assert_not_called()


if __name__ == "__main__":
  absltest.main()
