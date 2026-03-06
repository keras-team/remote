"""Tests for keras_remote.cli.infra.stack_manager."""

import os
from unittest import mock

from absl.testing import absltest
from pulumi.automation import errors as pulumi_errors

from keras_remote.cli.infra import stack_manager


class MakeStackNameTest(absltest.TestCase):
  def test_combines_project_and_cluster(self):
    self.assertEqual(
      stack_manager.make_stack_name("my-proj", "my-cluster"),
      "my-proj-my-cluster",
    )


class GetStackTest(absltest.TestCase):
  """Tests for get_stack with cluster-scoped stack names."""

  def setUp(self):
    super().setUp()
    self.enterContext(mock.patch.object(stack_manager, "_get_workspace"))

  @mock.patch("pulumi.automation.select_stack")
  def test_selects_existing_stack(self, mock_select):
    mock_stack = mock.MagicMock()
    mock_select.return_value = mock_stack

    config = mock.MagicMock()
    config.project = "proj"
    config.cluster_name = "cluster"
    config.zone = "us-central1-a"

    result = stack_manager.get_stack(lambda: None, config)

    self.assertIs(result, mock_stack)
    mock_select.assert_called_once()
    self.assertEqual(mock_select.call_args.kwargs["stack_name"], "proj-cluster")

  @mock.patch("pulumi.automation.create_stack")
  @mock.patch("pulumi.automation.select_stack")
  def test_creates_when_not_found(self, mock_select, mock_create):
    mock_select.side_effect = pulumi_errors.CommandError("not found")
    mock_stack = mock.MagicMock()
    mock_create.return_value = mock_stack

    config = mock.MagicMock()
    config.project = "proj"
    config.cluster_name = "cluster"
    config.zone = "us-central1-a"

    result = stack_manager.get_stack(lambda: None, config)

    self.assertIs(result, mock_stack)
    mock_create.assert_called_once()
    self.assertEqual(mock_create.call_args.kwargs["stack_name"], "proj-cluster")

  @mock.patch("pulumi.automation.select_stack")
  def test_sets_cluster_name_config(self, mock_select):
    """get_stack stores cluster_name as a stack config value."""
    mock_stack = mock.MagicMock()
    mock_select.return_value = mock_stack

    config = mock.MagicMock()
    config.project = "proj"
    config.cluster_name = "my-cluster"
    config.zone = "us-central1-a"

    stack_manager.get_stack(lambda: None, config)

    # Verify cluster_name was set as config.
    calls = {
      c[0][0]: c[0][1].value for c in mock_stack.set_config.call_args_list
    }
    self.assertEqual(calls["keras-remote:cluster_name"], "my-cluster")


class ActiveStackTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    import tempfile

    self.tmp_dir = tempfile.mkdtemp()
    self.active_file = os.path.join(self.tmp_dir, "active-stack")
    self.enterContext(
      mock.patch.object(stack_manager, "_ACTIVE_STACK_FILE", self.active_file)
    )

  def test_get_returns_none_when_no_file(self):
    self.assertIsNone(stack_manager.get_active_stack())

  def test_set_then_get(self):
    stack_manager.set_active_stack("proj-my-cluster")
    self.assertEqual(stack_manager.get_active_stack(), "proj-my-cluster")

  def test_set_overwrites_previous(self):
    stack_manager.set_active_stack("first")
    stack_manager.set_active_stack("second")
    self.assertEqual(stack_manager.get_active_stack(), "second")

  def test_get_returns_none_for_empty_file(self):
    with open(self.active_file, "w") as f:
      f.write("")
    self.assertIsNone(stack_manager.get_active_stack())

  def test_clear_active_stack(self):
    stack_manager.set_active_stack("my-stack")
    stack_manager.clear_active_stack()
    self.assertIsNone(stack_manager.get_active_stack())

  def test_clear_nonexistent_file_no_error(self):
    stack_manager.clear_active_stack()  # Should not raise.


class ResolveStackInfoTest(absltest.TestCase):
  """Tests for the shared resolve_stack_info function."""

  @mock.patch.object(stack_manager, "get_stack_config", return_value=None)
  @mock.patch.object(stack_manager, "get_stack_outputs")
  def test_reads_from_outputs(self, mock_outputs, _):
    mock_outputs.return_value = {
      "project": mock.MagicMock(value="proj"),
      "zone": mock.MagicMock(value="us-central1-a"),
      "cluster_name": mock.MagicMock(value="cluster"),
    }
    info = stack_manager.resolve_stack_info("proj-cluster")
    self.assertEqual(info.project, "proj")
    self.assertEqual(info.zone, "us-central1-a")
    self.assertEqual(info.cluster_name, "cluster")

  @mock.patch.object(stack_manager, "get_stack_config")
  @mock.patch.object(stack_manager, "get_stack_outputs", return_value=None)
  def test_falls_back_to_config(self, _, mock_config):
    mock_config.return_value = {
      "gcp:project": mock.MagicMock(value="proj"),
      "gcp:zone": mock.MagicMock(value="us-central1-a"),
      "keras-remote:cluster_name": mock.MagicMock(value="my-cluster"),
    }
    info = stack_manager.resolve_stack_info("proj-my-cluster")
    self.assertEqual(info.project, "proj")
    self.assertEqual(info.zone, "us-central1-a")
    self.assertEqual(info.cluster_name, "my-cluster")

  @mock.patch.object(stack_manager, "get_stack_config")
  @mock.patch.object(stack_manager, "get_stack_outputs")
  def test_partial_outputs_filled_from_config(self, mock_outputs, mock_config):
    """Outputs have project but not zone/cluster; config fills the gaps."""
    mock_outputs.return_value = {
      "project": mock.MagicMock(value="proj"),
    }
    mock_config.return_value = {
      "gcp:zone": mock.MagicMock(value="us-central1-a"),
      "keras-remote:cluster_name": mock.MagicMock(value="my-cluster"),
    }
    info = stack_manager.resolve_stack_info("proj-my-cluster")
    self.assertEqual(info.project, "proj")
    self.assertEqual(info.zone, "us-central1-a")
    self.assertEqual(info.cluster_name, "my-cluster")

  @mock.patch.object(stack_manager, "get_stack_config", return_value=None)
  @mock.patch.object(stack_manager, "get_stack_outputs", return_value=None)
  def test_returns_nones_when_stack_missing(self, *_):
    info = stack_manager.resolve_stack_info("nonexistent")
    self.assertIsNone(info.project)
    self.assertIsNone(info.zone)
    self.assertIsNone(info.cluster_name)


class ResolveFromActiveStackTest(absltest.TestCase):
  @mock.patch.object(stack_manager, "get_active_stack", return_value=None)
  def test_returns_nones_when_no_active_stack(self, _):
    result = stack_manager.resolve_from_active_stack()
    self.assertIsNone(result.project)
    self.assertIsNone(result.zone)
    self.assertIsNone(result.cluster_name)
    self.assertIsNone(result.stack_name)

  @mock.patch.object(stack_manager, "resolve_stack_info")
  @mock.patch.object(
    stack_manager, "get_active_stack", return_value="proj-cluster"
  )
  def test_delegates_to_resolve_stack_info(self, _, mock_info):
    mock_info.return_value = stack_manager.StackInfo(
      "proj", "us-central1-a", "cluster"
    )
    result = stack_manager.resolve_from_active_stack()
    self.assertEqual(result.project, "proj")
    self.assertEqual(result.zone, "us-central1-a")
    self.assertEqual(result.cluster_name, "cluster")
    self.assertEqual(result.stack_name, "proj-cluster")
    mock_info.assert_called_once_with("proj-cluster")


class RequireActiveStackTest(absltest.TestCase):
  @mock.patch.object(stack_manager, "resolve_from_active_stack")
  def test_raises_when_no_active_stack(self, mock_resolve):
    mock_resolve.return_value = stack_manager.ActiveStackResolution(
      None, None, None, None
    )
    with self.assertRaises(Exception) as ctx:
      stack_manager.require_active_stack()
    self.assertIn("No active stack set", str(ctx.exception))

  @mock.patch.object(stack_manager, "resolve_from_active_stack")
  def test_returns_resolution_when_active(self, mock_resolve):
    expected = stack_manager.ActiveStackResolution(
      "proj", "us-central1-a", "cluster", "proj-cluster"
    )
    mock_resolve.return_value = expected
    result = stack_manager.require_active_stack()
    self.assertEqual(result, expected)


class StackExistsTest(absltest.TestCase):
  @mock.patch.object(stack_manager, "_select_readonly_stack")
  def test_returns_true_when_found(self, mock_select):
    mock_select.return_value = mock.MagicMock()
    self.assertTrue(stack_manager.stack_exists("my-stack"))

  @mock.patch.object(stack_manager, "_select_readonly_stack")
  def test_returns_false_when_not_found(self, mock_select):
    mock_select.return_value = None
    self.assertFalse(stack_manager.stack_exists("nope"))


class GetStackOutputsTest(absltest.TestCase):
  @mock.patch.object(stack_manager, "_select_readonly_stack")
  def test_returns_outputs(self, mock_select):
    mock_stack = mock.MagicMock()
    mock_stack.outputs.return_value = {"key": "val"}
    mock_select.return_value = mock_stack
    self.assertEqual(stack_manager.get_stack_outputs("s"), {"key": "val"})

  @mock.patch.object(stack_manager, "_select_readonly_stack")
  def test_returns_none_when_missing(self, mock_select):
    mock_select.return_value = None
    self.assertIsNone(stack_manager.get_stack_outputs("s"))


class GetStackConfigTest(absltest.TestCase):
  @mock.patch.object(stack_manager, "_select_readonly_stack")
  def test_returns_config(self, mock_select):
    mock_stack = mock.MagicMock()
    mock_stack.get_all_config.return_value = {"k": "v"}
    mock_select.return_value = mock_stack
    self.assertEqual(stack_manager.get_stack_config("s"), {"k": "v"})

  @mock.patch.object(stack_manager, "_select_readonly_stack")
  def test_returns_none_when_missing(self, mock_select):
    mock_select.return_value = None
    self.assertIsNone(stack_manager.get_stack_config("s"))


class GetClusterNameFromOutputsTest(absltest.TestCase):
  def test_returns_name_when_present(self):
    stack = mock.MagicMock()
    stack.outputs.return_value = {
      "cluster_name": mock.MagicMock(value="my-cluster"),
    }
    self.assertEqual(
      stack_manager.get_cluster_name_from_outputs(stack), "my-cluster"
    )

  def test_returns_none_when_missing(self):
    stack = mock.MagicMock()
    stack.outputs.return_value = {}
    self.assertIsNone(stack_manager.get_cluster_name_from_outputs(stack))


if __name__ == "__main__":
  absltest.main()
