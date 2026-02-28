"""Tests for keras_remote.cli.commands.pool — add, remove, list."""

from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

from keras_remote.cli.commands.pool import pool
from keras_remote.cli.config import NodePoolConfig
from keras_remote.core.accelerators import GpuConfig, TpuConfig

# Patches applied to every test to bypass prerequisites and infrastructure.
_BASE_PATCHES = {
  "check_all": mock.patch("keras_remote.cli.commands.pool.check_all"),
  "create_program": mock.patch("keras_remote.cli.commands.pool.create_program"),
  "get_stack": mock.patch("keras_remote.cli.commands.pool.get_stack"),
  "get_current_node_pools": mock.patch(
    "keras_remote.cli.commands.pool.get_current_node_pools"
  ),
  "generate_pool_name": mock.patch(
    "keras_remote.cli.commands.pool.generate_pool_name",
    return_value="gpu-l4-abcd",
  ),
}


def _start_patches(test_case):
  mocks = {}
  for name, patcher in _BASE_PATCHES.items():
    mocks[name] = test_case.enterContext(patcher)
  return mocks


_ADD_ARGS = [
  "add",
  "--project",
  "test-project",
  "--zone",
  "us-central1-a",
  "--accelerator",
  "l4",
  "--yes",
]

_REMOVE_ARGS = [
  "remove",
  "--project",
  "test-project",
  "--zone",
  "us-central1-a",
  "gpu-l4-abcd",
  "--yes",
]

_LIST_ARGS = [
  "list",
  "--project",
  "test-project",
  "--zone",
  "us-central1-a",
]


class PoolAddTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    mock_stack = mock.MagicMock()
    mock_stack.up.return_value.summary.resource_changes = {"create": 1}
    self.mocks["get_stack"].return_value = mock_stack
    self.mocks["get_current_node_pools"].return_value = []

  def test_add_gpu_pool(self):
    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pool Added", result.output)

  def test_add_to_existing_pools(self):
    existing = NodePoolConfig(
      "tpu-v5p-1234",
      TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
    )
    self.mocks["get_current_node_pools"].return_value = [existing]

    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Total pools after add: 2", result.output)

  def test_add_cpu_rejected(self):
    result = self.runner.invoke(
      pool,
      [
        "add",
        "--project",
        "test-project",
        "--accelerator",
        "cpu",
        "--yes",
      ],
    )

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Cannot add a CPU node pool", result.output)


class PoolRemoveTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    mock_stack = mock.MagicMock()
    mock_stack.up.return_value.summary.resource_changes = {"delete": 1}
    self.mocks["get_stack"].return_value = mock_stack

  def test_remove_existing_pool(self):
    existing = NodePoolConfig(
      "gpu-l4-abcd",
      GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4"),
    )
    self.mocks["get_current_node_pools"].return_value = [existing]

    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pool Removed", result.output)
    self.assertIn("Remaining pools after remove: 0", result.output)

  def test_remove_nonexistent_pool_fails(self):
    self.mocks["get_current_node_pools"].return_value = []

    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("not found", result.output)


if __name__ == "__main__":
  absltest.main()
