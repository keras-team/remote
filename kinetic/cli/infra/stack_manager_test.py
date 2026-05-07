"""Tests for kinetic.cli.infra.stack_manager."""

from unittest import mock

from absl.testing import absltest

from kinetic.cli.config import InfraConfig
from kinetic.cli.infra import stack_manager


class GetStackTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    # Stub out the Pulumi automation API so the test only exercises the
    # backend wiring.
    self.mock_pulumi_cmd = self.enterContext(
      mock.patch.object(stack_manager.auto, "PulumiCommand")
    )
    self.mock_project_settings = self.enterContext(
      mock.patch.object(stack_manager.auto, "ProjectSettings")
    )
    self.mock_project_backend = self.enterContext(
      mock.patch.object(stack_manager.auto, "ProjectBackend")
    )
    self.mock_create_or_select_stack = self.enterContext(
      mock.patch.object(stack_manager.auto, "create_or_select_stack")
    )
    self.mock_local_workspace = self.enterContext(
      mock.patch.object(stack_manager.auto, "LocalWorkspaceOptions")
    )
    self.mock_ensure_gcs = self.enterContext(
      mock.patch.object(stack_manager, "ensure_gcs_backend")
    )

  def test_uses_per_project_gcs_bucket(self):
    config = InfraConfig(project="kinetic-proj", zone="z", cluster_name="c")

    stack_manager.get_stack(lambda: None, config)

    self.assertEqual(
      self.mock_project_backend.call_args.kwargs["url"],
      "gs://kinetic-proj-kinetic-state",
    )

  def test_ensures_bucket_with_project(self):
    config = InfraConfig(project="kinetic-proj", zone="z", cluster_name="c")

    stack_manager.get_stack(lambda: None, config)

    self.mock_ensure_gcs.assert_called_once_with("kinetic-proj")


if __name__ == "__main__":
  absltest.main()
