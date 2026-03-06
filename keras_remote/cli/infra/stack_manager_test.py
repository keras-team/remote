"""Tests for keras_remote.cli.infra.stack_manager — detect_resources_to_import."""

from unittest import mock

from absl.testing import absltest

from keras_remote.cli.infra.stack_manager import detect_resources_to_import


def _make_stack(tracked_resource_names):
  """Create a mock stack whose state contains the given resource names."""
  resources = [
    {"urn": f"urn:pulumi:stack::project::type::{name}"}
    for name in tracked_resource_names
  ]
  stack = mock.MagicMock()
  stack.export_stack.return_value.deployment = {"resources": resources}
  return stack


class TestDetectResourcesToImport(absltest.TestCase):
  """Tests for detect_resources_to_import."""

  def test_all_tracked_returns_empty(self):
    """When all shared resources are in state, nothing to import."""
    stack = _make_stack(
      [
        "keras-remote-repo",
        "keras-remote-jobs-bucket",
        "keras-remote-builds-bucket",
      ]
    )

    result = detect_resources_to_import(stack, "my-project", "us-central1-a")

    self.assertEqual(result, {})

  @mock.patch("keras_remote.cli.infra.stack_manager.gcs_storage")
  def test_untracked_bucket_exists_in_gcp(self, mock_storage):
    """Untracked bucket that exists in GCP is returned for import."""
    stack = _make_stack(
      [
        "keras-remote-repo",
        "keras-remote-builds-bucket",
      ]
    )
    mock_client = mock_storage.Client.return_value
    mock_client.bucket.return_value.exists.return_value = True

    result = detect_resources_to_import(stack, "my-project", "us-central1-a")

    self.assertIn("keras-remote-jobs-bucket", result)
    self.assertEqual(
      result["keras-remote-jobs-bucket"], "my-project-keras-remote-jobs"
    )

  @mock.patch("keras_remote.cli.infra.stack_manager.gcs_storage")
  def test_untracked_bucket_not_in_gcp(self, mock_storage):
    """Untracked bucket that doesn't exist in GCP is not imported."""
    stack = _make_stack(
      [
        "keras-remote-repo",
        "keras-remote-builds-bucket",
      ]
    )
    mock_client = mock_storage.Client.return_value
    mock_client.bucket.return_value.exists.return_value = False

    result = detect_resources_to_import(stack, "my-project", "us-central1-a")

    self.assertNotIn("keras-remote-jobs-bucket", result)

  @mock.patch("keras_remote.cli.infra.stack_manager.gcs_storage")
  def test_untracked_repo_exists_in_gcp(self, mock_storage):
    """Untracked AR repo that exists in GCP is returned for import."""
    stack = _make_stack(
      [
        "keras-remote-jobs-bucket",
        "keras-remote-builds-bucket",
      ]
    )
    # GCS client shouldn't be called since buckets are tracked.
    mock_storage.Client.return_value.bucket.return_value.exists.return_value = (
      False
    )

    with mock.patch(
      "google.cloud.artifactregistry_v1.ArtifactRegistryClient"
    ) as mock_ar:
      mock_ar.return_value.get_repository.return_value = mock.MagicMock()

      result = detect_resources_to_import(stack, "my-project", "us-central1-a")

    self.assertIn("keras-remote-repo", result)
    self.assertEqual(
      result["keras-remote-repo"],
      "projects/my-project/locations/us/repositories/keras-remote",
    )

  @mock.patch("keras_remote.cli.infra.stack_manager.gcs_storage")
  def test_untracked_repo_not_in_gcp(self, mock_storage):
    """Untracked AR repo that doesn't exist in GCP is not imported."""
    stack = _make_stack(
      [
        "keras-remote-jobs-bucket",
        "keras-remote-builds-bucket",
      ]
    )

    with mock.patch(
      "google.cloud.artifactregistry_v1.ArtifactRegistryClient"
    ) as mock_ar:
      mock_ar.return_value.get_repository.side_effect = Exception("not found")

      result = detect_resources_to_import(stack, "my-project", "us-central1-a")

    self.assertNotIn("keras-remote-repo", result)

  def test_export_stack_failure_returns_empty(self):
    """If export_stack() fails, return empty dict gracefully."""
    stack = mock.MagicMock()
    stack.export_stack.side_effect = Exception("state error")

    result = detect_resources_to_import(stack, "my-project", "us-central1-a")

    self.assertEqual(result, {})

  @mock.patch("keras_remote.cli.infra.stack_manager.gcs_storage")
  def test_multiple_untracked_resources(self, mock_storage):
    """Multiple untracked resources that exist in GCP are all returned."""
    stack = _make_stack([])  # nothing tracked

    mock_client = mock_storage.Client.return_value
    mock_client.bucket.return_value.exists.return_value = True

    with mock.patch(
      "google.cloud.artifactregistry_v1.ArtifactRegistryClient"
    ) as mock_ar:
      mock_ar.return_value.get_repository.return_value = mock.MagicMock()

      result = detect_resources_to_import(stack, "my-project", "us-central1-a")

    self.assertLen(result, 3)
    self.assertIn("keras-remote-repo", result)
    self.assertIn("keras-remote-jobs-bucket", result)
    self.assertIn("keras-remote-builds-bucket", result)

  @mock.patch("keras_remote.cli.infra.stack_manager.gcs_storage")
  def test_gcp_check_failure_skips_resource(self, mock_storage):
    """If GCP existence check raises, skip that resource gracefully."""
    stack = _make_stack([])  # nothing tracked

    mock_client = mock_storage.Client.return_value
    mock_client.bucket.return_value.exists.side_effect = Exception("api error")

    with mock.patch(
      "google.cloud.artifactregistry_v1.ArtifactRegistryClient"
    ) as mock_ar:
      mock_ar.return_value.get_repository.side_effect = Exception("api error")

      result = detect_resources_to_import(stack, "my-project", "us-central1-a")

    self.assertEqual(result, {})


if __name__ == "__main__":
  absltest.main()
