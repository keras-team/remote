"""Tests for kinetic.cli.infra.state_backend."""

from unittest import mock

from absl.testing import absltest
from google.api_core import exceptions as gax

from kinetic.cli.infra import state_backend


class StateBackendUrlTest(absltest.TestCase):
  def test_derives_from_project(self):
    self.assertEqual(
      state_backend.state_backend_url("my-proj"),
      "gs://my-proj-kinetic-state",
    )


class EnsureGcsBackendTest(absltest.TestCase):
  """ensure_gcs_backend is best-effort. It tries to create the bucket once;
  Conflict / Forbidden / PermissionDenied are silently swallowed so that
  collaborators with only object-level perms reach Pulumi, which surfaces
  a clean object-level error if access is actually wrong."""

  def _patch_client(self, bucket_mock):
    client_mock = mock.MagicMock()
    client_mock.bucket.return_value = bucket_mock
    return (
      mock.patch.object(
        state_backend.storage, "Client", return_value=client_mock
      ),
      client_mock,
    )

  def test_creates_with_versioning_and_ubla(self):
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    with patcher:
      state_backend.ensure_gcs_backend("my-proj")
    self.assertTrue(bucket.versioning_enabled)
    self.assertTrue(
      bucket.iam_configuration.uniform_bucket_level_access_enabled
    )
    client_mock.create_bucket.assert_called_once()

  def test_storage_client_pinned_to_project(self):
    bucket = mock.MagicMock()
    with mock.patch.object(state_backend.storage, "Client") as client_cls:
      client_cls.return_value.bucket.return_value = bucket
      state_backend.ensure_gcs_backend("kinetic-proj")
    client_cls.assert_called_once_with(project="kinetic-proj")

  def test_bucket_name_derived_from_project(self):
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    with patcher:
      state_backend.ensure_gcs_backend("my-proj")
    client_mock.bucket.assert_called_once_with("my-proj-kinetic-state")

  def test_conflict_swallowed_for_collaborators(self):
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    client_mock.create_bucket.side_effect = gax.Conflict("exists")
    with patcher:
      state_backend.ensure_gcs_backend("my-proj")  # no exception

  def test_forbidden_swallowed_for_collaborators(self):
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    client_mock.create_bucket.side_effect = gax.Forbidden("nope")
    with patcher:
      state_backend.ensure_gcs_backend("my-proj")  # no exception

  def test_permission_denied_swallowed(self):
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    client_mock.create_bucket.side_effect = gax.PermissionDenied("nope")
    with patcher:
      state_backend.ensure_gcs_backend("my-proj")  # no exception


if __name__ == "__main__":
  absltest.main()
