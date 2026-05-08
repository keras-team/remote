"""GCS state backend bucket lifecycle.

Kinetic's Pulumi state lives in a GCS bucket derived from the GCP project:
``gs://{project}-kinetic-state``. The first ``kinetic`` invocation for a
project creates the bucket; subsequent invocations reuse it.
"""

from google.api_core import exceptions as gax
from google.cloud import storage


def state_backend_url(project: str) -> str:
  """Pulumi backend URL for ``project``'s shared state bucket.

  GCS bucket names are globally unique; prefixing with the GCP project ID
  keeps team buckets distinct. Multiple clusters in one project share the
  bucket but get separate stacks (named ``{project}-{cluster}``).
  """
  return f"gs://{project}-kinetic-state"


def ensure_gcs_backend(project: str, *, location: str = "US") -> None:
  """Best-effort: create the state bucket if it does not exist.

  Idempotent. Versioning + uniform bucket-level access enabled, no public
  ACL. Pinned to the kinetic project so IAM/billing/ownership match the
  rest of the infrastructure.

  ``Conflict`` (bucket already exists), ``Forbidden``, and
  ``PermissionDenied`` are silently swallowed — collaborators with only
  ``roles/storage.objectAdmin`` lack ``storage.admin`` on the project but
  can still read/write state at the object level. Pulumi's first state
  read will surface a clean object-level error if access is wrong.
  """
  bucket_name = state_backend_url(project).removeprefix("gs://")
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)
  bucket.versioning_enabled = True
  bucket.iam_configuration.uniform_bucket_level_access_enabled = True
  try:
    client.create_bucket(bucket, location=location)
  except (gax.Conflict, gax.Forbidden, gax.PermissionDenied):
    return
