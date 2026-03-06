"""Cloud Storage operations for keras_remote."""

from __future__ import annotations

import os
import tempfile

from absl import logging
from google.cloud import storage

from keras_remote.data import Data
from keras_remote.infra.infra import get_default_project


def upload_artifacts(
  bucket_name: str,
  gcs_prefix: str,
  payload_path: str,
  context_path: str,
  project: str | None = None,
) -> None:
  """Upload execution artifacts to Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      gcs_prefix: Namespace-scoped prefix, e.g. "default/job-abc123"
      payload_path: Local path to payload.pkl
      context_path: Local path to context.zip
      project: GCP project ID (optional, uses env vars if not provided)
  """
  project = project or get_default_project()

  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)

  # Upload payload
  blob = bucket.blob(f"{gcs_prefix}/payload.pkl")
  blob.upload_from_filename(payload_path)
  logging.info(
    "Uploaded payload to gs://%s/%s/payload.pkl",
    bucket_name,
    gcs_prefix,
  )

  # Upload context
  blob = bucket.blob(f"{gcs_prefix}/context.zip")
  blob.upload_from_filename(context_path)
  logging.info(
    "Uploaded context to gs://%s/%s/context.zip",
    bucket_name,
    gcs_prefix,
  )

  # Get project ID for console link
  project = client.project
  logging.info(
    "View artifacts: https://console.cloud.google.com/storage/browser/%s/%s?project=%s",
    bucket_name,
    gcs_prefix,
    project,
  )


def download_result(
  bucket_name: str, gcs_prefix: str, project: str | None = None
) -> str:
  """Download result from Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      gcs_prefix: Namespace-scoped prefix, e.g. "default/job-abc123"
      project: GCP project ID (optional, uses env vars if not provided)

  Returns:
      Local path to downloaded result file
  """
  project = project or get_default_project()
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)

  blob = bucket.blob(f"{gcs_prefix}/result.pkl")
  safe_name = gcs_prefix.replace("/", "-")
  local_path = os.path.join(tempfile.gettempdir(), f"result-{safe_name}.pkl")
  blob.download_to_filename(local_path)
  logging.info(
    "Downloaded result from gs://%s/%s/result.pkl",
    bucket_name,
    gcs_prefix,
  )

  return local_path


def cleanup_artifacts(
  bucket_name: str, gcs_prefix: str, project: str | None = None
) -> None:
  """Clean up job artifacts from Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      gcs_prefix: Namespace-scoped prefix, e.g. "default/job-abc123"
      project: GCP project ID (optional, uses env vars if not provided)
  """
  project = project or get_default_project()
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)

  # Delete all blobs with gcs_prefix
  blobs = bucket.list_blobs(prefix=f"{gcs_prefix}/")
  deleted_count = 0
  for blob in blobs:
    blob.delete()
    deleted_count += 1

  if deleted_count > 0:
    logging.info(
      "Cleaned up %d artifacts from gs://%s/%s/",
      deleted_count,
      bucket_name,
      gcs_prefix,
    )


def upload_data(
  bucket_name: str,
  data: Data,
  project: str | None = None,
  namespace_prefix: str = "default",
) -> str:
  """Upload a Data object to GCS with content-based caching.

  For GCS Data: returns the original URI (no upload).
  For local Data: computes content hash, uploads on cache miss.

  Args:
      bucket_name: GCS bucket name.
      data: Data object to upload.
      project: GCP project ID (auto-detected if None).
      namespace_prefix: Namespace GCS prefix. Defaults to "default".

  Returns:
      GCS URI where the data is available.
  """
  if data.is_gcs:
    logging.info("Data already on GCS: %s", data.path)
    return data.path

  content_hash = data.content_hash()
  namespace_prefix = namespace_prefix.strip("/")
  cache_prefix = f"{namespace_prefix}/data-cache/{content_hash}"

  project = project or get_default_project()
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)

  # O(1) cache hit check via sentinel blob
  marker_blob = bucket.blob(f"{cache_prefix}/.cache_marker")
  if marker_blob.exists():
    gcs_uri = f"gs://{bucket_name}/{cache_prefix}"
    logging.info(
      "Data cache hit (hash=%s...): %s",
      content_hash[:12],
      gcs_uri,
    )
    return gcs_uri

  # Size warning for large local data
  total_size = _compute_total_size(data.path)
  if total_size > 10 * 1024**3:  # 10 GB
    size_gb = total_size / (1024**3)
    logging.warning(
      "Data at '%s' is %.1f GB. For large datasets, consider using "
      'a direct GCS URI (Data("gs://...")) with framework-native '
      "I/O (tf.data, grain) for better performance.",
      data._raw_path,
      size_gb,
    )

  # Cache miss — upload
  logging.info(
    "Uploading data (hash=%s...) to gs://%s/%s/",
    content_hash[:12],
    bucket_name,
    cache_prefix,
  )

  if data.is_dir:
    _upload_directory(bucket, data.path, cache_prefix)
  else:
    filename = os.path.basename(data.path)
    blob = bucket.blob(f"{cache_prefix}/{filename}")
    blob.upload_from_filename(data.path)

  # Write sentinel last — signals upload-complete
  marker_blob.upload_from_string("")
  logging.info("Data uploaded to gs://%s/%s/", bucket_name, cache_prefix)
  return f"gs://{bucket_name}/{cache_prefix}"


def _compute_total_size(path: str) -> int:
  """Compute total size in bytes of a file or directory."""
  if os.path.isfile(path):
    return os.path.getsize(path)
  total = 0
  for root, _dirs, files in os.walk(path):
    for fname in files:
      total += os.path.getsize(os.path.join(root, fname))
  return total


def _upload_directory(
  bucket: storage.Bucket, local_dir: str, gcs_prefix: str
) -> None:
  """Upload a local directory to GCS preserving structure."""
  for root, _dirs, files in os.walk(local_dir):
    for fname in files:
      local_path = os.path.join(root, fname)
      rel_path = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
      blob = bucket.blob(f"{gcs_prefix}/{rel_path}")
      blob.upload_from_filename(local_path)
