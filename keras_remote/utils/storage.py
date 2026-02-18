"""Cloud Storage operations for keras_remote."""

import os
import tempfile

from absl import logging
from google.cloud import storage


def _get_project():
  """Get project ID from environment or default gcloud config."""
  return os.environ.get("KERAS_REMOTE_PROJECT") or os.environ.get(
    "GOOGLE_CLOUD_PROJECT"
  )


def upload_artifacts(
  bucket_name, job_id, payload_path, context_path, project=None
):
  """Upload execution artifacts to Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      job_id: Unique job identifier
      payload_path: Local path to payload.pkl
      context_path: Local path to context.zip
      project: GCP project ID (optional, uses env vars if not provided)
  """
  project = project or _get_project()

  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)

  # Upload payload
  blob = bucket.blob(f"{job_id}/payload.pkl")
  blob.upload_from_filename(payload_path)
  logging.info(
    "Uploaded payload to gs://%s/%s/payload.pkl", bucket_name, job_id
  )

  # Upload context
  blob = bucket.blob(f"{job_id}/context.zip")
  blob.upload_from_filename(context_path)
  logging.info(
    "Uploaded context to gs://%s/%s/context.zip", bucket_name, job_id
  )

  # Get project ID for console link
  project = client.project
  logging.info(
    "View artifacts: https://console.cloud.google.com/storage/browser/%s/%s?project=%s",
    bucket_name,
    job_id,
    project,
  )


def download_result(bucket_name, job_id, project=None):
  """Download result from Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      job_id: Unique job identifier
      project: GCP project ID (optional, uses env vars if not provided)

  Returns:
      Local path to downloaded result file
  """
  project = project or _get_project()
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)

  blob = bucket.blob(f"{job_id}/result.pkl")
  local_path = os.path.join(tempfile.gettempdir(), f"result-{job_id}.pkl")
  blob.download_to_filename(local_path)
  logging.info(
    "Downloaded result from gs://%s/%s/result.pkl", bucket_name, job_id
  )

  return local_path


def cleanup_artifacts(bucket_name, job_id, project=None):
  """Clean up job artifacts from Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      job_id: Unique job identifier
      project: GCP project ID (optional, uses env vars if not provided)
  """
  project = project or _get_project()
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)

  # Delete all blobs with job_id prefix
  blobs = bucket.list_blobs(prefix=f"{job_id}/")
  deleted_count = 0
  for blob in blobs:
    blob.delete()
    deleted_count += 1

  if deleted_count > 0:
    logging.info(
      "Cleaned up %d artifacts from gs://%s/%s/",
      deleted_count,
      bucket_name,
      job_id,
    )
