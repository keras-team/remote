"""Cloud Storage operations for keras_remote."""

import os
import tempfile

from google.cloud import storage
from google.cloud.exceptions import NotFound

from keras_remote.infra import infra

logger = infra.logger


def _get_project():
    """Get project ID from environment or default gcloud config."""
    return os.environ.get("KERAS_REMOTE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def upload_artifacts(bucket_name, job_id, payload_path, context_path, location="us-central1", project=None):
    """Upload execution artifacts to Cloud Storage.

    Args:
        bucket_name: Name of the GCS bucket
        job_id: Unique job identifier
        payload_path: Local path to payload.pkl
        context_path: Local path to context.zip
        location: GCS bucket location (default: 'us-central1')
        project: GCP project ID (optional, uses env vars if not provided)
    """
    project = project or _get_project()

    # Ensure bucket exists
    _ensure_bucket(bucket_name, location=location, project=project)

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    # Upload payload
    blob = bucket.blob(f'{job_id}/payload.pkl')
    blob.upload_from_filename(payload_path)
    logger.info(f"Uploaded payload to gs://{bucket_name}/{job_id}/payload.pkl")

    # Upload context
    blob = bucket.blob(f'{job_id}/context.zip')
    blob.upload_from_filename(context_path)
    logger.info(f"Uploaded context to gs://{bucket_name}/{job_id}/context.zip")

    # Get project ID for console link
    project = client.project
    logger.info(f"View artifacts: https://console.cloud.google.com/storage/browser/{bucket_name}/{job_id}?project={project}")


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

    blob = bucket.blob(f'{job_id}/result.pkl')
    local_path = os.path.join(tempfile.gettempdir(), f'result-{job_id}.pkl')
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded result from gs://{bucket_name}/{job_id}/result.pkl")

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
    blobs = bucket.list_blobs(prefix=f'{job_id}/')
    deleted_count = 0
    for blob in blobs:
        blob.delete()
        deleted_count += 1

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} artifacts from gs://{bucket_name}/{job_id}/")


def _ensure_bucket(bucket_name, location="us-central1", project=None):
    """Create bucket if it doesn't exist.

    Args:
        bucket_name: Name of the GCS bucket
        location: GCS bucket location (default: 'us-central1')
        project: GCP project ID (optional, uses env vars if not provided)
    """
    project = project or _get_project()
    client = storage.Client(project=project)

    try:
        client.get_bucket(bucket_name)
    except NotFound:
        # Bucket doesn't exist, create it
        bucket = client.create_bucket(bucket_name, location=location)
        project = client.project
        logger.info(f"Created bucket: gs://{bucket_name} in location: {location}")
        logger.info(f"View bucket: https://console.cloud.google.com/storage/browser/{bucket_name}?project={project}")
