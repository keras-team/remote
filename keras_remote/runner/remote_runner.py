#!/usr/bin/env python3
"""Remote execution entrypoint for keras_remote.

This script runs on the remote TPU/GPU and executes the user's function.
Artifacts are downloaded from and uploaded to Cloud Storage (GCS).
"""

import os
import shutil
import sys
import tempfile
import traceback
import zipfile

import cloudpickle
from google.cloud import storage

# Base temp directory for remote execution artifacts
TEMP_DIR = tempfile.gettempdir()


def main():
    """Main entry point for remote execution.

    Usage: python remote_runner.py gs://bucket/context.zip gs://bucket/payload.pkl gs://bucket/result.pkl
    """

    if len(sys.argv) < 4:
        print("Usage: remote_runner.py <context_gcs> <payload_gcs> <result_gcs>")
        sys.exit(1)

    run_gcs_mode()


def run_gcs_mode():
    """Execute with Cloud Storage artifacts.

    Args from sys.argv:
        sys.argv[1]: GCS path to context.zip
        sys.argv[2]: GCS path to payload.pkl
        sys.argv[3]: GCS path to result.pkl (output)
    """
    context_gcs = sys.argv[1]
    payload_gcs = sys.argv[2]
    result_gcs = sys.argv[3]

    print(f"[REMOTE] Starting GCS execution mode", flush=True)

    # Define local paths using tempfile
    context_path = os.path.join(TEMP_DIR, "context.zip")
    payload_path = os.path.join(TEMP_DIR, "payload.pkl")
    result_path = os.path.join(TEMP_DIR, "result.pkl")
    workspace_dir = os.path.join(TEMP_DIR, "workspace")

    try:
        storage_client = storage.Client()

        # Download artifacts from Cloud Storage
        print(f"[REMOTE] Downloading artifacts...", flush=True)
        _download_from_gcs(storage_client, context_gcs, context_path)
        _download_from_gcs(storage_client, payload_gcs, payload_path)

        # Extract context
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
        os.makedirs(workspace_dir)

        with zipfile.ZipFile(context_path, "r") as zip_ref:
            zip_ref.extractall(workspace_dir)

        # Add workspace to Python path
        sys.path.insert(0, workspace_dir)

        # Load and deserialize payload
        print("[REMOTE] Loading function payload", flush=True)
        with open(payload_path, "rb") as f:
            payload = cloudpickle.load(f)

        func = payload["func"]
        args = payload["args"]
        kwargs = payload["kwargs"]
        env_vars = payload.get("env_vars", {})
        if env_vars:
            print(f"[REMOTE] Setting {len(env_vars)} environment variables", flush=True)
            os.environ.update(env_vars)

        # Execute function and capture result
        print(f"[REMOTE] Executing {func.__name__}()", flush=True)
        result = None
        exception = None

        try:
            result = func(*args, **kwargs)
            print(f"[REMOTE] Function completed successfully", flush=True)
        except BaseException as e:
            print(f"[REMOTE] ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            if isinstance(e, Exception):
                exception = e
            else:
                exception = RuntimeError(f"{type(e).__name__}: {e}")

        # Serialize result or exception
        result_payload = {
            "success": exception is None,
            "result": result if exception is None else None,
            "exception": exception,
            "traceback": traceback.format_exc() if exception else None
        }

        with open(result_path, "wb") as f:
            cloudpickle.dump(result_payload, f)

        # Upload result to Cloud Storage
        print(f"[REMOTE] Uploading result...", flush=True)
        _upload_to_gcs(storage_client, result_path, result_gcs)

        print("[REMOTE] Execution complete", flush=True)
        sys.exit(0 if exception is None else 1)

    except Exception as e:
        print(f"[REMOTE] FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def _download_from_gcs(client, gcs_path, local_path):
    """Download file from GCS.

    Args:
        client: Cloud Storage client
        gcs_path: GCS URI (gs://bucket/path)
        local_path: Local file path
    """
    # Parse gs://bucket/path format
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def _upload_to_gcs(client, local_path, gcs_path):
    """Upload file to GCS.

    Args:
        client: Cloud Storage client
        local_path: Local file path
        gcs_path: GCS URI (gs://bucket/path)
    """
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


if __name__ == "__main__":
    main()
