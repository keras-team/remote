"""Vertex AI job submission for keras_remote."""

from google.cloud import aiplatform

from keras_remote import accelerators, infra

logger = infra.logger


def submit_training_job(display_name, container_uri, accelerator,
                       machine_type, zone, project, job_id, bucket_name):
    """Submit custom training job to Vertex AI.

    Args:
        display_name: Job display name
        container_uri: Docker container image URI
        accelerator: TPU/GPU type (e.g., 'v3-8')
        machine_type: Machine type (optional, auto-detected if None)
        zone: GCP zone
        project: GCP project ID
        job_id: Unique job identifier
        bucket_name: GCS bucket name for artifacts

    Returns:
        Vertex AI CustomJob instance
    """
    # Initialize Vertex AI SDK
    region = _zone_to_region(zone)
    staging_bucket = f'gs://{bucket_name}'
    aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)

    # Parse accelerator configuration
    accel_config = _parse_accelerator(accelerator)

    # Determine machine type if not specified
    if not machine_type:
        machine_type = accel_config["machine_type"]

    # Create machine spec
    machine_spec = {
        "machine_type": machine_type,
    }

    # Add accelerator config if present
    if accel_config.get("accelerator_type"):
        machine_spec["accelerator_type"] = accel_config["accelerator_type"]
        machine_spec["accelerator_count"] = accel_config["accelerator_count"]

    # Create worker pool spec
    worker_pool_specs = [{
        "machine_spec": machine_spec,
        "replica_count": 1,
        "container_spec": {
            "image_uri": container_uri,
            "command": ["python3", "-u", "/app/remote_runner.py"],
            "args": [
                f'gs://{bucket_name}/{job_id}/context.zip',
                f'gs://{bucket_name}/{job_id}/payload.pkl',
                f'gs://{bucket_name}/{job_id}/result.pkl',
            ],
            "env": [
                {"name": "KERAS_BACKEND", "value": "jax"},
                {"name": "JAX_PLATFORMS", "value": accelerators.get_category(accelerator)},
            ]
        },
    }]

    # Create custom job
    logger.info(f"Submitting job to Vertex AI: {display_name}")
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f'gs://{bucket_name}/{job_id}/output',
    )

    # Submit job (non-blocking)
    job.run(sync=False)

    logger.info("Job submitted. Waiting for completion...")
    logger.info(f"View job at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project}")

    return job


def wait_for_job(job):
    """Wait for job to complete and return result.

    Args:
        job: Vertex AI CustomJob instance

    Returns:
        Job state
    """
    # Wait for job to complete
    job.wait()

    # Check job state
    state = job.state

    if state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED:
        print(f"[REMOTE] Job completed successfully")
        return "success"
    elif state == aiplatform.gapic.JobState.JOB_STATE_FAILED:
        print(f"[REMOTE] Job failed")
        raise RuntimeError(f"Vertex AI job failed: {job.error}")
    elif state == aiplatform.gapic.JobState.JOB_STATE_CANCELLED:
        print(f"[REMOTE] Job was cancelled")
        raise RuntimeError("Vertex AI job was cancelled")
    else:
        print(f"[REMOTE] Job ended with state: {state}")
        return str(state)


def _zone_to_region(zone):
    """Convert GCP zone to region.

    Args:
        zone: GCP zone (e.g., 'us-central1-a')

    Returns:
        Region (e.g., 'us-central1')
    """
    if not zone:
        return "us-central1"

    # Zone format: {region}-{zone_letter}
    parts = zone.rsplit("-", 1)
    return parts[0] if len(parts) > 1 else zone


def _parse_accelerator(accelerator):
    """Convert accelerator string to Vertex AI machine spec fields."""
    parsed = accelerators.parse_accelerator(accelerator)
    t = parsed.accelerator_type

    if t.category == "cpu":
        raise ValueError("CPU-only mode is not supported on Vertex AI. Specify a GPU or TPU.")

    if t.category == "tpu":
        if "{chips}" in t.vertex_tpu_template:
            return {
                "machine_type": t.vertex_tpu_template.format(chips=parsed.count),
                "accelerator_type": None,
                "accelerator_count": None,
            }
        return {
            "machine_type": t.vertex_tpu_template or "cloud-tpu",
            "accelerator_type": t.vertex_tpu_type,
            "accelerator_count": parsed.count,
        }

    # GPU
    machine_type = t.vertex_machines.get(parsed.count)
    if machine_type is None:
        raise ValueError(
            f"GPU count {parsed.count} not supported for '{t.short_name}' on Vertex AI."
        )
    return {
        "machine_type": machine_type,
        "accelerator_type": t.vertex_type,
        "accelerator_count": parsed.count,
    }
