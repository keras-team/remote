"""GKE job submission for keras_remote."""

from contextlib import suppress
import time

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from keras_remote.core.accelerators import TpuConfig
from keras_remote.core import accelerators
from keras_remote.infra import infra

logger = infra.logger


def submit_k8s_job(
    display_name,
    container_uri,
    accelerator,
    project,
    job_id,
    bucket_name,
    namespace="default",
):
    """Submit a Kubernetes Job to GKE cluster.

    Args:
        display_name: Job display name (used for K8s job name)
        container_uri: Docker container image URI
        accelerator: GPU type (e.g., 'l4', 'a100', 'nvidia-l4')
        project: GCP project ID
        job_id: Unique job identifier
        bucket_name: GCS bucket name for artifacts
        namespace: Kubernetes namespace (default: "default")

    Returns:
        kubernetes.client.V1Job object
    """
    # Load kubeconfig
    _load_kube_config()

    # Parse accelerator configuration
    accel_config = _parse_accelerator(accelerator)

    # Create job specification
    job_name = f"keras-remote-{job_id}"
    job = _create_job_spec(
        job_name=job_name,
        container_uri=container_uri,
        accel_config=accel_config,
        job_id=job_id,
        bucket_name=bucket_name,
        namespace=namespace,
    )

    # Submit job
    batch_v1 = client.BatchV1Api()

    try:
        created_job = batch_v1.create_namespaced_job(namespace=namespace, body=job)
        logger.info(f"Submitted K8s job: {job_name}")
        logger.info(f"View job with: kubectl get job {job_name} -n {namespace}")
        logger.info(
            f"View logs with: kubectl logs -l job-name={job_name} -n {namespace}"
        )
        return created_job
    except ApiException as e:
        if e.status == 403:
            raise RuntimeError(
                f"Permission denied creating K8s Job. Ensure your kubeconfig "
                f"has 'create' permission for Jobs in namespace '{namespace}'. "
                f"Run: kubectl auth can-i create jobs -n {namespace}"
            )
        elif e.status == 404:
            raise RuntimeError(
                f"Namespace '{namespace}' not found. Create it with: "
                f"kubectl create namespace {namespace}"
            )
        elif e.status == 409:
            raise RuntimeError(
                f"Job '{job_name}' already exists. "
                f"Clean up with: kubectl delete job {job_name} -n {namespace}"
            )
        else:
            raise RuntimeError(
                f"Kubernetes API error: {e.status} - {e.reason}: {e.body}"
            )


def wait_for_job(job, namespace="default", timeout=3600, poll_interval=10):
    """Wait for Kubernetes Job to complete.

    Args:
        job: Kubernetes Job object
        namespace: Kubernetes namespace
        timeout: Maximum time to wait in seconds (default: 1 hour)
        poll_interval: Time between status checks in seconds

    Returns:
        Job status: 'success'

    Raises:
        RuntimeError: If job fails or times out
    """
    _load_kube_config()
    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()

    job_name = job.metadata.name
    start_time = time.time()
    logged_running = False

    while True:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f"GKE job {job_name} timed out after {timeout}s")

        # Get job status
        try:
            job_status = batch_v1.read_namespaced_job_status(job_name, namespace)
        except ApiException as e:
            raise RuntimeError(f"Failed to read job status: {e.reason}")

        # Check completion conditions
        if job_status.status.succeeded and job_status.status.succeeded >= 1:
            print(f"[REMOTE] Job {job_name} completed successfully")
            return "success"

        if job_status.status.failed and job_status.status.failed >= 1:
            # Get pod logs for debugging
            _print_pod_logs(core_v1, job_name, namespace)
            raise RuntimeError(f"GKE job {job_name} failed")

        # Check for pod scheduling issues
        _check_pod_scheduling(core_v1, job_name, namespace)

        # Job still running
        if not logged_running:
            logger.info(f"Job {job_name} running...")
            logged_running = True

        time.sleep(poll_interval)


def cleanup_job(job_name, namespace="default"):
    """Delete completed Kubernetes Job and its pods.

    Args:
        job_name: Name of the Kubernetes Job
        namespace: Kubernetes namespace
    """
    _load_kube_config()
    batch_v1 = client.BatchV1Api()

    try:
        # Delete job with propagation policy to also delete pods
        batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=namespace,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )
        logger.info(f"Deleted K8s job: {job_name}")
    except ApiException as e:
        if e.status == 404:
            # Job already deleted
            pass
        else:
            logger.warning(f"Failed to delete job {job_name}: {e.reason}")


def _parse_accelerator(accelerator):
    """Convert accelerator string to GKE pod spec fields."""
    parsed = accelerators.parse_accelerator(accelerator)

    if parsed is None:
        return {
            "node_selector": {},
            "resource_limits": {},
            "resource_requests": {},
            "tolerations": [],
            "jax_platform": "cpu",
        }

    if isinstance(parsed, TpuConfig):
        return {
            "node_selector": {
                "cloud.google.com/gke-tpu-accelerator": parsed.gke_accelerator,
                "cloud.google.com/gke-tpu-topology": parsed.topology,
            },
            "resource_limits": {"google.com/tpu": str(parsed.chips)},
            "resource_requests": {"google.com/tpu": str(parsed.chips)},
            "tolerations": [
                {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}
            ],
            "jax_platform": "tpu",
        }

    # GpuConfig
    return {
        "node_selector": {"cloud.google.com/gke-accelerator": parsed.gke_label},
        "resource_limits": {"nvidia.com/gpu": str(parsed.count)},
        "resource_requests": {"nvidia.com/gpu": str(parsed.count)},
        "tolerations": [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ],
        "jax_platform": "gpu",
    }


def _load_kube_config():
    """Load Kubernetes configuration.

    Attempts to load config in order:
    1. In-cluster config (if running inside K8s)
    2. Kubeconfig from KUBECONFIG env or ~/.kube/config

    Raises:
        RuntimeError: If unable to load any configuration
    """
    try:
        # Try in-cluster config first (for running inside K8s)
        config.load_incluster_config()
        return
    except config.ConfigException:
        pass

    try:
        # Fall back to kubeconfig
        config.load_kube_config()
        return
    except config.ConfigException as e:
        raise RuntimeError(
            f"Failed to load Kubernetes configuration. "
            f"Ensure you have run 'gcloud container clusters get-credentials <cluster-name>' "
            f"or have a valid kubeconfig. Error: {e}"
        )


def _create_job_spec(
    job_name, container_uri, accel_config, job_id, bucket_name, namespace
):
    """Create Kubernetes Job specification.

    Args:
        job_name: Name for the K8s Job
        container_uri: Docker image URI
        accel_config: Accelerator configuration from _parse_accelerator_for_gke
        job_id: Unique job identifier
        bucket_name: GCS bucket for artifacts
        namespace: Kubernetes namespace

    Returns:
        V1Job object ready for creation
    """
    # Environment variables for remote_runner.py
    env_vars = [
        client.V1EnvVar(name="KERAS_BACKEND", value="jax"),
        client.V1EnvVar(
            name="JAX_PLATFORMS", value=accel_config.get("jax_platform", "gpu")
        ),
        client.V1EnvVar(name="JOB_ID", value=job_id),
        client.V1EnvVar(name="GCS_BUCKET", value=bucket_name),
    ]

    # Container specification
    container = client.V1Container(
        name="keras-remote-worker",
        image=container_uri,
        command=["python3", "-u", "/app/remote_runner.py"],
        args=[
            f"gs://{bucket_name}/{job_id}/context.zip",
            f"gs://{bucket_name}/{job_id}/payload.pkl",
            f"gs://{bucket_name}/{job_id}/result.pkl",
        ],
        env=env_vars,
        resources=client.V1ResourceRequirements(
            limits=accel_config["resource_limits"],
            requests=accel_config["resource_requests"],
        ),
    )

    # Build tolerations
    tolerations = [
        client.V1Toleration(
            key=t["key"],
            operator=t["operator"],
            effect=t["effect"],
        )
        for t in accel_config["tolerations"]
    ]

    # Pod template specification
    pod_spec_kwargs = {
        "containers": [container],
        "tolerations": tolerations if tolerations else None,
        "restart_policy": "Never",
    }
    # Only set node_selector if non-empty (for GPU nodes)
    if accel_config.get("node_selector"):
        pod_spec_kwargs["node_selector"] = accel_config["node_selector"]

    pod_template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "keras-remote", "job-id": job_id}),
        spec=client.V1PodSpec(**pod_spec_kwargs),
    )

    # Job specification
    job_spec = client.V1JobSpec(
        template=pod_template,
        backoff_limit=0,  # No retries - fail immediately
        ttl_seconds_after_finished=600,  # Auto-cleanup after 10 minutes
    )

    # Complete Job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=job_name,
            namespace=namespace,
            labels={"app": "keras-remote", "job-id": job_id},
        ),
        spec=job_spec,
    )

    return job


def _print_pod_logs(core_v1, job_name, namespace):
    """Print pod logs for debugging failed jobs."""
    with suppress(ApiException):
        pods = core_v1.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}"
        )

        for pod in pods.items:
            with suppress(ApiException):
                logs = core_v1.read_namespaced_pod_log(
                    pod.metadata.name, namespace, tail_lines=100
                )
                print(f"[REMOTE] Pod {pod.metadata.name} logs:\n{logs}")


def _check_pod_scheduling(core_v1, job_name, namespace):
    """Check for pod scheduling issues and raise helpful errors."""
    with suppress(ApiException):
        pods = core_v1.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}"
        )
        for pod in pods.items:
            if pod.status.phase == "Pending":
                for condition in pod.status.conditions or []:
                    if condition.type == "PodScheduled" and condition.status == "False":
                        msg = condition.message or ""
                        if "Insufficient nvidia.com/gpu" in msg:
                            raise RuntimeError(
                                "No GPU nodes available. Ensure your GKE cluster has a "
                                "node pool with the required GPU type and available capacity."
                            )
                        elif (
                            "didn't match Pod's node affinity/selector" in msg
                            or "node selector" in msg.lower()
                        ):
                            raise RuntimeError(
                                "No nodes match the GPU selector. Check that your node pool "
                                "has the correct GPU type label."
                            )
