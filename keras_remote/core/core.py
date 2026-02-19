import functools
import os

from keras_remote.backend.execution import (
  GKEBackend,
  PathwaysBackend,
  JobContext,
  execute_remote,
)


def run(
  accelerator="v3-8",
  container_image=None,
  zone=None,
  project=None,
  capture_env_vars=None,
  cluster=None,
  backend= None,
  namespace="default",
):
  """Execute function on remote TPU/GPU.

  Args:
    accelerator: TPU/GPU type (e.g., 'v3-8', 'v5litepod-4', 'l4', 'a100')
    container_image: Custom container image URI (optional)
    zone: GCP zone (default: from KERAS_REMOTE_ZONE or 'us-central1-a')
    project: GCP project (default: from KERAS_REMOTE_PROJECT)
    capture_env_vars: List of environment variable names or patterns (ending in *)
      to propagate to the remote environment. Defaults to None.
    cluster: GKE cluster name (default: from KERAS_REMOTE_GKE_CLUSTER)
    backend: Backend to use ('gke' or 'pathways')
    namespace: Kubernetes namespace (default: 'default')
  """

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # Capture environment variables
      env_vars = {}
      if capture_env_vars:
        for pattern in capture_env_vars:
          if pattern.endswith("*"):
            prefix = pattern[:-1]
            env_vars.update(
              {k: v for k, v in os.environ.items() if k.startswith(prefix)}
            )
          elif pattern in os.environ:
            env_vars[pattern] = os.environ[pattern]

      return _execute_on_gke(
        func,
        args,
        kwargs,
        accelerator,
        container_image,
        zone,
        project,
        cluster,
        namespace,
        env_vars,
      )
    if backend is None:
          try:
              accel_config = accelerators.parse_accelerator(accelerator)
              if isinstance(accel_config, accelerators.TpuConfig) and accel_config.num_nodes > 1:
                  resolved_backend = "pathways"
              else:
                  resolved_backend = "gke"
          except ValueError:
              resolved_backend = "gke"

    if resolved_backend == "gke":
      return _execute_on_gke(func, args, kwargs, accelerator, container_image, zone, project, cluster, namespace, env_vars)
    elif backend == "pathways":
      return _execute_on_pathways(func, args, kwargs, accelerator, container_image, zone, project, cluster, namespace, env_vars)
    else:
      raise ValueError(f"Unknown backend: {resolved_backend}. Use 'gke', 'pathways', or None for auto-detection")
    return wrapper

  return decorator


def _execute_on_gke(
  func,
  args,
  kwargs,
  accelerator,
  container_image,
  zone,
  project,
  cluster,
  namespace,
  env_vars,
):
  """Execute function on GKE cluster with GPU nodes."""
  # Get GKE-specific defaults
  if not cluster:
    cluster = os.environ.get("KERAS_REMOTE_GKE_CLUSTER")
  if not namespace:
    namespace = os.environ.get("KERAS_REMOTE_GKE_NAMESPACE", "default")

  ctx = JobContext.from_params(
    func, args, kwargs, accelerator, container_image, zone, project, env_vars
  )
  return execute_remote(ctx, GKEBackend(cluster=cluster, namespace=namespace))



def _execute_on_pathways(func, args, kwargs, accelerator, container_image, zone, project, cluster, namespace, env_vars):
    """Execute function on GKE cluster via ML Pathways."""
    if not cluster:
        cluster = os.environ.get("KERAS_REMOTE_GKE_CLUSTER")
    if not namespace:
        namespace = os.environ.get("KERAS_REMOTE_GKE_NAMESPACE", "default")

    ctx = JobContext.from_params(
        func, args, kwargs, accelerator, container_image, zone, project, env_vars
    )
    return execute_remote(ctx, PathwaysBackend(cluster=cluster, namespace=namespace))