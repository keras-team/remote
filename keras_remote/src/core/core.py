import datetime
import functools
import getpass
import inspect
import os
import shutil
import tempfile

from keras_remote.src.utils import packager
from keras_remote.src.infra import infra
from keras_remote.cli.constants import RESOURCE_NAME_PREFIX
from keras_remote.src.backend.execution import (
    JobContext,
    execute_remote,
    GKEBackend,
)
from keras_remote.src.api_export import keras_remote_export

logger = infra.logger

@keras_remote_export("keras_remote.run")
def run(accelerator="v3-8",
        container_image=None,
        zone=None,
        project=None,
        vm_name=None,
        capture_env_vars=None,
        backend="gke",
        cluster=None,
        namespace="default",
        ):
  """Execute function on remote TPU/GPU.

  Args:
    accelerator: TPU/GPU type (e.g., 'v3-8', 'v5litepod-4', 'l4', 'a100')
    container_image: Custom container image URI (optional)
    zone: GCP zone (default: from KERAS_REMOTE_ZONE or 'us-central1-a')
    project: GCP project (default: from KERAS_REMOTE_PROJECT)
    vm_name: Custom VM name (only for 'tpu-vm' backend)
    capture_env_vars: List of environment variable names or patterns (ending in *)
      to propagate to the remote environment. Defaults to None.
    backend: 'gke' (Kubernetes, recommended) or 'tpu-vm' (direct VM)
    cluster: GKE cluster name (only for 'gke' backend, default: from KERAS_REMOTE_GKE_CLUSTER)
    namespace: Kubernetes namespace (only for 'gke' backend, default: 'default')
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
                  env_vars.update({k: v for k, v in os.environ.items() if k.startswith(prefix)})
              elif pattern in os.environ:
                  env_vars[pattern] = os.environ[pattern]

      if backend == "gke":
        return _execute_on_gke(func, args, kwargs, accelerator, container_image, zone, project, cluster, namespace, env_vars)
      elif backend == "tpu-vm":
        return _execute_on_tpu_vm(func, args, kwargs, accelerator, container_image, zone, project, vm_name, env_vars)
      else:
        raise ValueError(f"Unknown backend: {backend}. Use 'gke' or 'tpu-vm'.")
    return wrapper
  return decorator


def _execute_on_tpu_vm(func, args, kwargs, accelerator, container_image, zone, project, vm_name, env_vars):
  """Implementation using direct TPU VM management."""
  with tempfile.TemporaryDirectory() as tmpdir:

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    context_zip_name = f'context_{timestamp}.zip'
    remote_context_zip_path = f'/tmp/{context_zip_name}'

    context_zip = os.path.join(tmpdir, context_zip_name)
    payload_pkl = os.path.join(tmpdir, "payload.pkl")
    remote_runner_py = os.path.join(tmpdir, "remote_runner.py")

    # 1. Create zip/pickle artifacts
    frame = inspect.stack()[2]  # Adjusted for wrapper depth
    module = inspect.getmodule(frame[0])
    if module:
      caller_path = os.path.dirname(os.path.abspath(module.__file__))
    else:
      caller_path = os.getcwd() # Fallback

    logger.info(f"Packaging directory: {caller_path}...")
    packager.zip_working_dir(caller_path, context_zip)
    logger.info(f"Context zip created at {context_zip}")

    logger.info(f"Capturing {len(env_vars)} environment variables...")
    logger.info("Serializing payload...")
    packager.save_payload(func, args, kwargs, env_vars, payload_pkl)
    logger.info(f"Payload pickle created at {payload_pkl}")

    # Copy remote_runner.py to tmpdir
    runner_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "runner")
    shutil.copy(
        os.path.join(runner_dir, "remote_runner.py"), remote_runner_py
    )

    # 2. Ensure TPU VM exists
    if vm_name:
      actual_vm_name = vm_name
    else:
      user = getpass.getuser()
      actual_vm_name = f"{RESOURCE_NAME_PREFIX}-{user}-{accelerator}"
    infra.ensure_tpu_vm(actual_vm_name, accelerator, container_image=container_image, zone=zone, project=project)

    # 3. Upload artifacts
    logger.info(f"Uploading files to {actual_vm_name}...")
    infra.scp_to_vm(actual_vm_name, context_zip, remote_context_zip_path, zone=zone, project=project)
    infra.scp_to_vm(actual_vm_name, payload_pkl, "/tmp/payload.pkl", zone=zone, project=project)
    infra.scp_to_vm(actual_vm_name, remote_runner_py, "/tmp/remote_runner.py", zone=zone, project=project)

    # Find and upload requirements.txt
    requirements_txt = None
    search_dir = caller_path
    while search_dir != "/":
        req_path = os.path.join(search_dir, "requirements.txt")
        if os.path.exists(req_path):
            requirements_txt = req_path
            break
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:  # Avoid infinite loop at root
            break
        search_dir = parent_dir

    if requirements_txt:
        logger.info(f"Using requirements.txt: {requirements_txt}")
        infra.scp_to_vm(actual_vm_name, requirements_txt, "/tmp/requirements.txt", zone=zone, project=project)
        use_requirements = True
    else:
        logger.info("No requirements.txt found.")
        use_requirements = False
    logger.info("Upload complete.")

    # 4. Execute remote_runner.py on the VM
    logger.info("Executing remote script...")
    result = infra.ssh_execute(actual_vm_name, "/tmp/remote_runner.py", context_zip_path=remote_context_zip_path, use_requirements=use_requirements, zone=zone, project=project, accelerator_type=accelerator)
    logger.info("Remote execution finished.")
    return result


def _execute_on_gke(func, args, kwargs, accelerator, container_image, zone, project, cluster, namespace, env_vars):
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
