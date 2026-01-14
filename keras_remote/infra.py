import subprocess
import sys
import json
import os
import logging
import shlex


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("keras_remote")


def get_default_zone():
  return os.environ.get("KERAS_REMOTE_ZONE", "us-central1-a")


def get_default_project():
  return os.environ.get("KERAS_REMOTE_PROJECT")


def run_cmd(cmd, stream=False):
  """Runs a shell command using subprocess.Popen, optionally streaming stdout."""
  logger.info(f"Running command: {' '.join(cmd)}")
  process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

  if stream:
    # Read stdout line by line
    for line in iter(process.stdout.readline, ''):
      if line.startswith('[REMOTE]'):
        sys.stdout.write(line)
        sys.stdout.flush()
      else:
        logger.info(line.strip())

    # Read stderr after stdout is closed
    stderr_lines = process.stderr.read()
    if stderr_lines:
      logger.error(f"STDERR: {stderr_lines}")

  stdout, stderr = process.communicate()

  if process.returncode != 0:
    logger.error(f"Error running command: {' '.join(cmd)}")
    if not stream:
      logger.error(f"STDOUT: {stdout}")
      logger.error(f"STDERR: {stderr}")
    raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)

  return stdout


def ensure_tpu_vm(name, accelerator_type, zone=None, project=None):
  """Ensures a TPU VM exists, creating it if necessary."""
  if zone is None:
    zone = get_default_zone()
  if project is None:
    project = get_default_project()

  try:
    list_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "list", f"--zone={zone}", "--format=json"]
    if project:
      list_cmd.append(f"--project={project}")
    
    output = run_cmd(list_cmd)
    vms = json.loads(output)
    if any(vm['name'].endswith(name) for vm in vms):
      logger.info(f"TPU VM {name} already exists.")
      return
  except subprocess.CalledProcessError:
    logger.info(f"Failed to list TPU VMs, assuming {name} does not exist.")
  except json.JSONDecodeError:
    logger.info(f"Failed to parse TPU VM list output, assuming {name} does not exist.")

  logger.info(f"Creating TPU VM {name}...")
  create_cmd = [
      "gcloud", "compute", "tpus", "tpu-vm", "create", name,
      f"--zone={zone}",
      f"--accelerator-type={accelerator_type}",
      "--version=tpu-vm-base"
  ]
  if project:
      create_cmd.append(f"--project={project}")

  run_cmd(create_cmd, stream=True)
  logger.info(f"TPU VM {name} created.")


def scp_to_vm(name, local, remote, zone=None, project=None):
  """Copies a local file to the remote VM."""
  if zone is None:
    zone = get_default_zone()
  if project is None:
    project = get_default_project()

  scp_cmd = [
      "gcloud", "compute", "tpus", "tpu-vm", "scp",
      local, f"{name}:{remote}",
      f"--zone={zone}", "--worker=all"
  ]
  if project:
      scp_cmd.append(f"--project={project}")

  run_cmd(scp_cmd)


def get_device_count(accelerator_type):
    """Determines the number of TPU chips (accel devices) per worker."""
    # Heuristic: Most v2/v3/v4 TPU VMs have 4 chips (8 cores) per worker.
    # Exceptions like v5e (litepod) can vary.
    if "v5litepod-1" in accelerator_type:
        return 1
    if "v5litepod-4" in accelerator_type:
        return 4
    if "v5litepod-8" in accelerator_type:
        return 8
    # Default to 4 for v2-8, v3-8, v4-8 and their pod slices (per worker)
    return 4


def ssh_execute(
    name: str,
    python_main_file: str,
    context_zip_path: str,
    use_requirements: bool = False,
    zone: str | None = None,
    project: str | None = None,
    accelerator_type: str = "v3-8",
) -> None:
  """Executes the remote script inside a Docker container on the VM."""
  if zone is None:
    zone = get_default_zone()
  if project is None:
    project = get_default_project()

  docker_image = "python:3.13-slim"
  num_devices = get_device_count(accelerator_type)
  device_flags = " ".join([f"--device /dev/accel{i}:/dev/accel{i}" for i in range(num_devices)])
  container_cmds = [
      "python3 -m pip install --upgrade pip",
  ]
  if use_requirements:
      container_cmds.append("python3 -m pip install -r /tmp/requirements.txt")

  container_cmds.extend([
      "python3 -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
      f"python3 -u {python_main_file} {context_zip_path}",
  ])

  # Join commands and quote safely for bash -c
  container_command = " && ".join(container_cmds)
  safe_container_command = shlex.quote(container_command)

  # Docker run command to be executed on the VM
  docker_run_cmd = (
      f"sudo docker run --rm "
      f"-v /tmp:/tmp "
      # Set environment variable
      # TODO: this shouldn't be hard-coded here
      f"-e KERAS_BACKEND=jax "
      # Expose TPU devices to the container
      f"{device_flags} "
      # Often needed for TPU access
      f"--privileged "
      f"{docker_image} "
      f"bash -c {safe_container_command}"
  )

  ssh_cmd = [
      "gcloud", "compute", "tpus", "tpu-vm", "ssh", name,
      f"--zone={zone}", "--worker=all"
  ]
  if project:
      ssh_cmd.append(f"--project={project}")

  ssh_cmd.append(f"--command={docker_run_cmd}")

  logger.info(f"Running script inside Docker container on {name}")
  stdout = run_cmd(ssh_cmd, stream=True)
  # TODO: Parse stdout to extract and deserialize the function result.
  return None
