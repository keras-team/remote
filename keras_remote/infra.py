import subprocess
import sys
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("keras_remote")

def get_default_zone():
  return os.environ.get("KERAS_REMOTE_ZONE", "us-central1-a")

def get_default_project():
  return os.environ.get("KERAS_REMOTE_PROJECT")

def run_cmd(cmd, stream=False):
  """Runs a shell command using subprocess.Popen, optionally streaming stdout."""
  logger.info(f"Running command: {cmd}")
  process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  
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
    logger.error(f"Error running command: {cmd}")
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

  project_flag = f"--project={project}" if project else ""

  try:
    list_cmd = f"gcloud compute tpus tpu-vm list --zone={zone} {project_flag} --format=json"
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
  create_cmd = (
      "gcloud compute tpus tpu-vm create"
      f" {name} --zone={zone} --accelerator-type={accelerator_type} --version=tpu-vm-base {project_flag}"
  )
  run_cmd(create_cmd, stream=True)
  logger.info(f"TPU VM {name} created.")

def scp_to_vm(name, local, remote, zone=None, project=None):
  """Copies a local file to the remote VM."""
  if zone is None:
    zone = get_default_zone()
  if project is None:
    project = get_default_project()
  
  project_flag = f"--project={project}" if project else ""

  scp_cmd = (
      "gcloud compute tpus tpu-vm scp"
      f" {local} {name}:{remote} --zone={zone} --worker=all {project_flag}"
  )
  run_cmd(scp_cmd)

def ssh_execute(name, command, context_zip_path, use_requirements=False, zone=None, project=None):
  """Executes the remote script inside a Docker container on the VM."""
  if zone is None:
    zone = get_default_zone()
  if project is None:
    project = get_default_project()

  project_flag = f"--project={project}" if project else ""

  docker_image = "python:3.13-slim"
  
  # Commands to run inside the container
  container_cmds = [
      "python3 -m pip install --upgrade pip", # Keep pip upgraded
      # Install from requirements.txt only if key imports fail
      "python3 -c 'import keras, numpy, cloudpickle' || python3 -m pip install -r /tmp/requirements.txt",
      # Install JAX/TPU only if jax import fails
      "python3 -c 'import jax; import jax.experimental.libtpu' || python3 -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
      f"python3 -u {command} {context_zip_path}"
  ]
  # Escape single quotes within each command for the bash -c '...' context
  escaped_cmds = [cmd.replace("'", "'\\''") for cmd in container_cmds]
  container_command = " && ".join(escaped_cmds)
  
  # Docker run command to be executed on the VM
  docker_run_cmd = (
      f"sudo docker run --rm "
      f"-v /tmp:/tmp "
      f"-e KERAS_BACKEND=jax "  # Set environment variable
      # Expose TPU devices to the container
      f"--device /dev/accel0:/dev/accel0 "
      f"--device /dev/accel1:/dev/accel1 "
      f"--device /dev/accel2:/dev/accel2 "
      f"--device /dev/accel3:/dev/accel3 "
      f"--privileged " # Often needed for TPU access
      f"{docker_image} "
      f"bash -c '{container_command}'"
  )
  
  ssh_cmd = (
      f"gcloud compute tpus tpu-vm ssh {name} --zone={zone} --worker=all {project_flag}"
      f" --command=\"{docker_run_cmd}\""
  )
  
  logger.info(f"Running script inside Docker container on {name}")
  run_cmd(ssh_cmd, stream=True)
