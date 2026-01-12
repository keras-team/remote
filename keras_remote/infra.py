import subprocess
import sys
import json
import os

def get_default_zone():
  return os.environ.get("KERAS_REMOTE_ZONE", "us-central1-a")

def run_cmd(cmd, stream=False):
  """Runs a shell command using subprocess.Popen, optionally streaming stdout."""
  print(f"Running command: {cmd}", flush=True)
  process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  
  if stream:
    # Read stdout line by line
    for line in iter(process.stdout.readline, ''):
      if line.startswith('[REMOTE]'):
        sys.stdout.write(line)
        sys.stdout.flush()
      else:
        print(line, end='') # Print other lines normally
    
    # Read stderr after stdout is closed
    stderr_lines = process.stderr.read()
    if stderr_lines:
      print(f"STDERR: {stderr_lines}", flush=True)
  
  stdout, stderr = process.communicate()
  
  if process.returncode != 0:
    print(f"Error running command: {cmd}", flush=True)
    if not stream: # If not streaming, stdout/stderr weren't printed yet
      print(f"STDOUT: {stdout}", flush=True)
      print(f"STDERR: {stderr}", flush=True)
    raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
  
  return stdout

def ensure_tpu_vm(name, accelerator_type, zone=None):
  """Ensures a TPU VM exists, creating it if necessary."""
  if zone is None:
    zone = get_default_zone()

  try:
    list_cmd = f"gcloud compute tpus tpu-vm list --zone={zone} --format=json"
    output = run_cmd(list_cmd)
    vms = json.loads(output)
    if any(vm['name'].endswith(name) for vm in vms):
      print(f"TPU VM {name} already exists.", flush=True)
      return
  except subprocess.CalledProcessError:
    print(f"Failed to list TPU VMs, assuming {name} does not exist.", flush=True)
  except json.JSONDecodeError:
    print(f"Failed to parse TPU VM list output, assuming {name} does not exist.", flush=True)

  print(f"Creating TPU VM {name}...", flush=True)
  create_cmd = (
      "gcloud compute tpus tpu-vm create"
      f" {name} --zone={zone} --accelerator-type={accelerator_type} --version=tpu-vm-base"
  )
  run_cmd(create_cmd, stream=True)
  print(f"TPU VM {name} created.", flush=True)

def scp_to_vm(name, local, remote, zone=None):
  """Copies a local file to the remote VM."""
  if zone is None:
    zone = get_default_zone()

  scp_cmd = (
      "gcloud compute tpus tpu-vm scp"
      f" {local} {name}:{remote} --zone={zone} --worker=all"
  )
  run_cmd(scp_cmd)

def ssh_execute(name, command, context_zip_path, use_requirements=False, zone=None):
  """Executes the remote script inside a Docker container on the VM."""
  if zone is None:
    zone = get_default_zone()

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
      f"gcloud compute tpus tpu-vm ssh {name} --zone={zone} --worker=all"
      f" --command=\"{docker_run_cmd}\""
  )
  
  print(f"Running script inside Docker container on {name}", flush=True)
  run_cmd(ssh_cmd, stream=True)
