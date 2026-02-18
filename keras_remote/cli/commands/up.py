"""keras-remote up command — provision infrastructure."""

import click

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from keras_remote.cli.infra.post_deploy import (
  configure_docker_auth,
  configure_kubectl,
  install_gpu_drivers,
)
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import get_stack
from keras_remote.cli.output import banner, config_summary, console, success
from keras_remote.cli.prerequisites_check import check_all
from keras_remote.cli.prompts import prompt_accelerator, resolve_project
from keras_remote.constants import zone_to_ar_location
from keras_remote.core import accelerators
from keras_remote.core.accelerators import GpuConfig


@click.command()
@click.option(
  "--project",
  envvar="KERAS_REMOTE_PROJECT",
  default=None,
  help="GCP project ID [env: KERAS_REMOTE_PROJECT]",
)
@click.option(
  "--zone",
  envvar="KERAS_REMOTE_ZONE",
  default=None,
  help=(f"GCP zone [env: KERAS_REMOTE_ZONE, default: {DEFAULT_ZONE}]"),
)
@click.option(
  "--accelerator",
  default=None,
  help="Accelerator spec: cpu, t4, l4, a100, a100-80gb, h100, "
  "v5litepod, v5p, v6e, v3",
)
@click.option(
  "--cluster-name",
  envvar="KERAS_REMOTE_CLUSTER",
  default=None,
  help="GKE cluster name [default: keras-remote-cluster]",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def up(project, zone, accelerator, cluster_name, yes):
  """Provision GCP infrastructure for keras-remote."""
  banner("keras-remote Setup")

  # Check prerequisites
  check_all()

  # Resolve configuration
  project = project or resolve_project()
  zone = zone or DEFAULT_ZONE
  cluster_name = cluster_name or DEFAULT_CLUSTER_NAME

  # Resolve accelerator (interactive if not provided)
  if accelerator and accelerator.strip().lower() == "cpu":
    accel_config = None
  elif accelerator:
    try:
      accel_config = accelerators.parse_accelerator(accelerator)
    except ValueError as e:
      raise click.BadParameter(str(e), param_hint="--accelerator") from e
  else:
    accel_config = prompt_accelerator()

  config = InfraConfig(
    project=project,
    zone=zone,
    cluster_name=cluster_name,
    accelerator=accel_config,
  )

  # Show summary and confirm
  config_summary(config)
  if not yes:
    click.confirm("\nProceed with setup?", abort=True)

  console.print()

  # Run Pulumi
  program = create_program(config)
  stack = get_stack(program, config)
  console.print("[bold]Provisioning infrastructure...[/bold]\n")
  result = stack.up(on_output=print)
  console.print()
  success(f"Pulumi update complete. {result.summary.resource_changes}")

  # Post-deploy steps
  ar_location = zone_to_ar_location(zone)
  console.print("\n[bold]Running post-deploy configuration...[/bold]\n")

  console.print("Configuring Docker authentication...")
  configure_docker_auth(ar_location)
  success("Docker authentication configured")

  console.print("Configuring kubectl access...")
  configure_kubectl(cluster_name, zone, project)
  success("kubectl configured")

  if isinstance(accel_config, GpuConfig):
    console.print("Installing NVIDIA GPU device drivers...")
    install_gpu_drivers()
    success("GPU driver installation initiated")

  # Final summary
  console.print()
  banner("Setup Complete")
  console.print()
  console.print("Add these environment variables to your shell config:")
  console.print(f"  export KERAS_REMOTE_PROJECT={project}")
  console.print(f"  export KERAS_REMOTE_ZONE={zone}")
  console.print(f"  export KERAS_REMOTE_CLUSTER={cluster_name}")
  console.print()
  console.print("View quotas:")
  console.print(
    f"  https://console.cloud.google.com/iam-admin/quotas?project={project}"
  )
  console.print()
