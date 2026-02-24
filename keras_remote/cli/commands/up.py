"""keras-remote up command — provision infrastructure."""

import subprocess

import click
import pulumi.automation as auto

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from keras_remote.cli.infra.post_deploy import (
  configure_docker_auth,
  configure_kubectl,
  install_gpu_drivers,
  install_lws,
)
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import get_stack
from keras_remote.cli.output import (
  banner,
  config_summary,
  console,
  success,
  warning,
)
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

  pulumi_failed = False
  try:
    result = stack.up(on_output=print)
    console.print()
    success(f"Pulumi update complete. {result.summary.resource_changes}")
  except auto.errors.CommandError as e:
    console.print()
    pulumi_failed = True
    warning(
      "Pulumi update encountered an issue"
      " (some resources may already exist):\n"
      f"  {e}\n"
      "Attempting post-deploy configuration anyway..."
    )

  # Post-deploy steps
  ar_location = zone_to_ar_location(zone)
  console.print("\n[bold]Running post-deploy configuration...[/bold]\n")

  steps = [
    ("Docker authentication", lambda: configure_docker_auth(ar_location)),
    (
      "kubectl configuration",
      lambda: configure_kubectl(
        cluster_name,
        zone,
        project,
      ),
    ),
    ("LWS CRD installation", install_lws),
  ]
  if isinstance(accel_config, GpuConfig):
    steps.append(("GPU driver installation", install_gpu_drivers))

  failures = []
  for name, fn in steps:
    console.print(f"{name}...")
    try:
      fn()
      success(f"{name} complete.")
    except subprocess.CalledProcessError as e:
      failures.append(name)
      warning(f"{name} failed: {e}")

  # Final summary
  console.print()
  if pulumi_failed or failures:
    banner("Setup Completed With Warnings")
    console.print()
    if pulumi_failed:
      warning("Pulumi provisioning encountered errors (see above).")
    if failures:
      warning(f"Post-deploy steps failed: {', '.join(failures)}")
    console.print()
    console.print(
      "You may re-run [bold]keras-remote up[/bold] to retry failed steps."
    )
  else:
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
