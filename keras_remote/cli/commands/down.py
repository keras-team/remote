"""keras-remote down command — tear down infrastructure."""

import click
import pulumi.automation as auto

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.constants import DEFAULT_ZONE
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import get_stack
from keras_remote.cli.output import banner, console, success, warning
from keras_remote.cli.prerequisites_check import check_all
from keras_remote.cli.prompts import resolve_project


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
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def down(project, zone, yes):
  """Tear down keras-remote GCP infrastructure."""
  banner("keras-remote Cleanup")

  check_all()

  project = project or resolve_project(allow_create=False)
  zone = zone or DEFAULT_ZONE

  # Warning
  console.print()
  warning(f"This will delete ALL keras-remote resources in project: {project}")
  console.print()
  console.print("This includes:")
  console.print("  - GKE cluster and node pools")
  console.print("  - Artifact Registry repository and images")
  console.print("  - Cloud Storage buckets (jobs and builds)")
  console.print("  - Enabled API services (left enabled)")
  console.print()

  if not yes:
    click.confirm("Are you sure you want to continue?", abort=True)

  console.print()

  config = InfraConfig(project=project, zone=zone)

  # Pulumi destroy
  try:
    # Minimal config to load the stack — accelerator is not
    # needed for destroy since the stack already has its state.
    program = create_program(config)
    stack = get_stack(program, config)
    console.print("[bold]Destroying Pulumi-managed resources...[/bold]\n")
    result = stack.destroy(on_output=print)
    console.print()
    success(f"Pulumi destroy complete. {result.summary.resource_changes}")
  except auto.errors.CommandError as e:
    warning(f"Pulumi destroy encountered an issue: {e}")

  # Summary
  console.print()
  banner("Cleanup Complete")
  console.print()
  console.print("Check manually for remaining resources:")
  console.print(
    f"  GKE: https://console.cloud.google.com/kubernetes/list?project={project}"
  )
  console.print(
    f"  Billing: https://console.cloud.google.com/billing?project={project}"
  )
  console.print()
