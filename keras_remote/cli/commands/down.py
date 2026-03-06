"""keras-remote down command — tear down infrastructure."""

import click
import pulumi.automation as auto

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import (
  clear_active_stack,
  get_stack,
  remove_stack,
  require_active_stack,
)
from keras_remote.cli.output import (
  banner,
  console,
  show_target_stack,
  success,
  warning,
)
from keras_remote.cli.prerequisites_check import check_all


@click.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def down(yes):
  """Tear down keras-remote GCP infrastructure."""
  banner("keras-remote Cleanup")

  check_all()

  # Stack name always comes from persisted state (set by `up` or `stacks set`).
  active = require_active_stack()
  show_target_stack(
    active.project, active.cluster_name, "active stack", destructive=True
  )

  # Warning
  console.print()
  warning(
    f"This will delete ALL keras-remote resources in project: {active.project}"
  )
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

  config = InfraConfig(
    project=active.project,
    zone=active.zone,
    cluster_name=active.cluster_name,
  )

  # Pulumi destroy
  try:
    program = create_program(config)
    stack = get_stack(program, config, stack_name=active.stack_name)
    console.print("[bold]Destroying Pulumi-managed resources...[/bold]\n")
    result = stack.destroy(on_output=print)
    console.print()
    success(f"Pulumi destroy complete. {result.summary.resource_changes}")
  except auto.errors.CommandError as e:
    warning(f"Pulumi destroy encountered an issue: {e}")

  # Remove the stack from local state and clear the active pointer.
  try:
    remove_stack(active.stack_name)
  except auto.errors.CommandError:
    clear_active_stack()

  # Summary
  console.print()
  banner("Cleanup Complete")
  console.print()
  console.print("Check manually for remaining resources:")
  console.print(
    f"  GKE: https://console.cloud.google.com/kubernetes/list?project={active.project}"
  )
  console.print(
    f"  Billing: https://console.cloud.google.com/billing?project={active.project}"
  )
  console.print()
