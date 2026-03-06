"""keras-remote status command — show current infrastructure state."""

import click
from pulumi.automation import CommandError

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import (
  get_stack,
  require_active_stack,
)
from keras_remote.cli.output import (
  banner,
  console,
  infrastructure_state,
  show_target_stack,
  warning,
)
from keras_remote.cli.prerequisites_check import check_all


@click.command()
def status():
  """Show current keras-remote infrastructure state."""
  banner("keras-remote Status")

  check_all()

  # Stack name always comes from persisted state (set by `up` or `stacks set`).
  active = require_active_stack()
  show_target_stack(active.project, active.cluster_name, "active stack")

  config = InfraConfig(
    project=active.project,
    zone=active.zone,
    cluster_name=active.cluster_name,
  )

  try:
    program = create_program(config)
    stack = get_stack(program, config, stack_name=active.stack_name)
  except CommandError as e:
    warning(f"No Pulumi stack found for project '{active.project}': {e}")
    console.print("Run 'keras-remote up' to provision infrastructure.")
    return

  console.print("\nRefreshing state...\n")
  try:
    stack.refresh(on_output=print)
  except CommandError as e:
    warning(f"Failed to refresh stack state: {e}")

  outputs = stack.outputs()

  if not outputs:
    warning("No infrastructure found. Run 'keras-remote up' first.")
    return

  infrastructure_state(outputs)
