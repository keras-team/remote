"""keras-remote status command — show current infrastructure state."""

import click
from pulumi.automation import CommandError

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import get_stack
from keras_remote.cli.output import (
  banner,
  console,
  infrastructure_state,
  warning,
)
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
@click.option(
  "--cluster-name",
  envvar="KERAS_REMOTE_CLUSTER",
  default=None,
  help="GKE cluster name [default: keras-remote-cluster]",
)
def status(project, zone, cluster_name):
  """Show current keras-remote infrastructure state."""
  banner("keras-remote Status")

  check_all()

  project = project or resolve_project()
  zone = zone or DEFAULT_ZONE
  cluster_name = cluster_name or DEFAULT_CLUSTER_NAME

  config = InfraConfig(project=project, zone=zone, cluster_name=cluster_name)

  try:
    program = create_program(config)
    stack = get_stack(program, config)
  except CommandError as e:
    warning(f"No Pulumi stack found for project '{project}': {e}")
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
