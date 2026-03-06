"""keras-remote status command — show current infrastructure state."""

import click

from keras_remote.cli.infra.state import load_state
from keras_remote.cli.output import (
  banner,
  console,
  infrastructure_state,
  warning,
)


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
  help="GCP zone [env: KERAS_REMOTE_ZONE]",
)
@click.option(
  "--cluster",
  "cluster_name",
  envvar="KERAS_REMOTE_CLUSTER",
  default=None,
  help="GKE cluster name [default: keras-remote-cluster]",
)
def status(project, zone, cluster_name):
  """Show current keras-remote infrastructure state."""
  banner("keras-remote Status")

  state = load_state(project, zone, cluster_name, allow_missing=True)

  if state.stack is None:
    warning("No Pulumi stack found.")
    console.print("Run 'keras-remote up' to provision infrastructure.")
    return

  outputs = state.stack.outputs()

  if not outputs:
    warning("No infrastructure found. Run 'keras-remote up' first.")
    return

  infrastructure_state(outputs)
