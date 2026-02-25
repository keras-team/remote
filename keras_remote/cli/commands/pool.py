"""keras-remote pool commands — add, remove, and list accelerator node pools."""

import click
import pulumi.automation as auto

from keras_remote.cli.config import InfraConfig, NodePoolConfig
from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import (
  get_current_node_pools,
  get_stack,
)
from keras_remote.cli.output import (
  banner,
  console,
  infrastructure_state,
  success,
  warning,
)
from keras_remote.cli.prerequisites_check import check_all
from keras_remote.cli.prompts import resolve_project
from keras_remote.core import accelerators
from keras_remote.core.accelerators import generate_pool_name


def _common_options(f):
  """Shared options for pool subcommands."""
  f = click.option(
    "--project",
    envvar="KERAS_REMOTE_PROJECT",
    default=None,
    help="GCP project ID [env: KERAS_REMOTE_PROJECT]",
  )(f)
  f = click.option(
    "--zone",
    envvar="KERAS_REMOTE_ZONE",
    default=None,
    help=f"GCP zone [env: KERAS_REMOTE_ZONE, default: {DEFAULT_ZONE}]",
  )(f)
  f = click.option(
    "--cluster",
    "cluster_name",
    envvar="KERAS_REMOTE_CLUSTER",
    default=None,
    help="GKE cluster name [default: keras-remote-cluster]",
  )(f)
  return f


def _resolve_common(project, zone, cluster_name):
  """Resolve common options to concrete values."""
  return (
    project or resolve_project(),
    zone or DEFAULT_ZONE,
    cluster_name or DEFAULT_CLUSTER_NAME,
  )


@click.group()
def pool():
  """Manage accelerator node pools."""


@pool.command("add")
@_common_options
@click.option(
  "--accelerator",
  required=True,
  help="Accelerator spec: t4, l4, a100, a100-80gb, h100, "
  "v5litepod, v5p, v6e, v3 (with optional count/topology)",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def pool_add(project, zone, cluster_name, accelerator, yes):
  """Add an accelerator node pool to the cluster."""
  banner("keras-remote Pool Add")

  check_all()
  project, zone, cluster_name = _resolve_common(project, zone, cluster_name)

  # Parse the accelerator spec.
  try:
    accel_config = accelerators.parse_accelerator(accelerator)
  except ValueError as e:
    raise click.BadParameter(str(e), param_hint="--accelerator") from e

  if accel_config is None:
    raise click.BadParameter(
      "Cannot add a CPU node pool. Use 'keras-remote up' instead.",
      param_hint="--accelerator",
    )

  new_pool_name = generate_pool_name(accel_config)
  new_pool = NodePoolConfig(new_pool_name, accel_config)

  # Load existing state.
  base_config = InfraConfig(
    project=project, zone=zone, cluster_name=cluster_name
  )
  program = create_program(base_config)
  stack = get_stack(program, base_config)

  console.print("\nRefreshing state...\n")
  try:
    stack.refresh(on_output=print)
  except auto.errors.CommandError as e:
    warning(f"Failed to refresh stack state: {e}")

  existing_pools = get_current_node_pools(stack)

  # Build the combined pool list.
  all_pools = existing_pools + [new_pool]

  console.print(f"\nAdding pool [bold]{new_pool_name}[/bold] ({accelerator})")
  console.print(f"Total pools after add: {len(all_pools)}\n")

  if not yes:
    click.confirm("Proceed?", abort=True)

  # Run Pulumi with the updated pool list.
  config = InfraConfig(
    project=project,
    zone=zone,
    cluster_name=cluster_name,
    node_pools=all_pools,
  )
  program = create_program(config)
  stack = get_stack(program, config)

  console.print("\n[bold]Provisioning...[/bold]\n")
  try:
    result = stack.up(on_output=print)
    console.print()
    success(f"Pulumi update complete. {result.summary.resource_changes}")
  except auto.errors.CommandError as e:
    console.print()
    warning(f"Pulumi update encountered an issue: {e}")

  console.print()
  banner("Pool Added")
  console.print()


@pool.command("remove")
@_common_options
@click.argument("pool_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def pool_remove(project, zone, cluster_name, pool_name, yes):
  """Remove an accelerator node pool from the cluster."""
  banner("keras-remote Pool Remove")

  check_all()
  project, zone, cluster_name = _resolve_common(project, zone, cluster_name)

  # Load existing state.
  base_config = InfraConfig(
    project=project, zone=zone, cluster_name=cluster_name
  )
  program = create_program(base_config)
  stack = get_stack(program, base_config)

  console.print("\nRefreshing state...\n")
  try:
    stack.refresh(on_output=print)
  except auto.errors.CommandError as e:
    warning(f"Failed to refresh stack state: {e}")

  existing_pools = get_current_node_pools(stack)

  # Find the pool to remove.
  remaining = [p for p in existing_pools if p.name != pool_name]
  if len(remaining) == len(existing_pools):
    existing_names = [p.name for p in existing_pools]
    raise click.ClickException(
      f"Node pool '{pool_name}' not found. "
      f"Existing pools: {', '.join(existing_names) or '(none)'}"
    )

  console.print(f"\nRemoving pool [bold]{pool_name}[/bold]")
  console.print(f"Remaining pools after remove: {len(remaining)}\n")

  if not yes:
    click.confirm("Proceed?", abort=True)

  # Run Pulumi with the reduced pool list.
  config = InfraConfig(
    project=project,
    zone=zone,
    cluster_name=cluster_name,
    node_pools=remaining,
  )
  program = create_program(config)
  stack = get_stack(program, config)

  console.print("\n[bold]Updating infrastructure...[/bold]\n")
  try:
    result = stack.up(on_output=print)
    console.print()
    success(f"Pulumi update complete. {result.summary.resource_changes}")
  except auto.errors.CommandError as e:
    console.print()
    warning(f"Pulumi update encountered an issue: {e}")

  console.print()
  banner("Pool Removed")
  console.print()


@pool.command("list")
@_common_options
def pool_list(project, zone, cluster_name):
  """List accelerator node pools on the cluster."""
  banner("keras-remote Node Pools")

  check_all()
  project, zone, cluster_name = _resolve_common(project, zone, cluster_name)

  base_config = InfraConfig(
    project=project, zone=zone, cluster_name=cluster_name
  )

  try:
    program = create_program(base_config)
    stack = get_stack(program, base_config)
  except auto.errors.CommandError as e:
    warning(f"No Pulumi stack found for project '{project}': {e}")
    console.print("Run 'keras-remote up' to provision infrastructure.")
    return

  console.print("\nRefreshing state...\n")
  try:
    stack.refresh(on_output=print)
  except auto.errors.CommandError as e:
    warning(f"Failed to refresh stack state: {e}")

  outputs = stack.outputs()
  if not outputs:
    warning("No infrastructure found. Run 'keras-remote up' first.")
    return

  infrastructure_state(outputs)
