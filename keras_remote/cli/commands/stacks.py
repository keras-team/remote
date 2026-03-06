"""keras-remote stacks commands — list and switch infrastructure stacks."""

import click
from pulumi.automation.errors import CommandError
from rich.table import Table

from keras_remote.cli.infra.stack_manager import (
  delete_stack,
  get_active_stack,
  list_stacks,
  resolve_stack_info,
  set_active_stack,
  stack_exists,
)
from keras_remote.cli.output import banner, console, success, warning


@click.group(invoke_without_command=True)
@click.pass_context
def stacks(ctx):
  """List and manage infrastructure stacks."""
  if ctx.invoked_subcommand is None:
    ctx.invoke(stacks_list)


@stacks.command("list")
def stacks_list():
  """List all keras-remote infrastructure stacks."""
  banner("keras-remote Stacks")

  summaries = list_stacks()

  if not summaries:
    warning("No stacks found. Run 'keras-remote up' to create one.")
    return

  active = get_active_stack()

  table = Table()
  table.add_column("Stack", style="bold", overflow="fold")
  table.add_column("Project", style="green", overflow="fold")
  table.add_column("Cluster", style="green", overflow="fold")
  table.add_column("Zone", style="dim", overflow="fold")
  table.add_column("Last Updated", style="dim", overflow="fold")

  for summary in summaries:
    marker = "* " if summary.name == active else ""
    info = resolve_stack_info(summary.name)

    table.add_row(
      f"{marker}{summary.name}",
      info.project or "",
      info.cluster_name or "",
      info.zone or "",
      str(summary.last_update or "never"),
    )

  console.print()
  console.print(table)
  if active:
    console.print(f"\n[dim]* = active stack ({active})[/dim]")
  console.print()


@stacks.command("set")
@click.argument("name")
def stacks_set(name):
  """Set the active stack by name.

  NAME is the stack name as shown in 'keras-remote stacks list'.
  """
  if not stack_exists(name):
    raise click.ClickException(
      f"Stack '{name}' not found. "
      "Run 'keras-remote stacks list' to see available stacks."
    )

  set_active_stack(name)
  success(f"Active stack set to '{name}'.")


@stacks.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def stacks_delete(name, yes):
  """Delete a stack and destroy its resources.

  NAME is the stack name as shown in 'keras-remote stacks list'.
  """
  if not stack_exists(name):
    raise click.ClickException(
      f"Stack '{name}' not found. "
      "Run 'keras-remote stacks list' to see available stacks."
    )

  warning(f"This will destroy all resources and remove stack '{name}'.")
  if not yes:
    click.confirm("Are you sure?", abort=True)

  console.print()
  try:
    delete_stack(name)
    success(f"Stack '{name}' deleted.")
  except CommandError as e:
    raise click.ClickException(f"Failed to delete stack '{name}': {e}") from e
