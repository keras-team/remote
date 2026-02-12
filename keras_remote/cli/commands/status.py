"""keras-remote status command — show current infrastructure state."""

import click
from pulumi.automation import CommandError
from rich.table import Table

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.constants import DEFAULT_ZONE
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import get_stack, refresh, get_outputs
from keras_remote.cli.output import console, banner, warning
from keras_remote.cli.prerequisites import check_all
from keras_remote.cli.prompts import resolve_config


@click.command()
@click.option("--project", envvar="KERAS_REMOTE_PROJECT", default=None,
              help="GCP project ID [env: KERAS_REMOTE_PROJECT]")
@click.option("--zone", envvar="KERAS_REMOTE_ZONE", default=None,
              help=("GCP zone [env: KERAS_REMOTE_ZONE,"
                    f" default: {DEFAULT_ZONE}]"))
def status(project, zone):
    """Show current keras-remote infrastructure state."""
    banner("keras-remote Status")

    check_all()

    project = project or resolve_config("project")
    zone = zone or DEFAULT_ZONE

    config = InfraConfig(project=project, zone=zone)

    try:
        program = create_program(config)
        stack = get_stack(program, config)
    except CommandError as e:
        warning(f"No Pulumi stack found for project '{project}': {e}")
        console.print("Run 'keras-remote up' to provision infrastructure.")
        return

    console.print("\nRefreshing state...\n")
    try:
        refresh(stack)
    except CommandError as e:
        warning(f"Failed to refresh stack state: {e}")

    outputs = get_outputs(stack)

    if not outputs:
        warning("No infrastructure found. Run 'keras-remote up' first.")
        return

    table = Table(title="Infrastructure State")
    table.add_column("Resource", style="bold")
    table.add_column("Value", style="green")

    for key, output in outputs.items():
        table.add_row(key, str(output.value))

    console.print()
    console.print(table)
    console.print()
