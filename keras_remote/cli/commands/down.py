"""keras-remote down command — tear down infrastructure."""

import click
import pulumi.automation as auto

from google.api_core import exceptions as api_exceptions
from google.cloud import tpu_v2

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.constants import DEFAULT_ZONE
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import get_stack, destroy
from keras_remote.cli.output import console, banner, success, warning
from keras_remote.cli.prerequisites import check_all
from keras_remote.cli.prompts import resolve_config


@click.command()
@click.option("--project", envvar="KERAS_REMOTE_PROJECT", default=None,
              help="GCP project ID [env: KERAS_REMOTE_PROJECT]")
@click.option("--zone", envvar="KERAS_REMOTE_ZONE", default=None,
              help=("GCP zone [env: KERAS_REMOTE_ZONE,"
                    f" default: {DEFAULT_ZONE}]"))
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--pulumi-only", is_flag=True,
              help="Only destroy Pulumi-managed resources (skip supplementary cleanup)")
def down(project, zone, yes, pulumi_only):
    """Tear down keras-remote GCP infrastructure."""
    banner("keras-remote Cleanup")

    check_all()

    project = project or resolve_config("project", allow_create=False)
    zone = zone or DEFAULT_ZONE

    # Warning
    console.print()
    warning(
        "This will delete ALL keras-remote resources "
        f"in project: {project}"
    )
    console.print()
    console.print("This includes:")
    console.print("  - GKE cluster and node pools")
    console.print("  - Artifact Registry repository and images")
    console.print("  - Cloud Storage buckets (jobs and builds)")
    console.print("  - Enabled API services (left enabled)")
    if not pulumi_only:
        console.print("  - TPU VMs (if any)")
    console.print()

    if not yes:
        click.confirm("Are you sure you want to continue?", abort=True)

    console.print()

    # Pulumi destroy
    try:
        # Minimal config to load the stack — accelerator is not
        # needed for destroy since the stack already has its state.
        config = InfraConfig(project=project, zone=zone)
        program = create_program(config)
        stack = get_stack(program, config)
        console.print("[bold]Destroying Pulumi-managed resources...[/bold]\n")
        result = destroy(stack)
        console.print()
        success(f"Pulumi destroy complete. {result.summary.resource_changes}")
    except auto.errors.CommandError as e:
        warning(f"Pulumi destroy encountered an issue: {e}")
        console.print("Continuing with supplementary cleanup...\n")

    # Supplementary cleanup
    if not pulumi_only:
        console.print("\n[bold]Running supplementary cleanup...[/bold]\n")
        _cleanup_tpu_vms(project, zone, console)

    # Summary
    console.print()
    banner("Cleanup Complete")
    console.print()
    console.print("Check manually for remaining resources:")
    console.print(
        f"  GKE: https://console.cloud.google.com/kubernetes/list"
        f"?project={project}"
    )
    console.print(
        f"  Billing: https://console.cloud.google.com/billing"
        f"?project={project}"
    )
    console.print()


def _is_api_disabled(exc):
    """Check if an exception indicates the API is not enabled."""
    msg = str(exc).lower()
    return "not enabled" in msg or "disabled" in msg or "not been used" in msg


def _cleanup_tpu_vms(project, zone, console):
    """Delete TPU VMs in the project."""
    console.print("Checking for TPU VMs...")
    client = tpu_v2.TpuClient()
    parent = f"projects/{project}/locations/{zone}"

    try:
        nodes = list(client.list_nodes(parent=parent))
    except api_exceptions.GoogleAPICallError as e:
        if _is_api_disabled(e):
            success("  Skipped (TPU API not enabled)")
        else:
            warning(f"  Failed to list TPU VMs: {e}")
        return

    if not nodes:
        success("  No TPU VMs found")
        return

    for node in nodes:
        short_name = node.name.split("/")[-1]
        try:
            operation = client.delete_node(name=node.name)
            operation.result()
            success(f"  Deleted TPU VM: {short_name}")
        except Exception as e:
            warning(f"  Failed to delete TPU VM: {short_name}: {e}")
