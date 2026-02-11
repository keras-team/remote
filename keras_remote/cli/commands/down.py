"""keras-remote down command — tear down infrastructure."""

import subprocess

import click

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
              help="GCP zone [env: KERAS_REMOTE_ZONE, default: us-central1-a]")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--pulumi-only", is_flag=True,
              help="Only destroy Pulumi-managed resources (skip supplementary cleanup)")
def down(project, zone, yes, pulumi_only):
    """Tear down keras-remote GCP infrastructure."""
    banner("keras-remote Cleanup")

    check_all()

    project = project or resolve_config("project")
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
    console.print("  - Enabled API services (left enabled)")
    if not pulumi_only:
        console.print("  - Cloud Storage buckets (jobs and builds)")
        console.print("  - TPU VMs (if any)")
        console.print("  - GKE clusters matching 'keras-remote-*'")
        console.print("  - Compute Engine VMs matching 'remote-*'")
    console.print()

    if not yes:
        click.confirm("Are you sure you want to continue?", abort=True)

    console.print()

    # Pulumi destroy
    try:
        # Build a minimal config to load the stack — accelerator is not
        # needed for destroy since the stack already has its state.
        config = {
            "project": project,
            "zone": zone,
            "cluster_name": "keras-remote-cluster",
            "accelerator": None,
        }
        program = create_program(config)
        stack = get_stack(program, config)
        console.print("[bold]Destroying Pulumi-managed resources...[/bold]\n")
        result = destroy(stack)
        console.print()
        success(f"Pulumi destroy complete. {result.summary.resource_changes}")
    except Exception as e:
        warning(f"Pulumi destroy encountered an issue: {e}")
        console.print("Continuing with supplementary cleanup...\n")

    # Supplementary cleanup
    if not pulumi_only:
        console.print("\n[bold]Running supplementary cleanup...[/bold]\n")
        _cleanup_buckets(project, console)
        _cleanup_artifact_registry(project, zone, console)
        _cleanup_tpu_vms(project, console)
        _cleanup_gke_clusters(project, console)
        _cleanup_compute_vms(project, console)

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


def _run_gcloud(args, suppress_errors=True):
    """Run a gcloud command and return (success, stdout)."""
    result = subprocess.run(
        ["gcloud"] + args,
        capture_output=True,
        text=True,
    )
    if suppress_errors:
        return result.returncode == 0, result.stdout.strip()
    return result.returncode == 0, result.stdout.strip()


def _cleanup_buckets(project, console):
    """Delete keras-remote Cloud Storage buckets."""
    console.print("Deleting Cloud Storage buckets...")
    for suffix in ("jobs", "builds"):
        bucket = f"gs://{project}-keras-remote-{suffix}"
        ok, _ = _run_gcloud(["storage", "rm", "-r", bucket])
        if ok:
            success(f"  Deleted {bucket}")
        else:
            warning(f"  {bucket} not found or already deleted")


def _cleanup_artifact_registry(project, zone, console):
    """Delete keras-remote Artifact Registry repository."""
    region = zone.rsplit("-", 1)[0] if zone and "-" in zone else "us-central1"
    ar_location = region.split("-")[0]

    console.print("Deleting Artifact Registry repository...")
    ok, _ = _run_gcloud([
        "artifacts", "repositories", "delete", "keras-remote",
        f"--location={ar_location}", f"--project={project}", "--quiet",
    ])
    if ok:
        success("  Deleted keras-remote repository")
    else:
        warning("  Repository not found or already deleted")


def _cleanup_tpu_vms(project, console):
    """Delete TPU VMs in the project."""
    console.print("Checking for TPU VMs...")
    ok, output = _run_gcloud([
        "compute", "tpus", "list", f"--project={project}",
        "--format=value(name,zone)",
    ])
    if not ok or not output:
        success("  No TPU VMs found")
        return

    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            vm, vm_zone = parts[0], parts[1]
            ok, _ = _run_gcloud([
                "compute", "tpus", "delete", vm,
                f"--zone={vm_zone}", f"--project={project}", "--quiet",
            ])
            if ok:
                success(f"  Deleted TPU VM: {vm}")
            else:
                warning(f"  Failed to delete TPU VM: {vm}")


def _cleanup_gke_clusters(project, console):
    """Delete GKE clusters matching keras-remote-* pattern."""
    console.print("Checking for GKE clusters (keras-remote-*)...")
    ok, output = _run_gcloud([
        "container", "clusters", "list", f"--project={project}",
        "--filter=name~^keras-remote-", "--format=value(name,location)",
    ])
    if not ok or not output:
        success("  No matching GKE clusters found")
        return

    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            cluster, location = parts[0], parts[1]
            ok, _ = _run_gcloud([
                "container", "clusters", "delete", cluster,
                f"--location={location}", f"--project={project}", "--quiet",
            ])
            if ok:
                success(f"  Deleted GKE cluster: {cluster}")
            else:
                warning(f"  Failed to delete GKE cluster: {cluster}")


def _cleanup_compute_vms(project, console):
    """Delete Compute Engine VMs matching remote-* pattern."""
    console.print("Checking for Compute Engine VMs (remote-*)...")
    ok, output = _run_gcloud([
        "compute", "instances", "list", f"--project={project}",
        "--filter=name~^remote-.*", "--format=value(name,zone)",
    ])
    if not ok or not output:
        success("  No matching Compute Engine VMs found")
        return

    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            vm, vm_zone = parts[0], parts[1]
            ok, _ = _run_gcloud([
                "compute", "instances", "delete", vm,
                f"--zone={vm_zone}", f"--project={project}", "--quiet",
            ])
            if ok:
                success(f"  Deleted VM: {vm}")
            else:
                warning(f"  Failed to delete VM: {vm}")
