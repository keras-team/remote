"""keras-remote up command — provision infrastructure."""

import click

from keras_remote.cli.constants import DEFAULT_ZONE, DEFAULT_CLUSTER_NAME
from keras_remote.cli.infra.accelerator_configs import GPU_CONFIGS, TPU_CONFIGS
from keras_remote.cli.infra.post_deploy import (
    configure_docker_auth,
    configure_kubectl,
    install_gpu_drivers,
)
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import get_stack, deploy
from keras_remote.cli.output import console, banner, success, config_summary
from keras_remote.cli.prerequisites import check_all
from keras_remote.cli.prompts import resolve_config, prompt_accelerator


@click.command()
@click.option("--project", envvar="KERAS_REMOTE_PROJECT", default=None,
              help="GCP project ID [env: KERAS_REMOTE_PROJECT]")
@click.option("--zone", envvar="KERAS_REMOTE_ZONE", default=None,
              help="GCP zone [env: KERAS_REMOTE_ZONE, default: us-central1-a]")
@click.option("--accelerator", default=None,
              help="Accelerator spec: cpu, t4, l4, a100, a100-80gb, h100, "
                   "v5litepod, v5p, v6e, v3")
@click.option("--cluster-name", envvar="KERAS_REMOTE_CLUSTER", default=None,
              help="GKE cluster name [default: keras-remote-cluster]")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def up(project, zone, accelerator, cluster_name, yes):
    """Provision GCP infrastructure for keras-remote."""
    banner("keras-remote Setup")

    # Check prerequisites
    check_all()

    # Resolve configuration
    project = project or resolve_config("project")
    zone = zone or DEFAULT_ZONE
    cluster_name = cluster_name or DEFAULT_CLUSTER_NAME

    # Resolve accelerator (interactive if not provided)
    if accelerator:
        accel_config = _parse_accelerator_flag(accelerator)
    else:
        accel_config = prompt_accelerator()

    config = {
        "project": project,
        "zone": zone,
        "cluster_name": cluster_name,
        "accelerator": accel_config,
    }

    # Show summary and confirm
    config_summary(config)
    if not yes:
        click.confirm("\nProceed with setup?", abort=True)

    console.print()

    # Run Pulumi
    program = create_program(config)
    stack = get_stack(program, config)
    console.print("[bold]Provisioning infrastructure...[/bold]\n")
    result = deploy(stack)
    console.print()
    success(f"Pulumi update complete. {result.summary.resource_changes}")

    # Post-deploy steps
    ar_location = _zone_to_ar_location(zone)
    console.print("\n[bold]Running post-deploy configuration...[/bold]\n")

    console.print("Configuring Docker authentication...")
    configure_docker_auth(ar_location)
    success("Docker authentication configured")

    console.print("Configuring kubectl access...")
    configure_kubectl(cluster_name, zone, project)
    success("kubectl configured")

    if accel_config and accel_config.get("category") == "gpu":
        console.print("Installing NVIDIA GPU device drivers...")
        install_gpu_drivers()
        success("GPU driver installation initiated")

    # Final summary
    console.print()
    banner("Setup Complete")
    console.print()
    console.print("Add these environment variables to your shell config:")
    console.print(f"  export KERAS_REMOTE_PROJECT={project}")
    console.print(f"  export KERAS_REMOTE_ZONE={zone}")
    console.print(f"  export KERAS_REMOTE_CLUSTER={cluster_name}")
    console.print()
    console.print("View quotas:")
    console.print(
        f"  https://console.cloud.google.com/iam-admin/quotas"
        f"?project={project}"
    )
    console.print()


def _zone_to_ar_location(zone):
    """Derive Artifact Registry multi-region location from a GCP zone."""
    region = zone.rsplit("-", 1)[0] if zone and "-" in zone else "us-central1"
    return region.split("-")[0]


def _parse_accelerator_flag(accelerator):
    """Parse a --accelerator flag value into an accel config dict."""
    accelerator = accelerator.strip().lower()

    if accelerator == "cpu":
        return None

    if accelerator in GPU_CONFIGS:
        return {"category": "gpu", **GPU_CONFIGS[accelerator]}

    # Check for TPU with topology: "v5litepod-2x2"
    for tpu_name, topologies in TPU_CONFIGS.items():
        if accelerator == tpu_name:
            # Use first (default) topology
            default_topo = next(iter(topologies))
            return {
                "category": "tpu",
                "pool_name": f"tpu-{tpu_name}-pool",
                "topology": default_topo,
                **topologies[default_topo],
            }
        if accelerator.startswith(tpu_name + "-"):
            topo = accelerator[len(tpu_name) + 1:]
            if topo in topologies:
                return {
                    "category": "tpu",
                    "pool_name": f"tpu-{tpu_name}-pool",
                    "topology": topo,
                    **topologies[topo],
                }

    raise click.BadParameter(
        f"Unknown accelerator: '{accelerator}'. "
        f"GPUs: {', '.join(GPU_CONFIGS.keys())}. "
        f"TPUs: {', '.join(TPU_CONFIGS.keys())} (e.g., v5litepod-2x2).",
        param_hint="--accelerator",
    )
