"""Interactive prompts for the keras-remote CLI."""

import os

import click

from keras_remote.cli.infra.accelerator_configs import GPU_CONFIGS, TPU_CONFIGS


def resolve_config(key):
    """Resolve a config value from env var or interactive prompt."""
    if key == "project":
        return _resolve_project()
    raise ValueError(f"Unknown config key: {key}")


def _resolve_project():
    """Resolve GCP project ID from env or prompt."""
    project = os.environ.get("KERAS_REMOTE_PROJECT")
    if project:
        return project
    return click.prompt("Enter your GCP project ID", type=str)


def prompt_accelerator():
    """Interactive accelerator selection menu.

    Returns:
        Config dict with accelerator details, or None for CPU.
    """
    accel_type = click.prompt(
        "\nWhat type of accelerator node pool?",
        type=click.Choice(["cpu", "gpu", "tpu"], case_sensitive=False),
        default="cpu",
    )

    if accel_type == "cpu":
        return None

    if accel_type == "gpu":
        return _prompt_gpu()

    if accel_type == "tpu":
        return _prompt_tpu()


def _prompt_gpu():
    """Prompt for GPU type selection."""
    click.echo()
    click.echo("Available GPU types:")
    gpu_names = list(GPU_CONFIGS.keys())
    for i, name in enumerate(gpu_names, 1):
        cfg = GPU_CONFIGS[name]
        click.echo(f"  {i}) {name:<12} ({cfg['gpu_type']})")

    gpu = click.prompt(
        "\nSelect GPU type",
        type=click.Choice(gpu_names, case_sensitive=False),
    )
    return {"category": "gpu", **GPU_CONFIGS[gpu]}


def _prompt_tpu():
    """Prompt for TPU type and topology selection."""
    click.echo()
    tpu_names = list(TPU_CONFIGS.keys())
    click.echo("Available TPU types:")
    for i, name in enumerate(tpu_names, 1):
        topologies = ", ".join(TPU_CONFIGS[name].keys())
        click.echo(f"  {i}) {name:<12} (topologies: {topologies})")

    tpu = click.prompt(
        "\nSelect TPU type",
        type=click.Choice(tpu_names, case_sensitive=False),
    )

    topologies = list(TPU_CONFIGS[tpu].keys())
    click.echo()
    click.echo(f"Available topologies for {tpu}:")
    for i, topo in enumerate(topologies, 1):
        cfg = TPU_CONFIGS[tpu][topo]
        click.echo(
            f"  {i}) {topo:<6} "
            f"(machine: {cfg['machine_type']}, nodes: {cfg['num_nodes']})"
        )

    default_topo = topologies[min(1, len(topologies) - 1)]
    topology = click.prompt(
        f"\nSelect topology for {tpu}",
        type=click.Choice(topologies),
        default=default_topo,
    )

    return {
        "category": "tpu",
        "pool_name": f"tpu-{tpu}-pool",
        "topology": topology,
        **TPU_CONFIGS[tpu][topology],
    }
