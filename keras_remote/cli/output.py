"""Rich console output helpers for the keras-remote CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def banner(text):
    """Display a styled banner."""
    console.print(Panel(f"  {text}", style="bold blue"))


def success(msg):
    """Display a success message."""
    console.print(f"[green]{msg}[/green]")


def warning(msg):
    """Display a warning message."""
    console.print(f"[yellow]{msg}[/yellow]")


def error(msg):
    """Display an error message."""
    console.print(f"[red]{msg}[/red]")


def config_summary(config):
    """Display a configuration summary table."""
    table = Table(title="Configuration Summary")
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Project", config["project"])
    table.add_row("Zone", config["zone"])
    table.add_row("Cluster Name", config["cluster_name"])

    accel = config.get("accelerator")
    if accel is None:
        table.add_row("Accelerator", "CPU only")
    elif accel["category"] == "gpu":
        table.add_row("Accelerator", f"GPU ({accel['gpu_type']})")
    elif accel["category"] == "tpu":
        table.add_row(
            "Accelerator",
            f"TPU ({accel['pool_name']}, topology: {accel['topology']})",
        )

    console.print()
    console.print(table)
