"""Rich console output helpers for the keras-remote CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from keras_remote.core.accelerators import GpuConfig, TpuConfig

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


_INFRA_LABELS = {
  "project": "Project",
  "zone": "Zone",
  "cluster_name": "Cluster Name",
  "cluster_endpoint": "Cluster Endpoint",
  "ar_registry": "Artifact Registry",
}

_GPU_LABELS = {
  "name": "GPU Type",
  "count": "GPU Count",
  "machine_type": "Machine Type",
  "node_pool": "Node Pool",
  "node_count": "Node Count",
}

_TPU_LABELS = {
  "name": "TPU Type",
  "chips": "TPU Chips",
  "topology": "Topology",
  "machine_type": "Machine Type",
  "node_pool": "Node Pool",
  "node_count": "Node Count",
}


def infrastructure_state(outputs):
  """Display infrastructure state from Pulumi stack outputs.

  Args:
      outputs: dict of key -> pulumi.automation.OutputValue from stack.outputs().
  """
  table = Table(title="Infrastructure State")
  table.add_column("Resource", style="bold")
  table.add_column("Value", style="green")

  for key, label in _INFRA_LABELS.items():
    if key in outputs:
      table.add_row(label, str(outputs[key].value))

  if "accelerator" not in outputs:
    table.add_row(
      "Accelerator",
      "[dim]Unknown (run 'keras-remote up' to refresh)[/dim]",
    )
  elif outputs["accelerator"].value is None:
    table.add_row("Accelerator", "CPU only")
  else:
    accel = outputs["accelerator"].value
    accel_type = accel.get("type", "Unknown")
    table.add_row("", "")
    table.add_row("Accelerator", accel_type)
    labels = _GPU_LABELS if accel_type == "GPU" else _TPU_LABELS
    for key, label in labels.items():
      if key in accel:
        table.add_row(f"  {label}", str(accel[key]))

  console.print()
  console.print(table)
  console.print()


def config_summary(config):
  """Display a configuration summary table."""
  table = Table(title="Configuration Summary")
  table.add_column("Setting", style="bold")
  table.add_column("Value", style="green")

  table.add_row("Project", config.project)
  table.add_row("Zone", config.zone)
  table.add_row("Cluster Name", config.cluster_name)

  accel = config.accelerator
  if accel is None:
    table.add_row("Accelerator", "CPU only")
  elif isinstance(accel, GpuConfig):
    table.add_row("Accelerator", f"GPU ({accel.gke_label})")
  elif isinstance(accel, TpuConfig):
    table.add_row(
      "Accelerator",
      f"TPU ({accel.name}, topology: {accel.topology})",
    )

  console.print()
  console.print(table)
