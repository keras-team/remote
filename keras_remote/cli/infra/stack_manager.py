"""Pulumi Automation API wrapper for keras-remote."""

import contextlib
import os
from typing import NamedTuple

import click
import pulumi.automation as auto

from keras_remote.cli.config import NodePoolConfig
from keras_remote.cli.constants import (
  PULUMI_ROOT,
  RESOURCE_NAME_PREFIX,
  STATE_DIR,
)
from keras_remote.core import accelerators

_ACTIVE_STACK_FILE = os.path.join(
  os.path.expanduser("~/.keras-remote"), "active-stack"
)

_NO_ACTIVE_STACK_MSG = (
  "No active stack set. Run 'keras-remote up' to create one "
  "or 'keras-remote stacks set <name>' to select an existing stack."
)


# ── Pulumi CLI / workspace helpers ──────────────────────────────────


def _get_pulumi_cmd():
  """Get or install the Pulumi CLI command."""
  try:
    return auto.PulumiCommand(root=PULUMI_ROOT)
  except Exception:  # noqa: BLE001
    click.echo("Pulumi CLI not found. Installing...")
    return auto.PulumiCommand.install(root=PULUMI_ROOT)


def _get_workspace():
  """Return a ready-to-use ``LocalWorkspace`` with local file backend."""
  os.makedirs(STATE_DIR, exist_ok=True)
  pulumi_cmd = _get_pulumi_cmd()
  project_settings = auto.ProjectSettings(
    name=RESOURCE_NAME_PREFIX,
    runtime="python",
    backend=auto.ProjectBackend(url=f"file://{STATE_DIR}"),
  )
  return auto.LocalWorkspace(
    project_settings=project_settings,
    env_vars={"PULUMI_CONFIG_PASSPHRASE": ""},
    pulumi_command=pulumi_cmd,
  )


def _select_readonly_stack(stack_name: str):
  """Select an existing stack with a no-op program (no Pulumi execution).

  Returns:
      A ``pulumi.automation.Stack``, or ``None`` if the stack does not exist.
  """
  ws = _get_workspace()
  opts = auto.LocalWorkspaceOptions(
    project_settings=ws.project_settings,
    env_vars={"PULUMI_CONFIG_PASSPHRASE": ""},
    pulumi_command=ws.pulumi_command,
  )
  try:
    return auto.select_stack(
      stack_name=stack_name,
      project_name=RESOURCE_NAME_PREFIX,
      program=lambda: None,
      opts=opts,
    )
  except auto.errors.CommandError:
    return None


# ── Stack naming ────────────────────────────────────────────────────


def make_stack_name(project: str, cluster_name: str) -> str:
  """Build the cluster-scoped Pulumi stack name."""
  return f"{project}-{cluster_name}"


# ── Core stack accessor ─────────────────────────────────────────────


def get_stack(program_fn, config, *, stack_name=None):
  """Create or select a Pulumi stack with local file backend.

  Uses a cluster-scoped stack name (``{project}-{cluster_name}``) so
  that multiple clusters can coexist within the same GCP project.

  Args:
      program_fn: Pulumi inline program callable.
      config: InfraConfig instance.
      stack_name: Explicit stack name to use. When ``None``, derived
          from ``config.project`` and ``config.cluster_name``.

  Returns:
      A pulumi.automation.Stack instance.
  """
  ws = _get_workspace()
  opts = auto.LocalWorkspaceOptions(
    project_settings=ws.project_settings,
    env_vars={"PULUMI_CONFIG_PASSPHRASE": ""},
    pulumi_command=ws.pulumi_command,
  )

  if stack_name is None:
    stack_name = make_stack_name(config.project, config.cluster_name)

  try:
    stack = auto.select_stack(
      stack_name=stack_name,
      project_name=RESOURCE_NAME_PREFIX,
      program=program_fn,
      opts=opts,
    )
  except auto.errors.CommandError:
    stack = auto.create_stack(
      stack_name=stack_name,
      project_name=RESOURCE_NAME_PREFIX,
      program=program_fn,
      opts=opts,
    )

  # Set GCP provider and cluster configuration on the stack.
  if config.project is not None:
    stack.set_config("gcp:project", auto.ConfigValue(value=config.project))
  if config.zone is not None:
    stack.set_config("gcp:zone", auto.ConfigValue(value=config.zone))
  if config.cluster_name is not None:
    stack.set_config(
      "keras-remote:cluster_name",
      auto.ConfigValue(value=config.cluster_name),
    )

  return stack


# ── Active stack persistence ────────────────────────────────────────


def get_active_stack() -> str | None:
  """Read the active stack name from ``~/.keras-remote/active-stack``.

  Returns:
      The stack name string, or ``None`` if no active stack is set.
  """
  try:
    with open(_ACTIVE_STACK_FILE) as f:
      name = f.read().strip()
      return name or None
  except FileNotFoundError:
    return None


def set_active_stack(stack_name: str) -> None:
  """Persist *stack_name* as the active stack."""
  os.makedirs(os.path.dirname(_ACTIVE_STACK_FILE), exist_ok=True)
  with open(_ACTIVE_STACK_FILE, "w") as f:
    f.write(stack_name + "\n")


def clear_active_stack() -> None:
  """Remove the active stack pointer."""
  with contextlib.suppress(FileNotFoundError):
    os.remove(_ACTIVE_STACK_FILE)


# ── Stack info resolution ──────────────────────────────────────────


class StackInfo(NamedTuple):
  """Resolved project / zone / cluster for a stack."""

  project: str | None
  zone: str | None
  cluster_name: str | None


def resolve_stack_info(stack_name: str) -> StackInfo:
  """Resolve project / zone / cluster_name for a given stack.

  Tries stack outputs first, then falls back to stack config
  (``gcp:project``, ``gcp:zone``, ``keras-remote:cluster_name``).

  This is the single source of truth for resolving stack metadata.
  """
  project = None
  zone = None
  cluster = None

  outputs = get_stack_outputs(stack_name)
  if outputs:
    project = outputs["project"].value if "project" in outputs else None
    zone = outputs["zone"].value if "zone" in outputs else None
    cluster = (
      outputs["cluster_name"].value if "cluster_name" in outputs else None
    )

  # Fall back to stack config for newly created stacks without outputs.
  if not project or not zone or not cluster:
    config = get_stack_config(stack_name)
    if config:
      if not project and "gcp:project" in config:
        project = str(config["gcp:project"].value)
      if not zone and "gcp:zone" in config:
        zone = str(config["gcp:zone"].value)
      if not cluster and "keras-remote:cluster_name" in config:
        cluster = str(config["keras-remote:cluster_name"].value)

  return StackInfo(project=project, zone=zone, cluster_name=cluster)


class ActiveStackResolution(NamedTuple):
  """Result of resolving the active stack."""

  project: str | None
  zone: str | None
  cluster_name: str | None
  stack_name: str | None


def resolve_from_active_stack() -> ActiveStackResolution:
  """Read project / zone / cluster from the active stack.

  Returns:
      An ``ActiveStackResolution`` — all fields are ``None`` if
      no active stack is set.
  """
  active = get_active_stack()
  if not active:
    return ActiveStackResolution(None, None, None, None)

  info = resolve_stack_info(active)
  return ActiveStackResolution(
    project=info.project,
    zone=info.zone,
    cluster_name=info.cluster_name,
    stack_name=active,
  )


def require_active_stack() -> ActiveStackResolution:
  """Like ``resolve_from_active_stack`` but raises on missing stack.

  Raises:
      click.ClickException: If no active stack is set.
  """
  result = resolve_from_active_stack()
  if not result.stack_name:
    raise click.ClickException(_NO_ACTIVE_STACK_MSG)
  return result


# ── Stack listing / querying ────────────────────────────────────────


def list_stacks():
  """List all Pulumi stacks in the local backend.

  Returns:
      A list of ``pulumi.automation.StackSummary`` objects.
  """
  ws = _get_workspace()
  return ws.list_stacks()


def stack_exists(stack_name: str) -> bool:
  """Check whether a stack exists in the local backend."""
  return _select_readonly_stack(stack_name) is not None


def get_stack_outputs(stack_name: str) -> dict | None:
  """Select an existing stack by name and return its outputs.

  The Pulumi program is *not* executed — a no-op is used instead.

  Returns:
      A dict of ``key → OutputValue``, or ``None`` if the stack
      cannot be loaded.
  """
  stack = _select_readonly_stack(stack_name)
  if stack is None:
    return None
  return stack.outputs()


def get_stack_config(stack_name: str) -> dict | None:
  """Select an existing stack by name and return its config.

  Returns:
      A dict of ``key → ConfigValue``, or ``None`` if the stack
      cannot be loaded.
  """
  stack = _select_readonly_stack(stack_name)
  if stack is None:
    return None
  return stack.get_all_config()


def delete_stack(stack_name: str) -> None:
  """Destroy resources and remove a Pulumi stack from the local backend.

  Also clears the active stack pointer if it matches.

  Args:
      stack_name: The stack name to delete.

  Raises:
      auto.errors.CommandError: If the stack cannot be found or destroyed.
  """
  ws = _get_workspace()
  opts = auto.LocalWorkspaceOptions(
    project_settings=ws.project_settings,
    env_vars={"PULUMI_CONFIG_PASSPHRASE": ""},
    pulumi_command=ws.pulumi_command,
  )

  stack = auto.select_stack(
    stack_name=stack_name,
    project_name=RESOURCE_NAME_PREFIX,
    program=lambda: None,
    opts=opts,
  )
  stack.destroy(on_output=print)
  stack.workspace.remove_stack(stack_name)

  # Clear active stack if it matches the deleted one.
  if get_active_stack() == stack_name:
    clear_active_stack()


def remove_stack(stack_name: str) -> None:
  """Remove a Pulumi stack from the local backend without destroying resources.

  Also clears the active stack pointer if it matches.

  Args:
      stack_name: The stack name to remove.

  Raises:
      auto.errors.CommandError: If the stack cannot be found.
  """
  ws = _get_workspace()
  ws.remove_stack(stack_name)

  if get_active_stack() == stack_name:
    clear_active_stack()


def get_cluster_name_from_outputs(stack) -> str | None:
  """Read the actual ``cluster_name`` from stack outputs.

  Returns:
      The cluster name string, or ``None`` if not available.
  """
  outputs = stack.outputs()
  if "cluster_name" in outputs:
    return outputs["cluster_name"].value
  return None


# ── Node pool helpers ───────────────────────────────────────────────


def get_current_node_pools(stack) -> list[NodePoolConfig]:
  """Read the current node pool list from Pulumi stack exports.

  Handles both the new ``accelerators`` key (list) and the legacy
  ``accelerator`` key (single dict or None).

  Args:
      stack: A ``pulumi.automation.Stack`` whose outputs have been
          populated (e.g. after ``stack.refresh()``).

  Returns:
      A list of :class:`NodePoolConfig` objects.
  """
  outputs = stack.outputs()

  # New format: list of accelerator dicts.
  if "accelerators" in outputs:
    accel_list = outputs["accelerators"].value
    if not accel_list:
      return []
    return [_export_to_node_pool(entry) for entry in accel_list]

  # Legacy format: single dict or None.
  if "accelerator" in outputs:
    accel = outputs["accelerator"].value
    if accel is None:
      return []
    return [_export_to_node_pool(accel)]

  return []


def _export_to_node_pool(entry: dict) -> NodePoolConfig:
  """Convert a stack export dict back to a NodePoolConfig."""
  pool_name = entry["node_pool"]
  accelerator: accelerators.GpuConfig | accelerators.TpuConfig
  if entry["type"] == "GPU":
    accelerator = accelerators.make_gpu(entry["name"], entry["count"])
  elif entry["type"] == "TPU":
    accelerator = accelerators.make_tpu(entry["name"], entry["chips"])
  else:
    raise ValueError(f"Unknown accelerator type in node pool export: {entry}")
  return NodePoolConfig(name=pool_name, accelerator=accelerator)
