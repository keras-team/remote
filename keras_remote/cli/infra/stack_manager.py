"""Pulumi Automation API wrapper for keras-remote."""

import os

import click
import pulumi.automation as auto
from google.cloud import storage as gcs_storage

from keras_remote.cli.config import NamespaceConfig, NodePoolConfig
from keras_remote.cli.constants import (
  PULUMI_ROOT,
  RESOURCE_NAME_PREFIX,
  STATE_BUCKET_SUFFIX,
  STATE_DIR,
)
from keras_remote.core import accelerators


def _ensure_state_bucket(project: str) -> str:
  """Ensure the GCS bucket for Pulumi state exists, creating if needed.

  Returns:
      The bucket name.
  """
  bucket_name = f"{project}-{STATE_BUCKET_SUFFIX}"
  client = gcs_storage.Client(project=project)
  bucket = client.bucket(bucket_name)
  if not bucket.exists():
    bucket.create(location="us")
  return bucket_name


def get_stack(program_fn, config):
  """Create or select a Pulumi stack with GCS remote backend.

  Args:
      program_fn: Pulumi inline program callable.
      config: InfraConfig instance.

  Returns:
      A pulumi.automation.Stack instance.
  """
  os.makedirs(STATE_DIR, exist_ok=True)

  # Auto-install the Pulumi CLI if not already present.
  try:
    pulumi_cmd = auto.PulumiCommand(root=PULUMI_ROOT)
  except Exception:  # noqa: BLE001
    click.echo("Pulumi CLI not found. Installing...")
    pulumi_cmd = auto.PulumiCommand.install(root=PULUMI_ROOT)

  # Ensure state bucket exists
  state_bucket = _ensure_state_bucket(config.project)

  # Use project ID as stack name so each GCP project gets its own stack
  stack_name = config.project

  project_settings = auto.ProjectSettings(
    name=RESOURCE_NAME_PREFIX,
    runtime="python",
    backend=auto.ProjectBackend(url=f"gs://{state_bucket}"),
  )

  stack = auto.create_or_select_stack(
    stack_name=stack_name,
    project_name=RESOURCE_NAME_PREFIX,
    program=program_fn,
    opts=auto.LocalWorkspaceOptions(
      project_settings=project_settings,
      env_vars={"PULUMI_CONFIG_PASSPHRASE": ""},
      pulumi_command=pulumi_cmd,
    ),
  )

  # Set GCP provider configuration on the stack
  stack.set_config("gcp:project", auto.ConfigValue(value=config.project))
  stack.set_config("gcp:zone", auto.ConfigValue(value=config.zone))

  return stack


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


def get_current_namespaces(stack) -> list[NamespaceConfig]:
  """Read the current namespace list from Pulumi stack exports.

  Args:
      stack: A ``pulumi.automation.Stack`` whose outputs have been
          populated (e.g. after ``stack.refresh()``).

  Returns:
      A list of :class:`NamespaceConfig` objects.
  """
  outputs = stack.outputs()
  if "namespaces" not in outputs:
    return []
  ns_list = outputs["namespaces"].value
  if not ns_list:
    return []
  return [
    NamespaceConfig(
      name=ns["name"],
      members=ns.get("members", []),
      gpus=ns.get("gpus"),
      tpus=ns.get("tpus"),
      cpu=ns.get("cpu"),
      memory=ns.get("memory"),
      max_jobs=ns.get("max_jobs"),
      max_lws=ns.get("max_lws"),
    )
    for ns in ns_list
  ]


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
