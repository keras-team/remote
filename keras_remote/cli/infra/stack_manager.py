"""Pulumi Automation API wrapper for keras-remote."""

import os
import shutil

import click
import pulumi.automation as auto

from keras_remote.cli.constants import RESOURCE_NAME_PREFIX, STATE_DIR


def get_stack(program_fn, config):
  """Create or select a Pulumi stack with local file backend.

  Args:
      program_fn: Pulumi inline program callable.
      config: InfraConfig instance.

  Returns:
      A pulumi.automation.Stack instance.
  """
  os.makedirs(STATE_DIR, exist_ok=True)

  # Auto-install the Pulumi CLI if not already present.
  if not shutil.which("pulumi"):
    click.echo("Pulumi CLI not found. Installing...")
  pulumi_cmd = auto.PulumiCommand.install()

  # Use project ID as stack name so each GCP project gets its own stack
  stack_name = config.project

  project_settings = auto.ProjectSettings(
    name=RESOURCE_NAME_PREFIX,
    runtime="python",
    backend=auto.ProjectBackend(url=f"file://{STATE_DIR}"),
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
