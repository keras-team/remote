"""Pulumi Automation API wrapper for keras-remote."""

import os

import pulumi.automation as auto

from keras_remote.cli.constants import PROJECT_NAME, STATE_DIR


def get_stack(program_fn, config):
    """Create or select a Pulumi stack with local file backend.

    Args:
        program_fn: Pulumi inline program callable.
        config: Dict with at least 'project' and 'zone' keys.

    Returns:
        A pulumi.automation.Stack instance.
    """
    state_dir = os.environ.get("KERAS_REMOTE_STATE_DIR", STATE_DIR)
    os.makedirs(state_dir, exist_ok=True)

    # Use project ID as stack name so each GCP project gets its own stack
    stack_name = config["project"]

    project_settings = auto.ProjectSettings(
        name=PROJECT_NAME,
        runtime="python",
        backend=auto.ProjectBackend(url=f"file://{state_dir}"),
    )

    stack = auto.create_or_select_stack(
        stack_name=stack_name,
        project_name=PROJECT_NAME,
        program=program_fn,
        opts=auto.LocalWorkspaceOptions(
            project_settings=project_settings,
            env_vars={"PULUMI_CONFIG_PASSPHRASE": ""},
        ),
    )

    # Set GCP provider configuration on the stack
    stack.set_config("gcp:project", auto.ConfigValue(value=config["project"]))
    stack.set_config("gcp:zone", auto.ConfigValue(value=config["zone"]))

    return stack


def deploy(stack, on_output=None):
    """Run pulumi up.

    Args:
        stack: A pulumi.automation.Stack instance.
        on_output: Callback for streaming output. Defaults to print.

    Returns:
        The UpResult from pulumi.
    """
    if on_output is None:
        on_output = print
    return stack.up(on_output=on_output)


def destroy(stack, on_output=None):
    """Run pulumi destroy.

    Args:
        stack: A pulumi.automation.Stack instance.
        on_output: Callback for streaming output. Defaults to print.

    Returns:
        The DestroyResult from pulumi.
    """
    if on_output is None:
        on_output = print
    return stack.destroy(on_output=on_output)


def refresh(stack, on_output=None):
    """Run pulumi refresh.

    Args:
        stack: A pulumi.automation.Stack instance.
        on_output: Callback for streaming output. Defaults to print.

    Returns:
        The RefreshResult from pulumi.
    """
    if on_output is None:
        on_output = print
    return stack.refresh(on_output=on_output)


def get_outputs(stack):
    """Get stack outputs.

    Args:
        stack: A pulumi.automation.Stack instance.

    Returns:
        Dict of output name -> OutputValue.
    """
    return stack.outputs()
