"""Prerequisite checks for the keras-remote CLI."""

import shutil
import subprocess

import click

from keras_remote.cli.output import warning


def check_gcloud():
    """Verify gcloud CLI is installed."""
    if not shutil.which("gcloud"):
        raise click.ClickException(
            "gcloud CLI not found. "
            "Install from: https://cloud.google.com/sdk/docs/install"
        )


def check_pulumi():
    """Verify Pulumi CLI is installed (required by Automation API)."""
    if not shutil.which("pulumi"):
        raise click.ClickException(
            "Pulumi CLI not found. "
            "Install from: https://www.pulumi.com/docs/install/"
        )


def check_kubectl():
    """Verify kubectl is installed."""
    if not shutil.which("kubectl"):
        raise click.ClickException(
            "kubectl not found. "
            "Install from: https://kubernetes.io/docs/tasks/tools/"
        )


def check_docker():
    """Verify Docker CLI is installed."""
    if not shutil.which("docker"):
        raise click.ClickException(
            "Docker not found. "
            "Install from: https://docs.docker.com/get-docker/"
        )


def check_gcloud_auth():
    """Check if gcloud Application Default Credentials are configured."""
    result = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True,
    )
    if result.returncode != 0:
        warning("Application Default Credentials not found.")
        click.echo("Running: gcloud auth application-default login")
        subprocess.run(
            ["gcloud", "auth", "application-default", "login"],
            check=True,
        )


def check_all():
    """Run all prerequisite checks."""
    check_gcloud()
    check_pulumi()
    check_kubectl()
    check_docker()
    check_gcloud_auth()
