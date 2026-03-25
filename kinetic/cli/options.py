"""Shared Click options for kinetic CLI commands."""

import click

from kinetic.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE


def common_options(f):
  """Shared --project, --zone, --cluster options for subcommands."""
  f = click.option(
    "--project",
    envvar="KINETIC_PROJECT",
    default=None,
    help="GCP project ID [env: KINETIC_PROJECT]",
  )(f)
  f = click.option(
    "--zone",
    envvar="KINETIC_ZONE",
    default=None,
    help=f"GCP zone [env: KINETIC_ZONE, default: {DEFAULT_ZONE}]",
  )(f)
  f = click.option(
    "--cluster",
    "cluster_name",
    envvar="KINETIC_CLUSTER",
    default=None,
    help=f"GKE cluster name [default: {DEFAULT_CLUSTER_NAME}]",
  )(f)
  return f
