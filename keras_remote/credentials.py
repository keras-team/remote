"""Credential verification and auto-setup for remote execution.

Ensures all required credentials (kubeconfig, GCP ADC, Docker auth) are
available before submitting jobs. Used by both the programmatic
``keras_remote.run()`` API and the CLI.

All functions raise ``RuntimeError`` on unrecoverable failures — callers
in the CLI layer convert these to ``click.ClickException`` as needed.
"""

import json
import os
import shutil
import subprocess
from typing import Optional

from absl import logging
from kubernetes import config

from keras_remote.constants import zone_to_ar_location


def ensure_credentials(
  project: str, zone: str, cluster: Optional[str] = None
) -> None:
  """Ensure all credentials needed for remote execution are available.

  Checks and auto-configures credentials in order:
  1. gcloud CLI (must be installed)
  2. gke-gcloud-auth-plugin (auto-install if missing)
  3. GCP Application Default Credentials (auto-login if missing)
  4. Kubeconfig for the target cluster (auto-configure if wrong/missing)
  5. Docker auth for Artifact Registry (auto-configure if missing)

  Args:
      project: GCP project ID.
      zone: GCP zone (e.g. ``us-central1-a``).
      cluster: GKE cluster name. If ``None``, uses ``KERAS_REMOTE_CLUSTER``
          env var, falling back to ``keras-remote-cluster``.

  Raises:
      RuntimeError: If a required credential cannot be configured.
  """
  ensure_gcloud()
  ensure_gke_auth_plugin()
  ensure_adc()
  ensure_kubeconfig(project, zone, cluster)
  ensure_docker_auth(zone)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def ensure_gcloud() -> None:
  """Verify gcloud CLI is installed."""
  if not shutil.which("gcloud"):
    raise RuntimeError(
      "gcloud CLI not found. "
      "Install from: https://cloud.google.com/sdk/docs/install"
    )


def ensure_gke_auth_plugin() -> None:
  """Verify gke-gcloud-auth-plugin is installed; auto-install if missing."""
  if shutil.which("gke-gcloud-auth-plugin"):
    return

  logging.info("gke-gcloud-auth-plugin not found. Installing...")
  try:
    subprocess.run(
      [
        "gcloud",
        "components",
        "install",
        "gke-gcloud-auth-plugin",
        "--quiet",
      ],
      check=True,
      capture_output=True,
    )
    logging.info("gke-gcloud-auth-plugin installed successfully.")
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      "Failed to install gke-gcloud-auth-plugin. "
      "Install manually: gcloud components install gke-gcloud-auth-plugin"
    ) from e


def ensure_adc() -> None:
  """Verify GCP Application Default Credentials are configured."""
  result = subprocess.run(
    ["gcloud", "auth", "application-default", "print-access-token"],
    capture_output=True,
  )
  if result.returncode == 0:
    return

  logging.info("Application Default Credentials not found. Running login...")
  try:
    subprocess.run(
      ["gcloud", "auth", "application-default", "login"],
      check=True,
    )
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      "Failed to configure Application Default Credentials. "
      "Run manually: gcloud auth application-default login"
    ) from e


def ensure_kubeconfig(
  project: str, zone: str, cluster: Optional[str] = None
) -> None:
  """Ensure kubeconfig is configured for the target GKE cluster.

  Loads the existing kubeconfig and verifies the active context points to the
  expected cluster (``gke_{project}_{zone}_{cluster}``).  If the context is
  wrong or kubeconfig is missing, runs ``gcloud container clusters
  get-credentials`` to configure it.

  When *cluster* is ``None``, any valid kubeconfig is accepted.
  """
  cluster_name = cluster or os.environ.get(
    "KERAS_REMOTE_CLUSTER", "keras-remote-cluster"
  )

  # Try loading existing kubeconfig and validate the active context.
  try:
    config.load_kube_config()
    contexts, active_context = config.list_kube_config_contexts()

    if active_context:
      active_cluster = active_context.get("context", {}).get("cluster", "")
      expected = f"gke_{project}_{zone}_{cluster_name}"

      if cluster is not None and active_cluster != expected:
        logging.info(
          "Active kubeconfig context '%s' does not match expected "
          "cluster '%s'. Reconfiguring...",
          active_cluster,
          expected,
        )
      else:
        # Either the cluster matches or no specific cluster was
        # requested — accept the current kubeconfig.
        return
    # No active context — fall through to reconfigure.
  except config.ConfigException:
    logging.info("No valid kubeconfig found. Configuring...")

  _configure_kubeconfig(cluster_name, zone, project)


def ensure_docker_auth(zone: str) -> None:
  """Configure Docker auth for Artifact Registry if not already set.

  This is **non-fatal**: Cloud Build uses its own service account, so a
  failure here logs a warning but does not block execution.
  """
  ar_host = f"{zone_to_ar_location(zone)}-docker.pkg.dev"

  docker_config_path = os.path.join(
    os.environ.get("DOCKER_CONFIG", os.path.expanduser("~/.docker")),
    "config.json",
  )
  if os.path.exists(docker_config_path):
    try:
      with open(docker_config_path, "r") as f:
        docker_config = json.load(f)
      if ar_host in docker_config.get("credHelpers", {}):
        return
    except (json.JSONDecodeError, KeyError):
      pass

  logging.info("Docker auth for %s not found. Configuring...", ar_host)
  try:
    subprocess.run(
      ["gcloud", "auth", "configure-docker", ar_host, "--quiet"],
      check=True,
      capture_output=True,
    )
    logging.info("Docker auth configured for %s.", ar_host)
  except subprocess.CalledProcessError as e:
    logging.warning(
      "Failed to configure Docker auth for %s. "
      "Cloud Build may still work with its service account. Error: %s",
      ar_host,
      e,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_kubeconfig(cluster_name: str, zone: str, project: str) -> None:
  """Run ``gcloud container clusters get-credentials``."""
  env = {**os.environ, "USE_GKE_GCLOUD_AUTH_PLUGIN": "True"}
  try:
    subprocess.run(
      [
        "gcloud",
        "container",
        "clusters",
        "get-credentials",
        cluster_name,
        f"--zone={zone}",
        f"--project={project}",
      ],
      check=True,
      env=env,
      capture_output=True,
    )
    logging.info("Kubeconfig configured for cluster '%s'.", cluster_name)
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      f"Failed to configure kubeconfig for cluster '{cluster_name}' "
      f"in zone '{zone}', project '{project}'. "
      f"Ensure the cluster exists and you have access. "
      f"Run manually: gcloud container clusters get-credentials "
      f"{cluster_name} --zone={zone} --project={project}"
    ) from e
