"""Default values and constants for the keras-remote CLI."""

import os
import re

from keras_remote.constants import (
  DEFAULT_CLUSTER_NAME,  # noqa: F401 — re-exported
  DEFAULT_ZONE,  # noqa: F401 — re-exported
)

RESOURCE_NAME_PREFIX = "keras-remote"
STATE_DIR = os.environ.get(
  "KERAS_REMOTE_STATE_DIR",
  os.path.expanduser("~/.keras-remote/pulumi"),
)
STATE_BUCKET_SUFFIX = "keras-remote-state"
PULUMI_ROOT = os.path.expanduser("~/.keras-remote/pulumi-cli")
REQUIRED_APIS = [
  "compute.googleapis.com",
  "cloudbuild.googleapis.com",
  "artifactregistry.googleapis.com",
  "storage.googleapis.com",
  "container.googleapis.com",
  "iam.googleapis.com",
]

# Namespace naming constraints (tightest: GCP SA "kr-{name}" max 30 chars)
NAMESPACE_MAX_LENGTH = 27
NAMESPACE_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
RESERVED_NAMESPACES = frozenset(
  {"default", "kube-system", "kube-public", "kube-node-lease"}
)

NVIDIA_DRIVER_DAEMONSET_URL = (
  "https://raw.githubusercontent.com/GoogleCloudPlatform/"
  "container-engine-accelerators/v1.0.20/"
  "nvidia-driver-installer/cos/daemonset-preloaded.yaml"
)

LWS_INSTALL_URL = "https://github.com/kubernetes-sigs/lws/releases/download/v0.5.1/manifests.yaml"

# Autoscaling upper bounds
MAX_CLUSTER_CPU = 1000
MAX_CLUSTER_MEMORY_GB = 64000
NODE_MAX_RUN_DURATION_SECONDS = 24 * 60 * 60  # 24 hours
