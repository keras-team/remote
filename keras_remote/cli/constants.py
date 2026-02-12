"""Default values and constants for the keras-remote CLI."""

import os

from keras_remote.constants import DEFAULT_ZONE  # noqa: F401 — re-exported
RESOURCE_NAME_PREFIX = "keras-remote"
DEFAULT_CLUSTER_NAME = f"{RESOURCE_NAME_PREFIX}-cluster"
STATE_DIR = os.environ.get(
    "KERAS_REMOTE_STATE_DIR",
    os.path.expanduser("~/.keras-remote/pulumi"),
)
PROJECT_NAME = "keras-remote"
STACK_NAME_PREFIX = "keras-remote"

REQUIRED_APIS = [
    "compute.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "container.googleapis.com",
]

NVIDIA_DRIVER_DAEMONSET_URL = (
    "https://raw.githubusercontent.com/GoogleCloudPlatform/"
    "container-engine-accelerators/master/"
    "nvidia-driver-installer/cos/daemonset-preloaded.yaml"
)
