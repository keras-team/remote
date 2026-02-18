"""Typed configuration for keras-remote CLI infrastructure commands."""

from dataclasses import dataclass

from keras_remote.core.accelerators import Accelerator
from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE


@dataclass
class InfraConfig:
    """Configuration for infrastructure provisioning/teardown."""

    project: str
    zone: str = DEFAULT_ZONE
    cluster_name: str = DEFAULT_CLUSTER_NAME
    accelerator: Accelerator = None
