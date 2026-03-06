"""Typed configuration for keras-remote CLI infrastructure commands."""

from dataclasses import dataclass, field
from typing import Optional, Union

from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from keras_remote.core.accelerators import GpuConfig, TpuConfig


@dataclass
class NodePoolConfig:
  """A named accelerator node pool."""

  name: str  # GKE node pool name, e.g. "gpu-l4-a3f2"
  accelerator: Union[GpuConfig, TpuConfig]


@dataclass
class NamespaceConfig:
  """Per-namespace isolation configuration."""

  name: str
  members: list[str] = field(default_factory=list)
  gpus: Optional[int] = None
  tpus: Optional[int] = None
  cpu: Optional[int] = None
  memory: Optional[str] = None
  max_jobs: Optional[int] = None
  max_lws: Optional[int] = None


@dataclass
class InfraConfig:
  """Configuration for infrastructure provisioning/teardown."""

  project: str
  zone: str = DEFAULT_ZONE
  cluster_name: str = DEFAULT_CLUSTER_NAME
  node_pools: list[NodePoolConfig] = field(default_factory=list)
  namespaces: list[NamespaceConfig] = field(default_factory=list)
