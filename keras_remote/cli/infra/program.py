"""Pulumi inline program for keras-remote infrastructure.

Defines all GCP resources needed for keras-remote: API services,
Artifact Registry, GKE cluster, and optional accelerator node pools.
"""

import pulumi
import pulumi_gcp as gcp

from keras_remote.cli.constants import REQUIRED_APIS, RESOURCE_NAME_PREFIX
from keras_remote.constants import zone_to_ar_location, zone_to_region
from keras_remote.core.accelerators import GpuConfig, TpuConfig

# OAuth scopes required by all node pools (including accelerator pools).
_BASE_OAUTH_SCOPES = [
  # Read/write access to GCS for storing checkpoints, datasets, and logs.
  "https://www.googleapis.com/auth/devstorage.full_control",
  # Write application logs to Cloud Logging.
  "https://www.googleapis.com/auth/logging.write",
  # Export metrics to Cloud Monitoring.
  "https://www.googleapis.com/auth/monitoring",
]

# Additional scopes for the default (system) node pool, which runs GKE
# control-plane components that need deeper platform integration.
_DEFAULT_POOL_OAUTH_SCOPES = _BASE_OAUTH_SCOPES + [
  # Report service status to Google Service Control.
  "https://www.googleapis.com/auth/servicecontrol",
  # Read managed-service configuration from Service Management.
  "https://www.googleapis.com/auth/service.management.readonly",
  # Send distributed traces to Cloud Trace.
  "https://www.googleapis.com/auth/trace.append",
]


def create_program(config):
  """Create a Pulumi inline program function closed over the config.

  Args:
      config: InfraConfig instance.

  Returns:
      A callable suitable for pulumi.automation.create_or_select_stack().
  """

  def pulumi_program():
    project_id = config.project
    zone = config.zone
    ar_location = zone_to_ar_location(zone)
    cluster_name = config.cluster_name
    accelerator = config.accelerator

    # 1. Enable GCP APIs
    enabled_apis = []
    for api in REQUIRED_APIS:
      svc = gcp.projects.Service(
        f"api-{api.split('.')[0]}",
        service=api,
        project=project_id,
        disable_on_destroy=False,
        disable_dependent_services=False,
      )
      enabled_apis.append(svc)

    # 2. Artifact Registry docker repository
    gcp.artifactregistry.Repository(
      "keras-remote-repo",
      repository_id="keras-remote",
      location=ar_location,
      format="DOCKER",
      description="keras-remote container images",
      project=project_id,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 3. Cloud Storage buckets
    region = zone_to_region(zone)

    gcp.storage.Bucket(
      "keras-remote-jobs-bucket",
      name=f"{project_id}-keras-remote-jobs",
      location=region,
      project=project_id,
      force_destroy=True,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    gcp.storage.Bucket(
      "keras-remote-builds-bucket",
      name=f"{project_id}-keras-remote-builds",
      location=ar_location,
      project=project_id,
      force_destroy=True,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 4. GKE Cluster
    cluster = gcp.container.Cluster(
      "keras-remote-cluster",
      name=cluster_name,
      location=zone,
      project=project_id,
      initial_node_count=1,
      remove_default_node_pool=False,
      node_config=gcp.container.ClusterNodeConfigArgs(
        machine_type="e2-standard-4",
        disk_size_gb=50,
        oauth_scopes=_DEFAULT_POOL_OAUTH_SCOPES,
      ),
      # Match setup.sh: --no-enable-autoupgrade
      release_channel=gcp.container.ClusterReleaseChannelArgs(
        channel="UNSPECIFIED",
      ),
      deletion_protection=False,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 5. Accelerator node pool (conditional)
    if isinstance(accelerator, GpuConfig):
      _create_gpu_node_pool(cluster, accelerator, zone, project_id)
    elif isinstance(accelerator, TpuConfig):
      _create_tpu_node_pool(cluster, accelerator, zone, project_id)

    # 6. Stack exports
    pulumi.export("project", project_id)
    pulumi.export("zone", zone)
    pulumi.export("cluster_name", cluster.name)
    pulumi.export("cluster_endpoint", cluster.endpoint)
    pulumi.export(
      "ar_registry",
      f"{ar_location}-docker.pkg.dev/{project_id}/keras-remote",
    )

    # 7. Accelerator node pool exports
    if isinstance(accelerator, GpuConfig):
      pulumi.export(
        "accelerator",
        {
          "type": "GPU",
          "name": accelerator.name,
          "count": accelerator.count,
          "machine_type": accelerator.machine_type,
          "node_pool": "gpu-pool",
          "node_count": 1,
        },
      )
    elif isinstance(accelerator, TpuConfig):
      pulumi.export(
        "accelerator",
        {
          "type": "TPU",
          "name": accelerator.name,
          "chips": accelerator.chips,
          "topology": accelerator.topology,
          "machine_type": accelerator.machine_type,
          "node_pool": f"tpu-{accelerator.name}-pool",
          "node_count": accelerator.num_nodes,
        },
      )
    else:
      pulumi.export("accelerator", None)

  return pulumi_program


def _create_gpu_node_pool(cluster, gpu: GpuConfig, zone, project_id):
  """Create a GPU-accelerated GKE node pool."""
  gcp.container.NodePool(
    "gpu-pool",
    name="gpu-pool",
    cluster=cluster.name,
    location=zone,
    project=project_id,
    node_count=1,
    node_config=gcp.container.NodePoolNodeConfigArgs(
      machine_type=gpu.machine_type,
      oauth_scopes=_BASE_OAUTH_SCOPES,
      guest_accelerators=[
        gcp.container.NodePoolNodeConfigGuestAcceleratorArgs(
          type=gpu.gke_label,
          count=1,
        ),
      ],
      labels={RESOURCE_NAME_PREFIX: "true"},
    ),
  )


def _create_tpu_node_pool(cluster, tpu: TpuConfig, zone, project_id):
  """Create a TPU GKE node pool."""
  pool_name = f"tpu-{tpu.name}-pool"
  # Single-host TPU slices (1 node) must not specify placement_policy;
  # multi-host slices require COMPACT placement with an explicit topology.
  placement = (
    gcp.container.NodePoolPlacementPolicyArgs(
      type="COMPACT",
      tpu_topology=tpu.topology,
    )
    if tpu.num_nodes > 1
    else None
  )
  gcp.container.NodePool(
    pool_name,
    name=pool_name,
    cluster=cluster.name,
    location=zone,
    project=project_id,
    node_count=tpu.num_nodes,
    node_config=gcp.container.NodePoolNodeConfigArgs(
      machine_type=tpu.machine_type,
      oauth_scopes=_BASE_OAUTH_SCOPES,
      labels={RESOURCE_NAME_PREFIX: "true"},
    ),
    placement_policy=placement,
  )
