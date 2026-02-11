"""Pulumi inline program for keras-remote infrastructure.

Defines all GCP resources needed for keras-remote: API services,
Artifact Registry, GKE cluster, and optional accelerator node pools.
"""

import pulumi
import pulumi_command as command
import pulumi_gcp as gcp

from keras_remote.cli.constants import REQUIRED_APIS


def create_program(config):
    """Create a Pulumi inline program function closed over the config.

    Args:
        config: Dict with keys: project, zone, cluster_name, accelerator.

    Returns:
        A callable suitable for pulumi.automation.create_or_select_stack().
    """

    def pulumi_program():
        project_id = config["project"]
        zone = config["zone"]
        region = zone.rsplit("-", 1)[0] if "-" in zone else "us-central1"
        ar_location = region.split("-")[0]  # e.g., "us"
        cluster_name = config["cluster_name"]
        accelerator = config.get("accelerator")

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
        ar_repo = gcp.artifactregistry.Repository(
            "keras-remote-repo",
            repository_id="keras-remote",
            location=ar_location,
            format="DOCKER",
            description="keras-remote container images",
            project=project_id,
            opts=pulumi.ResourceOptions(depends_on=enabled_apis),
        )

        # 3. GKE Cluster
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
                oauth_scopes=[
                    "https://www.googleapis.com/auth/devstorage.full_control",
                    "https://www.googleapis.com/auth/logging.write",
                    "https://www.googleapis.com/auth/monitoring",
                    "https://www.googleapis.com/auth/servicecontrol",
                    "https://www.googleapis.com/auth/service.management.readonly",
                    "https://www.googleapis.com/auth/trace.append",
                ],
            ),
            # Match setup.sh: --no-enable-autoupgrade
            release_channel=gcp.container.ClusterReleaseChannelArgs(
                channel="UNSPECIFIED",
            ),
            deletion_protection=False,
            opts=pulumi.ResourceOptions(depends_on=enabled_apis),
        )

        # 4. Accelerator node pool (conditional)
        if accelerator and accelerator.get("category") == "gpu":
            _create_gpu_node_pool(cluster, accelerator, zone, project_id)
        elif accelerator and accelerator.get("category") == "tpu":
            _create_tpu_node_pool(cluster, accelerator, zone, project_id)

        # 5. Stack exports
        pulumi.export("project", project_id)
        pulumi.export("zone", zone)
        pulumi.export("cluster_name", cluster.name)
        pulumi.export("cluster_endpoint", cluster.endpoint)
        pulumi.export(
            "ar_registry",
            f"{ar_location}-docker.pkg.dev/{project_id}/keras-remote",
        )

    return pulumi_program


def _create_gpu_node_pool(cluster, accelerator, zone, project_id):
    """Create a GPU-accelerated GKE node pool."""
    gcp.container.NodePool(
        "gpu-pool",
        name="gpu-pool",
        cluster=cluster.name,
        location=zone,
        project=project_id,
        node_count=1,
        node_config=gcp.container.NodePoolNodeConfigArgs(
            machine_type=accelerator["machine_type"],
            oauth_scopes=[
                "https://www.googleapis.com/auth/devstorage.full_control",
                "https://www.googleapis.com/auth/logging.write",
                "https://www.googleapis.com/auth/monitoring",
            ],
            guest_accelerators=[
                gcp.container.NodePoolNodeConfigGuestAcceleratorArgs(
                    type=accelerator["gpu_type"],
                    count=1,
                ),
            ],
        ),
    )


def _create_tpu_node_pool(cluster, accelerator, zone, project_id):
    """Create a TPU GKE node pool.

    Uses pulumi_command as a fallback if the Pulumi GCP provider does
    not support tpu_topology natively via placement_policy.
    """
    try:
        # Attempt native Pulumi GCP provider with placement_policy
        gcp.container.NodePool(
            accelerator["pool_name"],
            name=accelerator["pool_name"],
            cluster=cluster.name,
            location=zone,
            project=project_id,
            node_count=accelerator["num_nodes"],
            node_config=gcp.container.NodePoolNodeConfigArgs(
                machine_type=accelerator["machine_type"],
                oauth_scopes=[
                    "https://www.googleapis.com/auth/devstorage.full_control",
                    "https://www.googleapis.com/auth/logging.write",
                    "https://www.googleapis.com/auth/monitoring",
                ],
            ),
            placement_policy=gcp.container.NodePoolPlacementPolicyArgs(
                type="COMPACT",
                tpu_topology=accelerator["topology"],
            ),
        )
    except (TypeError, AttributeError):
        # Fallback: use pulumi-command to shell out to gcloud
        pool_name = accelerator["pool_name"]
        machine_type = accelerator["machine_type"]
        topology = accelerator["topology"]
        num_nodes = accelerator["num_nodes"]

        command.local.Command(
            f"tpu-node-pool-{pool_name}",
            create=cluster.name.apply(
                lambda name: (
                    f"gcloud container node-pools create {pool_name} "
                    f"--cluster={name} --zone={zone} --project={project_id} "
                    f"--machine-type={machine_type} "
                    f"--tpu-topology={topology} "
                    f"--num-nodes={num_nodes} "
                    f"--scopes=gke-default,storage-full"
                )
            ),
            delete=cluster.name.apply(
                lambda name: (
                    f"gcloud container node-pools delete {pool_name} "
                    f"--cluster={name} --zone={zone} --project={project_id} "
                    f"--quiet"
                )
            ),
            opts=pulumi.ResourceOptions(depends_on=[cluster]),
        )
