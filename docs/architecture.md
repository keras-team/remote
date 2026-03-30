# Architecture Overview

Kinetic automates the process of running Python functions on Google Cloud Platform (GCP) accelerators. It handles packaging, infrastructure provisioning, and execution management to provide a seamless experience for remote workloads.

## Execution Lifecycle

When a function decorated with `@kinetic.run()` or `@kinetic.submit()` is called, the system follows these steps:

1.  **Context Resolution**: Kinetic aggregates function parameters, environment variables, and local configurations into a unified `JobContext`.
2.  **Credential Validation**: The system verifies active `gcloud` and `kubectl` credentials, performing automatic configuration where necessary to ensure access to GCP services.
3.  **Artifact Preparation**:
    *   **Data Dependencies**: Local data paths are hashed and uploaded to Google Cloud Storage (GCS) if they are not already present in the content-addressed cache.
    *   **Function Serialization**: The decorated function and its closure are serialized using `cloudpickle`.
    *   **Project Packaging**: The local working directory is compressed into a ZIP archive, automatically excluding paths managed by the Data API.
4.  **Container Image Management**: Kinetic generates a hash of project dependencies (e.g., `requirements.txt` or `pyproject.toml`). If a corresponding image does not exist in Artifact Registry, Kinetic initiates a Cloud Build job to create it.
5.  **Job Submission**: Based on the requested accelerator type, Kinetic submits a Kubernetes Job (for GKE) or a LeaderWorkerSet (for multi-host Pathways) to the target cluster.
6.  **Remote Execution**: The remote pod pulls the container image, retrieves the serialized artifacts, mounts the required data volumes, and executes the function.
7.  **Result Retrieval**: Upon completion, the function's return value is retrieved from GCS, deserialized, and returned to the local Python process.

## Core Modules

### Runtime (`kinetic.core`)
The primary interface for users. It provides the `@run()` and `@submit()` decorators, manages job initialization, and coordinates with the backends.

### Data API (`kinetic.data`)
A system for managing data dependencies. It uses content-addressed storage to optimize uploads and handles the mapping of GCS URIs or local paths to the remote filesystem.

### Backends (`kinetic.backend`)
Kinetic supports two execution backends:
*   **GKE**: Used for standard CPU, single-GPU, and single-host TPU workloads. It manages standard Kubernetes Jobs and streams logs to the local console.
*   **Pathways**: Used for multi-host TPU topologies. It leverages the LeaderWorkerSet operator to manage distributed JAX runtimes across multiple nodes.

### Container Management (`kinetic.infra`)
Handles the creation of reproducible execution environments using Google Cloud Build. It ensures that the remote container environment satisfies all local code dependencies.

### Remote Runner (`kinetic.runner`)
The entry point script executed within the remote container. It is responsible for setting up the environment, resolving data mounts, and executing the serialized function payload.

## Backend Selection Logic

Kinetic automatically determines the appropriate backend based on the requested hardware:
*   **GKE Backend**: Selected for CPU, any GPU configuration, or single-node TPU slices (e.g., `v6e-8`).
*   **Pathways Backend**: Selected for multi-node TPU configurations (e.g., `v2-16`, `v3-32`, `v6e-2x4`) to support distributed execution.

Users can manually specify a backend by passing the `backend` parameter to the decorator.
