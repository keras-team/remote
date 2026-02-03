# Keras Remote

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Run Keras and JAX workloads on cloud TPUs and GPUs with a simple decorator. No infrastructure management required.

```python
import keras_remote

@keras_remote.run(accelerator="v3-8")
def train_model():
    import keras
    model = keras.Sequential([...])
    model.fit(x_train, y_train)
    return model.history.history["loss"][-1]

# Executes on TPU v3-8, returns the result
final_loss = train_model()
```

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Backends](#backends)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Supported Accelerators](#supported-accelerators)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Simple decorator API** — Add `@keras_remote.run()` to any function to execute it remotely
- **Automatic infrastructure** — No manual VM provisioning or teardown required
- **Result serialization** — Functions return actual values, not just logs
- **Multiple backends** — Choose between Vertex AI (managed), GKE, or direct TPU VM
- **Container caching** — Subsequent runs start in 2-4 minutes after initial build
- **Built-in monitoring** — View job status and logs in Google Cloud Console
- **Automatic cleanup** — Resources are released when jobs complete

## Installation

### From Source

```bash
git clone https://github.com/keras-team/keras-remote.git
cd keras-remote
pip install -e .
```

### Requirements

- Python 3.11+
- Google Cloud SDK (`gcloud`)
  - Run `gcloud auth login` and `gcloud auth application-default login`
- A Google Cloud project with billing enabled

## Quick Start

### 1. Configure Google Cloud

Run the automated setup script:

```bash
./setup.sh
```

The script will:

- Prompt for your GCP project ID
- Enable required APIs (Vertex AI, Cloud Build, Artifact Registry, Cloud Storage)
- Create the Artifact Registry repository
- Configure Docker authentication
- Verify the setup

### 2. Set Environment Variables

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export KERAS_REMOTE_PROJECT="your-project-id"
export KERAS_REMOTE_ZONE="us-central1-a"  # Optional
```

### 3. Run Your First Job

```python
import keras_remote

@keras_remote.run(accelerator="v3-8")
def hello_tpu():
    import jax
    return f"Running on {jax.devices()}"

result = hello_tpu()
print(result)
```

## Backends

Keras Remote supports three execution backends:

1. Vertex AI
2. Google Kubernetes Engine (GKE)
3. TPU VM

### Vertex AI

Fully managed infrastructure with automatic provisioning and cleanup.

```python
@keras_remote.run(accelerator="v3-8", backend="vertex-ai")
def train():
    ...
```

**Advantages:**

- Automatic resource cleanup
- Built-in monitoring and logging
- No infrastructure management

**Setup:** Run `./setup.sh` and select Vertex AI

### GKE

Execute workloads on existing Google Kubernetes Engine clusters with GPU support.

```python
@keras_remote.run(accelerator="l4", backend="gke")
def train():
    ...
```

**Advantages:**

- Use existing GKE infrastructure
- Support for GPU accelerators (T4, L4, A100, V100, H100)
- Lower overhead than Vertex AI

**Setup:** Run `./setup.sh` and select GKE.

### TPU VM

Direct TPU VM provisioning for maximum control.

```python
@keras_remote.run(accelerator="v3-8", backend="tpu-vm")
def train():
    ...
```

**Advantages:**

- Fastest warm start times
- Lowest cost
- Direct VM access

**Note:** This backend requires manual resource cleanup and does not support result serialization.

## Usage Examples

### Basic Computation

```python
import keras_remote

@keras_remote.run(accelerator="v3-8")
def compute(x, y):
    return x + y

result = compute(5, 7)
print(f"Result: {result}")  # Output: Result: 12
```

### Keras Model Training

```python
import keras_remote

@keras_remote.run(accelerator="v3-8")
def train_model():
    import keras
    import numpy as np

    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(10,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    x_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000, 1)

    history = model.fit(x_train, y_train, epochs=5, verbose=0)
    return history.history["loss"][-1]

final_loss = train_model()
print(f"Final loss: {final_loss}")
```

### Custom Dependencies

Create a `requirements.txt` in your project directory:

```text
tensorflow-datasets
pillow
scikit-learn
```

Keras Remote automatically detects and installs dependencies on the remote worker.

### Prebuilt Container Images

Skip container build time by using prebuilt images:

```python
@keras_remote.run(
    accelerator="v3-8",
    container_image="us-docker.pkg.dev/my-project/keras-remote/prebuilt:v1.0"
)
def train():
    ...
```

See [examples/Dockerfile.prebuilt](examples/Dockerfile.prebuilt) for a template.

## Configuration

### Environment Variables

| Variable               | Required | Default         | Description                        |
| ---------------------- | -------- | --------------- | ---------------------------------- |
| `KERAS_REMOTE_PROJECT` | Yes      | —               | Google Cloud project ID            |
| `KERAS_REMOTE_ZONE`    | No       | `us-central1-a` | Default compute zone               |
| `KERAS_REMOTE_CLUSTER` | No       | —               | GKE cluster name (for GKE backend) |

### Decorator Parameters

```python
@keras_remote.run(
    accelerator="v3-8",        # Required: TPU/GPU type
    backend="vertex-ai",       # "vertex-ai", "gke", or "tpu-vm"
    container_image=None,      # Custom container URI
    zone=None,                 # Override default zone
    project=None,              # Override default project
    vm_name=None,              # VM name (tpu-vm backend)
    cluster=None,              # GKE cluster name
    namespace="default"        # Kubernetes namespace (gke backend)
)
```

## Supported Accelerators

### TPUs (Vertex AI and TPU VM backends)

| Type           | Configurations                              |
| -------------- | ------------------------------------------- |
| TPU v2         | `v2-8`, `v2-32`                             |
| TPU v3         | `v3-8`, `v3-32`                             |
| TPU v5 Litepod | `v5litepod-1`, `v5litepod-4`, `v5litepod-8` |
| TPU v5p        | `v5p-8`, `v5p-16`                           |
| TPU v6e        | `v6e-8`, `v6e-16`                           |

### GPUs (Vertex AI and GKE backends)

| Type        | Aliases                     |
| ----------- | --------------------------- |
| NVIDIA T4   | `t4`, `nvidia-tesla-t4`     |
| NVIDIA L4   | `l4`, `nvidia-l4`           |
| NVIDIA V100 | `v100`, `nvidia-tesla-v100` |
| NVIDIA A100 | `a100`, `nvidia-tesla-a100` |
| NVIDIA H100 | `h100`, `nvidia-h100-80gb`  |

For multi-GPU configurations on GKE, append the count: `a100x4`, `l4x2`, etc.

## Monitoring

### Google Cloud Console

- **Vertex AI Jobs:** [console.cloud.google.com/vertex-ai/training/custom-jobs](https://console.cloud.google.com/vertex-ai/training/custom-jobs)
- **Cloud Build:** [console.cloud.google.com/cloud-build/builds](https://console.cloud.google.com/cloud-build/builds)
- **GKE Workloads:** [console.cloud.google.com/kubernetes/workload](https://console.cloud.google.com/kubernetes/workload)

### Command Line

```bash
# List Vertex AI jobs
gcloud ai custom-jobs list --region=us-central1 --project=$KERAS_REMOTE_PROJECT

# View job logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# List GKE jobs
kubectl get jobs -n default
```

## Troubleshooting

### Common Issues

#### "Project must be specified" error

```bash
export KERAS_REMOTE_PROJECT="your-project-id"
```

#### "404 Requested entity was not found" error

Enable required APIs and create the Artifact Registry repository:

```bash
gcloud services enable aiplatform.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com \
    --project=$KERAS_REMOTE_PROJECT

gcloud artifacts repositories create keras-remote \
    --repository-format=docker \
    --location=us \
    --project=$KERAS_REMOTE_PROJECT
```

#### Permission denied errors

Grant required IAM roles:

```bash
gcloud projects add-iam-policy-binding $KERAS_REMOTE_PROJECT \
    --member="user:your-email@example.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $KERAS_REMOTE_PROJECT \
    --member="user:your-email@example.com" \
    --role="roles/storage.admin"
```

#### Container build failures

Check Cloud Build logs:

```bash
gcloud builds list --project=$KERAS_REMOTE_PROJECT --limit=5
```

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Verify Setup

```bash
# Check authentication
gcloud auth list

# Check project
echo $KERAS_REMOTE_PROJECT

# Check APIs
gcloud services list --enabled --project=$KERAS_REMOTE_PROJECT \
    | grep -E "(aiplatform|cloudbuild|artifactregistry|storage)"

# Check Artifact Registry
gcloud artifacts repositories describe keras-remote \
    --location=us --project=$KERAS_REMOTE_PROJECT
```

## Resource Cleanup

Remove all Keras Remote resources to avoid charges:

```bash
./cleanup.sh
```

This removes:

- Cloud Storage buckets
- Artifact Registry repositories
- Running Vertex AI jobs
- GKE clusters (if created by setup)
- TPU VMs

## Documentation

- [Vertex AI Implementation Guide](VERTEX_AI_IMPLEMENTATION.md) — Technical architecture details
- [Prebuilt Images](examples/Dockerfile.prebuilt) — Custom container templates

## Contributing

Contributions are welcome. Please read our contributing guidelines before submitting pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Maintained by the Keras team at Google.**

- [Report Issues](https://github.com/keras-team/keras-remote/issues)
- [Discussions](https://github.com/keras-team/keras-remote/discussions)
