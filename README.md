# Keras Remote

Run Keras models remotely on TPU as seamlessly as running the same code locally.

## Prerequisites

1.  **Google Cloud SDK**: Install the `gcloud` CLI.
2.  **Authentication**: Run `gcloud auth login` and `gcloud auth application-default login`.
3.  **Permissions**: Ensure your GCP user has permissions to create and manage TPU VMs.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Demo

The `demo_train.py` script demonstrates how to run a Keras model on a remote TPU.

```bash
# Optional: Set your GCP project and zone
export KERAS_REMOTE_PROJECT="your-project-id"
export KERAS_REMOTE_ZONE="us-central1-f"  # or other zones like europe-west4-a

python demo_train.py
```

**Note:** TPU availability varies by zone. If you encounter a `RESOURCE_EXHAUSTED` error, try a different zone or accelerator type (e.g., `v2-8` vs `v3-8`).

The `@keras_remote.run` decorator handles:
1.  Packaging your local code.
2.  Provisioning (or finding) a TPU VM.
3.  Uploading your code and dependencies.
4.  Executing the function inside a Docker container on the TPU VM.