# Running vLLM on TPU with Kinetic

This guide explains how to run vLLM (versatile Large Language Model) serving and inference on Cloud TPUs using the Kinetic framework.

## Overview

Kinetic allows you to easily offload heavy vLLM workloads to Cloud TPUs. This is particularly useful for serving large models like Llama 3.1 that require significant compute and memory.

## Prerequisites

1.  **Kinetic Cluster**: You need a provisioned Kinetic cluster with TPU nodes (e.g., `v5litepod`).
2.  **Hugging Face Token**: If you are using gated models like Llama 3.1, you need a Hugging Face token (`HF_TOKEN`) with access to the model.

## Configuration

To run vLLM successfully on TPU via Kinetic, you need to handle dependencies and environment variables properly.

### 1. Dependencies

You need to ensure `vllm-tpu` is installed in the remote container. You can do this by creating a `requirements.txt` file in the directory of your script containing:

```text
vllm-tpu
```

Kinetic will detect this file and build a container with vLLM installed.

### 2. Environment Variables

You must pass the following environment variables to ensure correct execution:

-   `VLLM_TARGET_DEVICE="tpu"`: Tells vLLM to target TPU.
-   `VLLM_USE_V1="0"`: Forces vLLM to use the stable v0 engine (recommended for TPU currently).
-   `JAX_PLATFORMS="tpu,cpu"`: Allows JAX to see both TPU and CPU backends, avoiding initialization crashes.

You can use `capture_env_vars` in the `@kinetic.run` decorator to pass these from your local environment.

## Example

```{literalinclude} ../../examples/vllm_demo.py
:language: python
```

## Running the Example

To run the script, set the required environment variables locally so they get captured:

```bash
HF_TOKEN=your-hf-token \
python3 your_script.py
```
