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

```python
import os
import kinetic

@kinetic.run(
    accelerator="tpu-v5litepod",
    # Capture required environment variables
    capture_env_vars=["HF_TOKEN", "JAX_PLATFORMS"],
)
def run_vllm_inference():
    # Imports must happen inside the decorated function
    from vllm import LLM, SamplingParams

    # Configure environment for vLLM on TPU
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    os.environ["VLLM_USE_V1"] = "0"
    
    # Optional: ensure JAX sees CPU if needed by vLLM internals
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"

    model_id = "meta-llama/Llama-3.1-8B"

    print(f"Initializing vLLM with model: {model_id}")
    # Use tensor_parallel_size matching your TPU chip count (e.g., 4 for v5litepod-4)
    llm = LLM(model=model_id, tensor_parallel_size=4, max_model_len=2048)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    print("Generating completions...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nPrompt: {output.prompt!r}")
        print(f"Generated text: {output.outputs[0].text!r}")

if __name__ == "__main__":
    run_vllm_inference()
```

## Running the Example

To run the script, set the required environment variables locally so they get captured:

```bash
HF_TOKEN=your-hf-token \
python3 your_script.py
```
