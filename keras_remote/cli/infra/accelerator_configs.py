"""GPU and TPU machine type mappings for GKE node pools.

These configurations replicate the accelerator setup logic from the
original setup.sh script, mapping user-facing accelerator names to
the GCP machine types, GPU labels, and TPU topologies needed for
GKE node pool creation.
"""

GPU_CONFIGS = {
    "t4": {
        "gpu_type": "nvidia-tesla-t4",
        "machine_type": "n1-standard-4",
    },
    "l4": {
        "gpu_type": "nvidia-l4",
        "machine_type": "g2-standard-4",
    },
    "a100": {
        "gpu_type": "nvidia-tesla-a100",
        "machine_type": "a2-highgpu-1g",
    },
    "a100-80gb": {
        "gpu_type": "nvidia-a100-80gb",
        "machine_type": "a2-ultragpu-1g",
    },
    "h100": {
        "gpu_type": "nvidia-h100-80gb",
        "machine_type": "a3-highgpu-1g",
    },
}

TPU_CONFIGS = {
    "v5litepod": {
        "1x1": {"machine_type": "ct5lp-hightpu-1t", "num_nodes": 1},
        "2x2": {"machine_type": "ct5lp-hightpu-4t", "num_nodes": 1},
        "2x4": {"machine_type": "ct5lp-hightpu-8t", "num_nodes": 1},
    },
    "v5p": {
        "2x2": {"machine_type": "ct5p-hightpu-4t", "num_nodes": 2},
        "2x4": {"machine_type": "ct5p-hightpu-4t", "num_nodes": 4},
    },
    "v6e": {
        "2x2": {"machine_type": "ct6e-standard-4t", "num_nodes": 2},
        "2x4": {"machine_type": "ct6e-standard-4t", "num_nodes": 4},
    },
    "v3": {
        "2x2": {"machine_type": "ct3p-hightpu-4t", "num_nodes": 2},
        "4x4": {"machine_type": "ct3p-hightpu-4t", "num_nodes": 8},
    },
}
