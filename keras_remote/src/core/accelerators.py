"""Accelerator registry and parsing for keras-remote.

Single source of truth for all accelerator metadata — used by both the
runtime (gke_client, container_builder) and the CLI (up, prompts, program).
"""

import re
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class GpuConfig:
    """Fully resolved GPU accelerator configuration."""

    name: str  # "l4"
    count: int  # number of GPUs (1, 2, 4, …)
    gke_label: str  # "nvidia-l4" — K8s node selector value
    machine_type: str  # "g2-standard-4" — GKE node pool machine type


@dataclass(frozen=True)
class TpuConfig:
    """Fully resolved TPU accelerator configuration."""

    name: str  # "v5litepod"
    chips: int  # number of TPU chips (4, 8, …)
    topology: str  # "2x2" — TPU topology string
    gke_accelerator: str  # "tpu-v5-lite-podslice"
    machine_type: str  # "ct5lp-hightpu-4t"
    num_nodes: int  # GKE node pool node count


Accelerator = Union[GpuConfig, TpuConfig, None]


# Each GPU entry maps to its GKE label, provisioning machine type,
# and the set of supported multi-GPU counts.
GPUS: dict[str, dict] = {
    "l4": {
        "gke_label": "nvidia-l4",
        "machine_type": "g2-standard-4",
        "counts": (1, 2, 4),
    },
    "t4": {
        "gke_label": "nvidia-tesla-t4",
        "machine_type": "n1-standard-4",
        "counts": (1, 2, 4),
    },
    "v100": {
        "gke_label": "nvidia-tesla-v100",
        "machine_type": "n1-standard-8",
        "counts": (1, 2, 4, 8),
    },
    "a100": {
        "gke_label": "nvidia-tesla-a100",
        "machine_type": "a2-highgpu-1g",
        "counts": (1, 2, 4, 8),
    },
    "a100-80gb": {
        "gke_label": "nvidia-a100-80gb",
        "machine_type": "a2-ultragpu-1g",
        "counts": (1, 2, 4, 8),
    },
    "h100": {
        "gke_label": "nvidia-h100-80gb",
        "machine_type": "a3-highgpu-1g",
        "counts": (1, 2, 4, 8),
    },
}

# Reverse lookup: GKE label → canonical name (e.g. "nvidia-l4" → "l4").
_GPU_ALIASES: dict[str, str] = {spec["gke_label"]: name for name, spec in GPUS.items()}

# Each TPU entry contains its GKE accelerator label, default chip count,
# and a mapping of chip count → (topology, machine_type, num_nodes).
TPUS: dict[str, dict] = {
    "v2": {
        "gke_accelerator": "tpu-v2-podslice",
        "default_chips": 8,
        "topologies": {
            8: ("2x2", "ct2-hightpu-4t", 2),
            32: ("4x4", "ct2-hightpu-4t", 8),
        },
    },
    "v3": {
        "gke_accelerator": "tpu-v3-podslice",
        "default_chips": 8,
        "topologies": {
            8: ("2x2", "ct3p-hightpu-4t", 2),
            32: ("4x4", "ct3p-hightpu-4t", 8),
        },
    },
    "v5litepod": {
        "gke_accelerator": "tpu-v5-lite-podslice",
        "default_chips": 4,
        "topologies": {
            1: ("1x1", "ct5lp-hightpu-1t", 1),
            4: ("2x2", "ct5lp-hightpu-4t", 1),
            8: ("2x4", "ct5lp-hightpu-8t", 1),
        },
    },
    "v5p": {
        "gke_accelerator": "tpu-v5p-slice",
        "default_chips": 8,
        "topologies": {
            8: ("2x2", "ct5p-hightpu-4t", 2),
            16: ("2x4", "ct5p-hightpu-4t", 4),
        },
    },
    "v6e": {
        "gke_accelerator": "tpu-v6e-slice",
        "default_chips": 8,
        "topologies": {
            8: ("2x2", "ct6e-standard-4t", 2),
            16: ("2x4", "ct6e-standard-4t", 4),
        },
    },
}


# ── Parser ────────────────────────────────────────────────────────

_MULTI_GPU_RE = re.compile(r"^(.+?)x(\d+)$")  # "a100x4"
_TPU_CHIPS_RE = re.compile(r"^(v\d+\w*)-(\d+)$")  # "v3-8"
_TPU_TOPO_RE = re.compile(r"^(v\d+\w*)-(\d+x\d+)$")  # "v5litepod-2x2"


def parse_accelerator(accel_str: str) -> Accelerator:
    """Parse an accelerator string into a fully resolved config.

    Returns GpuConfig, TpuConfig, or None (for "cpu").

    Accepted formats:
        GPU:  "l4", "nvidia-l4", "a100x4", "a100-80gbx8"
        TPU:  "v3-8" (chip count), "v5litepod-2x2" (topology), "v5litepod" (default)
        CPU:  "cpu"
    """
    s = accel_str.strip().lower()

    if s == "cpu":
        return None

    # Direct GPU name: "l4", "a100-80gb"
    if s in GPUS:
        return _make_gpu(s, 1)

    # GPU alias: "nvidia-l4"
    if s in _GPU_ALIASES:
        return _make_gpu(_GPU_ALIASES[s], 1)

    # Multi-GPU: "a100x4", "l4x2"
    m = _MULTI_GPU_RE.match(s)
    if m:
        name = m.group(1)
        if name in GPUS:
            return _make_gpu(name, int(m.group(2)))
        if name in _GPU_ALIASES:
            return _make_gpu(_GPU_ALIASES[name], int(m.group(2)))

    # Direct TPU name (bare): "v5litepod" → default chips
    if s in TPUS:
        return _make_tpu(s, TPUS[s]["default_chips"])

    # TPU with topology string: "v5litepod-2x2"
    m = _TPU_TOPO_RE.match(s)
    if m and m.group(1) in TPUS:
        name = m.group(1)
        topo_str = m.group(2)
        for chips, (topo, _, _) in TPUS[name]["topologies"].items():
            if topo == topo_str:
                return _make_tpu(name, chips)

    # TPU with chip count: "v3-8", "v5litepod-4"
    m = _TPU_CHIPS_RE.match(s)
    if m and m.group(1) in TPUS:
        return _make_tpu(m.group(1), int(m.group(2)))

    raise ValueError(
        f"Unknown accelerator: '{accel_str}'. "
        f"GPUs: {', '.join(GPUS)} (use 'xN' for multi-GPU, e.g. 'a100x4'). "
        f"TPUs: {', '.join(TPUS)} (use '-N' for chips, e.g. 'v3-8', "
        f"or '-NxM' for topology, e.g. 'v5litepod-2x2')."
    )


def get_category(accel_str: str) -> str:
    """Return 'cpu', 'gpu', or 'tpu' for the given accelerator string."""
    result = parse_accelerator(accel_str)
    if result is None:
        return "cpu"
    if isinstance(result, GpuConfig):
        return "gpu"
    return "tpu"


def _make_gpu(name: str, count: int) -> GpuConfig:
    spec = GPUS[name]
    if count not in spec["counts"]:
        raise ValueError(
            f"GPU count {count} not supported for '{name}'. "
            f"Supported: {', '.join(str(c) for c in spec['counts'])}."
        )
    return GpuConfig(
        name=name,
        count=count,
        gke_label=spec["gke_label"],
        machine_type=spec["machine_type"],
    )


def _make_tpu(name: str, chips: int) -> TpuConfig:
    spec = TPUS[name]
    if chips not in spec["topologies"]:
        raise ValueError(
            f"Chip count {chips} not supported for '{name}'. "
            f"Supported: {', '.join(str(c) for c in spec['topologies'])}."
        )
    topo, machine_type, num_nodes = spec["topologies"][chips]
    return TpuConfig(
        name=name,
        chips=chips,
        topology=topo,
        gke_accelerator=spec["gke_accelerator"],
        machine_type=machine_type,
        num_nodes=num_nodes,
    )
