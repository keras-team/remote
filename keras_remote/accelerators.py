"""Canonical accelerator type registry and parsing utilities.

Single source of truth for accelerator metadata used by all backends
(GKE, Vertex AI, TPU VM) and the container builder.
"""

import re
from dataclasses import dataclass, field


@dataclass
class AcceleratorType:
    """Metadata for a single accelerator type."""

    short_name: str
    category: str  # "gpu", "tpu", or "cpu"
    aliases: tuple = ()

    # GPU — GKE node selector label (e.g., "nvidia-l4")
    gke_label: str = ""
    # GPU — Vertex AI accelerator enum (e.g., "NVIDIA_L4")
    vertex_type: str = ""
    # GPU — count → Vertex AI machine type (e.g., {1: "g2-standard-4", 2: ...})
    vertex_machines: dict = field(default_factory=dict)

    # TPU — default chip count when bare name is used (e.g., "v3" → 8 chips)
    default_chips: int = 0
    # TPU — Vertex AI accelerator enum (e.g., "TPU_V3"), empty if implicit
    vertex_tpu_type: str = ""
    # TPU — Vertex machine type template (e.g., "cloud-tpu" or "ct5lp-hightpu-{chips}t")
    vertex_tpu_template: str = ""
    # TPU — GKE node selector label (e.g., "tpu-v5-lite-podslice")
    gke_tpu_accelerator: str = ""
    # TPU — chip count → GKE topology (e.g., {8: "2x2", 32: "4x4"})
    gke_tpu_topologies: dict = field(default_factory=dict)


@dataclass
class ParsedAccelerator:
    """Result of parsing an accelerator string."""

    accelerator_type: AcceleratorType
    count: int  # GPU count or TPU chip count (0 for CPU)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CPU_TYPE = AcceleratorType(short_name="cpu", category="cpu")

_ACCELERATOR_TYPES = [
    AcceleratorType(
        short_name="l4",
        category="gpu",
        aliases=("nvidia-l4",),
        gke_label="nvidia-l4",
        vertex_type="NVIDIA_L4",
        vertex_machines={1: "g2-standard-4", 2: "g2-standard-24", 4: "g2-standard-48"},
    ),
    AcceleratorType(
        short_name="t4",
        category="gpu",
        aliases=("nvidia-tesla-t4",),
        gke_label="nvidia-tesla-t4",
        vertex_type="NVIDIA_TESLA_T4",
        vertex_machines={1: "n1-standard-4", 2: "n1-standard-8", 4: "n1-standard-16"},
    ),
    AcceleratorType(
        short_name="v100",
        category="gpu",
        aliases=("nvidia-tesla-v100",),
        gke_label="nvidia-tesla-v100",
        vertex_type="NVIDIA_TESLA_V100",
        vertex_machines={
            1: "n1-standard-8",
            2: "n1-standard-16",
            4: "n1-standard-32",
            8: "n1-standard-64",
        },
    ),
    AcceleratorType(
        short_name="a100",
        category="gpu",
        aliases=("nvidia-tesla-a100",),
        gke_label="nvidia-tesla-a100",
        vertex_type="NVIDIA_TESLA_A100",
        vertex_machines={
            1: "a2-highgpu-1g",
            2: "a2-highgpu-2g",
            4: "a2-highgpu-4g",
            8: "a2-highgpu-8g",
        },
    ),
    AcceleratorType(
        short_name="a100-80gb",
        category="gpu",
        aliases=("nvidia-a100-80gb",),
        gke_label="nvidia-a100-80gb",
        vertex_type="NVIDIA_A100_80GB",
        vertex_machines={
            1: "a2-ultragpu-1g",
            2: "a2-ultragpu-2g",
            4: "a2-ultragpu-4g",
            8: "a2-ultragpu-8g",
        },
    ),
    AcceleratorType(
        short_name="h100",
        category="gpu",
        aliases=("nvidia-h100-80gb",),
        gke_label="nvidia-h100-80gb",
        vertex_type="NVIDIA_H100_80GB",
        vertex_machines={
            1: "a3-highgpu-1g",
            2: "a3-highgpu-2g",
            4: "a3-highgpu-4g",
            8: "a3-highgpu-8g",
        },
    ),
    AcceleratorType(
        short_name="p100",
        category="gpu",
        aliases=("nvidia-tesla-p100",),
        gke_label="nvidia-tesla-p100",
        vertex_type="NVIDIA_TESLA_P100",
        vertex_machines={1: "n1-standard-4"},
    ),
    AcceleratorType(
        short_name="p4",
        category="gpu",
        aliases=("nvidia-tesla-p4",),
        gke_label="nvidia-tesla-p4",
        vertex_type="NVIDIA_TESLA_P4",
        vertex_machines={1: "n1-standard-4"},
    ),
    AcceleratorType(
        short_name="k80",
        category="gpu",
        aliases=("nvidia-tesla-k80",),
        gke_label="nvidia-tesla-k80",
        vertex_type="NVIDIA_TESLA_K80",
        vertex_machines={1: "n1-standard-4"},
    ),
    # --- TPU types ---
    AcceleratorType(
        short_name="v2",
        category="tpu",
        default_chips=8,
        vertex_tpu_type="TPU_V2",
        vertex_tpu_template="cloud-tpu",
        gke_tpu_accelerator="tpu-v2-podslice",
        gke_tpu_topologies={8: "2x2", 32: "4x4"},
    ),
    AcceleratorType(
        short_name="v3",
        category="tpu",
        default_chips=8,
        vertex_tpu_type="TPU_V3",
        vertex_tpu_template="cloud-tpu",
        gke_tpu_accelerator="tpu-v3-podslice",
        gke_tpu_topologies={8: "2x2", 32: "4x4"},
    ),
    AcceleratorType(
        short_name="v5litepod",
        category="tpu",
        default_chips=4,
        vertex_tpu_template="ct5lp-hightpu-{chips}t",
        gke_tpu_accelerator="tpu-v5-lite-podslice",
        gke_tpu_topologies={1: "1x1", 4: "2x2", 8: "2x4"},
    ),
    AcceleratorType(
        short_name="v5p",
        category="tpu",
        default_chips=8,
        vertex_tpu_type="TPU_V5_LITEPOD",
        vertex_tpu_template="cloud-tpu",
        gke_tpu_accelerator="tpu-v5p-slice",
        gke_tpu_topologies={8: "2x2", 16: "2x4"},
    ),
    AcceleratorType(
        short_name="v6e",
        category="tpu",
        default_chips=8,
        vertex_tpu_type="TPU_V6E",
        vertex_tpu_template="cloud-tpu",
        gke_tpu_accelerator="tpu-v6e-slice",
        gke_tpu_topologies={8: "2x2", 16: "2x4"},
    ),
]

# Flat lookup: maps every recognized name (lowercased) to its AcceleratorType.
_LOOKUP: dict[str, AcceleratorType] = {}
for _accel in _ACCELERATOR_TYPES:
    _LOOKUP[_accel.short_name] = _accel
    for _alias in _accel.aliases:
        _LOOKUP[_alias.lower()] = _accel

_MULTI_GPU_RE = re.compile(r"^(.+?)x(\d+)$")  # "a100x4", "a100-80gbx8"
_TPU_CHIPS_RE = re.compile(r"^(v\d+\w*)-(\d+)$")  # "v3-8", "v5litepod-4"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_accelerator(accel_str: str) -> ParsedAccelerator:
    """Parse an accelerator string into a canonical type and count.

    Handles: "cpu", "l4", "nvidia-l4", "a100x4", "a100-80gbx8", "v3-8", "v5litepod-4"
    """
    normalized = accel_str.strip().lower()

    if normalized == "cpu":
        return ParsedAccelerator(CPU_TYPE, 0)

    # Direct lookup ("l4", "nvidia-l4", "a100", "v3", ...)
    if normalized in _LOOKUP:
        t = _LOOKUP[normalized]
        return ParsedAccelerator(t, t.default_chips if t.category == "tpu" else 1)

    # Multi-GPU: "a100x4", "l4x2", "a100-80gbx8"
    m = _MULTI_GPU_RE.match(normalized)
    if m and m.group(1) in _LOOKUP:
        return ParsedAccelerator(_LOOKUP[m.group(1)], int(m.group(2)))

    # TPU chips: "v3-8", "v5litepod-4", "v6e-16"
    m = _TPU_CHIPS_RE.match(normalized)
    if m and m.group(1) in _LOOKUP:
        return ParsedAccelerator(_LOOKUP[m.group(1)], int(m.group(2)))

    gpu_names = ", ".join(a.short_name for a in _ACCELERATOR_TYPES if a.category == "gpu")
    tpu_names = ", ".join(a.short_name for a in _ACCELERATOR_TYPES if a.category == "tpu")
    raise ValueError(
        f"Unsupported accelerator: '{accel_str}'. "
        f"GPUs: {gpu_names} (use 'xN' for multi-GPU, e.g. 'a100x4'). "
        f"TPUs: {tpu_names} (use '-N' for chip count, e.g. 'v3-8')."
    )


def get_category(accel_str: str) -> str:
    """Return 'cpu', 'tpu', or 'gpu' for the given accelerator string."""
    return parse_accelerator(accel_str).accelerator_type.category
