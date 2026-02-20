"""Tests for keras_remote.core.accelerators — parser, registry, categories."""

import pytest

from keras_remote.core.accelerators import (
  _GPU_ALIASES,
  GPUS,
  TPUS,
  GpuConfig,
  TpuConfig,
  get_category,
  parse_accelerator,
)


class TestParseGpuDirect:
  def test_l4(self):
    result = parse_accelerator("l4")
    assert isinstance(result, GpuConfig)
    assert result.name == "l4"
    assert result.count == 1
    assert result.gke_label == "nvidia-l4"
    assert result.machine_type == "g2-standard-4"

  @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
  def test_all_gpu_types_parse_with_count_1(self, gpu_name):
    result = parse_accelerator(gpu_name)
    assert isinstance(result, GpuConfig)
    assert result.count == 1
    assert result.name == gpu_name


class TestParseGpuMultiCount:
  def test_a100x4(self):
    result = parse_accelerator("a100x4")
    assert isinstance(result, GpuConfig)
    assert result.name == "a100"
    assert result.count == 4

  def test_a100_80gbx4(self):
    result = parse_accelerator("a100-80gbx4")
    assert isinstance(result, GpuConfig)
    assert result.name == "a100-80gb"
    assert result.count == 4


class TestParseGpuAlias:
  def test_nvidia_tesla_t4(self):
    result = parse_accelerator("nvidia-tesla-t4")
    assert isinstance(result, GpuConfig)
    assert result.name == "t4"
    assert result.count == 1

  def test_nvidia_tesla_v100x4(self):
    result = parse_accelerator("nvidia-tesla-v100x4")
    assert isinstance(result, GpuConfig)
    assert result.name == "v100"
    assert result.count == 4


class TestParseGpuErrors:
  def test_l4x8_invalid_count(self):
    with pytest.raises(ValueError, match="not supported"):
      parse_accelerator("l4x8")


class TestParseTpuBare:
  def test_v5litepod(self):
    result = parse_accelerator("v5litepod")
    assert isinstance(result, TpuConfig)
    assert result.name == "v5litepod"
    assert result.chips == 4
    assert result.topology == "2x2"

  @pytest.mark.parametrize("tpu_name", list(TPUS.keys()))
  def test_all_tpu_types_parse_with_default_chips(self, tpu_name):
    result = parse_accelerator(tpu_name)
    assert isinstance(result, TpuConfig)
    assert result.name == tpu_name
    assert result.chips == TPUS[tpu_name].default_chips


class TestParseTpuChipCount:
  def test_v3_8(self):
    result = parse_accelerator("v3-8")
    assert isinstance(result, TpuConfig)
    assert result.name == "v3"
    assert result.chips == 8
    assert result.topology == "2x2"

  def test_v3_32(self):
    result = parse_accelerator("v3-32")
    assert isinstance(result, TpuConfig)
    assert result.name == "v3"
    assert result.chips == 32
    assert result.topology == "4x4"

  def test_v5litepod_1(self):
    result = parse_accelerator("v5litepod-1")
    assert isinstance(result, TpuConfig)
    assert result.chips == 1
    assert result.topology == "1x1"


class TestParseTpuTopology:
  def test_v5litepod_2x2(self):
    result = parse_accelerator("v5litepod-2x2")
    assert isinstance(result, TpuConfig)
    assert result.name == "v5litepod"
    assert result.chips == 4
    assert result.topology == "2x2"

  def test_v5litepod_1x1(self):
    result = parse_accelerator("v5litepod-1x1")
    assert isinstance(result, TpuConfig)
    assert result.chips == 1
    assert result.topology == "1x1"


class TestParseTpuErrors:
  def test_v3_16_invalid_chips(self):
    with pytest.raises(ValueError, match="not supported"):
      parse_accelerator("v3-16")

  def test_v5litepod_3x3_invalid_topology(self):
    with pytest.raises(ValueError, match="Unknown accelerator"):
      parse_accelerator("v5litepod-3x3")


class TestParseTpuConfigFields:
  def test_v3_8_full_config(self):
    result = parse_accelerator("v3-8")
    assert result.gke_accelerator == "tpu-v3-podslice"
    assert result.machine_type == "ct3p-hightpu-4t"
    assert result.num_nodes == 2


class TestParseCpu:
  def test_cpu(self):
    assert parse_accelerator("cpu") is None


class TestParseNormalizationAndErrors:
  def test_whitespace_and_case(self):
    result = parse_accelerator("  A100X4  ")
    assert isinstance(result, GpuConfig)
    assert result.name == "a100"
    assert result.count == 4

  def test_empty_string(self):
    with pytest.raises(ValueError, match="Unknown accelerator"):
      parse_accelerator("")

  def test_unknown_accelerator(self):
    with pytest.raises(ValueError, match="Unknown accelerator"):
      parse_accelerator("unknown")


class TestGetCategory:
  def test_cpu(self):
    assert get_category("cpu") == "cpu"

  def test_gpu(self):
    assert get_category("l4") == "gpu"

  def test_tpu(self):
    assert get_category("v5litepod") == "tpu"


class TestRegistryIntegrity:
  def test_all_gpus_have_nonempty_counts(self):
    for name, spec in GPUS.items():
      assert len(spec.counts) > 0, f"GPU '{name}' has empty counts"

  def test_all_tpus_have_nonempty_topologies(self):
    for name, spec in TPUS.items():
      assert len(spec.topologies) > 0, f"TPU '{name}' has empty topologies"

  def test_all_tpu_default_chips_valid(self):
    for name, spec in TPUS.items():
      assert spec.default_chips in spec.topologies, (
        f"TPU '{name}' default_chips={spec.default_chips} "
        f"not in topologies {list(spec.topologies.keys())}"
      )

  def test_all_gpus_have_gke_label_alias(self):
    for name, spec in GPUS.items():
      assert spec.gke_label in _GPU_ALIASES, (
        f"GPU '{name}' gke_label '{spec.gke_label}' not in aliases"
      )
      assert _GPU_ALIASES[spec.gke_label] == name
