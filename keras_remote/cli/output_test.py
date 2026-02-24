"""Tests for keras_remote.cli.output — infrastructure_state display."""

from unittest import mock

from absl.testing import absltest
from rich.console import Console

from keras_remote.cli import output


def _make_output(value):
  """Create a mock Pulumi OutputValue."""
  ov = mock.MagicMock()
  ov.value = value
  return ov


def _render(outputs):
  """Render infrastructure_state to a plain string."""
  buf = Console(file=None, force_terminal=False, width=120)
  with mock.patch.object(output, "console", buf):
    output.infrastructure_state(outputs)
  return buf.file.getvalue() if hasattr(buf.file, "getvalue") else ""


def _render_text(outputs):
  """Render infrastructure_state and capture output as text."""
  from io import StringIO

  sio = StringIO()
  buf = Console(file=sio, force_terminal=False, width=120)
  with mock.patch.object(output, "console", buf):
    output.infrastructure_state(outputs)
  return sio.getvalue()


class TestInfrastructureState(absltest.TestCase):
  """Verify infrastructure_state renders correctly."""

  def _base_outputs(self):
    return {
      "project": _make_output("my-project"),
      "zone": _make_output("us-central1-a"),
      "cluster_name": _make_output("keras-remote-cluster"),
      "cluster_endpoint": _make_output("34.123.45.67"),
      "ar_registry": _make_output("us-docker.pkg.dev/my-project/keras-remote"),
    }

  def test_gpu_accelerator(self):
    outputs = self._base_outputs()
    outputs["accelerator"] = _make_output(
      {
        "type": "GPU",
        "name": "l4",
        "count": 1,
        "machine_type": "g2-standard-4",
        "node_pool": "gpu-pool",
        "node_count": 1,
      }
    )
    text = _render_text(outputs)

    self.assertIn("my-project", text)
    self.assertIn("GPU", text)
    self.assertIn("l4", text)
    self.assertIn("g2-standard-4", text)
    self.assertIn("gpu-pool", text)

  def test_tpu_accelerator(self):
    outputs = self._base_outputs()
    outputs["accelerator"] = _make_output(
      {
        "type": "TPU",
        "name": "v5p",
        "chips": 8,
        "topology": "2x2x2",
        "machine_type": "ct5p-hightpu-4t",
        "node_pool": "tpu-v5p-pool",
        "node_count": 2,
      }
    )
    text = _render_text(outputs)

    self.assertIn("TPU", text)
    self.assertIn("v5p", text)
    self.assertIn("2x2x2", text)
    self.assertIn("ct5p-hightpu-4t", text)
    self.assertIn("tpu-v5p-pool", text)

  def test_cpu_only(self):
    outputs = self._base_outputs()
    outputs["accelerator"] = _make_output(None)
    text = _render_text(outputs)

    self.assertIn("CPU only", text)
    self.assertNotIn("GPU", text)
    self.assertNotIn("TPU", text)

  def test_missing_accelerator_key_backward_compat(self):
    outputs = self._base_outputs()
    # No "accelerator" key — simulates old stack
    text = _render_text(outputs)

    self.assertIn("Unknown", text)
    self.assertIn("keras-remote up", text)


if __name__ == "__main__":
  absltest.main()
