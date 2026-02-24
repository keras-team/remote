"""Tests for keras_remote.cli.infra.program — TPU node pool creation."""

from unittest import mock

from absl.testing import absltest, parameterized

from keras_remote.core.accelerators import TpuConfig

# Patch the pulumi_gcp module before importing program, so the module-level
# import inside program.py picks up the mock.
with mock.patch.dict("sys.modules", {"pulumi_gcp": mock.MagicMock()}):
  from keras_remote.cli.infra import program


class TestCreateTpuNodePool(parameterized.TestCase):
  """Verify _create_tpu_node_pool sets placement_policy correctly."""

  @parameterized.named_parameters(
    dict(
      testcase_name="v5p_multi_host",
      tpu=TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
      expect_placement=True,
    ),
    dict(
      testcase_name="v6e_multi_host",
      tpu=TpuConfig("v6e", 8, "2x4", "tpu-v6e-slice", "ct6e-standard-4t", 2),
      expect_placement=True,
    ),
    dict(
      testcase_name="v3_single_host",
      tpu=TpuConfig("v3", 4, "2x2", "tpu-v3-podslice", "ct3-hightpu-4t", 1),
      expect_placement=False,
    ),
    dict(
      testcase_name="v5litepod_single_host",
      tpu=TpuConfig(
        "v5litepod", 4, "2x2", "tpu-v5-lite-podslice", "ct5lp-hightpu-4t", 1
      ),
      expect_placement=False,
    ),
  )
  @mock.patch.object(program, "gcp")
  def test_placement_policy(self, gcp_mock, tpu, expect_placement):
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    program._create_tpu_node_pool(cluster, tpu, "us-central2-b", "my-project")

    call_kwargs = gcp_mock.container.NodePool.call_args
    placement = call_kwargs.kwargs.get(
      "placement_policy", call_kwargs[1].get("placement_policy")
    )

    if expect_placement:
      self.assertIsNotNone(placement)
      gcp_mock.container.NodePoolPlacementPolicyArgs.assert_called_once_with(
        type="COMPACT",
        tpu_topology=tpu.topology,
      )
    else:
      self.assertIsNone(placement)

  @mock.patch.object(program, "gcp")
  def test_node_count_matches_config(self, gcp_mock):
    tpu = TpuConfig("v5p", 16, "2x2x4", "tpu-v5p-slice", "ct5p-hightpu-4t", 4)
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    program._create_tpu_node_pool(cluster, tpu, "us-central2-b", "my-project")

    call_kwargs = gcp_mock.container.NodePool.call_args
    node_count = call_kwargs.kwargs.get(
      "node_count", call_kwargs[1].get("node_count")
    )
    self.assertEqual(node_count, 4)

  @mock.patch.object(program, "gcp")
  def test_pool_name_includes_tpu_name(self, gcp_mock):
    tpu = TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2)
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    program._create_tpu_node_pool(cluster, tpu, "us-central2-b", "my-project")

    positional_args = gcp_mock.container.NodePool.call_args[0]
    self.assertEqual(positional_args[0], "tpu-v5p-pool")


if __name__ == "__main__":
  absltest.main()
