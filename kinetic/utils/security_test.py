import os

from absl.testing import absltest

from kinetic.utils import security


class SecurityTest(absltest.TestCase):
  def setUp(self):
    self.test_file = "/tmp/test_security.pkl"
    if os.path.exists(self.test_file):
      os.remove(self.test_file)
    os.environ["KINETIC_SECRET"] = "test-secret"

  def tearDown(self):
    if os.path.exists(self.test_file):
      os.remove(self.test_file)

  def test_sign_and_verify(self):
    data = b"hello world"
    with open(self.test_file, "wb") as f:
      f.write(data)

    # Worker-side simulation (no namespace, relies on os.environ)
    security.sign_file(self.test_file)

    # File should be larger now
    self.assertEqual(os.path.getsize(self.test_file), len(data) + 32)

    # verify_file should return the original data and NOT modify the file
    returned_data = security.verify_file(self.test_file)
    self.assertEqual(returned_data, data)

    # File should still be signed (larger size)
    self.assertEqual(os.path.getsize(self.test_file), len(data) + 32)

  def test_verify_fail(self):
    data = b"hello world"
    with open(self.test_file, "wb") as f:
      f.write(data)

    # Worker-side simulation
    security.sign_file(self.test_file)

    # Tamper with data (overwrite first byte)
    with open(self.test_file, "r+b") as f:
      f.write(b"H")

    with self.assertRaisesRegex(RuntimeError, "Signature verification failed"):
      security.verify_file(self.test_file)

  def test_file_too_small(self):
    with open(self.test_file, "wb") as f:
      f.write(b"too small")

    with self.assertRaisesRegex(
      RuntimeError, "too small to contain a signature"
    ):
      security.verify_file(self.test_file)

  def test_client_ignores_env_var(self):
    """When namespace is provided (client-side), it should ignore os.environ."""
    data = b"hello world"
    with open(self.test_file, "wb") as f:
      f.write(data)

    # Even though KINETIC_SECRET is set in setUp, passing namespace
    # should trigger a fetch from K8s (which we can mock or let fail).
    from unittest.mock import patch

    with patch("kinetic.backend.k8s_utils.get_security_secret") as mock_get:
      mock_get.side_effect = Exception("k8s fetch triggered")
      security.sign_file(self.test_file, namespace="default")
      mock_get.assert_called_once_with("default")

  def test_no_secret_skips(self):
    del os.environ["KINETIC_SECRET"]
    data = b"hello world"
    with open(self.test_file, "wb") as f:
      f.write(data)

    # Should not raise anything
    security.verify_file(self.test_file)

    with open(self.test_file, "rb") as f:
      self.assertEqual(f.read(), data)


if __name__ == "__main__":
  absltest.main()
