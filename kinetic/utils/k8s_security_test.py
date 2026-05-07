import os
import base64
from unittest.mock import MagicMock, patch
from absl.testing import absltest
from kinetic.utils import security


class K8sSecurityTest(absltest.TestCase):
  def setUp(self):
    self.test_file = "/tmp/test_k8s_security.pkl"
    if os.path.exists(self.test_file):
      os.remove(self.test_file)
    if "KINETIC_SECRET" in os.environ:
      del os.environ["KINETIC_SECRET"]

  def tearDown(self):
    if os.path.exists(self.test_file):
      os.remove(self.test_file)

  @patch("kinetic.backend.k8s_utils.get_security_secret")
  def test_fetch_from_k8s(self, mock_get):
    # Mock k8s secret
    secret_bytes = b"k8s-generated-secret-key-32-bytes"
    encoded_secret = base64.b64encode(secret_bytes).decode("utf-8")
    mock_get.return_value = encoded_secret

    data = b"confidential data"
    with open(self.test_file, "wb") as f:
      f.write(data)

    # This should call k8s_utils.get_security_secret because namespace is provided
    security.sign_file(self.test_file, namespace="default")

    mock_get.assert_called_once_with("default")
    self.assertEqual(os.path.getsize(self.test_file), len(data) + 32)

    # Verify should also fetch from k8s
    mock_get.reset_mock()
    returned_data = security.verify_file(self.test_file, namespace="default")
    mock_get.assert_called_once_with("default")
    self.assertEqual(returned_data, data)

    # File should still be signed on disk
    self.assertEqual(os.path.getsize(self.test_file), len(data) + 32)


if __name__ == "__main__":
  absltest.main()
