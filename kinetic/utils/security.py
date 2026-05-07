import base64
import hashlib
import hmac
import os

from absl import logging

_KINETIC_SECRET_ENV = "KINETIC_SECRET"
_SIGNATURE_SIZE = 32  # HMAC-SHA256 produces a 32-byte digest


def get_secret(namespace: str | None = None) -> bytes | None:
  """Get the signing secret from the environment or Kubernetes.

  If namespace is provided (client side), it always fetches the secret
  from the Kubernetes cluster, ignoring local environment variables to
  ensure team consistency.

  If namespace is NOT provided (worker side), it reads the
  KINETIC_SECRET environment variable injected by the pod spec.
  """
  if namespace:
    from kinetic.backend import k8s_utils

    try:
      encoded_key = k8s_utils.get_security_secret(namespace)
      return base64.b64decode(encoded_key)
    except Exception as e:
      logging.warning("Failed to fetch signing key from Kubernetes: %s", e)
      return None

  # Remote worker side: Use environment variable injected by Pod Spec.
  secret = os.environ.get(_KINETIC_SECRET_ENV)
  if secret:
    return secret.encode("utf-8")

  return None


def sign_data(data: bytes, secret: bytes) -> bytes:
  """Sign data using HMAC-SHA256."""
  return hmac.new(secret, data, hashlib.sha256).digest()


def sign_file(file_path: str, namespace: str | None = None):
  """Sign a file and append the signature to it.

  The file format will be: [original data][signature (32 bytes)]
  """
  secret = get_secret(namespace=namespace)
  if not secret:
    return

  with open(file_path, "rb") as f:
    data = f.read()

  signature = sign_data(data, secret)

  with open(file_path, "ab") as f:
    f.write(signature)

  logging.info("Signed file: %s", file_path)


def verify_file(file_path: str, namespace: str | None = None) -> bytes:
  """Verify the signature of a file and return the original data.

  The file remains unchanged on disk. Raises RuntimeError if
  verification fails.
  """
  secret = get_secret(namespace=namespace)
  with open(file_path, "rb") as f:
    data_with_sig = f.read()

  if not secret:
    logging.warning("KINETIC_SECRET not set, skipping signature verification.")
    return data_with_sig

  if len(data_with_sig) < _SIGNATURE_SIZE:
    raise RuntimeError(
      f"File {file_path} is too small to contain a signature "
      f"({len(data_with_sig)} < {_SIGNATURE_SIZE} bytes)."
    )

  data = data_with_sig[:-_SIGNATURE_SIZE]
  expected_signature = data_with_sig[-_SIGNATURE_SIZE:]

  actual_signature = sign_data(data, secret)

  if not hmac.compare_digest(actual_signature, expected_signature):
    raise RuntimeError(
      f"Signature verification failed for {file_path}! "
      "This could indicate that the file was tampered with in transit "
      "or the signing keys do not match between client and worker."
    )

  logging.info("Verified signature for: %s", file_path)
  return data
