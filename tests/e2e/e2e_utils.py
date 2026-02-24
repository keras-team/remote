"""E2E test utilities — helpers for tests requiring GCP infrastructure."""

import os
import unittest


def skip_unless_e2e(reason="E2E_TESTS not set"):
  """Skip decorator for e2e tests unless E2E_TESTS env var is set."""
  return unittest.skipUnless(os.environ.get("E2E_TESTS"), reason)
