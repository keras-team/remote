import os


def get_default_project() -> str | None:
  """Get project ID from KINETIC_PROJECT or GOOGLE_CLOUD_PROJECT."""
  return os.environ.get("KINETIC_PROJECT") or os.environ.get(
    "GOOGLE_CLOUD_PROJECT"
  )
