"""Zone, region, and location constants for keras-remote."""

import os

ZONE_ENV_VAR = "KERAS_REMOTE_ZONE"
DEFAULT_ZONE = "us-central1-a"
DEFAULT_REGION = DEFAULT_ZONE.rsplit("-", 1)[0]  # "us-central1"


def get_default_zone():
  """Return zone from KERAS_REMOTE_ZONE env var, or DEFAULT_ZONE."""
  return os.environ.get(ZONE_ENV_VAR, DEFAULT_ZONE)


def zone_to_region(zone):
  """Convert a GCP zone to its region (e.g. 'us-central1-a' -> 'us-central1')."""
  return zone.rsplit("-", 1)[0] if zone and "-" in zone else DEFAULT_REGION


def zone_to_ar_location(zone):
  """Convert a GCP zone to Artifact Registry multi-region (e.g. 'us-central1-a' -> 'us')."""
  return zone_to_region(zone).split("-")[0]
