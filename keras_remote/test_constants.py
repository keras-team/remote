"""Tests for keras_remote.constants — zone/region/location helpers."""

import pytest

from keras_remote.constants import (
  DEFAULT_REGION,
  DEFAULT_ZONE,
  get_default_zone,
  zone_to_ar_location,
  zone_to_region,
)


class TestZoneToRegion:
  @pytest.mark.parametrize(
    "zone, expected_region",
    [
      ("us-central1-a", "us-central1"),
      ("us-central1-b", "us-central1"),
      ("us-east1-b", "us-east1"),
      ("us-east4-c", "us-east4"),
      ("us-west1-a", "us-west1"),
      ("us-west4-b", "us-west4"),
      ("europe-west1-b", "europe-west1"),
      ("europe-west4-b", "europe-west4"),
      ("asia-east1-c", "asia-east1"),
      ("asia-southeast1-a", "asia-southeast1"),
      ("me-west1-a", "me-west1"),
      ("southamerica-east1-b", "southamerica-east1"),
    ],
  )
  def test_zone_to_region(self, zone, expected_region):
    assert zone_to_region(zone) == expected_region

  @pytest.mark.parametrize(
    "zone",
    [
      "",
      None,
      "invalid",
    ],
  )
  def test_fallback_returns_default(self, zone):
    assert zone_to_region(zone) == DEFAULT_REGION


class TestZoneToArLocation:
  @pytest.mark.parametrize(
    "zone, expected_location",
    [
      ("us-central1-a", "us"),
      ("us-east1-b", "us"),
      ("us-west1-a", "us"),
      ("europe-west1-b", "europe"),
      ("europe-west4-b", "europe"),
      ("asia-east1-c", "asia"),
      ("asia-southeast1-a", "asia"),
      ("me-west1-a", "me"),
      ("southamerica-east1-b", "southamerica"),
    ],
  )
  def test_zone_to_ar_location(self, zone, expected_location):
    assert zone_to_ar_location(zone) == expected_location


class TestGetDefaultZone:
  @pytest.mark.parametrize(
    "env_value, expected_zone",
    [
      ("us-west1-b", "us-west1-b"),
      ("europe-west4-a", "europe-west4-a"),
      ("asia-east1-c", "asia-east1-c"),
    ],
  )
  def test_returns_env_var_when_set(
    self, monkeypatch, env_value, expected_zone
  ):
    # Temporarily set the env var so get_default_zone() picks it up.
    monkeypatch.setenv("KERAS_REMOTE_ZONE", env_value)
    assert get_default_zone() == expected_zone

  def test_returns_default_when_unset(self, monkeypatch):
    # Remove the env var if set
    # raising=False avoids KeyError when it's already absent.
    monkeypatch.delenv("KERAS_REMOTE_ZONE", raising=False)
    assert get_default_zone() == DEFAULT_ZONE

  @pytest.mark.parametrize(
    "constant, expected_value",
    [
      (DEFAULT_ZONE, "us-central1-a"),
      (DEFAULT_REGION, "us-central1"),
    ],
  )
  def test_default_constants(self, constant, expected_value):
    assert constant == expected_value
