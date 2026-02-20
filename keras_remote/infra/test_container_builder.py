"""Tests for keras_remote.infra.container_builder — hashing, Dockerfile gen, caching."""

import re
from unittest.mock import MagicMock

import pytest
from google.api_core import exceptions as google_exceptions

from keras_remote.infra.container_builder import (
  _generate_dockerfile,
  _hash_requirements,
  _image_exists,
  get_or_build_container,
)


class TestHashRequirements:
  def test_deterministic(self, tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26\n")

    h1 = _hash_requirements(str(req), "l4", "python:3.12-slim")
    h2 = _hash_requirements(str(req), "l4", "python:3.12-slim")
    assert h1 == h2

  def test_different_requirements_different_hash(self, tmp_path):
    req1 = tmp_path / "r1.txt"
    req1.write_text("numpy==1.26\n")
    req2 = tmp_path / "r2.txt"
    req2.write_text("scipy==1.12\n")

    h1 = _hash_requirements(str(req1), "l4", "python:3.12-slim")
    h2 = _hash_requirements(str(req2), "l4", "python:3.12-slim")
    assert h1 != h2

  def test_different_accelerator_different_hash(self, tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy\n")

    h1 = _hash_requirements(str(req), "l4", "python:3.12-slim")
    h2 = _hash_requirements(str(req), "v3-8", "python:3.12-slim")
    assert h1 != h2

  def test_different_base_image_different_hash(self, tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy\n")

    h1 = _hash_requirements(str(req), "l4", "python:3.12-slim")
    h2 = _hash_requirements(str(req), "l4", "python:3.11-slim")
    assert h1 != h2

  @pytest.mark.parametrize(
    "requirements_path",
    [None, "/nonexistent/path.txt"],
    ids=["none", "nonexistent"],
  )
  def test_missing_requirements_valid(self, requirements_path):
    h = _hash_requirements(requirements_path, "cpu", "python:3.12-slim")
    assert isinstance(h, str)
    assert len(h) == 64

  def test_returns_hex_string(self, tmp_path):
    req = tmp_path / "r.txt"
    req.write_text("keras\n")
    h = _hash_requirements(str(req), "l4", "python:3.12-slim")
    assert re.fullmatch(r"[0-9a-f]{64}", h)


class TestGenerateDockerfile:
  @pytest.mark.parametrize(
    ("accelerator_type", "expected", "not_expected"),
    [
      pytest.param("cpu", ["pip install jax"], ["cuda", "tpu"], id="cpu"),
      pytest.param("l4", ["jax[cuda12]"], [], id="gpu"),
      pytest.param("v3-8", ["jax[tpu]", "libtpu_releases"], [], id="tpu"),
    ],
  )
  def test_jax_install(self, accelerator_type, expected, not_expected):
    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      requirements_path=None,
      accelerator_type=accelerator_type,
    )
    for s in expected:
      assert s in content
    for s in not_expected:
      assert s not in content

  def test_with_requirements(self, tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy\n")

    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      requirements_path=str(req),
      accelerator_type="cpu",
    )
    assert "COPY requirements.txt" in content
    assert "pip install -r" in content

  def test_without_requirements(self):
    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      requirements_path=None,
      accelerator_type="cpu",
    )
    assert "COPY requirements.txt" not in content

  @pytest.mark.parametrize(
    "expected_substring",
    [
      pytest.param(
        "COPY remote_runner.py /app/remote_runner.py",
        id="remote_runner_copy",
      ),
      pytest.param("ENV KERAS_BACKEND=jax", id="keras_backend_env"),
    ],
  )
  def test_contains_expected_content(self, expected_substring):
    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      requirements_path=None,
      accelerator_type="cpu",
    )
    assert expected_substring in content

  def test_uses_base_image(self):
    content = _generate_dockerfile(
      base_image="python:3.11-bullseye",
      requirements_path=None,
      accelerator_type="cpu",
    )
    assert "FROM python:3.11-bullseye" in content


class TestImageExists:
  def test_returns_true_when_tag_found(self, mocker):
    mock_client = MagicMock()
    mocker.patch(
      "keras_remote.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    )
    result = _image_exists(
      "us-docker.pkg.dev/my-proj/keras-remote/base:l4-abc123",
      "my-proj",
    )
    assert result is True
    mock_client.get_tag.assert_called_once()

  @pytest.mark.parametrize(
    "side_effect",
    [
      pytest.param(google_exceptions.NotFound("nope"), id="not_found"),
      pytest.param(RuntimeError("unexpected"), id="other_error"),
    ],
  )
  def test_returns_false_on_error(self, mocker, side_effect):
    mock_client = MagicMock()
    mock_client.get_tag.side_effect = side_effect
    mocker.patch(
      "keras_remote.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    )
    result = _image_exists(
      "us-docker.pkg.dev/my-proj/keras-remote/base:l4-abc123",
      "my-proj",
    )
    assert result is False

  def test_correct_resource_name(self, mocker):
    mock_client = MagicMock()
    mocker.patch(
      "keras_remote.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    )
    _image_exists(
      "us-docker.pkg.dev/my-proj/keras-remote/base:v3-8-abc123def456",
      "my-proj",
    )
    call_args = mock_client.get_tag.call_args
    request = call_args.kwargs.get("request") or call_args[1].get("request")
    assert request.name == (
      "projects/my-proj/locations/us"
      "/repositories/keras-remote"
      "/packages/base/tags/v3-8-abc123def456"
    )


class TestGetOrBuildContainer:
  def test_returns_cached_when_image_exists(self, mocker):
    mocker.patch(
      "keras_remote.infra.container_builder._image_exists",
      return_value=True,
    )
    mock_build = mocker.patch(
      "keras_remote.infra.container_builder._build_and_push",
    )

    result = get_or_build_container(
      base_image="python:3.12-slim",
      requirements_path=None,
      accelerator_type="l4",
      project="test-proj",
      zone="us-central1-a",
    )

    mock_build.assert_not_called()
    assert "us-docker.pkg.dev/test-proj/keras-remote/base:" in result

  def test_builds_when_image_missing(self, mocker):
    mocker.patch(
      "keras_remote.infra.container_builder._image_exists",
      return_value=False,
    )
    mock_build = mocker.patch(
      "keras_remote.infra.container_builder._build_and_push",
      return_value="us-docker.pkg.dev/proj/keras-remote/base:l4-bbbbbbbbbbbb",
    )

    result = get_or_build_container(
      base_image="python:3.12-slim",
      requirements_path=None,
      accelerator_type="l4",
      project="proj",
      zone="us-central1-a",
    )

    mock_build.assert_called_once()
    assert result == "us-docker.pkg.dev/proj/keras-remote/base:l4-bbbbbbbbbbbb"

  def _get_image_uri(self, mocker, accelerator_type, project, zone):
    mocker.patch(
      "keras_remote.infra.container_builder._image_exists",
      return_value=True,
    )
    return get_or_build_container(
      base_image="python:3.12-slim",
      requirements_path=None,
      accelerator_type=accelerator_type,
      project=project,
      zone=zone,
    )

  def test_image_uri_format_tpu_europe(self, mocker):
    result = self._get_image_uri(mocker, "v3-8", "my-proj", "europe-west4-b")

    assert result.startswith("europe-docker.pkg.dev/my-proj/keras-remote/base:")
    tag = result.split(":")[-1]
    assert re.fullmatch(r"v3-8-[0-9a-f]{12}", tag)

  def test_image_uri_format_gpu_us(self, mocker):
    result = self._get_image_uri(mocker, "a100-80gb", "proj", "us-central1-a")

    assert result.startswith("us-docker.pkg.dev/proj/keras-remote/base:")
    tag = result.split(":")[-1]
    assert re.fullmatch(r"a100-80gb-[0-9a-f]{12}", tag)
