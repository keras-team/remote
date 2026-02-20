"""Integration tests for the full serialization roundtrip."""

import sys
import zipfile

from keras_remote.utils.packager import zip_working_dir


class TestZipAndExtract:
  def _zip_and_extract(self, src, tmp_path):
    """Zip a directory and extract it, returning the extraction path."""
    zip_path = str(tmp_path / "context.zip")
    zip_working_dir(str(src), zip_path)
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
      zf.extractall(str(extract_dir))
    return extract_dir

  def test_zip_extract_preserves_files(self, tmp_path):
    """Zip → extract roundtrip preserves file content."""
    src = tmp_path / "project"
    src.mkdir()
    (src / "main.py").write_text("x = 1")
    (src / "config.json").write_text('{"key": "val"}')

    extract_dir = self._zip_and_extract(src, tmp_path)

    assert (extract_dir / "main.py").read_text() == "x = 1"
    assert (extract_dir / "config.json").read_text() == '{"key": "val"}'

  def test_zip_extract_enables_imports(self, tmp_path):
    """Extracted workspace can be added to sys.path for imports."""
    src = tmp_path / "project"
    src.mkdir()
    (src / "helper.py").write_text("def greet(name):\n  return f'Hi {name}'")

    extract_dir = self._zip_and_extract(src, tmp_path)

    sys.path.insert(0, str(extract_dir))
    try:
      import helper

      assert helper.greet("World") == "Hi World"
    finally:
      sys.path.remove(str(extract_dir))
      # Clean up imported module
      sys.modules.pop("helper", None)
