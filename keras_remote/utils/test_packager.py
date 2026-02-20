"""Tests for keras_remote.utils.packager — zip and payload serialization."""

import os
import zipfile

import cloudpickle
import numpy as np
import pytest

from keras_remote.utils.packager import save_payload, zip_working_dir


class TestZipWorkingDir:
  def _zip_and_list(self, src, tmp_path):
    """Zip src directory and return the set of archive member names."""
    out = tmp_path / "context.zip"
    zip_working_dir(str(src), str(out))
    with zipfile.ZipFile(str(out)) as zf:
      return set(zf.namelist())

  def test_contains_all_files(self, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("a")
    (src / "b.txt").write_text("b")

    assert self._zip_and_list(src, tmp_path) == {"a.py", "b.txt"}

  def test_excludes_git_directory(self, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    git_dir = src / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config")
    (src / "main.py").write_text("code")

    names = self._zip_and_list(src, tmp_path)
    assert all(".git" not in n for n in names)
    assert "main.py" in names

  def test_excludes_pycache_directory(self, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    cache_dir = src / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "mod.cpython-312.pyc").write_bytes(b"\x00")
    (src / "mod.py").write_text("code")

    names = self._zip_and_list(src, tmp_path)
    assert all("__pycache__" not in n for n in names)
    assert "mod.py" in names

  def test_preserves_nested_structure(self, tmp_path):
    src = tmp_path / "src"
    sub = src / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "deep.py").write_text("deep")
    (src / "top.py").write_text("top")

    names = self._zip_and_list(src, tmp_path)
    assert "top.py" in names
    assert os.path.join("pkg", "sub", "deep.py") in names

  def test_empty_directory(self, tmp_path):
    src = tmp_path / "empty"
    src.mkdir()

    assert self._zip_and_list(src, tmp_path) == set()


class TestSavePayload:
  def _save_and_load(self, tmp_path, func, args=(), kwargs=None, env_vars=None):
    """Save a payload and load it back, returning the deserialized dict."""
    if kwargs is None:
      kwargs = {}
    if env_vars is None:
      env_vars = {}
    out = tmp_path / "payload.pkl"
    save_payload(func, args, kwargs, env_vars, str(out))
    with open(str(out), "rb") as f:
      return cloudpickle.load(f)

  def test_roundtrip_simple_function(self, tmp_path):
    def add(a, b):
      return a + b

    payload = self._save_and_load(
      tmp_path, add, args=(2, 3), env_vars={"KEY": "val"}
    )

    assert payload["func"](2, 3) == 5
    assert payload["args"] == (2, 3)
    assert payload["kwargs"] == {}
    assert payload["env_vars"] == {"KEY": "val"}

  def test_roundtrip_with_kwargs(self, tmp_path):
    def greet(name, greeting="Hello"):
      return f"{greeting}, {name}"

    payload = self._save_and_load(
      tmp_path, greet, args=("World",), kwargs={"greeting": "Hi"}
    )

    result = payload["func"](*payload["args"], **payload["kwargs"])
    assert result == "Hi, World"

  def test_roundtrip_lambda(self, tmp_path):
    payload = self._save_and_load(tmp_path, lambda x: x * 2, args=(5,))

    assert payload["func"](*payload["args"]) == 10

  def test_roundtrip_closure(self, tmp_path):
    multiplier = 7

    def make_closure(x):
      return x * multiplier

    payload = self._save_and_load(tmp_path, make_closure, args=(6,))

    assert payload["func"](*payload["args"]) == 42

  def test_roundtrip_numpy_args(self, tmp_path):
    def dot(a, b):
      return np.dot(a, b)

    arr_a = np.array([1.0, 2.0, 3.0])
    arr_b = np.array([4.0, 5.0, 6.0])

    payload = self._save_and_load(tmp_path, dot, args=(arr_a, arr_b))

    result = payload["func"](*payload["args"])
    assert result == pytest.approx(32.0)

  def test_roundtrip_complex_args(self, tmp_path):
    def identity(x):
      return x

    complex_arg = {
      "key": [1, 2, 3],
      "nested": {"a": True, "b": None},
      "tuple": (1, "two", 3.0),
    }

    payload = self._save_and_load(tmp_path, identity, args=(complex_arg,))

    assert payload["func"](*payload["args"]) == complex_arg
