import os
import zipfile

import cloudpickle

from keras_remote.data import Data


def zip_working_dir(base_dir, output_path, exclude_paths=None):
  """Zips the base_dir into output_path, excluding .git, __pycache__,
  and any paths in exclude_paths."""
  exclude_paths = exclude_paths or set()
  normalized_excludes = {os.path.normpath(p) for p in exclude_paths}

  with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(base_dir):
      # Exclude .git, __pycache__, and Data-referenced directories
      dirs[:] = [
        d
        for d in dirs
        if d not in [".git", "__pycache__"]
        and os.path.normpath(os.path.join(root, d)) not in normalized_excludes
      ]

      for file in files:
        file_path = os.path.join(root, file)
        if os.path.normpath(file_path) in normalized_excludes:
          continue
        archive_name = os.path.relpath(file_path, base_dir)
        zipf.write(file_path, archive_name)


def save_payload(func, args, kwargs, env_vars, output_path, volumes=None):
  """Uses cloudpickle to serialize the function, args, kwargs, and
  env_vars."""
  payload = {
    "func": func,
    "args": args,
    "kwargs": kwargs,
    "env_vars": env_vars,
  }
  if volumes:
    payload["volumes"] = volumes
  with open(output_path, "wb") as f:
    cloudpickle.dump(payload, f)


def extract_data_refs(args, kwargs):
  """Scan args and kwargs for Data objects at any nesting depth.

  Returns list of (data_obj, position_path) tuples.
  """
  refs = []
  for i, arg in enumerate(args):
    _scan_for_data(arg, ("arg", i), refs)
  for key, val in kwargs.items():
    _scan_for_data(val, ("kwarg", key), refs)
  return refs


def _scan_for_data(obj, path, refs):
  if isinstance(obj, Data):
    refs.append((obj, path))
  elif isinstance(obj, (list, tuple)):
    for i, item in enumerate(obj):
      _scan_for_data(item, path + (i,), refs)
  elif isinstance(obj, dict):
    for key, val in obj.items():
      _scan_for_data(val, path + (key,), refs)


def replace_data_with_refs(args, kwargs, ref_map):
  """Replace Data objects with serializable ref dicts.

  Args:
      ref_map: dict mapping id(Data) -> ref dict
  Returns:
      (new_args, new_kwargs) -- new tuples/dicts with Data replaced
  """
  new_args = tuple(_replace_in_value(a, ref_map) for a in args)
  new_kwargs = {k: _replace_in_value(v, ref_map) for k, v in kwargs.items()}
  return new_args, new_kwargs


def _replace_in_value(obj, ref_map):
  if isinstance(obj, Data) and id(obj) in ref_map:
    return ref_map[id(obj)]
  elif isinstance(obj, list):
    return [_replace_in_value(item, ref_map) for item in obj]
  elif isinstance(obj, tuple):
    return tuple(_replace_in_value(item, ref_map) for item in obj)
  elif isinstance(obj, dict):
    return {k: _replace_in_value(v, ref_map) for k, v in obj.items()}
  return obj
