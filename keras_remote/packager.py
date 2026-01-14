import os
import zipfile
import cloudpickle

def zip_working_dir(base_dir, output_path):
  """Zips the base_dir into output_path, excluding .git and __pycache__."""
  with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(base_dir):
      # Exclude .git and __pycache__ directories
      dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__']]

      for file in files:
        file_path = os.path.join(root, file)
        archive_name = os.path.relpath(file_path, base_dir)
        zipf.write(file_path, archive_name)


def save_payload(func, args, kwargs, output_path):
  """Uses cloudpickle to serialize the function, args, kwargs, and dummy env_vars."""
  payload = {
    'func': func,
    'args': args,
    'kwargs': kwargs,
  }
  with open(output_path, 'wb') as f:
    cloudpickle.dump(payload, f)
