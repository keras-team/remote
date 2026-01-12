
import sys
import os
import zipfile
import cloudpickle
import traceback
import argparse


WORKSPACE_DIR = '/tmp/workspace'
PAYLOAD_PKL = '/tmp/payload.pkl'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("context_zip", help="Path to the context zip file")
  args = parser.parse_args()
  context_zip_path = args.context_zip

  print(f"[REMOTE] Starting remote execution", flush=True)

  # 1. Unzip context_zip_path to WORKSPACE_DIR
  print(f"[REMOTE] Extracting {context_zip_path} to {WORKSPACE_DIR}", flush=True)
  os.makedirs(WORKSPACE_DIR, exist_ok=True)
  # Clear out old workspace contents if any
  for item in os.listdir(WORKSPACE_DIR):
      item_path = os.path.join(WORKSPACE_DIR, item)
      if os.path.isfile(item_path) or os.path.islink(item_path):
          os.unlink(item_path)
      elif os.path.isdir(item_path):
          import shutil
          shutil.rmtree(item_path)

  with zipfile.ZipFile(context_zip_path, 'r') as zip_ref:
    zip_ref.extractall(WORKSPACE_DIR)

  # 2. Add WORKSPACE_DIR to sys.path
  print(f"[REMOTE] Adding {WORKSPACE_DIR} to sys.path", flush=True)
  sys.path.insert(0, WORKSPACE_DIR)

  # 3. Load payload.pkl
  print(f"[REMOTE] Loading payload from {PAYLOAD_PKL}", flush=True)
  with open(PAYLOAD_PKL, 'rb') as f:
    payload = cloudpickle.load(f)

  func = payload['func']
  args = payload['args']
  kwargs = payload['kwargs']

  # 4. Execute the function
  print(f"[REMOTE] Executing function {func.__name__}", flush=True)
  try:
    result = func(*args, **kwargs)
    print(f"[REMOTE] Function execution completed. Result: {result}", flush=True)
  except Exception as e:
    print(f"[REMOTE] Error during function execution:", flush=True)
    traceback.print_exc()


if __name__ == "__main__":
  main()
