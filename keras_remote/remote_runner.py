import argparse
import cloudpickle
import os
import shutil
import sys
import traceback
import zipfile

WORKSPACE_DIR = '/tmp/workspace'
PAYLOAD_PKL = '/tmp/payload.pkl'

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("context_zip", help="Path to the context zip file")
  args = parser.parse_args()
  context_zip_path = args.context_zip

  print(f"Starting remote execution", flush=True)

  # 1. Unzip context_zip_path to WORKSPACE_DIR
  print(f"Extracting {context_zip_path} to {WORKSPACE_DIR}", flush=True)
  os.makedirs(WORKSPACE_DIR, exist_ok=True)
  # Clear out old workspace contents if any
  for item in os.listdir(WORKSPACE_DIR):
      item_path = os.path.join(WORKSPACE_DIR, item)
      if os.path.isfile(item_path) or os.path.islink(item_path):
          os.remove(item_path)
      elif os.path.isdir(item_path):
          shutil.rmtree(item_path)

  with zipfile.ZipFile(context_zip_path, 'r') as zip_ref:
    zip_ref.extractall(WORKSPACE_DIR)

  # 3. Load payload.pkl
  print(f"Loading payload from {PAYLOAD_PKL}", flush=True)
  with open(PAYLOAD_PKL, 'rb') as f:
    payload = cloudpickle.load(f)

  func = payload['func']
  args = payload['args']
  kwargs = payload['kwargs']
  # env_vars = payload['env_vars'] # Not used yet

  # 4. Execute the function
  print(f"Executing function {func.__name__}", flush=True)
  try:
    result = func(*args, **kwargs)
    print(f"Function execution completed. Result: {result}", flush=True)
    # TODO: Serialize result (e.g. base64 cloudpickle) and print to stdout for local capture.
  except Exception as e:
    print(f"Error during function execution:", flush=True)
    traceback.print_exc()

if __name__ == "__main__":
  main()
