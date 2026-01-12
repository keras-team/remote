import functools
import inspect
import os
import shutil
import tempfile

from keras_remote import packager
from keras_remote import infra

def run(accelerator='v3-8', zone=None):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      with tempfile.TemporaryDirectory() as tmpdir:

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        context_zip_name = f'context_{timestamp}.zip'
        remote_context_zip_path = f'/tmp/{context_zip_name}'

        context_zip = os.path.join(tmpdir, context_zip_name)
        payload_pkl = os.path.join(tmpdir, 'payload.pkl')
        remote_runner_py = os.path.join(tmpdir, 'remote_runner.py')

        # 1. Create zip/pickle artifacts
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        if module:
          caller_path = os.path.dirname(os.path.abspath(module.__file__))
        else:
          caller_path = os.getcwd() # Fallback

        print(f"Packaging directory: {caller_path}...", flush=True)
        packager.zip_working_dir(caller_path, context_zip)
        print(f"Context zip created at {context_zip}", flush=True)

        print("Serializing payload...", flush=True)
        packager.save_payload(func, args, kwargs, payload_pkl)
        print(f"Payload pickle created at {payload_pkl}", flush=True)

        # Copy remote_runner.py to tmpdir
        this_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(
            os.path.join(this_dir, "remote_runner.py"), remote_runner_py
        )

        # 2. Ensure TPU VM exists
        vm_name = f"remote-user-{accelerator}"
        infra.ensure_tpu_vm(vm_name, accelerator, zone=zone)

        # 3. Upload artifacts
        print(f"Uploading files to {vm_name}...", flush=True)
        infra.scp_to_vm(vm_name, context_zip, remote_context_zip_path, zone=zone)
        infra.scp_to_vm(vm_name, payload_pkl, '/tmp/payload.pkl', zone=zone)
        infra.scp_to_vm(vm_name, remote_runner_py, '/tmp/remote_runner.py', zone=zone)

        # Find and upload requirements.txt
        requirements_txt = None
        search_dir = caller_path
        while search_dir != "/":
            req_path = os.path.join(search_dir, "requirements.txt")
            if os.path.exists(req_path):
                requirements_txt = req_path
                break
            parent_dir = os.path.dirname(search_dir)
            if parent_dir == search_dir:  # Avoid infinite loop at root
                break
            search_dir = parent_dir

        if requirements_txt:
            print(f"Using requirements.txt: {requirements_txt}", flush=True)
            infra.scp_to_vm(vm_name, requirements_txt, '/tmp/requirements.txt', zone=zone)
            use_requirements = True
        else:
            print("No requirements.txt found.", flush=True)
            use_requirements = False
        print("Upload complete.", flush=True)

        # 4. Execute remote_runner.py on the VM
        print("Executing remote script...", flush=True)
        infra.ssh_execute(vm_name, '/tmp/remote_runner.py', context_zip_path=remote_context_zip_path, use_requirements=use_requirements, zone=zone)
        print("Remote execution finished.", flush=True)

    return wrapper
  return decorator
