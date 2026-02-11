# Keras Remote API design guidelines

These guidelines are meant to help focus design discussions and help us create delightful developer experiences for remote execution.

These are meant as guidelines, not rules: each decision should be debated in its own unique context.

---

## Design end-to-end workflows, not individual functions and classes.

When developing APIs, start by designing end-to-end workflows.

- The goal is to arrive at workflows that feel like they are purposefully designed and well-optimized. **Features only exist to support a workflow.** No feature should exist to provide a capability "just in case".
- **Every design review document should prominently feature a code example of one or two end-to-end workflows.**
- Ask: **in what workflows will this be used?**
- Ask: **do users really need to configure this parameter**, or can we infer it from the context (e.g., defaulting to the current gcloud project)?

### Example:

Instead of exposing low-level cluster creation and job submission as separate steps that the user must manually stitch together:

```python
# Bad: Fragmented workflow
cluster = create_cluster("my-cluster")
job = submit_job(cluster, my_function)
result = wait_for_result(job)
```

We design for the end-to-end user intent ("run this function on a TPU"):

```python
# Good: Workflow-centric
@keras_remote.run(accelerator="v3-8")
def my_function():
    ...
```

---

## Carefully weigh whether a new feature should be included.

Every feature has a maintenance cost.

- **It should be broadly useful.** Avoid niche infrastructure options unless critical.
- **It should be a best practice.** We don't support every experimental scheduler or accelerator immediately.
- **It should have an owner.**

---

## Seek to minimize cognitive load for our users.

- **Automate everything.** Infrastructure provisioning, Docker builds, and auth should be invisible.
- **Minimize actions & choices.** Sensible defaults are key (e.g., default zone `us-central1-a`, default project from gcloud config).
- **Design simple consistent workflows.**

### Practical rules:

- **No API should deal with internal implementation details.** Users shouldn't need to know about Kubernetes pods, node pools, or internal IP addresses unless absolutely necessary for advanced debugging.
- **Introduce as few new concepts as possible.** The mental model should be "running a function on an accelerator", not "managing a distributed system".
- **Objects that do interchangeable things should have identical APIs.** Switching from `backend="tpu-vm"` to `backend="gke"` should require minimal changes.
- **Plain Python types are preferable.** Use strings for accelerators (`"v3-8"`) rather than custom Enums.

### Naming:

- **The meaning of an argument should be clear.** `accelerator` is better than `machine_type` or `instance_config`.
- **Consistency is key.** modifying the `zone` argument should be consistent across all backends.
- **Avoid OverlyLongAndSpecificNamingPatterns.**

### Example:

```python
# Bad: Implementation details leaking into API
@keras_remote.run(
    tpu_topology="2x2",
    k8s_namespace="user-1",
    docker_image_repo="gcr.io/..."
)
```

```python
# Good: Focused on the user problem
@keras_remote.run(
    accelerator="v3-8",
    container_image="my-image"
)
```

---

## Balance expressivity vs. user-friendliness.

### Simple use cases should be simple, advanced use cases should be possible.

- **Don't increase cognitive load of common use cases.**
- **Make sure advanced users have a path.** For example, allow identifying a specific `cluster` or `service_account` for power users, but keep them optional.

### APIs should be strictly compartmentalized.

- The `keras_remote.run` decorator shouldn't handle model definition logic. It should only handle execution.

---

## Don't neglect error messages, docstrings, and documentation.

- **Catch user errors early.** Validate GCP project existence and quota before starting a long build.
- **Provide detailed feedback.**
    - Bad: `Error: 403 Forbidden`
    - Good: `Permission denied. Please ensure your account 'user@example.com' has the 'Storage Object Admin' role on bucket 'gs://my-bucket'.`
- **Show, don't tell.** Documentation should show code examples of running functions, not just list arguments.

### Error messages: a case study

Bad:
```
RuntimeError: Job failed.
```

Good:
```
RuntimeError: The remote job failed with exit code 1.
Logs from the worker:
...
ModuleNotFoundError: No module named 'tensorflow_datasets'
...
Tip: You can add 'tensorflow_datasets' to your requirements.txt file to install it on the remote worker.
```
