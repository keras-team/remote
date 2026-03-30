# Managing Async Jobs

By default, `@kinetic.run()` blocks your local process until the remote function finishes. For long-running training or large-scale sweeps, you can use the non-blocking `@kinetic.submit()` decorator to fire off jobs and manage them asynchronously.

## Submitting Jobs

Use `@kinetic.submit()` just like `@kinetic.run()`. It accepts the same parameters (accelerator, project, zone, etc.).

```python
import kinetic

@kinetic.submit(accelerator="v5e-1")
def train_model():
    # Long-running training code
    return result

# Returns a JobHandle immediately
job = train_model()
print(f"Submitted job: {job.job_id}")
```

## Monitoring Progress

A `JobHandle` provides several methods to track your job's lifecycle without blocking.

### Checking Status

You can poll the status of a job at any time.

```python
status = job.status()
print(f"Current status: {status.value}")  # e.g., 'PENDING', 'RUNNING', 'SUCCEEDED'
```

### Reading Logs

You can fetch recent log lines directly from the `JobHandle`.

```python
# Get the last 50 lines of logs
print(job.tail(n=50))
```

## Collecting Results

When you're ready to get the final return value, call `.result()`. This will block until the job completes.

```python
# Blocks until success or failure
final_loss = job.result()
print(f"Training finished with loss: {final_loss}")
```

## Reattaching to Jobs

If your local script crashes or you want to check on a job from a different machine, you can reattach to it using its unique ID.

```python
import kinetic

# From another session or machine
job = kinetic.attach("job-12345-67890")
print(f"Reattached to {job.func_name} ({job.status().value})")
```

## Listing Jobs

To see all jobs currently running or recently completed on your cluster, use `list_jobs()`.

```python
import kinetic

jobs = kinetic.list_jobs()
for j in jobs:
    print(f"{j.job_id}: {j.func_name} ({j.status().value})")
```

## Resource Cleanup

By default, Kinetic cleans up Kubernetes resources when a job succeeds. You can manually trigger cleanup via the handle.

```python
# Removes the k8s job and pod, and deletes GCS artifacts
job.cleanup(k8s=True, gcs=True)
```
