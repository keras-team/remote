# Custom Container Images

By default, Kinetic builds a container image for your job using Cloud Build. This build is cached by dependency hash, but if you have a complex environment or want to skip the build step entirely, you can provide a prebuilt image.

## Using a Prebuilt Image

Use the `container_image` parameter in the `@kinetic.run()` or `@kinetic.submit()` decorator to specify a custom URI.

```python
import kinetic

@kinetic.run(
    accelerator="v6e-8",
    container_image="us-docker.pkg.dev/my-project/kinetic/prebuilt:v1.0"
)
def train():
    # This job will start immediately, skipping the Cloud Build step
    ...
```

## Requirements for Custom Images

To work with Kinetic, your custom image must meet the following criteria:

1.  **Kinetic Installed**: The `kinetic` package must be installed in the image.
2.  **Entrypoint**: The image must include a compatible Python environment and the necessary dependencies for your function.
3.  **Permissions**: The GKE nodes must have permission to pull the image from the registry (e.g., Artifact Registry in the same GCP project).

## When to Use Custom Images

- **Fast Startup**: Skip the ~2-5 minute Cloud Build step for new dependency combinations.
- **Complex Dependencies**: If your environment requires non-Python system libraries (e.g., CUDA, specific C++ libs) that aren't in the default Kinetic template.
- **Corporate Compliance**: Use base images that have been vetted by your security or platform teams.

## Image Caching

Kinetic automatically tags and caches images it builds in your project's Artifact Registry. You can find these images in the `kn-{cluster-name}` repository in your GCP project.
