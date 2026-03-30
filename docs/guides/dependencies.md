# Managing Dependencies

Kinetic automatically ensures that your remote worker has all the libraries needed to execute your code.

## Automatic Detection

By default, Kinetic looks for dependency declarations in your current working directory and includes them in the container build.

### Supported Files

1.  **`requirements.txt`**: Standard pip requirements file.
2.  **`pyproject.toml`**: Project metadata file (extracts `project.dependencies`).

If both files exist, `requirements.txt` takes precedence.

## JAX & Accelerator Libraries

To prevent version conflicts with the pre-installed, hardware-optimized JAX runtime on remote nodes, Kinetic **automatically filters** JAX-related packages from your dependencies:

- `jax`
- `jaxlib`
- `libtpu`
- `libtpu-nightly`

### Keeping a JAX Dependency

If you have a specific reason to override the system JAX installation, you can force Kinetic to keep a dependency by appending `# kn:keep` to the line in your `requirements.txt`:

```text
jax==0.4.25 # kn:keep
```

## Adding New Dependencies

When you add a new library to your local project, Kinetic will detect the change in your `requirements.txt` or `pyproject.toml`, calculate a new dependency hash, and automatically trigger a new container build on the next `@kinetic.run()` call.

## Private Packages

If you need to install private packages or use a custom index, consider using a :doc:`custom container image <../advanced/containers>`.
