# Working with Data

The `kinetic.Data` API is the simplest way to manage your local and cloud data dependencies. It handles content-based hashing, upload caching, and remote path resolution so you don't have to manually manage storage or update paths in your code.

## The `Data` Class

The `Data` class wraps a local file, directory path, or a GCS URI (`gs://...`). When passed as a function argument, it resolves to a plain string path on the remote pod.

### Local Data (Files & Directories)

Kinetic automatically hashes the content of local data. Identical data is uploaded only once and cached across jobs.

```python
from kinetic import Data

@kinetic.run(accelerator="cpu")
def process_data(data_path):
    import os
    # data_path is a plain local path on the remote machine
    print(f"Reading from: {data_path}")
    return sorted(os.listdir(data_path))

# Passes a local directory to the remote function
process_data(Data("./my_dataset/"))
```

### Cloud Data (GCS URIs)

You can also point directly to data in GCS. Kinetic downloads the data locally to the pod before execution.

```python
from kinetic import Data

# gs:// paths resolve into local paths on the pod
process_data(Data("gs://my-bucket/training-set/"))
```

## Mounting Volumes

For training scripts with hardcoded paths, use the `volumes` parameter. This mounts `Data` objects at fixed absolute filesystem paths on the remote worker.

```python
from kinetic import Data

@kinetic.run(
    accelerator="v5e-4",
    volumes={"/data": Data("./dataset/")}
)
def train():
    # Available at the absolute path specified in 'volumes'
    import pandas as pd
    df = pd.read_csv("/data/train.csv")
    return len(df)
```

## Nested Data Structures

`Data` objects can be nested inside lists, dictionaries, or any other serializable structure. Kinetic recursively discovers and resolves them.

```python
from kinetic import Data

@kinetic.run(accelerator="cpu")
def train_multi(datasets):
    # 'datasets' is a list of plain local paths
    for d in datasets:
        print(f"Loading from {d}")

train_multi(datasets=[Data("./d1"), Data("./d2")])
```

## Content-Addressed Caching

Kinetic implements content-addressed caching for all local data uploads.

1. **Hash Calculation**: Kinetic calculates a SHA-256 hash over the contents of your local file or directory.
2. **Cache Check**: It checks `gs://{bucket}/data-cache/{hash}/` for a `.cache_marker` sentinel.
3. **Optimized Upload**: If the marker exists, the upload is skipped. This makes re-running jobs with the same data nearly instantaneous.

## Automatic Zip Exclusion

When you use `Data("./path/to/data")`, and that path is within your project root, Kinetic automatically excludes it from the `context.zip` payload. This prevents redundant uploads and keeps your project payload small.
