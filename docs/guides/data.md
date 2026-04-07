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

## FUSE Mounting

By default, Kinetic downloads data into the container before your function runs. For large datasets where you only need a subset of the files, pass `fuse=True` to lazily mount data from GCS instead. The data is read on demand — only the files you actually open are fetched from cloud storage.

```python
from kinetic import Data

# Large dataset mounted lazily — only files you read are fetched
@kinetic.run(
    accelerator="v5e-4",
    volumes={"/data": Data("gs://my-bucket/imagenet/", fuse=True)}
)
def train():
    import pandas as pd
    df = pd.read_csv("/data/train.csv")
    return len(df)
```

FUSE mounting works with both **volumes** and **function arguments**, and with both local paths and GCS URIs:

```python
# As a function argument — Kinetic auto-mounts and passes the path
@kinetic.run(accelerator="cpu")
def train(data_path):
    files = os.listdir(data_path)
    ...

train(Data("./my_dataset/", fuse=True))
```

### Single Files

Single files work transparently with `fuse=True`. Your function receives a direct file path, just like with downloaded data:

```python
@kinetic.run(accelerator="cpu")
def read_config(config_path):
    with open(config_path) as f:  # config_path points to the file, not a directory
        return json.load(f)

read_config(Data("./config.json", fuse=True))
```

### Mixing FUSE and Downloaded Data

You can freely combine FUSE-mounted and downloaded data in the same job:

```python
@kinetic.run(
    accelerator="v5e-4",
    volumes={
        "/data": Data("gs://my-bucket/large-dataset/", fuse=True),  # lazy mount
        "/config": Data("./small-config/"),                          # downloaded
    }
)
def train(extra_data):
    ...

train(Data("./labels.csv"))  # downloaded argument
```

### When to Use FUSE

| Scenario                                   | Recommended        |
| ------------------------------------------ | ------------------ |
| Large dataset, read a subset of files      | `fuse=True`        |
| Small dataset, read all files              | Default (download) |
| Streaming reads (e.g., `tf.data`, `grain`) | `fuse=True`        |
| Random access to many small files          | Default (download) |

### Prerequisites

FUSE mounting requires the GCS FUSE CSI driver addon on your GKE cluster. `kinetic up` enables it by default.

## Content-Addressed Caching

Kinetic implements content-addressed caching for all local data uploads.

1. **Hash Calculation**: Kinetic calculates a SHA-256 hash over the contents of your local file or directory.
2. **Cache Check**: It checks for a sentinel blob at `gs://{bucket}/{namespace}/data-markers/{hash}` (a separate prefix from the data, so it never appears interferes with the actual data).
3. **Optimized Upload**: If the marker exists, the upload is skipped. This makes re-running jobs with the same data nearly instantaneous.

## Automatic Zip Exclusion

When you use `Data("./path/to/data")`, and that path is within your project root, Kinetic automatically excludes it from the `context.zip` payload. This prevents redundant uploads and keeps your project payload small.
