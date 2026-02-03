"""
Example: Using keras_remote with Vertex AI backend

This demonstrates the new Vertex AI-based execution backend for keras_remote.

Note: These examples use dynamic container building. To skip the 5-8 minute build time,
you can use prebuilt images with the container_image parameter. See:
- examples/example_prebuilt_image.py for usage example
- examples/Dockerfile.prebuilt for building your own prebuilt image
"""

import keras_remote


# Example 1: Simple function execution on TPU v2-8
# Option A: Dynamic build (current - will build container on first run)
@keras_remote.run(accelerator="nvidia-tesla-t4", backend="vertex-ai")
# Option B: Use prebuilt image (skip build time - see examples/example_prebuilt_image.py)
# @keras_remote.run(
#     accelerator="nvidia-tesla-t4",
#     backend="vertex-ai",
#     container_image="us-central1-docker.pkg.dev/my-project/keras-remote/prebuilt:v1.0"
# )
def simple_computation(x, y):
    """Simple addition that runs on remote TPU."""
    result = x + y
    print(f"Computing {x} + {y} = {result}")
    return result


# Example 2: Keras model training on TPU
# Note: Add container_image parameter to use a prebuilt image instead of building
@keras_remote.run(accelerator="nvidia-tesla-t4", backend="vertex-ai")
def train_simple_model():
    """Train a simple Keras model on remote TPU."""
    import keras
    import numpy as np

    # Create a simple model
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(10,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    # Generate some dummy data
    x_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000, 1)

    # Train the model
    print("Training model on TPU...")
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

    print(f"Final loss: {history.history["loss"][-1]}")
    return history.history["loss"][-1]


def main():
    """Run examples."""
    print("=" * 60)
    print("Keras Remote - Vertex AI Backend Examples")
    print("=" * 60)

    # Example 1: Simple computation
    print("\n--- Example 1: Simple Computation ---")
    print("Running simple_computation(10, 20) on Vertex AI...")
    try:
        result = simple_computation(10, 20)
        print(f"✓ Result: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Example 2: Model training
    print("\n--- Example 2: Keras Model Training ---")
    print("Training a simple model on TPU v2-8...")
    try:
        final_loss = train_simple_model()
        print(f"✓ Model trained. Final loss: {final_loss}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Prerequisites:
    # 1. Set KERAS_REMOTE_PROJECT environment variable to your GCP project ID
    # 2. Authenticate with: gcloud auth application-default login
    # 3. Enable required APIs:
    #    - Vertex AI API
    #    - Cloud Build API
    #    - Artifact Registry API
    #    - Cloud Storage API

    import os

    if not os.environ.get("KERAS_REMOTE_PROJECT"):
        print("ERROR: KERAS_REMOTE_PROJECT environment variable not set")
        print("Please set it to your GCP project ID:")
        print("  export KERAS_REMOTE_PROJECT=your-project-id")
        exit(1)

    main()
