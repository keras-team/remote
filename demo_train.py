import os
import socket

print("[LOCAL] Setting KERAS_BACKEND to jax")
os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import jax
from keras_remote import core as keras_remote

@keras_remote.run(accelerator='v3-8')
def train_keras_jax_model():
  host = socket.gethostname()
  print(f"[REMOTE] Running on host: {host}")
  print(f"[REMOTE] Keras version: {keras.__version__}")
  print(f"[REMOTE] Keras backend: {keras.config.backend()}")
  print(f"[REMOTE] JAX version: {jax.__version__}")
  print(f"[REMOTE] JAX devices: {jax.devices()}")

  num_classes = 10
  input_shape = (28, 28, 1)

  model = keras.Sequential(
      [
          keras.layers.Input(shape=input_shape),
          keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
          keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
          keras.layers.MaxPooling2D(pool_size=(2, 2)),
          keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
          keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
          keras.layers.GlobalAveragePooling2D(),
          keras.layers.Dropout(0.5),
          keras.layers.Dense(num_classes, activation="softmax"),
      ]
  )
  print("[REMOTE] Model defined.")

  model.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(),
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      metrics=[
          keras.metrics.SparseCategoricalAccuracy(name="acc"),
      ],
  )
  print("[REMOTE] Model compiled.")

  # Dummy data
  num_samples = 128
  x_train = np.random.rand(num_samples, *input_shape).astype(np.float32)
  y_train = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)

  print("[REMOTE] Starting model.fit...")
  model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=2)
  print("[REMOTE] Model.fit finished.")

  return f"Keras JAX training complete on {host}"

if __name__ == "__main__":
  print("Starting Keras JAX demo...")
  result = train_keras_jax_model()
  print(f"Demo result: {result}")
