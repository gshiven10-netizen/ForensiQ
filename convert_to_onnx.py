"""
Convert weights.weights.h5 to model.onnx for lightweight inference on Render.
Run this once locally: python convert_to_onnx.py
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import tf2onnx
import onnx

base_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(base_dir, "weights.weights.h5")
onnx_path    = os.path.join(base_dir, "model.onnx")

print(f"Loading model from {weights_path} ...")

@tf.keras.utils.register_keras_serializable()
class CompatibleDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

try:
    model = tf_load_model(weights_path, custom_objects={'Dense': CompatibleDense}, compile=False)
    print("✅ Model loaded via tf_load_model")
except Exception as e:
    print(f"⚠️  Full load failed ({e}), using fallback architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.load_weights(weights_path)
    print("✅ Fallback architecture loaded")

print("Converting to ONNX via SavedModel ...")
import tempfile, subprocess, sys

# Save as SavedModel first, then convert
saved_dir = os.path.join(base_dir, "saved_model_tmp")
model.export(saved_dir)
print(f"✅ Exported SavedModel to {saved_dir}")

# Convert using tf2onnx CLI  
result = subprocess.run([
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", saved_dir,
    "--output", onnx_path,
    "--opset", "13"
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    sys.exit(1)
