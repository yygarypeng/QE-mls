import os
import tensorflow as tf
import tf2onnx
import onnx

# Print the current working directory
print("Current working directory:", os.getcwd())

# Define the path to the model
path = "/root/work/QE-mls/8th_trial/ww_resregressor_result/"
model_path = path + "ww_resregressor.keras"

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No file or directory found at {model_path}")

# Load the Keras model
model = tf.keras.models.load_model(model_path)

# Convert the Keras model to ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(model)

# Save the ONNX model
onnx.save(onnx_model, 'ww_resregressor.onnx')

print("Model successfully converted and saved as 'ww_resregressor.onnx'")