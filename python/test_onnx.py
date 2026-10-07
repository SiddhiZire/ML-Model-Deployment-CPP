import onnxruntime as ort
import numpy as np


# ============================================================
# 1. Load the ONNX model
# ============================================================

session = ort.InferenceSession(
    "models/traffic_sign_cnn.onnx"
)

print("ONNX model loaded successfully!")


# ============================================================
# 2. Get model input information
# ============================================================

input_info = session.get_inputs()[0]

print("Input name:", input_info.name)
print("Input shape:", input_info.shape)


# ============================================================
# 3. Create a sample input image
# ============================================================

# Model expects:
# Batch size = 1
# Channels = 3
# Height = 32
# Width = 32

sample_image = np.random.rand(
    1, 3, 32, 32
).astype(np.float32)


# ============================================================
# 4. Run ONNX inference
# ============================================================

outputs = session.run(
    None,
    {
        input_info.name: sample_image
    }
)


# ============================================================
# 5. Get prediction
# ============================================================

prediction = np.argmax(
    outputs[0],
    axis=1
)[0]

print("Predicted class:", prediction)

print("ONNX inference successful!")