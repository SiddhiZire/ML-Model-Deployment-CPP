import torch
import os

from train import TrafficSignCNN


# ============================================================
# 1. Create the model
# ============================================================

model = TrafficSignCNN()


# ============================================================
# 2. Load the trained model
# ============================================================

model.load_state_dict(
    torch.load(
        "models/traffic_sign_cnn.pth",
        weights_only=True
    )
)

model.eval()

print("Trained model loaded successfully!")


# ============================================================
# 3. Create dummy input
# ============================================================

dummy_input = torch.randn(1, 3, 32, 32)


# ============================================================
# 4. Export model to ONNX
# ============================================================

torch.onnx.export(
    model,
    dummy_input,
    "models/traffic_sign_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)


# ============================================================
# 5. Confirm export
# ============================================================

print("ONNX model exported successfully!")

print(
    "Saved at: models/traffic_sign_cnn.onnx"
)