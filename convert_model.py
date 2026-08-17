import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import os

print("Loading EfficientNet-B0...")
model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)

model_path = r"C:\Users\evere\Downloads\best_efnetb0_pretrained (1).pth"
state_dict = torch.load(model_path, map_location='cpu')

if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.")

dummy_input = torch.randn(1, 3, 224, 224)

# Use relative paths to avoid weird unicode issues in Windows CMD
str_pt_path = "model.pt"
print("Tracing model to TorchScript...")
traced_model = torch.jit.trace(model, dummy_input)
# Save for Lite Interpreter which is required for PyTorch Mobile / Flutter
try:
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_model = optimize_for_mobile(traced_model)
    optimized_model._save_for_lite_interpreter(str_pt_path)
    print("Saved as optimized Lite model for mobile!")
except Exception as e:
    print("Could not optimize for mobile, saving standard TorchScript...", e)
    traced_model.save(str_pt_path)

print(f"TorchScript model saved: {os.path.abspath(str_pt_path)}")

str_onnx_path = "model.onnx"
print("Exporting model to ONNX...")
torch.onnx.export(model,
                  dummy_input,
                  str_onnx_path,
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print(f"ONNX model saved: {os.path.abspath(str_onnx_path)}")
