import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2

# Grad-CAM library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = FastAPI(title="Skin Lesion Grad-CAM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the model
device = torch.device("cpu")
print("Loading EfficientNet-B2...")
model = models.efficientnet_b2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)

state_dict = torch.load("best_model.pth", map_location=device)
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict)
model.eval()

# 2. Setup Grad-CAM
# For EfficientNet, the last conv layer is usually features[-1]
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# 3. Setup transforms
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/explain")
async def explain_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Original image as float32 for visualization (normalized 0-1)
        rgb_img = cv2.resize(np.array(pil_image), (260, 260))
        rgb_img = np.float32(rgb_img) / 255.0
        
        # Preprocess for model
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Create overlay
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Convert back to Base64
        vis_image = Image.fromarray(visualization)
        buffer = io.BytesIO()
        vis_image.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {"success": True, "heatmap_base64": img_str}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting Grad-CAM Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
