import torch
import torchvision.models as models
import torch.nn as nn

try:
    model = models.efficientnet_b2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    sd = torch.load("best_model.pth", map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    model.load_state_dict(sd)
    print("Loaded as B2")
except Exception as e:
    print("Failed B2:", e)
    
    try:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
        sd = torch.load("best_model.pth", map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd)
        print("Loaded as B0")
    except Exception as e2:
        print("Failed B0:", e2)
