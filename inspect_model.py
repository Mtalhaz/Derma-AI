import torch

model_path = r"C:\Users\evere\Downloads\best_efnetb0_pretrained (1).pth"
try:
    state_dict = torch.load(model_path, map_location='cpu')
    print("Successfully loaded state_dict.")
    
    # Check what kind of state_dict we have
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        print("Found nested 'state_dict' key.")

    keys = list(state_dict.keys())
    print("\nFirst 10 keys:")
    for k in keys[:10]:
        print(k)
        
    print("\nLast 10 keys:")
    for k in keys[-10:]:
        print(k)

except Exception as e:
    print(f"Error loading model: {e}")
