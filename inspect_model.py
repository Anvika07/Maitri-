import torch

# Path to your trained model
model_path = "fatigue_model_nthuddd.pth"

# Load the checkpoint
checkpoint = torch.load(model_path, map_location="cpu")

# If it’s wrapped inside a dict like {'model_state_dict': ...}, unpack it
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

print("\n🔍 Model keys found in checkpoint:\n")
for key in state_dict.keys():
    print(key)

print("\n✅ Total keys found:", len(state_dict.keys()))
