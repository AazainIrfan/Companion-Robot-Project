import torch

# Ensure CUDA is available and enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)