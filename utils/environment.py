# utils/environment.py
import torch

def device():
    """
    Returns the best available device for PyTorch: 'mps', 'cuda', or 'cpu'.
    """
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print("device=", device)
    return device

