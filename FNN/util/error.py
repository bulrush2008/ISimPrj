import torch

def LInfError(pred: torch.Tensor, target: torch.Tensor):
    pred, target = _ensure_same_device(pred, target)
    return torch.max(torch.abs(pred - target), dim=0).values

def L2Error(pred: torch.Tensor, target: torch.Tensor):
    pred, target = _ensure_same_device(pred, target)
    return torch.sqrt(torch.mean((pred - target)**2, dim=0))

def L1Error(pred: torch.Tensor, target: torch.Tensor):
    pred, target = _ensure_same_device(pred, target)
    return torch.mean(torch.abs(pred - target), dim=0)

def _ensure_same_device(pred: torch.Tensor, target: torch.Tensor):
    """Move both tensors to the better device (GPU > MPS > CPU)"""
    if pred.device == target.device:
        return pred, target
    
    # Choose the better device
    device = _get_better_device(pred.device, target.device)
    return pred.to(device), target.to(device)

def _get_better_device(device1, device2):
    """Return the better device: CUDA > MPS > CPU"""
    devices = [device1, device2]
    
    # Prefer CUDA
    for device in devices:
        if device.type == 'cuda':
            return device
    
    # Then prefer MPS
    for device in devices:
        if device.type == 'mps':
            return device
    
    # Fall back to CPU
    return device1