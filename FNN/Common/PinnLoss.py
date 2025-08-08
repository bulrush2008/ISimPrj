import torch
from .util.Gradient import gradient

def incompressibility_loss(velocity: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    inps:
        velocity: (BATCH, 3)
        coords: (BATCH, 3)
    The loss reflects the Divergence(velocity) = 0 physics constraint
    """
    # coords should already have requires_grad=True from train_step
    du_dx = gradient(velocity[:, 0], coords)[0][:,0]
    dv_dy = gradient(velocity[:, 1], coords)[0][:,1]
    dw_dz = gradient(velocity[:, 2], coords)[0][:,2]
    div = torch.sum(torch.stack([du_dx, dv_dy, dw_dz], dim=1), dim=1)
    return torch.mean(div**2)