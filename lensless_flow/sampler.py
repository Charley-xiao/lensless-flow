import torch
from .physics import data_consistency_step

@torch.no_grad()
def sample_with_physics_guidance(model, y, H, steps: int, dc_step: float, dc_steps: int,
                                init_noise_std: float = 1.0, disable_physics: bool = False):
    """
    Euler integrate dx/dt = v_theta(t, x, y), t: 0->1, with DC correction per step.
    """
    device = y.device
    x = torch.randn_like(y) * init_noise_std  # start from noise in image space

    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(y.shape[0])
        dt = ts[i + 1] - ts[i]

        v = model(x, y, t)
        x = x + dt * v

        # physics-guided step: enforce Hx ≈ y
        if not disable_physics:
            x = data_consistency_step(x, y, H, step=dc_step, iters=dc_steps)

    return x
