import torch


def _dc_refine(x: torch.Tensor, y: torch.Tensor, H, step: float, iters: int) -> torch.Tensor:
    """
    Gradient descent on ||H x - y||^2 performed in float32 for stability.
    x, y: [B,C,H,W]
    """
    if step <= 0 or iters <= 0:
        return x

    x_f = x.float()
    y_f = y.float()

    for _ in range(iters):
        r = H.forward(x_f) - y_f
        grad = 2.0 * H.adjoint(r)
        x_f = x_f - float(step) * grad

    return x_f.to(dtype=x.dtype)


@torch.no_grad()
def sample_with_physics_guidance(
    model,
    y,
    H,
    steps: int,
    dc_step: float,
    dc_steps: int,
    init_noise_std: float = 1.0,
    denom_min: float = 0.05,
    clamp_x: bool = True,
    disable_physics: bool = False,
):
    """
    Heun / RK2 sampling for BTB x-prediction model.

    Model outputs x_pred:
      x_pred = model(z_t, y, t)
      v(z_t,t) = (x_pred - z_t) / (1 - t)

    Heun step:
      z_euler = z + dt * v(z,t)
      z_next  = z + dt/2 * ( v(z,t) + v(z_euler,t+dt) )

    Optional DC step after each RK2 step.
    """
    device = y.device
    B = y.shape[0]

    # initial state z0 ~ N(0, I)
    z = init_noise_std * torch.randn_like(y)

    # time grid
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=y.dtype)

    for k in range(steps):
        t0 = ts[k]
        t1 = ts[k + 1]
        dt = (t1 - t0)

        t0b = torch.full((B,), float(t0), device=device, dtype=y.dtype)
        t1b = torch.full((B,), float(t1), device=device, dtype=y.dtype)

        # --- v0 at (z, t0)
        x0_pred = model(z, y, t0b)
        den0 = (1.0 - t0b).clamp_min(denom_min).view(B, 1, 1, 1)
        v0 = (x0_pred - z) / den0

        # predictor (Euler)
        z_euler = z + dt * v0
        if clamp_x:
            z_euler = z_euler.clamp(0.0, 1.0)

        # --- v1 at (z_euler, t1)
        x1_pred = model(z_euler, y, t1b)
        den1 = (1.0 - t1b).clamp_min(denom_min).view(B, 1, 1, 1)
        v1 = (x1_pred - z_euler) / den1

        # corrector (Heun)
        z = z + (dt * 0.5) * (v0 + v1)

        # optional DC refinement (do it after the RK2 update)
        if not disable_physics and dc_steps > 0 and dc_step > 0:
            z = _dc_refine(z, y, H, step=dc_step, iters=dc_steps)

        if clamp_x:
            z = z.clamp(0.0, 1.0)

    return z
