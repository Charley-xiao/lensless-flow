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
    pred_type: str = "btb",   # "btb" or "vanilla"
):
    """
    Heun / RK2 sampling that supports two model parameterizations:

    pred_type="vanilla":
        model outputs v_pred directly:
            v = model(z_t, y, t)

    pred_type="btb":
        model outputs x_pred and we convert to velocity:
            x_pred = model(z_t, y, t)
            v = (x_pred - z_t) / (1 - t)   with denom clamp

    Then solve ODE with Heun:
        z_euler = z + dt * v(z,t)
        z_next  = z + dt/2 * ( v(z,t) + v(z_euler,t+dt) )

    Optional DC after each RK2 step.
    """
    pred_type = str(pred_type).lower()
    assert pred_type in ["btb", "vanilla"], f"pred_type must be 'btb' or 'vanilla', got {pred_type}"

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
        out0 = model(z, y, t0b)
        if pred_type == "vanilla":
            v0 = out0
        else:
            den0 = (1.0 - t0b).clamp_min(denom_min).view(B, 1, 1, 1)
            v0 = (out0 - z) / den0

        # predictor (Euler)
        z_euler = z + dt * v0
        if clamp_x:
            z_euler = z_euler.clamp(0.0, 1.0)

        # --- v1 at (z_euler, t1)
        out1 = model(z_euler, y, t1b)
        if pred_type == "vanilla":
            v1 = out1
        else:
            den1 = (1.0 - t1b).clamp_min(denom_min).view(B, 1, 1, 1)
            v1 = (out1 - z_euler) / den1

        # corrector (Heun)
        z = z + (dt * 0.5) * (v0 + v1)

        # optional DC refinement (after RK2 update)
        if not disable_physics and dc_steps > 0 and dc_step > 0:
            z = _dc_refine(z, y, H, step=dc_step, iters=dc_steps)

        if clamp_x:
            z = z.clamp(0.0, 1.0)

    return z
