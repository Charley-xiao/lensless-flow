import torch


@torch.no_grad()
def suggested_dc_step(Hop, safety=0.5, eps=1e-12):
    # Hop.otf: [1,C,H,W] complex
    L = (Hop.otf.abs() ** 2).amax().item()
    # for grad of ||H x - y||^2, gradient has factor 2
    step = safety / (2.0 * (L + eps))
    return float(step), float(L)


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W] in any range
    returns: [B,1,H,W]
    Using standard Rec.601 luma weights.
    """
    # weights sum to 1
    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def _dc_refine_rgb(x: torch.Tensor, y: torch.Tensor, H, step: float, iters: int) -> torch.Tensor:
    """
    Original DC: per-channel DC on full [B,C,H,W].
    Gradient descent on ||H x - y||^2 performed in float32 for stability.
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


def _dc_refine_luma(x: torch.Tensor, y: torch.Tensor, H, step: float, iters: int) -> torch.Tensor:
    """
    Luma-only DC (recommended for RGB):
      - Compute luma(x) and luma(y)
      - Do DC update ONLY on luma channel with a *single-channel* H (use channel 0 OTF)
      - Apply the luma delta back to all RGB channels equally.

    Assumptions:
      - x,y are [B,3,H,W] (RGB). If C!=3, we fall back to rgb DC.
      - H is FFTConvOperator with buffer H.otf of shape [1,C,H,W].
        We use only H.otf[:,0:1] as luma forward/adjoint.
    """
    if step <= 0 or iters <= 0:
        return x

    if x.shape[1] != 3 or y.shape[1] != 3:
        # Not RGB: fallback to original
        return _dc_refine_rgb(x, y, H, step, iters)

    # Work in float32 for stability
    x_f = x.float()
    y_f = y.float()

    # Build a "single-channel view" operator using only channel 0 OTF.
    # This avoids constraining colors with potentially mismatched per-channel OTFs.
    otf1 = H.otf[:, 0:1, ...]  # [1,1,H,W] complex

    def H1_forward(u):
        U = torch.fft.fft2(u)
        V = U * otf1
        return torch.fft.ifft2(V).real

    def H1_adjoint(v):
        V = torch.fft.fft2(v)
        U = V * torch.conj(otf1)
        return torch.fft.ifft2(U).real

    for _ in range(iters):
        # luma residual
        xY = _rgb_to_luma(x_f)   # [B,1,H,W]
        yY = _rgb_to_luma(y_f)   # [B,1,H,W]
        rY = H1_forward(xY) - yY
        gradY = 2.0 * H1_adjoint(rY)  # [B,1,H,W]

        # update luma only
        xY_new = xY - float(step) * gradY
        deltaY = xY_new - xY  # [B,1,H,W]

        # add luma delta equally back to RGB channels
        x_f = x_f + deltaY.repeat(1, 3, 1, 1)

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
    dc_mode: str = "luma",    # "luma" (recommended) or "rgb"
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

    Optional DC after each RK2 step.

    dc_mode:
      - "rgb": original per-channel DC on RGB
      - "luma": luma-only DC, then apply delta to all channels (reduces weird color shifts)
    """
    pred_type = str(pred_type).lower()
    assert pred_type in ["btb", "vanilla"], f"pred_type must be 'btb' or 'vanilla', got {pred_type}"

    dc_mode = str(dc_mode).lower()
    assert dc_mode in ["rgb", "luma"], f"dc_mode must be 'rgb' or 'luma', got {dc_mode}"

    device = y.device
    B = y.shape[0]

    # initial state z0 ~ N(0, I)
    z = init_noise_std * torch.randn_like(y)

    # time grid
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=y.dtype)

    # Auto DC step if user didn't specify (or set <=0)
    if (not disable_physics) and dc_steps > 0 and (dc_step is None or dc_step <= 0):
        safety = float(getattr(H, "dc_safety", 0.5))  # optional
        dc_step, L = suggested_dc_step(H, safety=safety)
        print(f"Auto DC step: {dc_step:.4e} (L={L:.4e}, safety={safety})")

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
            if dc_mode == "luma":
                z = _dc_refine_luma(z, y, H, step=dc_step, iters=dc_steps)
            else:
                z = _dc_refine_rgb(z, y, H, step=dc_step, iters=dc_steps)

        if clamp_x:
            z = z.clamp(0.0, 1.0)

    return z