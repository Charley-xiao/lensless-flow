import torch


def _model_uses_time_conditioning(model) -> bool:
    wrapped = getattr(model, "_orig_mod", model)
    wrapped = getattr(wrapped, "module", wrapped)
    return bool(getattr(wrapped, "use_time_conditioning", True))


@torch.no_grad()
def suggested_dc_step(Hop, safety=0.5, eps=1e-12):
    # Hop.otf: [1,C,H,W] complex
    L = (Hop.otf.abs() ** 2).amax().item()
    # for grad of ||H x - y||^2, gradient has factor 2
    step = safety / (2.0 * (L + eps))
    return float(step), float(L)


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


def _maybe_record_trajectory_state(
    trajectory: list[dict] | None,
    step_idx: int,
    time_value: float,
    state: torch.Tensor,
) -> None:
    if trajectory is None:
        return
    trajectory.append(
        {
            "step": int(step_idx),
            "time": float(time_value),
            "state": state.detach().clone(),
        }
    )


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
    dc_mode: str = "rgb",    # deprecated
    solver: str = "heun",    # "heun" (rk2) or "euler" (rk1)
    trajectory: list[dict] | None = None,
):
    """
    ODE sampling with optional physics-guided data-consistency (DC).

    solver="heun": Heun / RK2 (predictor-corrector)
    solver="euler": Euler / RK1 (single step)

    pred_type="vanilla":
        model outputs v directly: v = model(z_t, y, t), or model(z_t, y)
        when time conditioning is disabled.

    pred_type="btb":
        model outputs x_pred and we convert to velocity:
            x_pred = model(z_t, y, t)
            v = (x_pred - z_t) / (1 - t)   with denom clamp

    Optional DC after each solver step.

    If `trajectory` is provided, it is populated in-place with dictionaries
    containing the post-step state:
      {"step": int, "time": float, "state": Tensor[B,C,H,W]}
    The initial latent at t=0 is recorded as step 0.
    """
    pred_type = str(pred_type).lower()
    assert pred_type in ["btb", "vanilla"], f"pred_type must be 'btb' or 'vanilla', got {pred_type}"

    dc_mode = str(dc_mode).lower()
    assert dc_mode in ["rgb", "luma"], f"dc_mode must be 'rgb' or 'luma', got {dc_mode}"

    solver = str(solver).lower()
    assert solver in ["heun", "euler"], f"solver must be 'heun' or 'euler', got {solver}"

    use_time_conditioning = _model_uses_time_conditioning(model)
    if pred_type == "btb" and not use_time_conditioning:
        raise ValueError(
            "pred_type='btb' requires time conditioning because x-prediction depends on t. "
            "Use pred_type='vanilla' or enable model.use_time_conditioning."
        )

    device = y.device
    B = y.shape[0]

    # initial state z0 ~ N(0, I)
    z = init_noise_std * torch.randn_like(y)

    # time grid
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=y.dtype)
    _maybe_record_trajectory_state(trajectory, step_idx=0, time_value=float(ts[0]), state=z)

    # Auto DC step if user didn't specify (or set <=0)
    if (not disable_physics) and dc_steps > 0 and (dc_step is None or dc_step <= 0):
        safety = float(getattr(H, "dc_safety", 0.5))  # optional
        dc_step, L = suggested_dc_step(H, safety=safety)
        print(f"Auto DC step: {dc_step:.4e} (L={L:.4e}, safety={safety})")

    def _velocity(z_in: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
        out = model(z_in, y, t_b if use_time_conditioning else None)
        if pred_type == "vanilla":
            return out
        den = (1.0 - t_b).clamp_min(denom_min).view(B, 1, 1, 1)
        return (out - z_in) / den

    for k in range(steps):
        t0 = ts[k]
        t1 = ts[k + 1]
        dt = (t1 - t0)

        t0b = torch.full((B,), float(t0), device=device, dtype=y.dtype)
        t1b = torch.full((B,), float(t1), device=device, dtype=y.dtype)

        # v0 at (z, t0)
        v0 = _velocity(z, t0b)

        if solver == "euler":
            # Euler (RK1)
            z = z + dt * v0
            if clamp_x:
                z = z.clamp(0.0, 1.0)
        else:
            # Heun / RK2
            z_euler = z + dt * v0
            if clamp_x:
                z_euler = z_euler.clamp(0.0, 1.0)

            v1 = _velocity(z_euler, t1b)
            z = z + (dt * 0.5) * (v0 + v1)

        # optional DC refinement (after solver update)
        if not disable_physics and dc_steps > 0 and dc_step > 0: # and k <= steps // 3:
            z = _dc_refine_rgb(z, y, H, step=dc_step, iters=dc_steps)

        if clamp_x:
            z = z.clamp(0.0, 1.0)
        _maybe_record_trajectory_state(trajectory, step_idx=k + 1, time_value=float(t1), state=z)

    return z
