from dataclasses import dataclass

import torch


@dataclass
class FlowSourceSample:
    x_source: torch.Tensor
    x_init: torch.Tensor | None
    source_mode: str


def normalize_source_mode(mode: str | None) -> str:
    if mode is None:
        return "gaussian"

    key = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "gaussian": "gaussian",
        "noise": "gaussian",
        "standard": "gaussian",
        "measurement": "measurement_initialized",
        "measurement_init": "measurement_initialized",
        "measurement_initialized": "measurement_initialized",
        "measurement_initialised": "measurement_initialized",
        "backprojection": "measurement_initialized",
        "adjoint": "measurement_initialized",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported cfm.source.mode='{mode}'. "
            "Expected 'gaussian' or 'measurement_initialized'."
        )
    return aliases[key]


def normalize_init_method(method: str | None) -> str:
    if method is None:
        return "adjoint"

    key = str(method).strip().lower().replace("-", "_")
    aliases = {
        "admm": "admm",
        "admm_box": "admm",
        "box_admm": "admm",
        "adjoint": "adjoint",
        "backprojection": "adjoint",
        "h_t_y": "adjoint",
        "measurement": "measurement",
        "identity": "measurement",
        "y": "measurement",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported cfm.source.init='{method}'. "
            "Expected 'admm', 'adjoint', or 'measurement'."
        )
    return aliases[key]


def normalize_admm_start(method: str | None) -> str:
    if method is None:
        return "adjoint"

    key = str(method).strip().lower().replace("-", "_")
    aliases = {
        "adjoint": "adjoint",
        "backprojection": "adjoint",
        "h_t_y": "adjoint",
        "measurement": "measurement",
        "identity": "measurement",
        "y": "measurement",
        "zero": "zeros",
        "zeros": "zeros",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported cfm.source.admm.start='{method}'. "
            "Expected 'adjoint', 'measurement', or 'zeros'."
        )
    return aliases[key]


def _sample_dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.ndim))


def normalize_measurement_init(
    x_init: torch.Tensor,
    mode: str | None = "max",
    eps: float = 1e-8,
) -> torch.Tensor:
    mode = "max" if mode is None else str(mode).strip().lower().replace("-", "_")

    if mode in {"none", "raw"}:
        return x_init

    if mode == "clamp":
        return x_init.clamp(0.0, 1.0)

    dims = _sample_dims(x_init)
    if mode in {"max", "amax"}:
        scale = x_init.amax(dim=dims, keepdim=True).clamp_min(float(eps))
        return (x_init / scale).clamp(0.0, 1.0)

    if mode == "absmax":
        scale = x_init.abs().amax(dim=dims, keepdim=True).clamp_min(float(eps))
        return (x_init / scale).clamp(0.0, 1.0)

    if mode == "minmax":
        x_min = x_init.amin(dim=dims, keepdim=True)
        x_max = x_init.amax(dim=dims, keepdim=True)
        return ((x_init - x_min) / (x_max - x_min).clamp_min(float(eps))).clamp(
            0.0,
            1.0,
        )

    raise ValueError(
        f"Unsupported cfm.source.normalize='{mode}'. "
        "Expected 'max', 'minmax', 'absmax', 'clamp', or 'none'."
    )


def measurement_initialization(
    y: torch.Tensor,
    H,
    method: str | None = "adjoint",
    normalize: str | None = "max",
    admm_steps: int = 5,
    admm_inner_steps: int = 1,
    admm_rho: float = 0.1,
    admm_step_size: float = 0.001,
    admm_start: str | None = "adjoint",
    admm_start_normalize: str | None = "max",
) -> torch.Tensor:
    method = normalize_init_method(method)

    with torch.no_grad():
        if method == "admm":
            x_init = admm_measurement_initialization(
                y=y,
                H=H,
                steps=admm_steps,
                inner_steps=admm_inner_steps,
                rho=admm_rho,
                step_size=admm_step_size,
                start=admm_start,
                start_normalize=admm_start_normalize,
            )
        elif method == "adjoint":
            if H is None:
                raise ValueError("cfm.source.init='adjoint' requires a forward operator H.")
            x_init = H.adjoint(y.float()).to(dtype=y.dtype)
        elif method == "measurement":
            x_init = y
        else:
            raise ValueError(f"Unsupported measurement initialization method: {method}")

    return normalize_measurement_init(x_init, mode=normalize)


def _admm_initial_x(
    y: torch.Tensor,
    H,
    start: str | None,
    start_normalize: str | None,
) -> torch.Tensor:
    start = normalize_admm_start(start)

    if start == "adjoint":
        if H is None:
            raise ValueError("cfm.source.admm.start='adjoint' requires a forward operator H.")
        x = H.adjoint(y.float()).to(dtype=y.dtype)
        return normalize_measurement_init(x, mode=start_normalize)

    if start == "measurement":
        return normalize_measurement_init(y, mode=start_normalize)

    if start == "zeros":
        return torch.zeros_like(y)

    raise ValueError(f"Unsupported ADMM start method: {start}")


def admm_measurement_initialization(
    y: torch.Tensor,
    H,
    steps: int = 5,
    inner_steps: int = 1,
    rho: float = 0.1,
    step_size: float = 0.001,
    start: str | None = "adjoint",
    start_normalize: str | None = "max",
) -> torch.Tensor:
    """
    Coarse constrained ADMM initializer for x_init = P(y, H_nominal).

    We solve a small number of approximate iterations for:
        min_x 0.5 ||H(x) - y||^2 + I_[0,1](z), subject to x = z.

    The repo's practical forward model normalizes H(x), so the x-update uses a
    few gradient steps with H.adjoint as the backprojection direction instead of
    a closed-form FFT inverse.
    """
    if H is None:
        raise ValueError("cfm.source.init='admm' requires a forward operator H.")

    steps = max(0, int(steps))
    inner_steps = max(1, int(inner_steps))
    rho = float(rho)
    step_size = float(step_size)
    if rho <= 0:
        raise ValueError("cfm.source.admm.rho must be positive.")
    if step_size <= 0:
        raise ValueError("cfm.source.admm.step_size must be positive.")

    x = _admm_initial_x(y, H, start=start, start_normalize=start_normalize).float()
    y_f = y.float()
    z = x.clamp(0.0, 1.0)
    u = torch.zeros_like(z)

    for _ in range(steps):
        for _ in range(inner_steps):
            residual = H.forward(x) - y_f
            data_grad = H.adjoint(residual).float()
            penalty_grad = rho * (x - z + u)
            x = x - step_size * (data_grad + penalty_grad)

        z = (x + u).clamp(0.0, 1.0)
        u = u + x - z

    return z.to(dtype=y.dtype)


def sample_flow_source(
    y: torch.Tensor,
    H,
    mode: str | None,
    noise_std: float,
    init_method: str | None = "adjoint",
    init_normalize: str | None = "max",
    x_like: torch.Tensor | None = None,
    admm_steps: int = 5,
    admm_inner_steps: int = 1,
    admm_rho: float = 0.1,
    admm_step_size: float = 0.001,
    admm_start: str | None = "adjoint",
    admm_start_normalize: str | None = "max",
) -> FlowSourceSample:
    source_mode = normalize_source_mode(mode)
    noise_like = y if x_like is None else x_like
    noise = float(noise_std) * torch.randn_like(noise_like)

    if source_mode == "gaussian":
        return FlowSourceSample(
            x_source=noise,
            x_init=None,
            source_mode=source_mode,
        )

    x_init = measurement_initialization(
        y=y,
        H=H,
        method=init_method,
        normalize=init_normalize,
        admm_steps=admm_steps,
        admm_inner_steps=admm_inner_steps,
        admm_rho=admm_rho,
        admm_step_size=admm_step_size,
        admm_start=admm_start,
        admm_start_normalize=admm_start_normalize,
    )
    return FlowSourceSample(
        x_source=x_init + noise,
        x_init=x_init,
        source_mode=source_mode,
    )


def source_cfg(cfg: dict) -> dict:
    return dict(cfg.get("cfm", {}).get("source", {}) or {})


def source_mode_from_cfg(cfg: dict) -> str:
    return normalize_source_mode(source_cfg(cfg).get("mode", "gaussian"))


def source_sigma0_from_cfg(cfg: dict) -> float:
    cfg_source = source_cfg(cfg)
    return float(cfg_source.get("sigma0", cfg.get("sample", {}).get("init_noise_std", 1.0)))


def source_sampler_kwargs_from_cfg(cfg: dict) -> dict:
    cfg_source = source_cfg(cfg)
    admm_cfg = dict(cfg_source.get("admm", {}) or {})
    return {
        "source_mode": normalize_source_mode(cfg_source.get("mode", "gaussian")),
        "source_init": normalize_init_method(cfg_source.get("init", "adjoint")),
        "source_init_normalize": cfg_source.get("normalize", "max"),
        "source_admm_steps": int(admm_cfg.get("steps", 5)),
        "source_admm_inner_steps": int(admm_cfg.get("inner_steps", 1)),
        "source_admm_rho": float(admm_cfg.get("rho", 0.1)),
        "source_admm_step_size": float(admm_cfg.get("step_size", 0.001)),
        "source_admm_start": normalize_admm_start(admm_cfg.get("start", "adjoint")),
        "source_admm_start_normalize": admm_cfg.get("start_normalize", "max"),
    }
