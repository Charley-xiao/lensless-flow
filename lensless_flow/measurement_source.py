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
            "Expected 'adjoint' or 'measurement'."
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
) -> torch.Tensor:
    method = normalize_init_method(method)

    if method == "adjoint":
        if H is None:
            raise ValueError("cfm.source.init='adjoint' requires a forward operator H.")
        x_init = H.adjoint(y.float()).to(dtype=y.dtype)
    elif method == "measurement":
        x_init = y
    else:
        raise ValueError(f"Unsupported measurement initialization method: {method}")

    return normalize_measurement_init(x_init, mode=normalize)


def sample_flow_source(
    y: torch.Tensor,
    H,
    mode: str | None,
    noise_std: float,
    init_method: str | None = "adjoint",
    init_normalize: str | None = "max",
    x_like: torch.Tensor | None = None,
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
    return {
        "source_mode": normalize_source_mode(cfg_source.get("mode", "gaussian")),
        "source_init": normalize_init_method(cfg_source.get("init", "adjoint")),
        "source_init_normalize": cfg_source.get("normalize", "max"),
    }
