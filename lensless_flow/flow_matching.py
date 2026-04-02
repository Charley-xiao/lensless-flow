from dataclasses import dataclass

import torch

try:
    from torchcfm.conditional_flow_matching import (
        ConditionalFlowMatcher,
        ExactOptimalTransportConditionalFlowMatcher,
    )
except ImportError as exc:
    raise ImportError(
        "torchcfm is required for lensless_flow.flow_matching. "
        "Install it with `pip install torchcfm` or `pip install -r requirements.txt`."
    ) from exc


SUPPORTED_FLOW_MATCHERS = ("rectified", "ot_cfm")


@dataclass
class FlowMatchSample:
    t: torch.Tensor
    x_t: torch.Tensor
    v_target: torch.Tensor
    y_cond: torch.Tensor
    matcher_name: str


def normalize_flow_matcher_name(name: str | None) -> str:
    if name is None:
        return "rectified"

    key = str(name).strip().lower()
    aliases = {
        "cfm": "rectified",
        "conditional_flow_matching": "rectified",
        "independent": "rectified",
        "independent_cfm": "rectified",
        "rectified": "rectified",
        "rectified_flow": "rectified",
        "rf": "rectified",
        "ot": "ot_cfm",
        "ot-cfm": "ot_cfm",
        "ot_cfm": "ot_cfm",
        "otcfm": "ot_cfm",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported cfm.matcher='{name}'. "
            f"Expected one of {SUPPORTED_FLOW_MATCHERS} (aliases: {tuple(sorted(aliases))})."
        )
    return aliases[key]


def build_flow_matcher(name: str):
    """
    Build a TorchCFM matcher for this project.

    We intentionally keep sigma=0 because the rest of this codebase assumes the
    deterministic straight-line path x_t = (1 - t) x_src + t x_tgt for both the
    physics loss and x-pred parameterization during sampling.
    """
    matcher_name = normalize_flow_matcher_name(name)
    if matcher_name == "rectified":
        matcher = ConditionalFlowMatcher(sigma=0.0)
    elif matcher_name == "ot_cfm":
        matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    else:
        raise ValueError(f"Unknown matcher_name={matcher_name}")

    matcher.lensless_flow_name = matcher_name
    return matcher


def sample_t(batch_size: int, t_min: float, t_max: float, device):
    t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
    return t


def sample_flow_matching_training_batch(
    x_target: torch.Tensor,
    y_cond: torch.Tensor,
    t: torch.Tensor,
    flow_matcher,
    noise_std: float = 1.0,
) -> FlowMatchSample:
    """
    Sample a deterministic straight-line CFM training tuple using TorchCFM.

    `rectified` keeps the independent noise-data pairing, while `ot_cfm`
    replaces that pairing with TorchCFM's exact minibatch OT coupling and moves
    the conditioning measurement alongside the matched target image.
    """
    matcher_name = getattr(flow_matcher, "lensless_flow_name", None)
    if matcher_name is None:
        if isinstance(flow_matcher, ExactOptimalTransportConditionalFlowMatcher):
            matcher_name = "ot_cfm"
        elif isinstance(flow_matcher, ConditionalFlowMatcher):
            matcher_name = "rectified"
        else:
            raise TypeError(f"Unsupported flow_matcher type: {type(flow_matcher)!r}")
    matcher_name = normalize_flow_matcher_name(matcher_name)
    x_source = torch.randn_like(x_target) * noise_std

    if matcher_name == "ot_cfm":
        t_out, x_t, v_target, _, y_matched = flow_matcher.guided_sample_location_and_conditional_flow(
            x0=x_source,
            x1=x_target,
            y0=None,
            y1=y_cond,
            t=t,
            return_noise=False,
        )
    else:
        t_out, x_t, v_target = flow_matcher.sample_location_and_conditional_flow(
            x0=x_source,
            x1=x_target,
            t=t,
            return_noise=False,
        )
        y_matched = y_cond

    return FlowMatchSample(
        t=t_out,
        x_t=x_t,
        v_target=v_target,
        y_cond=y_matched,
        matcher_name=matcher_name,
    )


def x0_from_xt_v(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
    """
    Recover the target image for deterministic straight-line CFM / OT-CFM.

    With x_t = (1 - t) x_src + t x_tgt and v = x_tgt - x_src, we have
    x_tgt = x_t + (1 - t) v.
    """
    t_img = t[:, None, None, None]
    return x_t + (1.0 - t_img) * v
