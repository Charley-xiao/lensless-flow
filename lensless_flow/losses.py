import torch
import torch.nn.functional as F
from .flow_matching import x0_from_xt_v
from .metrics import _validated_region_mask

def cfm_loss(v_pred, v_star):
    return F.mse_loss(v_pred, v_star)


def region_balanced_cfm_loss(v_pred, v_star, mask, foreground_weight=0.75,
                             balance_mix=0.5):
    """Blend ordinary velocity MSE with foreground/background balanced MSE.

    Regional means normalize H,W separately for each image and channel before
    averaging B,C. The mask is detached and may have one or C channels. An
    image/channel with either region absent uses ordinary MSE. Float32 regional
    reductions keep small mask denominators safe under mixed precision; mix=0
    returns the original ``cfm_loss`` exactly. Statistics are detached scalars.

    Target-derived masks define a task-weighted supervised surrogate, and do not
    generally preserve the original CFM population-optimal vector field.
    """
    import math

    foreground_weight, balance_mix = float(foreground_weight), float(balance_mix)
    if not math.isfinite(foreground_weight) or not 0 <= foreground_weight <= 1:
        raise ValueError("foreground_weight must be finite and in [0,1].")
    if not math.isfinite(balance_mix) or not 0 <= balance_mix <= 1:
        raise ValueError("balance_mix must be finite and in [0,1].")
    if v_pred.shape != v_star.shape or v_pred.device != v_star.device:
        raise ValueError("v_pred and v_star must have matching shapes and devices.")
    mask = _validated_region_mask(mask, v_pred)
    error = (v_pred.float() - v_star.float()).square()
    dims = (-2, -1)
    pixel_count = v_pred.shape[-2] * v_pred.shape[-1]
    fg_count = mask.sum(dims)
    bg_count = (1.0 - mask).sum(dims)
    fg_sum = (error * mask).sum(dims)
    bg_sum = (error * (1.0 - mask)).sum(dims)
    ordinary = error.mean(dims)
    fg_mse = fg_sum / fg_count.clamp_min(1e-12)
    bg_mse = bg_sum / bg_count.clamp_min(1e-12)
    valid = (fg_count > 0) & (bg_count > 0)
    balanced = foreground_weight * fg_mse + (1.0 - foreground_weight) * bg_mse
    balanced = torch.where(valid, balanced, ordinary)
    loss = ((1.0 - balance_mix) * ordinary + balance_mix * balanced).mean()
    if balance_mix == 0:
        loss = cfm_loss(v_pred, v_star)

    ordinary_fg = fg_sum / pixel_count
    weighted_fg = ((1.0 - balance_mix) * ordinary_fg
                   + balance_mix * torch.where(valid, foreground_weight * fg_mse, ordinary_fg))

    def present_mean(values, present):
        count = present.sum()
        mean = torch.where(present, values, 0.0).sum() / count.clamp_min(1)
        return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))

    stats = {
        "unweighted_mse": ordinary.mean(),
        "foreground_mse": present_mean(fg_mse, fg_count > 0),
        "background_mse": present_mean(bg_mse, bg_count > 0),
        "foreground_fraction": mask.mean(),
        "ordinary_foreground_contribution": ordinary_fg.mean(),
        "weighted_foreground_contribution": weighted_fg.mean(),
        "valid_mask_fraction": valid.float().mean(),
    }
    return loss, {name: value.detach() for name, value in stats.items()}


def physics_loss_from_v(x_t, v_pred, t, y_meas, H):
    """
    Predict the target image from (x_t, v_pred) and penalize ||H x_hat - y||^2.

    This assumes the deterministic straight-line path used by both the
    rectified-flow and OT-CFM matchers wired into this project.
    """
    x0_hat = x0_from_xt_v(x_t, v_pred, t)
    resid = H.forward(x0_hat) - y_meas
    return (resid ** 2).mean()
