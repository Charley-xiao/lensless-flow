import torch
import torch.nn.functional as F
from .flow_matching import x0_from_xt_v

def cfm_loss(v_pred, v_star):
    return F.mse_loss(v_pred, v_star)

def physics_loss_from_v(x_t, v_pred, t, y_meas, H):
    """
    Predict x0 from (x_t, v_pred) then penalize measurement residual ||H x0 - y||^2.
    This is a *training-time* physics regularizer that complements test-time guidance.
    """
    x0_hat = x0_from_xt_v(x_t, v_pred, t)
    resid = H.forward(x0_hat) - y_meas
    return (resid ** 2).mean()
