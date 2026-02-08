import torch

def sample_t(batch_size: int, t_min: float, t_max: float, device):
    t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
    return t

def cfm_forward(x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor, noise_std: float = 1.0):
    """
    Straight path interpolation between noise x1 and data x0:
      x_t = (1-t) * x1 + t * x0
    with x1 ~ N(0, noise_std^2 I)
    Target velocity for this path:
      v* = d/dt x_t = x0 - x1
    """
    x1 = torch.randn_like(x0) * noise_std
    t_img = t[:, None, None, None]
    x_t = (1.0 - t_img) * x1 + t_img * x0
    v_star = x0 - x1
    return x_t, v_star, x1

def x0_from_xt_v(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
    """
    From x_t = (1-t)x1 + t x0, v = x0 - x1
    => x0 = x_t + (1-t) v
    """
    t_img = t[:, None, None, None]
    return x_t + (1.0 - t_img) * v
