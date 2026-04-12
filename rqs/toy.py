import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
)

# =========================================================
# 0. Config
# =========================================================
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    train_size: int = 20000
    val_size: int = 4000
    batch_size: int = 256

    hidden_dim: int = 128
    num_layers: int = 3

    lr: float = 1e-3
    steps: int = 4000
    print_every: int = 200

    # Euler steps for ODE sampling
    nfe: int = 100

    # Toy-data noise
    obs_noise_std: float = 0.03
    vertical_std: float = 0.12
    sigma_fm: float = 0.0  # use deterministic interpolation first


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# =========================================================
# 1. Toy paired dataset: x=(s*a, b), y=(a, b)+noise
#    Same shape: both are 2D vectors
# =========================================================
@torch.no_grad()
def sample_paired_data(n, device):
    # hidden sign => ambiguity
    s = torch.randint(0, 2, (n,), device=device).float() * 2 - 1  # in {-1, +1}
    a = 0.6 + 0.8 * torch.rand(n, device=device)                  # positive magnitude
    b = cfg.vertical_std * torch.randn(n, device=device)

    x = torch.stack([s * a, b], dim=1)

    # y loses sign information in the first coordinate
    y_clean = torch.stack([a, b], dim=1)
    y = y_clean + cfg.obs_noise_std * torch.randn_like(y_clean)

    return x, y


x_train, y_train = sample_paired_data(cfg.train_size, cfg.device)
x_val, y_val = sample_paired_data(cfg.val_size, cfg.device)


# =========================================================
# 2. Simple conditional vector field network
#    input: z_t, t, cond
#    output: v(z_t, t, cond)
# =========================================================
class MLPVectorField(nn.Module):
    def __init__(self, data_dim=2, cond_dim=2, hidden_dim=128, num_layers=3):
        super().__init__()
        in_dim = data_dim + cond_dim + 1

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [data_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, z, t, cond):
        if t.ndim == 1:
            t = t[:, None]
        inp = torch.cat([z, t, cond], dim=1)
        return self.net(inp)


# =========================================================
# 3. Simple Euler ODE solver
# =========================================================
@torch.no_grad()
def euler_integrate(model, z0, cond, nfe):
    z = z0.clone()
    dt = 1.0 / nfe
    batch = z.shape[0]

    for i in range(nfe):
        t = torch.full((batch,), i / nfe, device=z.device, dtype=z.dtype)
        v = model(z, t, cond)
        z = z + dt * v
    return z


# =========================================================
# 4. Train two models
#    A: noise -> x | y      using TargetConditionalFlowMatcher
#    B: y -> x              using ConditionalFlowMatcher
# =========================================================
class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        self.model_noise2x = MLPVectorField(
            data_dim=2, cond_dim=2,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers
        ).to(cfg.device)

        self.model_y2x = MLPVectorField(
            data_dim=2, cond_dim=2,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers
        ).to(cfg.device)

        self.fm_noise2x = TargetConditionalFlowMatcher(sigma=cfg.sigma_fm)
        self.fm_y2x = ConditionalFlowMatcher(sigma=cfg.sigma_fm)

        self.opt_noise2x = optim.Adam(self.model_noise2x.parameters(), lr=cfg.lr)
        self.opt_y2x = optim.Adam(self.model_y2x.parameters(), lr=cfg.lr)

    def sample_batch(self):
        idx = torch.randint(0, x_train.shape[0], (self.cfg.batch_size,), device=self.cfg.device)
        xb = x_train[idx]
        yb = y_train[idx]
        return xb, yb

    def train_step_noise2x(self, xb, yb):
        """
        Scheme 1:
            condition on y, start from Gaussian noise, flow to x
        """
        # x0 here is only used as a shape carrier for sampling noise path in TargetCFM
        x0 = torch.randn_like(xb)
        t, xt, ut = self.fm_noise2x.sample_location_and_conditional_flow(x0, xb)

        pred = self.model_noise2x(xt, t, yb)
        loss = ((pred - ut) ** 2).mean()

        self.opt_noise2x.zero_grad()
        loss.backward()
        self.opt_noise2x.step()
        return loss.item()

    def train_step_y2x(self, xb, yb):
        """
        Scheme 2:
            directly bridge from y to x
        """
        t, xt, ut = self.fm_y2x.sample_location_and_conditional_flow(yb, xb)

        # still feed y as explicit condition for fairness
        pred = self.model_y2x(xt, t, yb)
        loss = ((pred - ut) ** 2).mean()

        self.opt_y2x.zero_grad()
        loss.backward()
        self.opt_y2x.step()
        return loss.item()

    def train(self):
        hist_a, hist_b = [], []

        for step in range(1, self.cfg.steps + 1):
            xb, yb = self.sample_batch()

            loss_a = self.train_step_noise2x(xb, yb)
            loss_b = self.train_step_y2x(xb, yb)

            hist_a.append(loss_a)
            hist_b.append(loss_b)

            if step % self.cfg.print_every == 0:
                print(
                    f"step {step:4d} | "
                    f"noise->x loss: {sum(hist_a[-self.cfg.print_every:]) / self.cfg.print_every:.6f} | "
                    f"y->x loss: {sum(hist_b[-self.cfg.print_every:]) / self.cfg.print_every:.6f}"
                )

        return hist_a, hist_b

    @torch.no_grad()
    def sample_from_noise2x(self, y, num_samples=None):
        if num_samples is None:
            num_samples = y.shape[0]
        z0 = torch.randn(num_samples, 2, device=self.cfg.device)
        return euler_integrate(self.model_noise2x, z0, y, self.cfg.nfe)

    @torch.no_grad()
    def sample_from_y2x(self, y):
        z0 = y.clone()
        return euler_integrate(self.model_y2x, z0, y, self.cfg.nfe)


exp = Experiment(cfg)
loss_a, loss_b = exp.train()


# =========================================================
# 5. Evaluation
# =========================================================
@torch.no_grad()
def evaluate_single_sample_mse(exp, x_val, y_val):
    """
    This metric favors deterministic reconstruction,
    so it is NOT the main metric in the ambiguous case.
    We report it only as a reference.
    """
    pred_a = exp.sample_from_noise2x(y_val, num_samples=y_val.shape[0])
    pred_b = exp.sample_from_y2x(y_val)

    mse_a = ((pred_a - x_val) ** 2).mean().item()
    mse_b = ((pred_b - x_val) ** 2).mean().item()
    return mse_a, mse_b


@torch.no_grad()
def fixed_y_probe(exp, y_value=(1.0, 0.0), num_samples=2000):
    """
    Probe one ambiguous condition y repeatedly.
    For the true conditional distribution, x should have two modes:
    approximately (+1, 0) and (-1, 0).
    """
    y = torch.tensor(y_value, device=cfg.device, dtype=torch.float32)[None, :].repeat(num_samples, 1)

    samples_a = exp.sample_from_noise2x(y, num_samples=num_samples)
    samples_b = exp.sample_from_y2x(y)

    return samples_a.cpu(), samples_b.cpu()


mse_a, mse_b = evaluate_single_sample_mse(exp, x_val, y_val)
print(f"\nSingle-sample MSE (reference only)")
print(f"noise -> x | y : {mse_a:.6f}")
print(f"y -> x         : {mse_b:.6f}")


# =========================================================
# 6. Plot
# =========================================================
samples_a, samples_b = fixed_y_probe(exp, y_value=(1.0, 0.0), num_samples=2000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Ground truth conditional support for y=(1,0):
# x should be near (+1,0) or (-1,0)
gt_left = torch.tensor([-1.0, 0.0])
gt_right = torch.tensor([1.0, 0.0])

axes[0].scatter(samples_a[:, 0], samples_a[:, 1], s=5, alpha=0.35)
axes[0].scatter([gt_left[0], gt_right[0]], [gt_left[1], gt_right[1]], s=120, marker="x")
axes[0].set_title("Scheme 1: noise -> x | y")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].axis("equal")
axes[0].grid(True, alpha=0.2)

axes[1].scatter(samples_b[:, 0], samples_b[:, 1], s=5, alpha=0.35)
axes[1].scatter([gt_left[0], gt_right[0]], [gt_left[1], gt_right[1]], s=120, marker="x")
axes[1].set_title("Scheme 2: y -> x")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
axes[1].axis("equal")
axes[1].grid(True, alpha=0.2)

# training curves
axes[2].plot(loss_a, label="noise->x|y")
axes[2].plot(loss_b, label="y->x")
axes[2].set_title("Training loss")
axes[2].set_xlabel("step")
axes[2].set_ylabel("MSE on conditional flow")
axes[2].legend()
axes[2].grid(True, alpha=0.2)

plt.tight_layout()
plt.show()


# =========================================================
# 7. Optional quantitative diversity check
# =========================================================
@torch.no_grad()
def sign_statistics(samples):
    # proportion on left/right
    x1 = samples[:, 0]
    right_ratio = (x1 > 0).float().mean().item()
    left_ratio = (x1 < 0).float().mean().item()
    return left_ratio, right_ratio, x1.std().item()

left_a, right_a, std_a = sign_statistics(samples_a)
left_b, right_b, std_b = sign_statistics(samples_b)

print("\nProbe y=(1,0) repeated sampling:")
print(f"Scheme 1 noise->x|y: left={left_a:.3f}, right={right_a:.3f}, std(x1)={std_a:.3f}")
print(f"Scheme 2 y->x      : left={left_b:.3f}, right={right_b:.3f}, std(x1)={std_b:.3f}")