import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= max(embed_dim // 2, 1)
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_hw: tuple[int, int]) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    grid_h = np.arange(grid_hw[0], dtype=np.float32)
    grid_w = np.arange(grid_hw[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, indexing="xy")
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_hw[0], grid_hw[1])

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class TimestepEmbedder(nn.Module):
    """
    SiT-style sinusoidal timestep embedding followed by a small MLP.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class PatchEmbed2D(nn.Module):
    """
    Patchify an image with a strided conv and return [B, N, D] tokens.
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int | tuple[int, int]):
        super().__init__()
        self.patch_size = _pair(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(bsz, num_tokens, channels)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_dim, bias=True)
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class SiTBlock(nn.Module):
    """
    Transformer block with adaLN-zero conditioning, following SiT.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: tuple[int, int], out_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = int(out_channels)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * self.out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class ConditionalSiT(nn.Module):
    """
    SiT-style conditional backbone for lensless flow matching.

    We patchify both the noisy state x_t and the measurement y with separate
    patch embedders, sum them with a fixed 2D sin-cos position embedding, and
    condition all transformer blocks with adaLN-zero timestep modulation.
    """

    def __init__(
        self,
        img_channels: int,
        im_hw: tuple[int, int],
        patch_size: int | tuple[int, int] = 16,
        hidden_size: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_time_conditioning: bool = True,
    ):
        super().__init__()
        self.img_channels = int(img_channels)
        self.im_hw = (int(im_hw[0]), int(im_hw[1]))
        self.patch_size = _pair(patch_size)
        self.hidden_size = int(hidden_size)
        self.use_time_conditioning = bool(use_time_conditioning)

        ph, pw = self.patch_size
        padded_h = _ceil_div(self.im_hw[0], ph) * ph
        padded_w = _ceil_div(self.im_hw[1], pw) * pw
        self.padded_hw = (padded_h, padded_w)
        self.grid_hw = (padded_h // ph, padded_w // pw)
        self.num_patches = self.grid_hw[0] * self.grid_hw[1]

        self.x_embedder = PatchEmbed2D(self.img_channels, self.hidden_size, self.patch_size)
        self.y_embedder = PatchEmbed2D(self.img_channels, self.hidden_size, self.patch_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.blocks = nn.ModuleList(
            [SiTBlock(self.hidden_size, num_heads=int(num_heads), mlp_ratio=float(mlp_ratio)) for _ in range(int(depth))]
        )
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, out_channels=self.img_channels)

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_hw)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for embedder in [self.x_embedder, self.y_embedder]:
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            if embedder.proj.bias is not None:
                nn.init.constant_(embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = self.padded_hw[0] - x.shape[-2]
        pad_w = self.padded_hw[1] - x.shape[-1]
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        gh, gw = self.grid_hw
        ph, pw = self.patch_size
        x = x.reshape(bsz, gh, gw, ph, pw, self.img_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(bsz, self.img_channels, gh * ph, gw * pw)
        return x[..., : self.im_hw[0], : self.im_hw[1]]

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if x_t.shape[-2:] != self.im_hw or y.shape[-2:] != self.im_hw:
            raise ValueError(
                f"ConditionalSiT expected inputs with spatial size {self.im_hw}, "
                f"got x_t={tuple(x_t.shape[-2:])}, y={tuple(y.shape[-2:])}"
            )

        x_tokens = self.x_embedder(self._pad_input(x_t))
        y_tokens = self.y_embedder(self._pad_input(y))
        tokens = x_tokens + y_tokens + self.pos_embed.to(dtype=x_tokens.dtype, device=x_tokens.device)

        if self.use_time_conditioning:
            if t is None:
                raise ValueError("t must be provided when use_time_conditioning=True.")
            c = self.t_embedder(t)
        else:
            c = x_t.new_zeros((x_t.shape[0], self.hidden_size))

        for block in self.blocks:
            tokens = block(tokens, c)

        patches = self.final_layer(tokens, c)
        return self._unpatchify(patches)
