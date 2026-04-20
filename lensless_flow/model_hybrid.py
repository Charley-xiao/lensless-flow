import torch
import torch.nn as nn
import torch.nn.functional as F

from lensless_flow.model_unet import TimeEmbedding, _match_spatial, _pick_groups


def _modulate_tokens(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class LocalFeatureBlock(nn.Module):
    """
    Lightweight MBConv-style residual block with optional time FiLM.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_ch: int | None = None,
        expand_ratio: float = 2.0,
    ):
        super().__init__()
        hidden = max(out_ch, int(round(out_ch * expand_ratio)))

        self.in_norm = nn.GroupNorm(_pick_groups(in_ch), in_ch)
        self.expand = nn.Conv2d(in_ch, hidden, kernel_size=1)
        self.hidden_norm1 = nn.GroupNorm(_pick_groups(hidden), hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.hidden_norm2 = nn.GroupNorm(_pick_groups(hidden), hidden)
        self.se = SqueezeExcite(hidden)
        self.project = nn.Conv2d(hidden, out_ch, kernel_size=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(out_ch) * 1.0e-3)
        self.time_proj = None if t_ch is None else nn.Linear(t_ch, hidden * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = F.silu(self.in_norm(x))
        h = self.expand(h)
        h = F.silu(self.hidden_norm1(h))
        h = self.depthwise(h)
        h = self.hidden_norm2(h)

        if self.time_proj is not None:
            if t_emb is None:
                raise ValueError("t_emb must be provided for time-conditioned LocalFeatureBlock.")
            shift, scale = self.time_proj(F.silu(t_emb)).chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = F.silu(h)
        h = self.se(h)
        h = self.project(h)
        h = h * self.layer_scale.view(1, -1, 1, 1)
        return self.skip(x) + h


class ConditionFusion(nn.Module):
    """
    Fuse measurement features into the evolving latent with a learned gate.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = _match_spatial(cond, x)
        joint = torch.cat([x, cond], dim=1)
        return x + self.gate(joint) * self.mix(joint)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class CrossTransformerBlock(nn.Module):
    """
    Adaptive-LN transformer block with self-attention on the latent and
    cross-attention into measurement tokens at the bottleneck.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        t_ch: int,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")

        hidden = max(int(round(channels * mlp_ratio)), channels)

        self.latent_pos = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.cond_pos = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.context_norm = nn.LayerNorm(channels)

        self.norm3 = nn.LayerNorm(channels, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        self.ada_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_ch, channels * 9),
        )
        nn.init.zeros_(self.ada_norm[-1].weight)
        nn.init.zeros_(self.ada_norm[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x = x + self.latent_pos(x)
        cond_tokens = (cond + self.cond_pos(cond)).flatten(2).transpose(1, 2)
        x_tokens = x.flatten(2).transpose(1, 2)

        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ffn, scale_ffn, gate_ffn = self.ada_norm(
            t_emb
        ).chunk(9, dim=1)

        sa_in = _modulate_tokens(self.norm1(x_tokens), shift_sa, scale_sa)
        sa_out = self.self_attn(sa_in, sa_in, sa_in, need_weights=False)[0]
        x_tokens = x_tokens + gate_sa.unsqueeze(1) * sa_out

        ca_in = _modulate_tokens(self.norm2(x_tokens), shift_ca, scale_ca)
        context = self.context_norm(cond_tokens)
        ca_out = self.cross_attn(ca_in, context, context, need_weights=False)[0]
        x_tokens = x_tokens + gate_ca.unsqueeze(1) * ca_out

        ffn_in = _modulate_tokens(self.norm3(x_tokens), shift_ffn, scale_ffn)
        x_tokens = x_tokens + gate_ffn.unsqueeze(1) * self.mlp(ffn_in)

        return x_tokens.transpose(1, 2).reshape(b, c, h, w)


class ConditionalHybridFormer(nn.Module):
    """
    Small hybrid backbone for conditional flow matching.

    Design:
    - lightweight local encoder/decoder blocks with depthwise separable convs
    - a dedicated measurement pyramid instead of only raw concatenation
    - transformer-style self/cross attention only at the coarsest scale
    """

    def __init__(
        self,
        img_channels: int = 3,
        base_ch: int = 32,
        channel_mults=(1, 2, 4, 6),
        num_res_blocks: int = 2,
        bottleneck_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        attn_pool: int = 4,
        use_time_conditioning: bool = True,
    ):
        super().__init__()
        if len(channel_mults) < 2:
            raise ValueError("channel_mults must contain at least two stages.")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1.")
        if bottleneck_depth < 1:
            raise ValueError("bottleneck_depth must be >= 1.")
        if attn_pool < 1:
            raise ValueError("attn_pool must be >= 1.")

        self.img_channels = int(img_channels)
        self.use_time_conditioning = bool(use_time_conditioning)
        self.attn_pool = int(attn_pool)
        self.tdim = max(64, base_ch * 2)
        self.time = TimeEmbedding(self.tdim)
        t_ch = self.time.out_dim

        dims = [int(base_ch * mult) for mult in channel_mults]
        self.total_downsample = 2 ** (len(dims) - 1)

        self.input_proj = nn.Conv2d(2 * self.img_channels, dims[0], kernel_size=3, padding=1)
        self.cond_proj = nn.Conv2d(self.img_channels, dims[0], kernel_size=3, padding=1)

        self.encoder = nn.ModuleList()
        cur = dims[0]
        cond_cur = dims[0]
        for i, dim in enumerate(dims):
            latent_blocks = nn.ModuleList()
            cond_blocks = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                latent_blocks.append(LocalFeatureBlock(cur if block_idx == 0 else dim, dim, t_ch=t_ch))
                cond_blocks.append(LocalFeatureBlock(cond_cur if block_idx == 0 else dim, dim, t_ch=None))
                cur = dim
                cond_cur = dim

            stage = nn.ModuleDict(
                {
                    "latent_blocks": latent_blocks,
                    "cond_blocks": cond_blocks,
                    "fusion": ConditionFusion(dim),
                }
            )
            if i < len(dims) - 1:
                stage["down"] = Downsample(dim, dims[i + 1])
                stage["cond_down"] = Downsample(dim, dims[i + 1])
                cur = dims[i + 1]
                cond_cur = dims[i + 1]
            self.encoder.append(stage)

        self.mid_latent = LocalFeatureBlock(cur, cur, t_ch=t_ch)
        self.mid_cond = LocalFeatureBlock(cur, cur, t_ch=None)
        self.global_proj = nn.Conv2d(cur, cur, kernel_size=1)
        self.transformer = nn.ModuleList(
            [CrossTransformerBlock(cur, num_heads=num_heads, t_ch=t_ch, mlp_ratio=mlp_ratio) for _ in range(bottleneck_depth)]
        )

        self.decoder = nn.ModuleList()
        for dim in reversed(dims[:-1]):
            stage = nn.ModuleDict(
                {
                    "up": Upsample(cur, dim),
                    "merge": nn.Conv2d(dim * 3, dim, kernel_size=1),
                    "blocks": nn.ModuleList([LocalFeatureBlock(dim, dim, t_ch=t_ch) for _ in range(num_res_blocks)]),
                }
            )
            self.decoder.append(stage)
            cur = dim

        self.out_norm = nn.GroupNorm(_pick_groups(cur), cur)
        self.out_conv = nn.Conv2d(cur, self.img_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        factor = self.total_downsample
        pad_h = (factor - (h % factor)) % factor
        pad_w = (factor - (w % factor)) % factor
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        return F.pad(x, (0, pad_w, 0, pad_h)), (h, w)

    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_time_conditioning:
            if t is None:
                raise ValueError("t must be provided when use_time_conditioning=True.")
            t_emb = self.time(t)
        else:
            t_emb = x_t.new_zeros((x_t.shape[0], self.time.out_dim))

        x_t, orig_hw = self._pad_to_multiple(x_t)
        y, _ = self._pad_to_multiple(y)

        latent = self.input_proj(torch.cat([x_t, y], dim=1))
        cond = self.cond_proj(y)

        skips: list[torch.Tensor] = []
        cond_skips: list[torch.Tensor] = []
        for stage_idx, stage in enumerate(self.encoder):
            for block in stage["latent_blocks"]:
                latent = block(latent, t_emb)
            for block in stage["cond_blocks"]:
                cond = block(cond, None)

            latent = stage["fusion"](latent, cond)

            if stage_idx < len(self.encoder) - 1:
                skips.append(latent)
                cond_skips.append(cond)
                latent = stage["down"](latent)
                cond = stage["cond_down"](cond)

        latent = self.mid_latent(latent, t_emb)
        cond = self.mid_cond(cond, None)

        bottleneck_hw = latent.shape[-2:]
        if self.attn_pool > 1:
            pooled_hw = (
                max(1, (bottleneck_hw[0] + self.attn_pool - 1) // self.attn_pool),
                max(1, (bottleneck_hw[1] + self.attn_pool - 1) // self.attn_pool),
            )
            latent_global = F.adaptive_avg_pool2d(latent, pooled_hw)
            cond_global = F.adaptive_avg_pool2d(cond, pooled_hw)
        else:
            latent_global = latent
            cond_global = cond

        for block in self.transformer:
            latent_global = block(latent_global, cond_global, t_emb)

        if self.attn_pool > 1:
            latent = latent + F.interpolate(
                self.global_proj(latent_global),
                size=bottleneck_hw,
                mode="bilinear",
                align_corners=False,
            )
        else:
            latent = latent_global

        for stage in self.decoder:
            latent = stage["up"](latent)
            skip = skips.pop()
            cond_skip = cond_skips.pop()
            latent = _match_spatial(latent, skip)
            cond_skip = _match_spatial(cond_skip, skip)
            latent = torch.cat([latent, skip, cond_skip], dim=1)
            latent = stage["merge"](latent)
            for block in stage["blocks"]:
                latent = block(latent, t_emb)

        out = self.out_conv(F.silu(self.out_norm(latent)))
        return out[..., : orig_hw[0], : orig_hw[1]]
