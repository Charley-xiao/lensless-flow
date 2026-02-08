# lensless_flow/model_unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    Pick a GroupNorm group count that divides num_channels.
    Falls back to 1 (LayerNorm-like) if needed.
    """
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return max(g, 1)


class TimeEmbedding(nn.Module):
    """
    Sin/cos timestep embedding + MLP, similar to diffusion UNets.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B], assumed in [0, 1]
        returns: [B, dim*4]
        """
        half = self.dim // 2
        # log-spaced frequencies
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=t.dtype)
            * (math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class ResBlock(nn.Module):
    """
    GN + SiLU + Conv + time conditioning + GN + SiLU + Conv with skip.
    """
    def __init__(self, in_ch: int, out_ch: int, t_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_pick_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(t_ch, out_ch)

        self.norm2 = nn.GroupNorm(_pick_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


def _match_spatial(h: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Make h have the same H,W as ref by padding or cropping.
    """
    Ht, Wt = h.shape[-2], h.shape[-1]
    Hr, Wr = ref.shape[-2], ref.shape[-1]

    # pad if too small
    pad_h = max(Hr - Ht, 0)
    pad_w = max(Wr - Wt, 0)
    if pad_h > 0 or pad_w > 0:
        h = F.pad(h, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)

    # crop if too large
    if h.shape[-2] > Hr:
        h = h[..., :Hr, :]
    if h.shape[-1] > Wr:
        h = h[..., :, :Wr]

    return h


class SimpleCondUNet(nn.Module):
    """
    Conditional U-Net for v_theta(t, x_t, y).

    Inputs:
      x_t: [B, C, H, W]
      y:   [B, C, H, W]
      t:   [B] in [0,1]

    Output:
      v:   [B, C, H, W]
    """
    def __init__(
        self,
        img_channels: int = 3,
        base_ch: int = 64,
        channel_mults=(1, 2, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        assert num_res_blocks >= 1, "num_res_blocks must be >= 1"

        self.img_channels = img_channels
        self.tdim = 128
        self.time = TimeEmbedding(self.tdim)

        t_ch = self.tdim * 4

        # Input: concat [x_t, y] along channels => 2C
        self.in_conv = nn.Conv2d(2 * img_channels, base_ch, kernel_size=3, padding=1)

        # Encoder
        chs = [base_ch * m for m in channel_mults]
        self.down_stages = nn.ModuleList()
        cur = base_ch
        for ch in chs:
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(cur, ch, t_ch))
                cur = ch
            down = Downsample(cur)
            self.down_stages.append(nn.ModuleDict({"blocks": blocks, "down": down}))

        # Bottleneck
        self.mid1 = ResBlock(cur, cur, t_ch)
        self.mid2 = ResBlock(cur, cur, t_ch)

        # Decoder
        self.up_stages = nn.ModuleList()
        for ch in reversed(chs):
            up = Upsample(cur)

            # IMPORTANT: only the FIRST block after concatenation sees (cur + ch) channels.
            blocks = nn.ModuleList()
            blocks.append(ResBlock(cur + ch, ch, t_ch))
            cur = ch
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(cur, cur, t_ch))

            self.up_stages.append(nn.ModuleDict({"up": up, "blocks": blocks}))

        self.out_norm = nn.GroupNorm(_pick_groups(cur), cur)
        self.out_conv = nn.Conv2d(cur, img_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # time embedding
        t_emb = self.time(t)  # [B, t_ch]

        # input stem
        h = torch.cat([x_t, y], dim=1)
        h = self.in_conv(h)

        # encoder with skips
        skips = []
        for stage in self.down_stages:
            for blk in stage["blocks"]:
                h = blk(h, t_emb)
            skips.append(h)
            h = stage["down"](h)

        # bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # decoder
        for stage in self.up_stages:
            h = stage["up"](h)
            skip = skips.pop()
            h = _match_spatial(h, skip)
            h = torch.cat([h, skip], dim=1)
            for blk in stage["blocks"]:
                h = blk(h, t_emb)

        # output head
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h
