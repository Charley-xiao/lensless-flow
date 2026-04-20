import torch
import torch.nn as nn
import torch.nn.functional as F

from lensless_flow.model_unet import TimeEmbedding


class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm applied independently at each spatial location.
    """

    def __init__(self, channels: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Lightweight restoration block inspired by NAFNet.
    """

    def __init__(
        self,
        channels: int,
        t_ch: int | None = None,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
        )
        self.pw2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels * 2, kernel_size=1)
        self.ffn2 = nn.Conv2d(ffn_channels, channels, kernel_size=1)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.time_proj = None if t_ch is None else nn.Linear(t_ch, dw_channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        residual = x

        h = self.norm1(x)
        h = self.pw1(h)
        h = self.dwconv(h)

        if self.time_proj is not None:
            if t_emb is None:
                raise ValueError("t_emb must be provided for time-conditioned NAFBlock.")
            shift, scale = self.time_proj(F.silu(t_emb)).chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.sg(h)
        h = h * self.sca(h)
        h = self.pw2(h)
        h = self.dropout1(h)

        x = residual + h * self.beta

        h = self.norm2(x)
        h = self.ffn1(h)
        h = self.sg(h)
        h = self.ffn2(h)
        h = self.dropout2(h)

        return x + h * self.gamma


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        if out_ch % 4 != 0:
            raise ValueError(f"out_ch={out_ch} must be divisible by 4 for PixelUnshuffle downsampling.")
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, padding=1),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ConditionalNAFNet(nn.Module):
    """
    Conditional NAFNet-style encoder-decoder for flow reconstruction.

    The model keeps the restoration-friendly local inductive bias of NAFNet,
    while accepting the repo-standard signature: forward(x_t, y, t).
    """

    def __init__(
        self,
        img_channels: int = 3,
        base_ch: int = 32,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        middle_blocks: int = 4,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
        use_time_conditioning: bool = True,
    ):
        super().__init__()
        if len(channel_mults) < 2:
            raise ValueError("channel_mults must contain at least two stages.")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1.")
        if middle_blocks < 1:
            raise ValueError("middle_blocks must be >= 1.")

        self.img_channels = int(img_channels)
        self.use_time_conditioning = bool(use_time_conditioning)

        self.tdim = max(64, base_ch * 2)
        self.time = TimeEmbedding(self.tdim)
        t_ch = self.time.out_dim

        dims = [int(base_ch * mult) for mult in channel_mults]
        self.total_downsample = 2 ** (len(dims) - 1)

        self.intro = nn.Conv2d(2 * self.img_channels, dims[0], kernel_size=3, padding=1)
        self.ending = nn.Conv2d(dims[0], self.img_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, dim in enumerate(dims):
            self.encoders.append(
                nn.ModuleList(
                    [
                        NAFBlock(
                            channels=dim,
                            t_ch=t_ch,
                            dw_expand=dw_expand,
                            ffn_expand=ffn_expand,
                            dropout=dropout,
                        )
                        for _ in range(num_res_blocks)
                    ]
                )
            )
            if i < len(dims) - 1:
                self.downs.append(Downsample(dim, dims[i + 1]))

        self.middle = nn.ModuleList(
            [
                NAFBlock(
                    channels=dims[-1],
                    t_ch=t_ch,
                    dw_expand=dw_expand,
                    ffn_expand=ffn_expand,
                    dropout=dropout,
                )
                for _ in range(middle_blocks)
            ]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(Upsample(dims[i], dims[i - 1]))
            self.decoders.append(
                nn.ModuleList(
                    [
                        NAFBlock(
                            channels=dims[i - 1],
                            t_ch=t_ch,
                            dw_expand=dw_expand,
                            ffn_expand=ffn_expand,
                            dropout=dropout,
                        )
                        for _ in range(num_res_blocks)
                    ]
                )
            )

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

        x = torch.cat([x_t, y], dim=1)
        x = self.intro(x)

        skips: list[torch.Tensor] = []
        for stage_idx, blocks in enumerate(self.encoders):
            for block in blocks:
                x = block(x, t_emb)
            if stage_idx < len(self.downs):
                skips.append(x)
                x = self.downs[stage_idx](x)

        for block in self.middle:
            x = block(x, t_emb)

        for up, blocks in zip(self.ups, self.decoders):
            x = up(x)
            skip = skips.pop()
            x = x + skip
            for block in blocks:
                x = block(x, t_emb)

        x = self.ending(x)
        return x[..., : orig_hw[0], : orig_hw[1]]
