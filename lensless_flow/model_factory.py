import torch.nn as nn

from lensless_flow.model_sit import ConditionalSiT
from lensless_flow.model_unet import (
    SimpleCondUNet,
    resolve_baseline_use_time_conditioning,
    resolve_use_time_conditioning,
)


def normalize_model_name(name: str | None) -> str:
    if name is None:
        return "unet"
    name = str(name).strip().lower()
    aliases = {
        "unet": "unet",
        "simplecondunet": "unet",
        "simple_cond_unet": "unet",
        "sit": "sit",
        "conditionalsit": "sit",
        "conditional_sit": "sit",
    }
    if name not in aliases:
        raise ValueError(f"Unsupported model.name/model_name='{name}'")
    return aliases[name]


def resolve_model_name(cfg: dict | None = None, checkpoint_state: dict | None = None, default: str = "unet") -> str:
    if isinstance(checkpoint_state, dict):
        if "model_name" in checkpoint_state:
            return normalize_model_name(checkpoint_state["model_name"])
        saved_cfg = checkpoint_state.get("cfg")
        if isinstance(saved_cfg, dict):
            return resolve_model_name(saved_cfg, checkpoint_state=None, default=default)

    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(model_cfg, dict):
        for key in ["name", "arch", "type"]:
            if key in model_cfg:
                return normalize_model_name(model_cfg[key])

    return normalize_model_name(default)


def build_flow_model(
    cfg,
    img_channels: int,
    im_hw: tuple[int, int],
    device,
    checkpoint_state: dict | None = None,
) -> nn.Module:
    model_name = resolve_model_name(cfg, checkpoint_state=checkpoint_state, default="unet")
    if model_name == "unet":
        model = SimpleCondUNet(
            img_channels=img_channels,
            base_ch=cfg["model"]["base_channels"],
            channel_mults=tuple(cfg["model"]["channel_mults"]),
            num_res_blocks=cfg["model"]["num_res_blocks"],
            use_time_conditioning=resolve_use_time_conditioning(cfg, checkpoint_state),
        )
    elif model_name == "sit":
        model = ConditionalSiT(
            img_channels=img_channels,
            im_hw=im_hw,
            patch_size=cfg["model"].get("patch_size", 16),
            hidden_size=cfg["model"].get("hidden_size", 512),
            depth=cfg["model"].get("depth", 12),
            num_heads=cfg["model"].get("num_heads", 8),
            mlp_ratio=cfg["model"].get("mlp_ratio", 4.0),
            use_time_conditioning=resolve_use_time_conditioning(cfg, checkpoint_state),
        )
    else:
        raise AssertionError(f"Unhandled model_name={model_name}")

    return model.to(device)


def build_baseline_unet(
    cfg,
    img_channels: int,
    device,
    checkpoint_state: dict | None = None,
) -> nn.Module:
    return SimpleCondUNet(
        img_channels=img_channels,
        base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        use_time_conditioning=resolve_baseline_use_time_conditioning(checkpoint_state),
    ).to(device)


def load_checkpoint_state_dict(model: nn.Module, state) -> None:
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
