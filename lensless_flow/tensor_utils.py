import torch

def to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert common dataset tensor layouts to NCHW.

    Supported:
      - HW              -> 1x1xH xW
      - HWC             -> 1xC xH xW
      - CHW             -> 1xC xH xW
      - BHWC            -> BxC xH xW
      - BCHW            -> BxC xH xW
      - B1HWC (your case) -> BxC xH xW  (i.e., (B,1,H,W,C))
      - B1CHW            -> BxC xH xW  (i.e., (B,1,C,H,W))
    """
    if not torch.is_tensor(x):
        raise TypeError(f"to_nchw expects a torch.Tensor, got {type(x)}")

    if x.ndim == 2:
        return x[None, None, ...]

    if x.ndim == 3:
        # CHW or HWC
        if x.shape[0] in (1, 3):          # CHW
            return x[None, ...]
        if x.shape[-1] in (1, 3):         # HWC
            return x.permute(2, 0, 1)[None, ...]
        raise ValueError(f"Ambiguous 3D tensor shape {tuple(x.shape)}; expected CHW or HWC.")

    if x.ndim == 4:
        # BCHW or BHWC
        if x.shape[1] in (1, 3):          # BCHW
            return x
        if x.shape[-1] in (1, 3):         # BHWC
            return x.permute(0, 3, 1, 2)
        raise ValueError(f"Ambiguous 4D tensor shape {tuple(x.shape)}; expected BCHW or BHWC.")

    if x.ndim == 5:
        # Common in some lensless datasets: (B,1,H,W,C) or (B,1,C,H,W)
        if x.shape[1] == 1 and x.shape[-1] in (1, 3):
            # B1HWC -> BHWC -> BCHW
            x = x.squeeze(1)              # (B,H,W,C)
            return x.permute(0, 3, 1, 2)  # (B,C,H,W)

        if x.shape[1] == 1 and x.shape[2] in (1, 3):
            # B1CHW -> BCHW
            return x.squeeze(1)           # (B,C,H,W)

        raise ValueError(f"Unsupported 5D tensor shape {tuple(x.shape)}; expected B1HWC or B1CHW.")

    raise ValueError(f"Unsupported tensor ndim={x.ndim}, shape={tuple(x.shape)}")
