"""Exercise epoch logging without a network service or a large training job."""
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from lensless_flow.config import load_config
from scripts import train


class TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(-0.5))

    def forward(self, state, measurement, time):
        return state * self.scale


def test_epoch_metrics_share_commit_and_state_checkpoint_roundtrips(tmp_path, monkeypatch):
    cfg = load_config("configs/pld_rml_unet64_flow_200ep.yaml")
    cfg["device"] = "cpu"
    cfg["data"].update(dataset="diffusercam", num_workers=0)
    cfg["eval"].update(protocol=None, subset_size=2, preview_dir=None)
    cfg["cfm"]["loss"]["spatial"]["enabled"] = False
    cfg["train"].update(epochs=2, batch_size=2, amp=False,
                         checkpoint_dir=str(tmp_path), metrics_path=str(tmp_path / "metrics.jsonl"))
    cfg["wandb"].update(log_every=1, eval_batches=2, log_artifacts=False)
    cfg["sample"]["steps"] = 2
    dataset = TensorDataset(torch.zeros(2, 16, 16, 1), torch.zeros(2, 16, 16, 1))
    monkeypatch.setattr(train, "make_dataloader", lambda **kwargs:
                        (dataset, DataLoader(dataset, batch_size=kwargs["batch_size"])))
    monkeypatch.setattr(train, "build_flow_model", lambda **kwargs: TinyFlow())
    monkeypatch.setattr(train, "build_forward_operator_from_dataset", lambda *args, **kwargs: None)

    history, pending = [], {}
    current_step = 0

    def log(values, step, commit=None):
        nonlocal current_step
        assert step >= current_step, "W&B would discard a log at an already committed step"
        if step > current_step and pending:
            history.append((current_step, dict(pending)))
            pending.clear()
        current_step = step
        pending.update(values)
        if commit is not False:
            history.append((step, dict(pending)))
            pending.clear()
            current_step = step + 1

    fake = SimpleNamespace(init=lambda **kwargs: None, log=log, finish=lambda: None,
                           run=SimpleNamespace(summary={}))
    monkeypatch.setattr(train, "wandb", fake)
    train.main(cfg)
    epochs = [values for _, values in history if "epoch/train_loss" in values]
    assert len(epochs) == 2
    assert [values["epoch"] for values in epochs] == [1, 2]
    assert all("eval/ssim" in values and "sched/lr_after" in values for values in epochs)
    saved = torch.load(tmp_path / "last_state.pt", weights_only=True)
    assert saved["epoch"] == 2
    assert saved["optimizer"]["state"][0]["step"].item() == 2
    assert (tmp_path / "best.pt").exists()
