import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _move(batch, device):
    return (
        batch["rgb"].to(device),
        batch["lidar"].to(device),
        batch["y"].to(device),
    )

@torch.no_grad()
def evaluate_loss_and_f1(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    ys, ps = [], []

    for batch in loader:
        rgb, lidar, y = _move(batch, device)
        logits = model(rgb, lidar)
        loss = ce(logits, y)
        total += loss.item() * y.size(0)
        n += y.size(0)

        pred = torch.argmax(logits, dim=1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_all = np.concatenate(ys) if ys else np.array([])
    p_all = np.concatenate(ps) if ps else np.array([])
    f1 = float(f1_score(y_all, p_all, average="macro")) if len(y_all) else 0.0
    return total / max(n, 1), f1

def train_task3(model, train_loader, val_loader, *, device, epochs=10, lr=1e-3, wandb_run=None):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    params = count_params(model)
    if wandb_run is not None:
        wandb_run.config.update({
            "epochs": epochs,
            "lr": lr,
            "params": params,
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
        }, allow_val_change=True)

    epoch_times, peak_mems = [], []
    train_losses, val_losses, val_f1s = [], [], []

    for ep in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        model.train()
        total, n = 0.0, 0

        for batch in train_loader:
            rgb, lidar, y = _move(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(rgb, lidar)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            n += y.size(0)

        tr_loss = total / max(n, 1)
        va_loss, va_f1 = evaluate_loss_and_f1(model, val_loader, device)

        dt = time.time() - t0
        mem = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_f1s.append(va_f1)
        epoch_times.append(dt)
        peak_mems.append(mem)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": ep,
                "train/loss": tr_loss,
                "val/loss": va_loss,
                "val/f1_macro": va_f1,
                "perf/sec_per_epoch": dt,
                "perf/gpu_mem_mb_peak": mem,
            })

        print(f"epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | f1 {va_f1:.3f} | {dt:.2f}s | {mem:.0f} MB")

    sec_per_epoch_avg = float(sum(epoch_times) / len(epoch_times))
    gpu_mem_mb_peak = float(max(peak_mems) if peak_mems else 0.0)

    summary = {
        "val_loss_final": float(val_losses[-1]),
        "val_f1_final": float(val_f1s[-1]),
        "params": int(params),
        "sec_per_epoch_avg": sec_per_epoch_avg,
        "gpu_mem_mb_peak": gpu_mem_mb_peak,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1s": val_f1s,
    }

    if wandb_run is not None:
        wandb_run.summary.update({
            "val_loss_final": summary["val_loss_final"],
            "val_f1_final": summary["val_f1_final"],
            "sec_per_epoch_avg": summary["sec_per_epoch_avg"],
            "gpu_mem_mb_peak": summary["gpu_mem_mb_peak"],
            "params": summary["params"],
        })

    return summary
