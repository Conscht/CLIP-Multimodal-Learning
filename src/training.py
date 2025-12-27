import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _move(batch, device):
    return (
        batch["rgb"].to(device),
        batch["lidar"].to(device),
        batch["y"].to(device),
    )

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for batch in loader:
        rgb, lidar, y = _move(batch, device)
        logits = model(rgb, lidar)
        loss = ce(logits, y)
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)

def train_task3(model, train_loader, val_loader, *, device, epochs=10, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    epoch_times = []
    peak_mems = []
    train_losses = []
    val_losses = []

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
        va_loss = evaluate(model, val_loader, device)

        dt = time.time() - t0
        mem = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        epoch_times.append(dt)
        peak_mems.append(mem)

        print(f"epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | {dt:.2f}s | {mem:.0f} MB")

    return {
        "val_loss_final": val_losses[-1],
        "params": count_params(model),
        "sec_per_epoch_avg": float(sum(epoch_times) / len(epoch_times)),
        "gpu_mem_mb_peak": float(max(peak_mems) if peak_mems else 0.0),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
