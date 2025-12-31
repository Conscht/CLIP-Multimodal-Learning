import time
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

import numpy as np
import torch
import wandb

import torch
import torch.nn.functional as F



@torch.no_grad()
def log_sample_predictions(model, loader, device, wandb_run, *, n=5, class_names=("cube", "sphere")):
    """
    Logs at least n sample predictions to W&B as a table.
    Uses RGB image visualization; also logs LiDAR as a grayscale image.
    """
    model.eval()

    rows = []
    # get a single batch
    batch = next(iter(loader))
    rgb = batch["rgb"].to(device)
    lidar = batch["lidar"].to(device)
    y = batch["y"].to(device)

    logits = model(rgb, lidar)
    pred = torch.argmax(logits, dim=1)

    n = min(n, rgb.shape[0])

    for i in range(n):
        # RGB: assume [C,H,W] float in [0,1], C can be 3 or 4
        rgb_i = rgb[i].detach().cpu().float()
        rgb_i = rgb_i[:3]  # take first 3 channels if RGBA
        rgb_img = (rgb_i.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # LiDAR: visualize first channel as grayscale
        li = lidar[i].detach().cpu().float()
        if li.ndim == 3:
            li0 = li[0]
        else:
            li0 = li
        li0 = li0.numpy()
        lo, hi = np.percentile(li0, [1, 99])
        li0 = np.clip((li0 - lo) / (hi - lo + 1e-6), 0, 1)
        lidar_img = (li0 * 255).astype(np.uint8)

        y_true = int(y[i].item())
        y_pred = int(pred[i].item())

        rows.append([
            wandb.Image(rgb_img, caption="RGB"),
            wandb.Image(lidar_img, caption="LiDAR"),
            class_names[y_true],
            class_names[y_pred],
        ])

    table = wandb.Table(columns=["rgb", "lidar", "y_true", "y_pred"], data=rows)
    wandb_run.log({"samples/predictions": table})


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

def train_task3(model, train_loader, val_loader, *, device, epochs=10, lr=1e-3, wandb_run=None, ckpt_path: str):
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
    best_val = None
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
        if ckpt_path is not None and (best_val is None or va_loss < best_val):
            best_val = va_loss
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": va_loss,
                    "val_f1": va_f1,
                },
                ckpt_path,
            )
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
    if wandb_run is not None:
        log_sample_predictions(model, val_loader, device, wandb_run, n=5)

    return summary

import matplotlib.pyplot as plt

def train_cilp(model, train_loader, val_loader, *, device, epochs=20, lr=1e-3, wandb_run=None):
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.parameters(), lr=lr)  # <- Adam (no weight decay)

    @torch.no_grad()
    def eval_loader(loader):
        model.eval()
        total, n = 0.0, 0
        for batch in loader:
            rgb = batch["rgb"].to(device).float()
            lidar = batch["lidar"].to(device).float()
            logits = model(rgb, lidar)
            loss = clip_contrastive_loss(logits)
            bs = rgb.size(0)
            total += loss.item() * bs
            n += bs
        return total / max(n, 1)

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0

        for batch in train_loader:
            rgb = batch["rgb"].to(device).float()
            lidar = batch["lidar"].to(device).float()

            opt.zero_grad(set_to_none=True)
            logits = model(rgb, lidar)
            loss = clip_contrastive_loss(logits)
            loss.backward()
            opt.step()

            bs = rgb.size(0)
            total += loss.item() * bs
            n += bs

        tr = total / max(n, 1)
        va = eval_loader(val_loader)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": ep,
                "train/contrastive_loss": tr,
                "val/contrastive_loss": va,
                "lr": opt.param_groups[0]["lr"],
            })

        print(f"epoch {ep:02d} | train {tr:.4f} | val {va:.4f}")

    return model


def train_projector(cilp, projector, train_loader, val_loader, *, device, epochs=20, lr=1e-3, wandb_run=None):
    cilp.eval()  # freeze encoders
    projector = projector.to(device)
    opt = torch.optim.AdamW(projector.parameters(), lr=lr)
    mse = nn.MSELoss()

    def step(loader, train=False):
        total, n = 0.0, 0
        if train: projector.train()
        else: projector.eval()
        for batch in loader:
            rgb = batch["rgb"].to(device)
            lidar = batch["lidar"].to(device)
            with torch.no_grad():
                z_rgb = cilp.rgb(rgb)
                z_lid = cilp.lidar(lidar)
            pred = projector(z_rgb)
            loss = mse(pred, z_lid)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total += loss.item() * rgb.size(0)
            n += rgb.size(0)
        return total / max(n, 1)

    for ep in range(1, epochs+1):
        tr = step(train_loader, train=True)
        va = step(val_loader, train=False)
        if wandb_run is not None:
            wandb_run.log({"epoch": ep, "train/projector_mse": tr, "val/projector_mse": va})
        print(f"epoch {ep:02d} | train mse {tr:.4f} | val mse {va:.4f}")

    return projector

from sklearn.metrics import accuracy_score

def train_embedding_classifier(cilp, proj, clf, train_loader, val_loader, *, device, epochs=10, lr=1e-3, wandb_run=None):
    cilp.eval()
    proj.eval()
    clf = clf.to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def eval_acc(loader):
        clf.eval()
        ys, ps = [], []
        total, n = 0.0, 0
        for batch in loader:
            rgb = batch["rgb"].to(device)
            y = batch["y"].to(device)
            z = cilp.rgb(rgb)
            z_hat = proj(z)
            logits = clf(z_hat)
            loss = ce(logits, y)
            total += loss.item() * y.size(0)
            n += y.size(0)
            pred = torch.argmax(logits, dim=1)
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        return total / max(n,1), float(accuracy_score(ys, ps))

    for ep in range(1, epochs+1):
        clf.train()
        total, n = 0.0, 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            y = batch["y"].to(device)
            with torch.no_grad():
                z = cilp.rgb(rgb)
                z_hat = proj(z)
            logits = clf(z_hat)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            n += y.size(0)

        tr = total / max(n,1)
        va_loss, va_acc = eval_acc(val_loader)

        if wandb_run is not None:
            wandb_run.log({"epoch": ep, "train/clf_loss": tr, "val/clf_loss": va_loss, "val/acc": va_acc})

        print(f"epoch {ep:02d} | train {tr:.4f} | val {va_loss:.4f} | acc {va_acc:.4f}")

    return clf


def clip_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    CLIP-style symmetric contrastive loss.
    logits: [B, B] where logits[i, j] = sim(rgb_i, lidar_j) / T
    Positive pairs are on the diagonal.
    """
    B = logits.size(0)
    targets = torch.arange(B, device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)        # rgb -> lidar
    loss_t2i = F.cross_entropy(logits.t(), targets)    # lidar -> rgb
    return 0.5 * (loss_i2t + loss_t2i)