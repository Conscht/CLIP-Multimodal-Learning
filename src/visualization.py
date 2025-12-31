from pathlib import Path
from typing import Dict, List

import numpy as np
import fiftyone as fo
import torch
import wandb

def build_grouped_dataset(
    name: str,
    pairs: List,
    group_field: str = "group",
    overwrite: bool = True,
) -> fo.Dataset:
    """
    Creates a grouped dataset with slices:
      - rgb
      - lidar
    Stores classification as `gt` (rubric-safe).
    """
    if overwrite and fo.dataset_exists(name):
        fo.delete_dataset(name)

    ds = fo.Dataset(name)
    ds.add_group_field(group_field, default="rgb")

    for p in pairs:
        label_str = "cube" if p.label == 0 else "sphere"
        g = fo.Group()

        rgb = fo.Sample(
            filepath=str(p.rgb_path),
            gt=fo.Classification(label=label_str),
            pair_id=p.pair_id,
        )
        rgb[group_field] = g.element("rgb")

        lidar = fo.Sample(
            filepath=str(p.lidar_path),
            gt=fo.Classification(label=label_str),
            pair_id=p.pair_id,
        )
        lidar[group_field] = g.element("lidar")

        ds.add_samples([rgb, lidar])

    ds.save()
    return ds


def rgb_image_stats(rgb_paths: List[Path], max_n: int = 200) -> Dict[str, object]:
    """
    Returns:
      - most common shape
      - dtype counts
      - shape counts
    """
    from PIL import Image

    shapes = []
    dtype_counts: Dict[str, int] = {}
    shape_counts: Dict[str, int] = {}

    for p in rgb_paths[: min(max_n, len(rgb_paths))]:
        arr = np.array(Image.open(p))
        shapes.append(arr.shape)
        dtype_counts[str(arr.dtype)] = dtype_counts.get(str(arr.dtype), 0) + 1
        shape_counts[str(arr.shape)] = shape_counts.get(str(arr.shape), 0) + 1

    most_common_shape = None
    if shape_counts:
        most_common_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]

    return {
        "checked": len(shapes),
        "most_common_shape": most_common_shape,
        "dtype_counts": dtype_counts,
        "shape_counts": shape_counts,
    }


@torch.no_grad()
def log_task5_predictions(cilp, proj, clf, loader, device, wandb_run, n=5, class_names=("cube","sphere")):
    cilp.eval(); proj.eval(); clf.eval()

    batch = next(iter(loader))
    rgb = batch["rgb"].to(device)
    lidar = batch["lidar"].to(device)
    y = batch["y"].to(device)

    z = cilp.rgb(rgb)
    z_hat = proj(z)
    logits = clf(z_hat)
    pred = torch.argmax(logits, dim=1)

    n = min(n, rgb.size(0))
    rows = []

    for i in range(n):
        rgb_i = rgb[i].detach().cpu().float()[:3]
        rgb_img = (rgb_i.permute(1,2,0).numpy() * 255).astype(np.uint8)

        li = lidar[i].detach().cpu().float()
        li0 = li[0] if li.ndim == 3 else li
        li0 = li0.numpy()
        lo, hi = np.percentile(li0, [1, 99])
        li0 = np.clip((li0 - lo) / (hi - lo + 1e-6), 0, 1)
        lidar_img = (li0 * 255).astype(np.uint8)

        rows.append([
            wandb.Image(rgb_img, caption="RGB"),
            wandb.Image(lidar_img, caption="LiDAR"),
            class_names[int(y[i])],
            class_names[int(pred[i])],
            float(torch.softmax(logits[i], dim=0).max().item())
        ])

    table = wandb.Table(columns=["rgb","lidar","y_true","y_pred","conf"], data=rows)
    wandb_run.log({"samples/task5_predictions": table})
from pathlib import Path

import numpy as np
import torch
import wandb

def colorize_heatmap(S, cmap_name="viridis"):
    import matplotlib.cm as cm

    S_min, S_max = float(S.min()), float(S.max())
    S01 = (S - S_min) / (S_max - S_min + 1e-8)

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(S01)              # H x W x 4
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


@torch.no_grad()
def log_similarity_matrix_artifact(
    model,
    loader,
    device,
    wandb_run,
    *,
    n: int = 32,
    artifact_name: str = "similarity_matrix",
    artifact_type: str = "analysis",
    key_prefix: str = "samples",
):
    """
    Computes an image-vs-lidar cosine similarity matrix on a single batch and logs:
      - a heatmap image to W&B
      - the matrix as a W&B artifact (.npy and .csv)

    Assumptions:
      - batch provides: batch["rgb"], batch["lidar"]
      - model exposes encoders: model.rgb_encoder(x) and model.lidar_encoder(x)
        OR provides a method model.encode_rgb / model.encode_lidar (see below)
    """
    model.eval()

    batch = next(iter(loader))
    rgb = batch["rgb"][:n].to(device)
    lidar = batch["lidar"][:n].to(device)
    if "pair_id" in batch:
        pair_ids = list(batch["pair_id"][:n])  # list of strings
        order = np.argsort(pair_ids)
        rgb = rgb[order]
        lidar = lidar[order]

    # --- get embeddings (adapt this block to your model API) ---
    if hasattr(model, "encode_rgb") and hasattr(model, "encode_lidar"):
        z_rgb = model.encode_rgb(rgb)
        z_lidar = model.encode_lidar(lidar)
    elif hasattr(model, "rgb") and hasattr(model, "lidar"):
        # common in your code: cilp.rgb(...) / cilp.lidar(...)
        z_rgb = model.rgb(rgb)
        z_lidar = model.lidar(lidar)
    elif hasattr(model, "rgb_encoder") and hasattr(model, "lidar_encoder"):
        z_rgb = model.rgb_encoder(rgb)
        z_lidar = model.lidar_encoder(lidar)
    else:
        raise AttributeError(
            "Model must expose encode_rgb/encode_lidar OR rgb/lidar OR rgb_encoder/lidar_encoder"
        )

    # flatten if needed: [B,C,H,W] -> [B, C*H*W]
    z_rgb = z_rgb.flatten(1)
    z_lidar = z_lidar.flatten(1)

    # normalize for cosine sim
    z_rgb = torch.nn.functional.normalize(z_rgb, dim=1)
    z_lidar = torch.nn.functional.normalize(z_lidar, dim=1)

    # cosine similarity matrix [B,B]
    S = (z_rgb @ z_lidar.T).detach().cpu().numpy()

    # --- log quick visual as heatmap image ---
    # create a simple grayscale image without matplotlib
    S_min, S_max = float(S.min()), float(S.max())
    S_img = colorize_heatmap(S)

    wandb_run.log({f"{key_prefix}/similarity_matrix": wandb.Image(S_img, caption=f"cosine sim [{S_min:.3f},{S_max:.3f}]")})
    wandb_run.summary[f"{key_prefix}/similarity_matrix_range"] = {"min": S_min, "max": S_max}

    # --- save to temp files and log as artifact ---
    art = wandb.Artifact(artifact_name, type=artifact_type)

    npy_path = Path("similarity_matrix.npy")
    csv_path = Path("similarity_matrix.csv")

    np.save(npy_path, S)
    np.savetxt(csv_path, S, delimiter=",")

    art.add_file(str(npy_path))
    art.add_file(str(csv_path))

    wandb_run.log_artifact(art)

    # clean up local temp files (optional)
    try:
        npy_path.unlink()
        csv_path.unlink()
    except Exception:
        pass

    return S
