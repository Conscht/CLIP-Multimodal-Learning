from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class PairRecord:
    rgb_path: Path
    lidar_path: Path
    label: int          # 0=cube, 1=sphere
    class_name: str     # "cubes" or "spheres"
    stem: str           # "0000"
    pair_id: str        # "cubes_0000"


class AssessmentPairs:
    """
    Expects:
      <root>/cubes/rgb/*.png
      <root>/cubes/lidar/*.npy
      <root>/spheres/rgb/*.png
      <root>/spheres/lidar/*.npy
    Matches by filename stem.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.classes = ["cubes", "spheres"]
        self.class_to_label = {"cubes": 0, "spheres": 1}

    def load_pairs(self) -> List[PairRecord]:
        out: List[PairRecord] = []
        for cls in self.classes:
            rgb_dir = self.root / cls / "rgb"
            lidar_dir = self.root / cls / "lidar"
            if not rgb_dir.exists() or not lidar_dir.exists():
                raise FileNotFoundError(f"Missing folders for class '{cls}': {rgb_dir} or {lidar_dir}")

            rgb_files = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.jpeg")))

            for rgb_path in rgb_files:
                stem = rgb_path.stem
                lidar_path = lidar_dir / f"{stem}.npy"
                if not lidar_path.exists():
                    continue

                pair_id = f"{cls}_{stem}"
                out.append(
                    PairRecord(
                        rgb_path=rgb_path,
                        lidar_path=lidar_path,
                        label=self.class_to_label[cls],
                        class_name=cls,
                        stem=stem,
                        pair_id=pair_id,
                    )
                )
        return out


def train_val_split(pairs: List[PairRecord], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[PairRecord], List[PairRecord]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = int(round(len(pairs) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return train, val


def class_counts(pairs: List[PairRecord]) -> Dict[str, int]:
    counts = {"cubes": 0, "spheres": 0}
    for p in pairs:
        counts[p.class_name] += 1
    return counts


from typing import Optional
import torch
from torch.utils.data import Dataset
from PIL import Image

class AssessmentTorchDataset(Dataset):
    """
    Wraps List[PairRecord] into a PyTorch Dataset yielding:
      {"rgb": Tensor[C,H,W], "lidar": Tensor[C,H,W], "y": LongTensor[]}
    """
    def __init__(self, pairs: List[PairRecord], rgb_mode: str = "RGBA"):
        self.pairs = pairs
        self.rgb_mode = rgb_mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]

        # RGB
        img = Image.open(p.rgb_path).convert(self.rgb_mode)
        rgb = np.array(img).astype(np.float32) / 255.0      # [H,W,C]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)        # [C,H,W]

        # LiDAR
        lidar = np.load(p.lidar_path).astype(np.float32)
        if lidar.ndim == 2:
            lidar = lidar[None, :, :]                       # [1,H,W]
        elif lidar.ndim == 3:
            # if [H,W,C] -> [C,H,W]
            if lidar.shape[-1] in (1,2,3,4) and lidar.shape[0] not in (1,2,3,4):
                lidar = np.transpose(lidar, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected lidar shape: {lidar.shape} for {p.lidar_path}")

        lidar = torch.from_numpy(lidar)

        y = torch.tensor(p.label, dtype=torch.long)
        return {"rgb": rgb, "lidar": lidar, "y": y}

import numpy as np
from typing import List

def stratified_subsample(pairs: List[PairRecord], frac: float = 0.10, seed: int = 42) -> List[PairRecord]:
    rng = np.random.default_rng(seed)

    cubes   = [p for p in pairs if p.class_name == "cubes"]
    spheres = [p for p in pairs if p.class_name == "spheres"]

    k_c = max(1, int(round(len(cubes) * frac)))
    k_s = max(1, int(round(len(spheres) * frac)))

    cubes_sel = rng.choice(len(cubes), size=k_c, replace=False)
    sph_sel   = rng.choice(len(spheres), size=k_s, replace=False)

    out = [cubes[i] for i in cubes_sel] + [spheres[i] for i in sph_sel]
    rng.shuffle(out)
    return out



import fiftyone as fo
import numpy as np
from pathlib import Path
from PIL import Image

def lidar_to_png(lidar_npy: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.load(lidar_npy)
    # Reduce to 2D for visualization
    if arr.ndim == 3:
        # if (H,W,C) or (C,H,W), take first channel safely
        if arr.shape[0] in (1,2,3,4) and arr.shape[1] != arr.shape[0]:
            arr2 = arr[0]
        else:
            arr2 = arr[..., 0]
    elif arr.ndim == 2:
        arr2 = arr
    else:
        raise ValueError(f"Unexpected LiDAR shape: {arr.shape} for {lidar_npy}")

    # Normalize to 0..255 for PNG preview
    a = arr2.astype(np.float32)
    lo, hi = np.nanpercentile(a, (1, 99))
    if hi <= lo:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a) + 1e-6)
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    img = (a * 255).astype(np.uint8)

    out_path = out_dir / f"{lidar_npy.stem}.png"
    Image.fromarray(img, mode="L").save(out_path)
    return out_path

def build_grouped_dataset(name: str, pairs, group_field="group", overwrite=True):
    if overwrite and name in fo.list_datasets():
        fo.delete_dataset(name)

    ds = fo.Dataset(name)
    ds.add_group_field(group_field, default="rgb")

    preview_dir = Path("/content/lidar_previews")  # fast local previews in Colab

    for p in pairs:
        # make LiDAR preview PNG for the App
        lidar_png = lidar_to_png(p.lidar_path, preview_dir)

        rgb = fo.Sample(
            filepath=str(p.rgb_path),
            ground_truth=fo.Classification(label="cube" if p.class_name == "cubes" else "sphere"),
            pair_id=p.pair_id,
            class_name=p.class_name,
            rgb_path=str(p.rgb_path),
            lidar_npy_path=str(p.lidar_path),
        )

        lidar = fo.Sample(
            filepath=str(lidar_png),
            ground_truth=fo.Classification(label="cube" if p.class_name == "cubes" else "sphere"),
            pair_id=p.pair_id,
            class_name=p.class_name,
            rgb_path=str(p.rgb_path),
            lidar_npy_path=str(p.lidar_path),
        )

        g = fo.Group()
        rgb[group_field] = g.element("rgb")
        lidar[group_field] = g.element("lidar")

        ds.add_samples([rgb, lidar])

    ds.save()
    return ds

def make_loaders(root: Path, frac=0.1, seed=42, val_ratio=0.2, batch_size=64, num_workers=2):
    pairs = AssessmentPairs(root).load_pairs()
    sub = stratified_subsample(pairs, frac=frac, seed=seed)
    train_pairs, val_pairs = train_val_split(sub, val_ratio=val_ratio, seed=seed)
    train_ds = AssessmentTorchDataset(train_pairs)
    val_ds = AssessmentTorchDataset(val_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return pairs, sub, train_pairs, val_pairs, train_loader, val_loader
