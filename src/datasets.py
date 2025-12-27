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
