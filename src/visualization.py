from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import fiftyone as fo


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
