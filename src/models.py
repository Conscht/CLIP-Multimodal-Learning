import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# MaxPool (Task 3 baseline)
# -------------------------

class EarlyStem(nn.Module):
    """Conv1-Conv2 blocks up to a mid-level feature map (MaxPool version)."""
    def __init__(self, in_ch: int):
        super().__init__()
        k, p = 3, 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 50, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(50, 100, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),     # 32 -> 16
        )

    def forward(self, x):
        return self.net(x)  # [B,100,16,16]


class SharedTrunk(nn.Module):
    """Shared conv blocks after fusion to produce an embedding (MaxPool version)."""
    def __init__(self, in_ch: int, emb_size: int = 200, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        k, p = 3, 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 200, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(200, 200, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),    # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, emb_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        z = self.proj(x)
        return F.normalize(z, dim=1) if self.normalize else z


class Embedder(nn.Module):
    """Late-fusion embedder (MaxPool version)."""
    def __init__(self, in_ch: int, emb_size: int = 200, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        k, p = 3, 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(50, 100, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),     # 32 -> 16
            nn.Conv2d(100, 200, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),    # 16 -> 8
            nn.Conv2d(200, 200, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),    # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, emb_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        z = self.proj(x)
        return F.normalize(z, dim=1) if self.normalize else z


class LateFusionClassifier(nn.Module):
    def __init__(
        self,
        rgb_in_ch: int = 4,
        lidar_in_ch: int = 4,
        emb_size: int = 200,
        num_classes: int = 2,
        normalize_embeddings: bool = False,
    ):
        super().__init__()
        self.rgb = Embedder(rgb_in_ch, emb_size, normalize=normalize_embeddings)
        self.lidar = Embedder(lidar_in_ch, emb_size, normalize=normalize_embeddings)
        self.head = nn.Sequential(
            nn.Linear(2 * emb_size, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, rgb, lidar):
        x = torch.cat([self.rgb(rgb), self.lidar(lidar)], dim=1)
        return self.head(x)


class IntermediateFusionClassifier(nn.Module):
    """
    TRUE intermediate fusion: fuse feature maps after EarlyStem, then SharedTrunk.
    fusion ∈ {"concat","add","hadamard"}
    """
    def __init__(
        self,
        fusion: str,
        rgb_in_ch: int = 4,
        lidar_in_ch: int = 4,
        emb_size: int = 200,
        num_classes: int = 2,
        normalize_embeddings: bool = False,
    ):
        super().__init__()
        assert fusion in ("concat", "add", "hadamard")
        self.fusion = fusion

        self.rgb_stem = EarlyStem(rgb_in_ch)
        self.lidar_stem = EarlyStem(lidar_in_ch)

        fused_ch = 200 if fusion == "concat" else 100
        self.shared = SharedTrunk(in_ch=fused_ch, emb_size=emb_size, normalize=normalize_embeddings)

        self.head = nn.Sequential(
            nn.Linear(emb_size, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, rgb, lidar):
        fr = self.rgb_stem(rgb)
        fl = self.lidar_stem(lidar)

        if self.fusion == "concat":
            f = torch.cat([fr, fl], dim=1)
        elif self.fusion == "add":
            f = fr + fl
        else:
            f = fr * fl

        z = self.shared(f)
        return self.head(z)


# -------------------------
# StridedConv (Task 4)
# -------------------------

class EarlyStemStrided(nn.Module):
    """Strided version of EarlyStem (replaces MaxPool2d with stride-2 convs)."""
    def __init__(self, in_ch: int):
        super().__init__()
        k, p = 3, 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 50,  k, stride=2, padding=p), nn.ReLU(),  # 64 -> 32
            nn.Conv2d(50,  100, k, stride=2, padding=p), nn.ReLU(),    # 32 -> 16
        )

    def forward(self, x):
        return self.net(x)  # [B,100,16,16]


class SharedTrunkStrided(nn.Module):
    """Strided version of SharedTrunk (replaces MaxPool2d with stride-2 convs)."""
    def __init__(self, in_ch: int, emb_size: int = 200, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        k, p = 3, 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 200, k, stride=2, padding=p), nn.ReLU(),  # 16 -> 8
            nn.Conv2d(200, 200, k, stride=2, padding=p), nn.ReLU(),    # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, emb_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        z = self.proj(x)
        return F.normalize(z, dim=1) if self.normalize else z


class EmbedderStrided(nn.Module):
    """Strided version of Embedder (replaces all MaxPool2d with stride-2 convs)."""
    def __init__(self, in_ch: int, emb_size: int = 200, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        k, p = 3, 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50,  k, stride=2, padding=p), nn.ReLU(),  # 64 -> 32
            nn.Conv2d(50,  100, k, stride=2, padding=p), nn.ReLU(),    # 32 -> 16
            nn.Conv2d(100, 200, k, stride=2, padding=p), nn.ReLU(),    # 16 -> 8
            nn.Conv2d(200, 200, k, stride=2, padding=p), nn.ReLU(),    # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, emb_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        z = self.proj(x)
        return F.normalize(z, dim=1) if self.normalize else z


# -------------------------
# Builder used by Task 3/4
# -------------------------

def build_fusion_model(
    arch: str,
    *,
    rgb_in_ch: int,
    lidar_in_ch: int,
    emb_size: int = 200,
    num_classes: int = 2,
    fusion: str | None = None,          # only for intermediate
    pool_type: str = "maxpool",         # "maxpool" or "strided"
    normalize_embeddings: bool = False,
):
    """
    Task 3: pool_type="maxpool"
    Task 4: compare pool_type="maxpool" vs "strided"
    """
    assert arch in ("late", "intermediate")
    assert pool_type in ("maxpool", "strided")

    if pool_type == "maxpool":
        Stem = EarlyStem
        Trunk = SharedTrunk
        Emb = Embedder
    else:
        Stem = EarlyStemStrided
        Trunk = SharedTrunkStrided
        Emb = EmbedderStrided

    if arch == "late":
        # build directly with the correct embedder type
        model = LateFusionClassifier(
            rgb_in_ch=rgb_in_ch,
            lidar_in_ch=lidar_in_ch,
            emb_size=emb_size,
            num_classes=num_classes,
            normalize_embeddings=normalize_embeddings,
        )
        model.rgb = Emb(rgb_in_ch, emb_size, normalize=normalize_embeddings)
        model.lidar = Emb(lidar_in_ch, emb_size, normalize=normalize_embeddings)
        return model

    # intermediate
    assert fusion in ("concat", "add", "hadamard")
    fused_ch = 200 if fusion == "concat" else 100

    model = IntermediateFusionClassifier(
        fusion=fusion,
        rgb_in_ch=rgb_in_ch,
        lidar_in_ch=lidar_in_ch,
        emb_size=emb_size,
        num_classes=num_classes,
        normalize_embeddings=normalize_embeddings,
    )

    # swap BOTH stems + trunk to match "replace each MaxPool2d"
    model.rgb_stem = Stem(rgb_in_ch)
    model.lidar_stem = Stem(lidar_in_ch)
    model.shared = Trunk(in_ch=fused_ch, emb_size=emb_size, normalize=normalize_embeddings)

    return model


# -------------------------
# Task 5 components (ok)
# -------------------------

import math
import torch
import torch.nn as nn

class CILP(nn.Module):
    def __init__(self, embedder_rgb: nn.Module, embedder_lidar: nn.Module, temperature: float = 0.07):
        super().__init__()
        self.rgb = embedder_rgb
        self.lidar = embedder_lidar
        # CLIP learns logit scale; init to 1/T
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def forward(self, rgb, lidar):
        z_rgb = self.rgb(rgb)
        z_lid = self.lidar(lidar)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return (z_rgb @ z_lid.t()) * scale


class Projector(nn.Module):
    def __init__(self, dim: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, dim),
        )

    def forward(self, z):
        return self.net(z)


class EmbeddingClassifier(nn.Module):
    def __init__(self, dim: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, z):
        return self.net(z)
