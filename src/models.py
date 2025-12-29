import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyStem(nn.Module):
    """Conv1-Conv2 blocks up to a mid-level feature map."""
    def __init__(self, in_ch: int):
        super().__init__()
        k, p = 3, 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 50, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(50, 100, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),     # 32 -> 16
        )

    def forward(self, x):
        return self.net(x)  # [B,100,16,16] if input is [B,C,64,64]


class SharedTrunk(nn.Module):
    """Shared conv blocks after fusion to produce an embedding."""
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
    """Late-fusion embedder producing an embedding from a single modality."""
    def __init__(self, in_ch: int, emb_size: int = 200, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        k, p = 3, 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(50, 100, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(100, 200, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(200, 200, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),
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
    """
    Late fusion: separate encoders -> concatenate embeddings -> classifier head.
    For Task 3 (classification), keep normalize_embeddings=False.
    For Task 5 (contrastive), you may set normalize_embeddings=True.
    """
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
        a = self.rgb(rgb)
        b = self.lidar(lidar)
        x = torch.cat([a, b], dim=1)
        return self.head(x)


class IntermediateFusionClassifier(nn.Module):
    """
    TRUE intermediate fusion: fuse feature maps after EarlyStem, then SharedTrunk.
    fusion ∈ {"concat","add","hadamard"}

    For Task 3 (classification), keep normalize_embeddings=False.
    For Task 5 (contrastive-style embeddings), you may set normalize_embeddings=True.
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

        fused_ch = 200 if fusion == "concat" else 100  # concat doubles channels
        self.shared = SharedTrunk(in_ch=fused_ch, emb_size=emb_size, normalize=normalize_embeddings)

        self.head = nn.Sequential(
            nn.Linear(emb_size, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, rgb, lidar):
        fr = self.rgb_stem(rgb)       # [B,100,16,16]
        fl = self.lidar_stem(lidar)   # [B,100,16,16]

        if self.fusion == "concat":
            f = torch.cat([fr, fl], dim=1)   # [B,200,16,16]
        elif self.fusion == "add":
            f = fr + fl                      # [B,100,16,16]
        else:
            f = fr * fl                      # [B,100,16,16]

        z = self.shared(f)                   # [B,emb]
        return self.head(z)

class EmbedderStrided(nn.Module):
    """Embedder using stride-2 convolutions instead of MaxPool2d."""
    def __init__(self, in_ch: int, emb_size: int = 200, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        k, p = 3, 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, k, stride=2, padding=p), nn.ReLU(),   # 64 -> 32
            nn.Conv2d(50, 100, k, stride=2, padding=p), nn.ReLU(),     # 32 -> 16
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
