import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, in_ch: int, emb_size: int = 200):
        super().__init__()
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
        return F.normalize(z, dim=1)

class LateFusionClassifier(nn.Module):
    def __init__(self, rgb_in_ch: int, lidar_in_ch: int, emb_size: int = 200, num_classes: int = 2):
        super().__init__()
        self.rgb = Embedder(rgb_in_ch, emb_size)
        self.lidar = Embedder(lidar_in_ch, emb_size)
        self.head = nn.Sequential(
            nn.Linear(2 * emb_size, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, lidar):
        a = self.rgb(rgb)
        b = self.lidar(lidar)
        x = torch.cat([a, b], dim=1)
        return self.head(x)

class IntermediateFusionClassifier(nn.Module):
    def __init__(self, fusion: str, rgb_in_ch: int, lidar_in_ch: int, emb_size: int = 200, num_classes: int = 2):
        super().__init__()
        assert fusion in ("concat", "add", "hadamard")
        self.fusion = fusion
        self.rgb = Embedder(rgb_in_ch, emb_size)
        self.lidar = Embedder(lidar_in_ch, emb_size)

        in_dim = 2 * emb_size if fusion == "concat" else emb_size
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, lidar):
        a = self.rgb(rgb)
        b = self.lidar(lidar)
        if self.fusion == "concat":
            x = torch.cat([a, b], dim=1)
        elif self.fusion == "add":
            x = a + b
        else:
            x = a * b
        return self.head(x)
