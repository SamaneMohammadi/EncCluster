"""MLP-Mixer (Tolstikhin et al., 2021), compact version. forward() returns logits."""

import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, token_dim), nn.GELU(), nn.Linear(token_dim, num_patches))
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_dim), nn.GELU(), nn.Linear(channel_dim, dim))

    def forward(self, x):
        x = x + self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.norm2(x))


class MLPMixer(nn.Module):
    def __init__(self, num_classes=10, image_size=32, patch_size=4, dim=128,
                 depth=4, token_dim=64, channel_dim=256):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(3, dim, patch_size, patch_size)
        self.blocks = nn.Sequential(
            *[MixerBlock(dim, num_patches, token_dim, channel_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)   # (B, num_patches, dim)
        x = self.norm(self.blocks(x)).mean(dim=1)
        return self.head(x)
