"""ConvNeXt (Liu et al., 2022), compact CIFAR version. forward() returns logits."""

import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dwconv(x)))))


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=10, dims=(64, 128, 256), depths=(2, 2, 2)):
        super().__init__()
        self.stem = nn.Conv2d(3, dims[0], 3, padding=1)
        stages = []
        for i, (d, n) in enumerate(zip(dims, depths)):
            if i > 0:
                stages.append(nn.Conv2d(dims[i - 1], d, 2, stride=2))   # downsample
            stages += [Block(d) for _ in range(n)]
        self.stages = nn.Sequential(*stages)
        self.norm = nn.GroupNorm(1, dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.head(self.norm(x).mean(dim=(-2, -1)))
