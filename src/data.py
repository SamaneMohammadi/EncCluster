"""CIFAR-10 loading and IID federated partition."""

import numpy as np
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import config

_MEAN, _STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def _tf(train):
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    return T.Compose(aug + [T.ToTensor(), T.Normalize(_MEAN, _STD)])


def get_cifar10(root="./data_files", train=True):
    return torchvision.datasets.CIFAR10(root, train=train, download=True, transform=_tf(train))


def make_client_loaders(dataset, num_clients, batch_size=config.BATCH_SIZE, seed=config.SEED):
    idx = np.arange(len(dataset)); np.random.default_rng(seed).shuffle(idx)
    parts = [list(s) for s in np.array_split(idx, num_clients)]
    loaders = [DataLoader(Subset(dataset, p), batch_size=batch_size, shuffle=True) for p in parts]
    return loaders, [len(p) for p in parts]


def test_loader(root="./data_files", batch_size=256):
    return DataLoader(get_cifar10(root, train=False), batch_size=batch_size, shuffle=False)
