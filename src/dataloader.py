import os
import torch
import torchvision
import numpy as np
import requests
import gdown
import tarfile
from tqdm import tqdm
from PIL import Image

def get_cifar10(train_transform=None, test_transform=None):
	data_train = torchvision.datasets.CIFAR10(os.path.join(os.environ['TORCH_DATA_DIR'],'cifar10'), train=True, download=True, transform=train_transform)
	data_test = torchvision.datasets.CIFAR10(os.path.join(os.environ['TORCH_DATA_DIR'],'cifar10'), train=False, download=False, transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'CIFAR-10'

def get_cifar100(train_transform=None, test_transform=None):
	data_train = torchvision.datasets.CIFAR100(os.path.join(os.environ['TORCH_DATA_DIR'],'cifar100'), train=True, download=True, transform=train_transform)
	data_test = torchvision.datasets.CIFAR100(os.path.join(os.environ['TORCH_DATA_DIR'],'cifar100'), train=False, download=False, transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'CIFAR-100'

def get_emnist(train_transform=None, test_transform=None):
	data_train = torchvision.datasets.EMNIST(os.path.join(os.environ['TORCH_DATA_DIR'],'emnist'), split='letters', train=True, download=True, transform=train_transform)
	data_test = torchvision.datasets.EMNIST(os.path.join(os.environ['TORCH_DATA_DIR'],'emnist'), split='letters', train=False, download=True, transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'EMNIST'

def get_fashion_mnist(train_transform=None, test_transform=None):
	data_train = torchvision.datasets.FashionMNIST(os.path.join(os.environ['TORCH_DATA_DIR'],'fashionmnist'), train=True, download=True, transform=train_transform)
	data_test = torchvision.datasets.FashionMNIST(os.path.join(os.environ['TORCH_DATA_DIR'],'fashionmnist'), train=False, download=True, transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'Fashion-MNIST'

def get_svhn(train_transform=None, test_transform=None):

	class SVHN(torchvision.datasets.SVHN):
		def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
			super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
			self.targets = self.labels

	data_train = SVHN(os.path.join(os.environ['TORCH_DATA_DIR'],'svhn'), split='train', download=True, transform=train_transform)
	data_test = SVHN(os.path.join(os.environ['TORCH_DATA_DIR'],'svhn'), split='test', download=True, transform=test_transform)
	return data_train, data_test, len(np.unique(data_train.labels)), 'SVHN'

DATASETS = {
	'cifar10': get_cifar10,
	'cifar100': get_cifar100,
	'emnist': get_emnist,
	'svhn': get_svhn,
	'fashion_mnist': get_fashion_mnist,
}
