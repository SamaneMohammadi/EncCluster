import os
import sys
import torch
import torchvision
from torchvision.transforms import v2
import contextlib
import numpy as np
sys.path.append(f"{os.environ['TORCH_DATA_DIR']}")
from dataloader import DATASETS as datasets

def split_fn(ds, num_clients, client_ids, num_classes, skewness=0.1, seed=42):
	np.random.seed(seed)
	partitions = {}
	idxs, num_samples, idx_batch = np.array(ds.targets), len(ds), [[] for _ in range(num_clients)]
	for k in range(num_classes):
		idx_k = np.where(idxs==k)[0]
		np.random.shuffle(idx_k)
		proportions = np.random.dirichlet(np.repeat(skewness, num_clients))
		proportions = np.array([p * (len(idx_j) < num_samples/num_clients) for p, idx_j in zip(proportions, idx_batch)])
		proportions = proportions / proportions.sum()
		proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
		idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
	for j in range(num_clients):
		np.random.shuffle(idx_batch[j])
		partitions[j] = idx_batch[j]
	return {cid: torch.utils.data.Subset(ds, partitions[i])  for i, cid in enumerate(client_ids)}

def get_transforms(ds_name='cifar10', model_name='resnet20', nx=32):

	if model_name=='resnet20':

		train_transforms = torchvision.transforms.Compose([
			torchvision.transforms.RandomCrop(32, padding=4),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
		])

		test_transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
		])

	else:
		MEAN_STD = {
			'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
			'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
			'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
		}

		POLICY = {
			'cifar10': v2.AutoAugmentPolicy.CIFAR10,
			'cifar100': v2.AutoAugmentPolicy.CIFAR10,
			'svhn': v2.AutoAugmentPolicy.SVHN,
		}

		def _convert_image_to_rgb(image):
			return image.convert("RGB")

		train_transforms = v2.Compose([
				v2.Resize(nx, antialias=True, interpolation=v2.InterpolationMode.BICUBIC),
				v2.RandomCrop(nx, padding=4),
				_convert_image_to_rgb,
				v2.AutoAugment(POLICY[ds_name]),
				torchvision.transforms.ToTensor(),
				v2.Normalize(*MEAN_STD[ds_name]),
			])

		test_transforms = v2.Compose([
				v2.Resize(nx, antialias=True, interpolation=v2.InterpolationMode.BICUBIC),
				v2.CenterCrop(nx),
				_convert_image_to_rgb,
				torchvision.transforms.ToTensor(),
				v2.Normalize(*MEAN_STD[ds_name]),
			])

	return train_transforms,test_transforms

def load_data(name='cifar10', model_name='resnet20', input_res=32, skewness=0.2, id=0, num_clients=10,
	batch_size=128, return_eval_ds=False, use_cutmix=True, seed=42):

	train_transform, test_transform = get_transforms(ds_name=name, model_name=model_name, nx=input_res)

	with contextlib.redirect_stdout(None):
		ds_train, ds_test, num_classes, _ = datasets[name](train_transform, test_transform)

	if return_eval_ds:
		ds_test = torch.utils.data.DataLoader(ds_test, batch_size=8*batch_size, shuffle=False, num_workers=8, pin_memory=True)
		return (ds_test, num_classes, len(ds_test))

	if num_clients>1:
		ds_train = split_fn(ds=ds_train, num_clients=num_clients, client_ids=np.arange(num_clients),
			num_classes=num_classes, skewness=skewness, seed=seed)[id]

	collate_fn=None
	if use_cutmix:
		cutmix = v2.RandomChoice([v2.CutMix(num_classes=num_classes)], p=[0.5])
		def collate_fn(batch):
			return cutmix(*torch.utils.data.default_collate(batch))

	ds_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn,)

	return (ds_train, num_classes, len(ds_train))

'''
if __name__ == "__main__":
	(ds, num_classes, num_samples) = load_data(name='cifar10', input_res=32, num_clients=10, batch_size=128, return_eval_ds=False, seed=42)
	for x,y in ds: print(x.shape, y)
'''