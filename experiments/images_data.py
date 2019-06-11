import os

import torch
import numpy as np
from torchvision import transforms as tvt
from torchvision import datasets
from torch.utils.data import Subset, random_split

import data
from experiments import autils

class Preprocess:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_bins = 2 ** self.num_bits

    def __call__(self, img):
        if img.dtype == torch.uint8:
            img = img.float() # Already in [0,255]
        else:
            img = img * 255. # [0,1] -> [0,255]

        if self.num_bits != 8:
            img = torch.floor(img / 2 ** (8 - self.num_bits)) # [0, 255] -> [0, num_bins - 1]

        # Uniform dequantization.
        img = img + torch.rand_like(img)

        return img

    def inverse(self, inputs):
        # Discretize the pixel values.
        inputs = torch.floor(inputs)
        # Convert to a float in [0, 1].
        inputs = inputs * (256 / self.num_bins) / 255
        inputs = torch.clamp(inputs, 0, 1)
        return inputs

class RandomHorizontalFlipTensor(object):
    """Random horizontal flip of a CHW image tensor."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        assert img.dim() == 3
        if np.random.rand() < self.p:
            return img.flip(2) # Flip the width dimension, assuming img shape is CHW.
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def dataset_root(dataset_name):
    return os.path.join(autils.get_dataset_root(), dataset_name)

def get_data(dataset, num_bits, train=True, valid_frac=None):
    train_dataset = None
    valid_dataset = None
    test_dataset = None

    if train:
        assert valid_frac is not None

    if dataset == 'imagenet-64-fast':
        root = dataset_root('imagenet64_fast')
        c, h, w = (3, 64, 64)

        if train:
            train_dataset = data.ImageNet64Fast(
                root=root,
                train=True,
                download=True,
                transform=Preprocess(num_bits)
            )

            num_train = len(train_dataset)
            valid_size = int(np.floor(num_train * valid_frac))
            train_size = num_train - valid_size
            train_dataset, valid_dataset = random_split(train_dataset,
                                                        (train_size, valid_size))
        else:
            test_dataset = data.ImageNet64Fast(
                root=root,
                train=False,
                download=True,
                transform=Preprocess(num_bits)
            )

    elif dataset == 'cifar-10-fast' or dataset == 'cifar-10':
        root = dataset_root('cifar-10')
        c, h, w = (3, 32, 32)

        if dataset == 'cifar-10-fast':
            dataset_class = data.CIFAR10Fast
            train_transform = tvt.Compose([
                RandomHorizontalFlipTensor(),
                Preprocess(num_bits)
            ])
            test_transform = Preprocess(num_bits)
        else:
            dataset_class = datasets.CIFAR10
            train_transform=tvt.Compose([
                tvt.RandomHorizontalFlip(),
                tvt.ToTensor(),
                Preprocess(num_bits)
            ])
            test_transform = tvt.Compose([
                tvt.ToTensor(),
                Preprocess(num_bits)
            ])

        if train:
            train_dataset = dataset_class(
                root=root,
                train=True,
                download=True,
                transform=train_transform
            )

            valid_dataset = dataset_class(
                root=root,
                train=True,
                transform=test_transform # Note different transform.
            )

            num_train = len(train_dataset)
            indices = torch.randperm(num_train).tolist()
            valid_size = int(np.floor(valid_frac * num_train))
            train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

            train_dataset = Subset(train_dataset, train_idx)
            valid_dataset = Subset(valid_dataset, valid_idx)
        else:
            test_dataset = dataset_class(
                root=root,
                train=False,
                download=True,
                transform=test_transform
            )
    elif dataset == 'imagenet-32' or dataset == 'imagenet-64':
        if dataset == 'imagenet-32':
            root = dataset_root('imagenet32')
            c, h, w = (3, 32, 32)
            dataset_class = data.ImageNet32
        else:
            root = dataset_root('imagenet64')
            c, h, w = (3, 64, 64)
            dataset_class = data.ImageNet64

        if train:
            train_dataset = dataset_class(
                root=root,
                train=True,
                download=True,
                transform=tvt.Compose([
                    tvt.ToTensor(),
                    Preprocess(num_bits)
                ])
            )

            num_train = len(train_dataset)
            valid_size = int(np.floor(num_train * valid_frac))
            train_size = num_train - valid_size
            train_dataset, valid_dataset = random_split(train_dataset,
                                                        (train_size, valid_size))
        else:
            test_dataset = dataset_class(
                root=root,
                train=False,
                download=True,
                transform=tvt.Compose([
                    tvt.ToTensor(),
                    Preprocess(num_bits)
                ])
            )
    elif dataset == 'celeba-hq-64-fast':
        root = dataset_root('celeba_hq_64_fast')
        c, h, w = (3, 64, 64)

        train_transform = tvt.Compose([
            RandomHorizontalFlipTensor(),
            Preprocess(num_bits)
        ])
        test_transform = Preprocess(num_bits)

        if train:
            train_dataset = data.CelebAHQ64Fast(
                root=root,
                train=True,
                download=True,
                transform=train_transform
            )

            valid_dataset = data.CelebAHQ64Fast(
                root=root,
                train=True,
                transform=test_transform # Note different transform.
            )

            num_train = len(train_dataset)
            indices = torch.randperm(num_train).tolist()
            valid_size = int(np.floor(valid_frac * num_train))
            train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

            train_dataset = Subset(train_dataset, train_idx)
            valid_dataset = Subset(valid_dataset, valid_idx)
        else:
            test_dataset = data.CelebAHQ64Fast(
                root=root,
                train=False,
                download=True,
                transform=test_transform
            )

    # if dataset_name == 'fashion-mnist' or dataset_name == 'mnist':
    #     base_transforms = [
    #         tvt.Pad((pad, pad)),
    #         tvt.ToTensor()
    #     ]
    #
    #     root = dataset_root(dataset_name)
    #
    #     c, h, w = (1, 28 + 2 * pad, 28 + 2 * pad)
    #
    #     if dataset_name == 'fashion-mnist':
    #         dataset_cls = datasets.FashionMNIST
    #         base_transforms.insert(0, tvt.RandomHorizontalFlip())
    #     else:
    #         dataset_cls = datasets.MNIST
    #
    #     dataset = dataset_cls(
    #         root=root,
    #         train=train,
    #         transform=tvt.Compose(
    #             base_transforms + [Preprocess(num_bits, jitter=train)]
    #         ),
    #         download=True
    #     )
    #
    # elif dataset_name == 'celeba-64':
    #     if not train:
    #         raise RuntimeError('No test set for CelebA.')
    #
    #     root = dataset_root('celeba')
    #     c, h, w = (3, 64, 64)
    #     dataset = data.CelebA(
    #         root=root,
    #         transform=tvt.Compose([
    #             tvt.CenterCrop(148),
    #             tvt.Resize(64),
    #             tvt.RandomHorizontalFlip(),
    #             tvt.ToTensor(),
    #             Preprocess(num_bits)
    #         ]),
    #         download=True
    #     )
    #
    # elif dataset_name == 'celeba-hq-64':
    #     if not train:
    #         raise RuntimeError('No test set for CelebA.')
    #
    #     root = dataset_root('celeba-hq')
    #     c, h, w = (3, 64, 64)
    #     dataset = data.CelebAHQ(
    #         root=root,
    #         transform=tvt.Compose([
    #             tvt.Resize(64),
    #             tvt.RandomHorizontalFlip(),
    #             tvt.ToTensor(),
    #             Preprocess(num_bits)
    #         ]),
    #         download=True
    #     )
    #

    else:
        raise RuntimeError('Unknown dataset')

    if train:
        return train_dataset, valid_dataset, (c, h, w)
    else:
        return test_dataset, (c, h, w)
