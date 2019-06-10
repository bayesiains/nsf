import os

import numpy as np
import torch

from torch.utils import data
from torchvision.datasets.folder import (default_loader,
                                         has_file_allowed_extension,
                                         IMG_EXTENSIONS)

from .plane import GaussianDataset
from .plane import CrescentDataset
from .plane import CrescentCubedDataset
from .plane import SineWaveDataset
from .plane import AbsDataset
from .plane import SignDataset
from .plane import FourCircles
from .plane import DiamondDataset
from .plane import TwoSpiralsDataset
from .plane import CheckerboardDataset
from .plane import FaceDataset
from .gas import GasDataset
from .power import PowerDataset
from .hepmass import HEPMASSDataset
from .miniboone import MiniBooNEDataset
from .bsds300 import BSDS300Dataset


def load_dataset(name, split, frac=None):
    """Loads and returns a requested dataset.

    Args:
        name: string, the name of the dataset.
        split: one of 'train', 'val' or 'test', the dataset split.
        frac: float between 0 and 1 or None, the fraction of the dataset to be returned.
            If None, defaults to the whole dataset.

    Returns:
        A Dataset object, the requested dataset.

    Raises:
         ValueError: If any of the arguments has an invalid value.
    """

    if split not in ['train', 'val', 'test']:
        raise ValueError('Split must be one of \'train\', \'val\' or \'test\'.')

    if frac is not None and (frac < 0 or frac > 1):
        raise ValueError('Frac must be between 0 and 1.')

    try:
        return {
            'power': PowerDataset,
            'gas': GasDataset,
            'hepmass': HEPMASSDataset,
            'miniboone': MiniBooNEDataset,
            'bsds300': BSDS300Dataset
        }[name](split=split, frac=frac)

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


def get_uci_dataset_range(dataset_name):
    """
    Returns the per dimension (min, max) range for a specified UCI dataset.

    :param dataset_name:
    :return:
    """
    train_dataset = load_dataset(dataset_name, split='train')
    val_dataset = load_dataset(dataset_name, split='val')
    test_dataset = load_dataset(dataset_name, split='test')
    train_min, train_max = np.min(train_dataset.data, axis=0), np.max(train_dataset.data, axis=0)
    val_min, val_max = np.min(val_dataset.data, axis=0), np.max(val_dataset.data, axis=0)
    test_min, test_max = np.min(test_dataset.data, axis=0), np.max(test_dataset.data, axis=0)
    min_ = np.minimum(train_min, np.minimum(val_min, test_min))
    max_ = np.maximum(train_max, np.maximum(val_max, test_max))
    return np.array((min_, max_))


def get_uci_dataset_max_abs_value(dataset_name):
    """
    Returns the max of the absolute values of a specified UCI dataset.

    :param dataset_name:
    :return:
    """
    range_ = get_uci_dataset_range(dataset_name)
    return np.max(np.abs(range_))


def load_plane_dataset(name, num_points, flip_axes=False):
    """Loads and returns a plane dataset.

    Args:
        name: string, the name of the dataset.
        num_points: int, the number of points the dataset should have,
        flip_axes: bool, flip x and y axes if True.

    Returns:
        A Dataset object, the requested dataset.

    Raises:
         ValueError: If `name` an unknown dataset.
    """

    try:
        return {
            'gaussian': GaussianDataset,
            'crescent': CrescentDataset,
            'crescent_cubed': CrescentCubedDataset,
            'sine_wave': SineWaveDataset,
            'abs': AbsDataset,
            'sign': SignDataset,
            'four_circles': FourCircles,
            'diamond': DiamondDataset,
            'two_spirals': TwoSpiralsDataset,
            'checkerboard': CheckerboardDataset,
        }[name](num_points=num_points, flip_axes=flip_axes)

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


def load_face_dataset(name, num_points, flip_axes=False):
    """Loads and returns a face dataset.

    Args:
        name: string, the name of the dataset.
        num_points: int, the number of points the dataset should have,
        flip_axes: bool, flip x and y axes if True.

    Returns:
        A Dataset object, the requested dataset.
    """
    return FaceDataset(num_points=num_points, name=name, flip_axes=flip_axes)

def batch_generator(loader, num_batches=int(1e10)):
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return


class InfiniteLoader(data.DataLoader):
    """A data loader that can load a dataset repeatedly."""

    def __init__(self, num_epochs=None, *args, **kwargs):
        """Constructor.

        Args:
            dataset: A `Dataset` object to be loaded.
            batch_size: int, the size of each batch.
            shuffle: bool, whether to shuffle the dataset after each epoch.
            drop_last: bool, whether to drop last batch if its size is less than
                `batch_size`.
            num_epochs: int or None, number of epochs to iterate over the dataset.
                If None, defaults to infinity.
        """
        super().__init__(
            *args, **kwargs
        )
        self.finite_iterable = super().__iter__()
        self.counter = 0
        self.num_epochs = float('inf') if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = super().__iter__()
            return next(self.finite_iterable)

    def __iter__(self):
        return self

    def __len__(self):
        return None

def load_num_batches(loader, num_batches):
    """A generator that returns num_batches batches from the loader, irrespective of the length
    of the dataset."""
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return

class UnlabelledImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = self.find_images(os.path.join(root))

    def __getitem__(self, index):
        path = self.paths[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        # Add a bogus label to be compatible with standard image datasets.
        return image, torch.tensor([0.])

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def find_images(dir):
        paths = []
        for fname in sorted(os.listdir(dir)):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(dir, fname)
                paths.append(path)
        return paths
