import numpy as np

import os
import zipfile

import torch
from torch.utils.data import Dataset

from data import UnlabelledImageFolder
from data.download import download_file_from_google_drive


class CelebA(UnlabelledImageFolder):
    """Unlabelled standard CelebA dataset, the aligned version."""
    GOOGLE_DRIVE_FILE_ID = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    ZIP_FILE_NAME = 'img_align_celeba.zip'

    def __init__(self, root, transform=None, download=False):
        if download:
            self.download(root)
        super(CelebA, self).__init__(os.path.join(root, self.img_dir),
                                     transform=transform)

    @property
    def img_dir(self):
        return 'img_align_celeba'

    def download(self, root):
        if os.path.isdir(os.path.join(root, self.img_dir)):
            return # Downloaded already

        os.makedirs(root, exist_ok=True)

        zip_file = os.path.join(root, self.ZIP_FILE_NAME)

        print('Downloading {}...'.format(os.path.basename(zip_file)))
        download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID, zip_file)

        print('Extracting {}...'.format(os.path.basename(zip_file)))
        with zipfile.ZipFile(zip_file, 'r') as fp:
            fp.extractall(root)

        os.remove(zip_file)


class CelebAHQ(CelebA):
    """Unlabelled high quality CelebA dataset with 256x256 images."""
    GOOGLE_DRIVE_FILE_ID = '1psLniAvAvyDgJV8DBk7cvTZ9EasB_2tZ'
    ZIP_FILE_NAME = 'celeba-hq-256.zip'

    def __init__(self, root, transform=None, train=True, download=False):
        self.train = train
        super().__init__(root, transform=transform, download=download)

    @property
    def img_dir(self):
        if self.train:
            return 'celeba-hq-256/train-png'
        else:
            return 'celeba-hq-256/validation-png'

class CelebAHQ64Fast(Dataset):
    GOOGLE_DRIVE_FILE_ID = {
        'train': '1bcaqMKWzJ-2ca7HCQrUPwN61lfk115TO',
        'valid': '1WfE64z9FNgOnLliGshUDuCrGBfJSwf-t'
    }

    NPY_NAME = {
        'train': 'train.npy',
        'valid': 'valid.npy'
    }

    def __init__(self, root, train=True, download=False, transform=None):
        self.transform = transform
        self.root = root

        if download:
            self._download()

        tag = 'train' if train else 'valid'
        npy_data = np.load(os.path.join(root, self.NPY_NAME[tag]))
        self.data = torch.from_numpy(npy_data) # Shouldn't make a copy.

    def __getitem__(self, index):
        img = self.data[index, ...]

        if self.transform is not None:
            img = self.transform(img)

        # Add a bogus label to be compatible with standard image datasets.
        return img, torch.tensor([0.])

    def __len__(self):
        return self.data.shape[0]

    def _download(self):
        os.makedirs(self.root, exist_ok=True)

        for tag in ['train','valid']:
            npy = os.path.join(self.root, self.NPY_NAME[tag])
            if not os.path.isfile(npy):
                print('Downloading {}...'.format(self.NPY_NAME[tag]))
                download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID[tag], npy)
