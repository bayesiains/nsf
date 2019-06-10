import os
import zipfile

import torch
from torch.utils.data import Dataset
from data import UnlabelledImageFolder
from data.download import download_file_from_google_drive
import numpy as np

class ImageNet32(UnlabelledImageFolder):
    GOOGLE_DRIVE_FILE_ID = '1TXsg8TP5SfsSL6Gk39McCkZu9rhSQnNX'
    UNZIPPED_DIR_NAME = 'imagenet32'
    UNZIPPED_TRAIN_SUBDIR = 'train_32x32'
    UNZIPPED_VAL_SUBDIR = 'valid_32x32'

    def __init__(self, root, train=True, download=False, transform=None):
        if download:
            self._download(root)

        img_dir = 'train' if train else 'val'
        super(ImageNet32, self).__init__(os.path.join(root, img_dir),
                                         transform=transform)

    def _download(self, root):
        if os.path.isdir(os.path.join(root, 'train')):
            return  # Downloaded already

        os.makedirs(root, exist_ok=True)

        zip_file = os.path.join(root, self.UNZIPPED_DIR_NAME + '.zip')

        print('Downloading {}...'.format(os.path.basename(zip_file)))
        download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID, zip_file)

        print('Extracting {}...'.format(os.path.basename(zip_file)))
        with zipfile.ZipFile(zip_file, 'r') as fp:
            fp.extractall(root)
        os.remove(zip_file)

        os.rename(os.path.join(root, self.UNZIPPED_DIR_NAME, self.UNZIPPED_TRAIN_SUBDIR),
                  os.path.join(root, 'train'))
        os.rename(os.path.join(root, self.UNZIPPED_DIR_NAME, self.UNZIPPED_VAL_SUBDIR),
                  os.path.join(root, 'val'))
        os.rmdir(os.path.join(root, self.UNZIPPED_DIR_NAME))


class ImageNet64(ImageNet32):
    GOOGLE_DRIVE_FILE_ID = '1NqpYnfluJz9A2INgsn16238FUfZh9QwR'
    UNZIPPED_DIR_NAME = 'imagenet64'
    UNZIPPED_TRAIN_SUBDIR = 'train_64x64'
    UNZIPPED_VAL_SUBDIR = 'valid_64x64'

class ImageNet64Fast(Dataset):
    GOOGLE_DRIVE_FILE_ID = {
        'train': '15AMmVSX-LDbP7LqC3R9Ns0RPbDI9301D',
        'valid': '1Me8EhsSwWbQjQ91vRG1emkIOCgDKK4yC'
    }

    NPY_NAME = {
        'train': 'train_64x64.npy',
        'valid': 'valid_64x64.npy'
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
