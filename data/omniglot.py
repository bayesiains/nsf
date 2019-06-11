import numpy as np
import os
import torch

import utils

from PIL import Image
from scipy.io import loadmat
from torch.utils import data
from torchvision import transforms as tvtransforms


class OmniglotDataset(data.Dataset):
    def __init__(self, split='train', transform=None):
        self.transform = transform
        path = os.path.join(utils.get_data_root(), 'omniglot', 'omniglot.mat')
        rawdata = loadmat(path)

        if split == 'train':
            self.data = rawdata['data'].T.reshape(-1, 28, 28)
            self.targets = rawdata['target'].T
        elif split == 'test':
            self.data = rawdata['testdata'].T.reshape(-1, 28, 28)
            self.targets = rawdata['testtarget'].T
        else:
            raise ValueError

    def __getitem__(self, item):
        image, target = self.data[item], self.targets[item]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


def main():
    transform = tvtransforms.Compose([
        tvtransforms.ToTensor(),
        tvtransforms.Lambda(torch.bernoulli)
    ])
    dataset = OmniglotDataset(split='test', transform=transform)
    loader = data.DataLoader(dataset, batch_size=16)
    batch = next(iter(loader))[0]
    from matplotlib import pyplot as plt
    from experiments import cutils
    from torchvision.utils import make_grid
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    cutils.gridimshow(make_grid(batch, nrow=4), ax)
    plt.show()


if __name__ == '__main__':
    main()
