import torch
from torchvision.datasets import CIFAR10

class CIFAR10Fast(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

        self.data = self.data.transpose((0, 3, 1, 2)) # HWC -> CHW.
        self.data = torch.from_numpy(self.data) # Shouldn't make a copy.
        assert self.data.dtype == torch.uint8

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Don't convert to PIL Image, just to convert back later: slow.

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
