"""
Run the following commands before running this file
wget http://image-net.org/small/train_64x64.tar
wget http://image-net.org/small/valid_64x64.tar
tar -xvf train_64x64.tar
tar -xvf valid_64x64.tar
"""

import numpy as np
import scipy.ndimage
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

def process_images(*, path, outfile):
    assert os.path.exists(path), "Input path doesn't exist"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    print('Number of valid images is:', len(files))
    imgs = []
    for i in tqdm(range(len(files))):
        img = scipy.ndimage.imread(join(path, files[i]))

        assert isinstance(img, np.ndarray)

        img = img.astype('uint8')

        # HWC -> CHW, for use in PyTorch
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 64, 64)

        imgs.append(img)

    imgs = np.asarray(imgs).astype('uint8')
    assert imgs.shape[1:] == (3, 64, 64)

    np.save(outfile, imgs)

if __name__ == '__main__':
    process_images(path='./train_64x64', outfile='./train_64x64.npy')
    process_images(path='./valid_64x64', outfile='./valid_64x64.npy')
