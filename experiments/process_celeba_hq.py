import numpy as np
import scipy.ndimage
from scipy.misc import imresize
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
        assert img.shape == (256, 256, 3)
        assert isinstance(img, np.ndarray)

        img = img.astype('uint8')
        img = imresize(img, (64, 64))

        assert img.dtype == np.uint8
        assert img.shape == (64, 64, 3)

        # HWC -> CHW, for use in PyTorch
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 64, 64)

        imgs.append(img)

    imgs = np.asarray(imgs).astype('uint8')
    assert imgs.shape[1:] == (3, 64, 64)

    np.save(outfile, imgs)

if __name__ == '__main__':
    process_images(path='./train-png', outfile='./train.npy')
    process_images(path='./validation-png', outfile='./valid.npy')
