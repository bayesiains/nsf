import numpy as np
import os
import pandas as pd
import utils

from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def load_gas():
    def load_data(file):
        data = pd.read_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    def get_correlation_numbers(data):
        C = data.corr()
        A = C > 0.98
        B = A.sum(axis=1)
        return B

    def load_data_and_clean(file):
        data = load_data(file)
        B = get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)
        data = (data - data.mean()) / data.std()

        return data.values

    def load_data_and_clean_and_split(file):
        data = load_data_and_clean(file)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    return load_data_and_clean_and_split(
        file=os.path.join(utils.get_data_root(), 'gas', 'ethylene_CO.pickle')
    )


def save_splits():
    train, val, test = load_gas()
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join(utils.get_data_root(), 'gas', '{}.npy'.format(name))
        np.save(file, data)


class GasDataset(Dataset):
    def __init__(self, split='train', frac=None):
        path = os.path.join(utils.get_data_root(), 'gas', '{}.npy'.format(split))
        self.data = np.load(path).astype(np.float32)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n


def main():
    dataset = GasDataset(split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    print(np.where(dataset.data == dataset.data.max()))
    fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    axs = axs.reshape(-1)
    for i, dimension in enumerate(dataset.data.T):
        print(i)
        axs[i].hist(dimension, bins=100)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
