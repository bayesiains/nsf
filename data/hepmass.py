import numpy as np
import os
import pandas as pd
import utils

from collections import Counter
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def load_hepmass():
    def load_data(path):

        data_train = pd.read_csv(filepath_or_buffer=os.path.join(path, '1000_train.csv'),
                                 index_col=False)
        data_test = pd.read_csv(filepath_or_buffer=os.path.join(path, '1000_test.csv'),
                                index_col=False)

        return data_train, data_test

    def load_data_no_discrete(path):
        """Loads the positive class examples from the first 10% of the dataset."""
        data_train, data_test = load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data_ set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        return data_train, data_test

    def load_data_no_discrete_normalised(path):

        data_train, data_test = load_data_no_discrete(path)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_test

    def load_data_no_discrete_normalised_as_array(path):

        data_train, data_test = load_data_no_discrete_normalised(path)
        data_train, data_test = data_train.values, data_test.values

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array(
            [i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array(
            [i for i in range(data_test.shape[1]) if i not in features_to_remove])]

        N = data_train.shape[0]
        N_validate = int(N * 0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    return load_data_no_discrete_normalised_as_array(
        path=os.path.join(utils.get_data_root(), 'hepmass')
    )


def save_splits():
    train, val, test = load_hepmass()
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join(utils.get_data_root(), 'hepmass', '{}.npy'.format(name))
        np.save(file, data)


class HEPMASSDataset(Dataset):
    def __init__(self, split='train', frac=None):
        path = os.path.join(utils.get_data_root(), 'hepmass', '{}.npy'.format(split))
        self.data = np.load(path).astype(np.float32)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n


def main():
    dataset = HEPMASSDataset(split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    plt.hist(dataset.data.reshape(-1), bins=250)
    plt.show()


if __name__ == '__main__':
    main()
