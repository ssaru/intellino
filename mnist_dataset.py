import os
import pickle

import cv2
import numpy as np

from intellino.utils.data.dataset import Dataset
from utils import *


class MNIST(Dataset):
    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root: str, train: bool, download: bool):
        """
        `MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
        super(MNIST, self).__init__()

        self.root = root
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        with open(os.path.join(self.processed_folder, data_file), 'rb') as f:
            self.data, self.targets = pickle.load(f)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Type casting to uint8. cause Intellino Hardware support only 8bit-integer
        img, target = self.data[index].astype(np.uint8), int(self.targets[index])

        # image resize for limitation about length of vector in Intellino
        img = cv2.resize(img, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)

        # Vector Should be flatten for sending data to Intellino Hardware
        img = img.reshape(1, -1).squeeze()

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_idx(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_idx(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_idx(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_idx(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            pickle.dump(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            pickle.dump(test_set, f)

        print('Done!')


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from PIL import Image

    mnist = MNIST('./mnist_data/', train=True, download=False)

    image, target = mnist.__getitem__(0)

    plt.figure()
    plt.imshow(image)
    plt.title(str(target))
    plt.show()

    image, target = mnist.__getitem__(1)

    plt.figure()
    plt.imshow(image)
    plt.title(str(target))
    plt.show()

    image, target = mnist.__getitem__(2)

    plt.figure()
    plt.imshow(image)
    plt.title(str(target))
    plt.show()
