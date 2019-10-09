import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class ModelNet40(Dataset):
    """
    The `ModelNet40 <https://modelnet.cs.princeton.edu/>`_ dataset.

    """

    name = 'modelnet40'
    url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    basedir = 'modelnet40_ply_hdf5_2048'

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.samples = []
        self.targets = []
        self.transform = transform

        if download:
            self.download()

        splitfiles = 'train_files.txt' if self.train else 'test_files.txt'
        with open(os.path.join(self.root, self.name, splitfiles)) as fp:
            filelist = [os.path.basename(x.strip()) for x in fp]

        # now load the data from .h5 files
        for filename in filelist:
            h5 = h5py.File(os.path.join(self.root, self.name, filename), 'r')
            assert 'data' in h5 and 'label' in h5
            self.samples.append(np.array(h5['data'][:]))
            self.targets.append(np.array(h5['label'][:]))
        self.samples = np.concatenate(self.samples, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        self.targets = np.squeeze(self.targets).astype(np.int64)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        if self.transform is not None:
            sample, target = self.transform(sample, target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def download(self):
        download_and_extract_archive(self.url, self.root, remove_finished=True)
        os.rename(os.path.join(self.root, self.basedir), os.path.join(self.root, self.name))
