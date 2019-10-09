import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class S3DIS(Dataset):
    """
    The `S3DIS <http://buildingparser.stanford.edu/dataset.html>`_ dataset.

    """

    url = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
    basedir = 'indoor3d_sem_seg_hdf5_data'

    def __init__(self, root, train=True, test_area=5, transform=None, download=False):
        self.root = root
        self.train = train
        self.test_area = test_area
        self.transform = transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            flist = self.splits['train']
        else:
            flist = self.splits['test']

        self.samples = []
        self.targets = []

        for filename, md5 in flist:
            h5 = h5py.File(os.path.join(self.root, self.basedir, filename), 'r')
            assert 'data' in h5 and 'label' in h5
            self.samples.append(np.array(h5['data'][:]))
            self.targets.append(np.array(h5['label'][:]))
            h5.close()
        self.samples = np.concatenate(self.samples, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        self.targets = np.squeeze(self.targets).astype(np.int64)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, i):
        sample = self.samples[i]
        target = self.targets[i]
        if self.transform is not None:
            sample, target = self.transform(sample, target)
        return sample, target

    def download(self):
        if not self._check_integrity():
            download_and_extract_archive(self.url, self.root)

    def _check_integrity(self):
        for filename, md5 in (self.splits['train'] + self.splits['test']):
            fpath = os.path.join(self.root, self.basedir, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
