import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class S3DIS(Dataset):
    """
    The `S3DIS <http://buildingparser.stanford.edu/dataset.html>`_ dataset.

    """

    name = 's3dis'
    url = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
    basedir = 'indoor3d_sem_seg_hdf5_data'

    def __init__(self, root, train=True, test_area=5, transform=None, download=False):
        self.root = root
        self.train = train
        self.test_area = test_area
        self.samples = []
        self.targets = []
        self.transform = transform

        if download:
            self.download()

        with open(os.path.join(self.root, self.name, 'all_files.txt')) as fp:
            filelist = [os.path.basename(x.strip()) for x in fp]
        with open(os.path.join(self.root, self.name, 'room_filelist.txt')) as fp:
            rooms = [x.strip() for x in fp]

        # now load the data from .h5 files
        for filename in filelist:
            h5 = h5py.File(os.path.join(self.root, self.name, filename), 'r')
            assert 'data' in h5 and 'label' in h5
            self.samples.append(np.array(h5['data'][:]))
            self.targets.append(np.array(h5['label'][:]))
        self.samples = np.concatenate(self.samples, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        self.targets = np.squeeze(self.targets).astype(np.int64)

        # filter point cloud not in area of interest
        area = 'Area_' + str(self.test_area)
        indices = [i for i, room in enumerate(rooms) if area in room]
        if self.train:
            indices = list(set(range(len(rooms))) - set(indices))
        self.samples = self.samples[indices][..., :3]
        self.targets = self.targets[indices]

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
