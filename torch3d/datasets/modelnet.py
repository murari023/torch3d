import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class ModelNet40(Dataset):
    """
    The `ModelNet40 <https://modelnet.cs.princeton.edu/>`_ dataset.

    """

    name = "modelnet40"
    url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
    basedir = "modelnet40_ply_hdf5_2048"
    splits = {
        "train": [
            ("ply_data_train0.h5", "3176385ffc31a7b6b5af22191fd920d1"),
            ("ply_data_train1.h5", "e3f613fb500559403b34925112754dc4"),
            ("ply_data_train2.h5", "0c56e233a090ff87c3049d4ce08e7d8b"),
            ("ply_data_train3.h5", "9d2af465adfa33a3285c369f3ca66c45"),
            ("ply_data_train4.h5", "dff38de489b2c41bfaeded86c2208984")
        ],
        "test": [
            ("ply_data_test0.h5", "e9732e6d83b09e79e9a7617df058adee"),
            ("ply_data_test1.h5", "aba4b12a67c34391cc3c015a6f08ed4b")
        ]
    }
    categories = [
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox"
    ]

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 download=False,
                 categories=None):
        self.root = root
        self.train = train
        self.transform = transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            flist = self.splits["train"]
        else:
            flist = self.splits["test"]

        self.data = []
        self.labels = []

        for filename, md5 in flist:
            h5 = h5py.File(os.path.join(self.root, self.name, filename), "r")
            assert "data" in h5 and "label" in h5
            self.data.append(np.array(h5["data"][:]))
            self.labels.append(np.array(h5["label"][:]))
            h5.close()
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.labels = np.squeeze(self.labels).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        points = self.data[i]
        label = self.labels[i]
        if self.transform is not None:
            points, label = self.transform(points, label)
        return points, label

    def download(self):
        if not self._check_integrity():
            download_and_extract_archive(self.url, self.root)
            os.rename(os.path.join(self.root, self.basedir),
                      os.path.join(self.root, self.name))

    def _check_integrity(self):
        flist = self.splits["train"] + self.splits["test"]
        for filename, md5 in flist:
            fpath = os.path.join(self.root, self.name, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
