import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class ShapeNetPart(Dataset):
    """
    The ShapeNet part segmentation dataset.

    """

    name = "shapenetpart"
    url = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
    basedir = "hdf5_data"
    splits = {
        "train": [
            ("ply_data_train0.h5", "e7152ff588ae6c87eb1156823df05855"),
            ("ply_data_train1.h5", "b4894a082211418b61b5a707fedb4f56"),
            ("ply_data_train2.h5", "508eeeee96053b90388520c37df3a8b8"),
            ("ply_data_train3.h5", "88574c3d5c61d0f3156b9e02cd6cda03"),
            ("ply_data_train4.h5", "418cb01104740bf1353b792331cb5878"),
            ("ply_data_train5.h5", "26e65c8827b08f7c340cd03f902e27e8")
        ],
        "val": [
            ("ply_data_val0.h5", "628b4b3cbc17765de2114d104e51b9c9")
        ],
        "test": [
            ("ply_data_test0.h5", "fa3fb32b179128ede32c2c948ed83efc"),
            ("ply_data_test1.h5", "5eb63ae378831c665282c8f22b6c1249")
        ]
    }
    cat2synset = {
        "airplane": "02691156",
        "bag": "02773838",
        "cap": "02954340",
        "car": "02958343",
        "chair": "03001627",
        "earphone": "03261776",
        "guitar": "03467517",
        "knife": "03624134",
        "lamp": "03636649",
        "laptop": "03642806",
        "motorbike": "03790512",
        "mug": "03797390",
        "pistol": "03948459",
        "rocket": "04099429",
        "skateboard": "04225987",
        "table": "04379243"
    }

    def __init__(self,
                 root,
                 split="train",
                 transform=None,
                 download=False,
                 categories=None):
        self.root = root
        self.split = split
        self.transform = transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        flist = self.splits[split]

        self.points = []
        self.labels = []
        self.parts = []

        for filename, md5 in flist:
            h5 = h5py.File(os.path.join(self.root, self.name, filename), "r")
            assert "data" in h5 and "label" in h5 and "pid" in h5
            self.points.append(np.array(h5["data"][:]))
            self.labels.append(np.array(h5["label"][:]))
            self.parts.append(np.array(h5["pid"][:]))
            h5.close()
        self.points = np.concatenate(self.points, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.parts = np.concatenate(self.parts, axis=0)
        self.labels = np.squeeze(self.labels).astype(np.int64)
        self.parts = np.astype(np.int64)

    def __getitem__(self, i):
        pcd = self.points[i]
        label = self.labels[i]
        part = self.parts[i]
        if self.transform is not None:
            pcd, label, part = self.transform(pcd, label, part)
        return pcd, label, part

    def download(self):
        if not self._check_integrity():
            download_and_extract_archive(self.url, self.root)
            os.rename(os.path.join(self.root, self.basedir),
                      os.path.join(self.root, self.name))

    def _check_integrity(self):
        flist = self.splits["train"] + self.splits["val"] + self.splits["test"]
        for filename, md5 in flist:
            fpath = os.path.join(self.root, self.name, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
