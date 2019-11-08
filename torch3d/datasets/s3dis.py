import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class S3DIS(Dataset):
    """
    The `S3DIS <http://buildingparser.stanford.edu/dataset.html>`_ dataset.

    """

    name = "s3dis"
    url = "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"
    basedir = "indoor3d_sem_seg_hdf5_data"
    flist = [
        ("ply_data_all_0.h5", "ec71baeb8b8cf19f75e626225974ae1d"),
        ("ply_data_all_1.h5", "4d61066842bbbc383e0a9a8e34414630"),
        ("ply_data_all_2.h5", "f64d91e4b7084b6b9b9e6cb9511ac767"),
        ("ply_data_all_3.h5", "205bef85da42f335edab1026f1b6d13c"),
        ("ply_data_all_4.h5", "ee7f32ecd3ea433a3dc4be1a42a417d2"),
        ("ply_data_all_5.h5", "4427e15c3817d83b47846f481ead2d31"),
        ("ply_data_all_6.h5", "56454ba836a77934ca22726666b1f8b4"),
        ("ply_data_all_7.h5", "795c57b238a1addcf63357826985ffbe"),
        ("ply_data_all_8.h5", "57795c1f4206acd14b629909971cb5b6"),
        ("ply_data_all_9.h5", "bd5be19c7719bad69788cf3349f46b6e"),
        ("ply_data_all_10.h5", "24d9c96585a8763f416283f7b1032330"),
        ("ply_data_all_11.h5", "5aba80c6d26f4553a3ba44a5b29d8391"),
        ("ply_data_all_12.h5", "1298f9487271a303cd8728e0eb6db431"),
        ("ply_data_all_13.h5", "9787f12ea1844b584d75d704f1bf15ec"),
        ("ply_data_all_14.h5", "e683dbefbaa6d89291f05e67c128c331"),
        ("ply_data_all_15.h5", "f431f58fe6460bfb842e43da14d363e4"),
        ("ply_data_all_16.h5", "5b46e283f0a5601994c3967466b10b8f"),
        ("ply_data_all_17.h5", "05643a31973875bfc580ba82a19195b9"),
        ("ply_data_all_18.h5", "ca258a7159a892c9d888bcaf8bd621c0"),
        ("ply_data_all_19.h5", "1f5dd15150274c5810c714624102eeed"),
        ("ply_data_all_20.h5", "6d44469f525825d17c046217e4ce6750"),
        ("ply_data_all_21.h5", "aedc2ec2984a058b9d914ba6d9b65159"),
        ("ply_data_all_22.h5", "a01ae14ceff1165ad49e1973570c08c5"),
        ("ply_data_all_23.h5", "67e8c5a7179babe18f110ebea1d3e3b7")
    ]
    categories = [
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter"
    ]

    def __init__(self,
                 root,
                 train=True,
                 test_area=5,
                 transform=None,
                 download=False):
        self.root = root
        self.train = train
        self.test_area = test_area
        self.transform = transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        self.data = []
        self.labels = []

        for filename, md5 in self.flist:
            h5 = h5py.File(os.path.join(self.root, self.name, filename), "r")
            assert "data" in h5 and "label" in h5
            self.data.append(np.array(h5["data"][:]))
            self.labels.append(np.array(h5["label"][:]))
            h5.close()
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0).squeeze()

        # Filter point cloud not in area of interest
        with open(os.path.join(self.root, self.name, "room_filelist.txt")) as fp:
            rooms = [x.strip() for x in fp]
        area = "Area_" + str(self.test_area)
        indices = [i for i, room in enumerate(rooms) if area in room]
        if self.train:
            indices = list(set(range(len(rooms))) - set(indices))
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        self.labels = self.labels.astype(np.int64)

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
        for filename, md5 in (self.flist):
            fpath = os.path.join(self.root, self.name, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
