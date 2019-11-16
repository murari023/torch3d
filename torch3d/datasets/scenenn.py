import os
import h5py
import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class SceneNN(VisionDataset):
    """
    The `SceneNN <http://scenenn.net/>`_ dataset.

    """

    name = "scenenn"
    url = "http://103.24.77.34:8080/scenenn/home/cvpr18/data/scenenn_seg_76.zip"
    basedir = "scene_seg_76"
    splits = {
        "train": [
            "005",
            "014",
            "015",
            "016",
            "025",
            "036",
            "038",
            "041",
            "045",
            "047",
            "052",
            "054",
            "057",
            "061",
            "062",
            "066",
            "071",
            "073",
            "078",
            "080",
            "084",
            "087",
            "089",
            "096",
            "098",
            "109",
            "201",
            "202",
            "209",
            "217",
            "223",
            "225",
            "227",
            "231",
            "234",
            "237",
            "240",
            "243",
            "249",
            "251",
            "255",
            "260",
            "263",
            "265",
            "270",
            "276",
            "279",
            "286",
            "294",
            "308",
            "522",
            "609",
            "613",
            "614",
            "623",
            "700",
        ],
        "test": [
            "011",
            "021",
            "065",
            "032",
            "093",
            "246",
            "086",
            "069",
            "206",
            "252",
            "273",
            "527",
            "621",
            "076",
            "082",
            "049",
            "207",
            "213",
            "272",
            "074",
        ],
    }

    def __init__(
        self,
        root,
        train=True,
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super(SceneNN, self).__init__(root, transforms, transform, target_transform)
        self.train = train

        if self.train:
            self.split = self.splits["train"]
        else:
            self.split = self.splits["test"]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        self.data = []
        self.targets = []

        for scn in self.split:
            basename = "scenenn_seg_" + scn + ".hdf5"
            h5 = h5py.File(os.path.join(self.root, self.name, basename), "r")
            assert "data" in h5 and "label" in h5
            self.data.append(np.array(h5["data"][:]))
            self.targets.append(np.array(h5["label"][:]))
            h5.close()

        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        pcd = self.data[i]
        target = self.targets[i]
        if self.transforms is not None:
            pcd, target = self.transforms(pcd, target)
        return pcd, target

    def download(self):
        if not self._check_integrity():
            download_and_extract_archive(self.url, self.root)
            os.rename(
                os.path.join(self.root, self.basedir),
                os.path.join(self.root, self.name),
            )

    def _check_integrity(self):
        for scn in self.split:
            basename = "scenenn_seg_" + scn + ".hdf5"
            if not os.path.exists(os.path.join(self.root, self.name, basename)):
                print(os.path.join(self.root, self.name, basename))
                return False
        return True
