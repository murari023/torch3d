import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class ShapeNetPart(Dataset):
    """
    The ShapeNet part segmentation dataset.

    """

    url = "https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip"
    basedir = "shapenetcore_partanno_segmentation_benchmark_v0"
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

    def __init__(self, root, split="train", transform=None, download=False, categories=None):
        self.root = root
        self.split = split
        self.transform = transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        self.dataset = []
        self.targets = []

    def download(self):
        if not self._check_integrity():
            download_and_extract_archive(self.url, self.root)
            os.rename(os.path.join(self.root, self.basedir),
                      os.path.join(self.root, self.__class__.__name__))

    def _check_integrity(self):
        for _, d in self.cat2synset.items():
            fpath = os.path.join(self.root, self.__class__.__name__, d)
            if not os.path.exists(fpath):
                return False
            return True
