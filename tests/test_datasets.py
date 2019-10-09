import shutil
import pytest
import tempfile
import numpy as np
import torch3d.datasets as datasets


class TestDataset:
    def test_modelnet40(self):
        root = tempfile.mkdtemp()
        dataset = datasets.ModelNet40(root, download=True)
        assert len(dataset) == 9840
        shutil.rmtree(root)
        sample, target = dataset[0]
        assert isinstance(sample, np.ndarray)
        assert isinstance(target, np.int64)

    def test_s3dis(self):
        root = tempfile.mkdtemp()
        dataset = datasets.S3DIS(root, download=True)
        shutil.rmtree(root)
        sample, target = dataset[0]
        assert isinstance(sample, np.ndarray)
        assert isinstance(target, np.ndarray)
