import shutil
import pytest
import tempfile
import numpy as np
import torch3d.datasets as datasets


class TestDatasets:
    def test_modelnet40(self):
        root = "data"
        dataset = datasets.ModelNet40(root, download=True)
        assert len(dataset) == 9840
        points, target = dataset[0]
        assert isinstance(points, np.ndarray)

    def test_s3dis(self):
        root = "data"
        dataset = datasets.S3DIS(root, download=True)
        assert len(dataset) == 16733
        points, target = dataset[0]
        assert isinstance(points, np.ndarray)
        assert isinstance(target, np.ndarray)
