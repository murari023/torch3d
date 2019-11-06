import torch
import torch3d.datasets as datasets


ROOT_DIR = "data"


class TestModelnet40:
    def test_download(self):
        dataset = datasets.ModelNet40(ROOT_DIR, download=True)
        assert dataset._check_integrity()


class TestS3DIS:
    def test_download(self):
        dataset = datasets.S3DIS(ROOT_DIR, train=False, download=True)
        assert dataset._check_integrity()


class TestShapeNetPart:
    def test_download(self):
        dataset = datasets.ShapeNetPart(ROOT_DIR, split="train", download=True)
        assert dataset._check_integrity()
