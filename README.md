Torch3d
=======

[![PyPI](https://img.shields.io/pypi/v/torch3d)](https://pypi.org/project/torch3d)
[![Downloads](https://pepy.tech/badge/torch3d)](https://pepy.tech/project/torch3d)

The Torch3d package consists of datasets, model architectures, and common
operations for 3D deep learning.

Why Torch3d?
------------

For 3D domain, there is currently no official support from PyTorch that likes
[torchvision](https://github.com/pytorch/vision) for images. Torch3d aims to
fill this gap by streamlining the prototyping process of deep neural networks
on discrete 3D domain. Currently, it focuses on deep learning methods on 3D
point clouds.

The following network architectures are currently included:
- **PointNet** from Qi et al. (CVPR 2017) [[arXiv](https://arxiv.org/abs/1612.00593)]
- **PoinNet++** from Qi et al. (NeurIPS 2017) [[arXiv](https://arxiv.org/abs/1706.02413)]
- **PointCNN** from Li et al. (NeurIPS 2018) [[arXiv](https://arxiv.org/abs/1801.07791)]

Installation
------------

Required PyTorch 1.2 or newer. Some other dependencies are:
- torchvision
- h5py

From PyPi:
```bash
pip install torch3d
```

From source:
```bash
git clone https://github.com/pqhieu/torch3d
cd torch3d
pip install --editable .
```

**Note**: Some of the operations only support CUDA.

Roadmap
-------

**0.3.0**
- [ ] Improve documentation
- [ ] More tutorials/examples

**0.2.0**
- [x] PointNet++ model
- [x] ShapeNetPart dataset

**0.1.0**
- [x] PointCNN model
- [x] Publish on PyPi
