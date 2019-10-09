Torch3d
=======

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
- **PointCNN** from Li et al. (NeurIPS 2018) [[arXiv](https://arxiv.org/abs/1801.07791)]

Installation
------------

Torch3d requires PyTorch 1.2 or newer. Some other dependencies are:
- torchvision (only needed to download datasets, may consider dropping it later)
- h5py

From source:

```bash
git clone https://github.com/pqhieu/torch3d
cd torch3d
python setup.py install
```

Roadmap
-------

**v0.1**
- [ ] PointCNN models
- [ ] Publish on PyPi
- [ ] Improve documentation
