import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


requirements = [
    "h5py",
    "numpy",
    "torch",
    "torchvision"
]
ext_modules = []

__version__ = "0.1.0"
url = "https://github.com/pqhieu/torch3d"

setup(
    name="torch3d",
    version=__version__,
    description="Datasets and network architectures for 3D deep learning in PyTorch",
    author="Quang-Hieu Pham",
    author_email="pqhieu1192@gmail.com",
    url="{}/archive/{}.tar.gz".format(url, __version__),
    install_requires=requirements,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages()
)
