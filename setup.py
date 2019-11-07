import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


requirements = [
    "h5py",
    "numpy",
    "torch",
    "torchvision"
]

sources = []
ext_modules = []
define_macros = []
extension = CppExtension
sources += glob.glob(os.path.join("torch3d", "csrc", "cpu", "*.cpp"))

if CUDA_HOME is not None:
    extension = CUDAExtension
    sources += glob.glob(os.path.join("torch3d", "csrc", "*.cpp"))
    sources += glob.glob(os.path.join("torch3d", "csrc", "cuda", "*.cu"))
    define_macros += [('WITH_CUDA', None)]

ext_modules += [
    extension(
        "torch3d._C",
        sources,
        define_macros=define_macros
    )
]

__version__ = "0.2.0"
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
