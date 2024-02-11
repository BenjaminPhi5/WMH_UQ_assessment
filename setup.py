# setup for the trustworthai python library with required packages

from setuptools import find_packages, setup

setup(
    name="trustworthai",
    version="0.1.0",
    packages=find_packages(exclude=["examples", "misc experiments", "slurm_cluster", "other_works_demos"]),

    install_requires = [
        "numpy",
        "statsmodels",
        "nibabel",
        "SimpleITK",
        "pyrobex",
        "itkwidgets",
        "natsort",
        "seaborn",
        "matplotlib",
        "torch",
        "torchvision",
        "deepspeed",
        "tqdm",
        "connected-components-3d",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "kornia",
        "tensorboard",
        "pytorch-lightning",
        "torchinfo" # probably shouldn't have this in any production repo as it might break and then people wouldn't
        # use any of this code sad face.
    ]

)
