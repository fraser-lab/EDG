"""
Setup module for ADP3D.
"""

import os
import sys

from setuptools import setup, find_packages

sys.path.insert(0, f"{os.path.dirname(__file__)}/adp3d")

import adp3d

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "adp3d")

setup(
    name="model_angelo",
    entry_points={
        "console_scripts": [
            "adp3d = adp3d.__main__:main",
        ],
    },
    packages=find_packages(),
    version=adp3d.__version__,
    install_requires=[
        "tqdm",
        "scipy",
        "biopython>=1.81",
        "einops",
        "matplotlib",
        "numpy",
        "pandas",
        "chroma",
        "loguru",
    ],
)