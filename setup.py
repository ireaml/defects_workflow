"""This is a setup.py script to install ShakeNBreak"""

import os
import warnings

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

path_to_file = os.path.dirname(os.path.abspath(__file__))

setup(
    name="defects_workflow",
    version="0.0.1",
    description="Package to run defect calculations with AiiDA",
    author="Irea Mosquera-Lois",
    author_email="i.mosquera-lois22@imperial.ac.uk",
    maintainer="Irea Mosquera-Lois",
    maintainer_email="i.mosquera-lois22@imperial.ac.uk",
    readme="README.md",  # PyPI readme
    license="MIT",
    license_files=("LICENSE",),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="chemistry pymatgen dft defects",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pymatgen>=2022.10.22",
        "pymatgen-analysis-defects>=2022.10.28",
        "ase",
        "pandas>=1.1.0",
        "seaborn",
        "monty",
    ]
)

