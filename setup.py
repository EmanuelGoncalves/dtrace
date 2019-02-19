#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().split("\n")

version = {}
with open("dtrace/__init__.py") as f:
    exec(f.read(), version)

included_files = {"dtrace": []}

setuptools.setup(
    name="dtrace",
    version=version['__version__'],
    author="Emanuel Goncalves",
    author_email="eg14@sanger.ac.uk",
    long_description=long_description,
    description="Integration of drug sensitivity and gene essentiality screens inform on drug mode-of-action",
    long_description_content_type="text/markdown",
    url="https://github.com/EmanuelGoncalves/dtrace",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data=included_files,
    install_requires=requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ),
)
