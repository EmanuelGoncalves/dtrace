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

included_files = {"dtrace": [
    "data/meta/drugsheet_20190210.xlsx",

    "data/drug/screening_set_384_all_owners_fitted_data_20180308_updated.csv",
    "data/drug/fitted_rapid_screen_1536_v1.2.1_20181026_updated.csv",

    "data/crispr/sanger_depmap18_fc_corrected.csv",
    "data/crispr/sanger_depmap18_fc_ess_aucs.csv",
    "data/crispr/broad_depmap18q4_fc_corrected.csv",
    "data/crispr/broad_depmap18q4_fc_ess_aucs.csv",
    "data/crispr/depmap19Q1_essential_genes.txt",
    "data/crispr/projectscore_essential_genes.txt",

    "data/meta/model_list_2018-09-28_1452.csv",
    "data/meta/growth_rates_rapid_screen_1536_v1.2.2_20181113.csv",
    "data/meta/samples_origin.csv",

    "data/genomic/PANCAN_mobem.csv",

    "data/genomic/rnaseq_voom.csv.gz",
    "data/genomic/rnaseq_rpkm.csv.gz",

    "data/ppi/9606.protein.links.full.v10.5.txt",
    "data/ppi/9606.protein.aliases.v10.5.txt",
    "data/ppi/BIOGRID-ORGANISM-Homo_sapiens-3.4.157.tab2.txt",

    "data/genomic/WES_variants.csv.gz",
]}

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
