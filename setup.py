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
    "data/drug_crispr_icon.png",
    "data/drug_crispr_icon.pdf",

    "data/drug/DrugResponse_MaxC_v1.5.1_20191108.csv",
    "data/drug/DrugResponse_IC50_v1.5.1_20191108.csv",

    "data/crispr/CRISPR_corrected_qnorm_20191108.csv",
    "data/crispr/CRISPR_Institute_Origin_20191108.csv",

    "data/meta/DrugSheet_20191106.csv",
    "data/meta/GrowthRates_v1.3.0_20190222.csv",
    "data/meta/SamplesOrigin_20191106.csv",
    "data/meta/ModelList_20191106.csv",

    "data/genomic/PANCAN_mobem.csv",
    "data/genomic/rnaseq_voom.csv.gz",
    "data/genomic/rnaseq_rpkm.csv.gz",
    "data/genomic/WES_variants.csv.gz",
    "data/genomic/copynumber_total_new_map.csv.gz",

    "data/ppi/9606.protein.links.full.v10.5.txt",
    "data/ppi/9606.protein.aliases.v10.5.txt",
    "data/ppi/BIOGRID-ORGANISM-Homo_sapiens-3.4.157.tab2.txt",

    "data/klaeger_et_al_catds_most_potent.csv",
    "data/klaeger_et_al_idmap.csv",
    "data/klaeger_et_al_catds.csv",

    "data/PCA_GExp_row_vex.csv",
    "data/PCA_GExp_row_pcs.csv",
    "data/PCA_GExp_column_vex.csv",
    "data/PCA_GExp_column_pcs.csv",

    "data/PCA_drug_row_vex.csv",
    "data/PCA_drug_row_pcs.csv",
    "data/PCA_drug_column_vex.csv",
    "data/PCA_drug_column_pcs.csv",

    "data/PCA_CRISPR_row_vex.csv",
    "data/PCA_CRISPR_row_pcs.csv",
    "data/PCA_CRISPR_column_vex.csv",
    "data/PCA_CRISPR_column_pcs.csv",

    "data/growth_drug_correlation.csv",
    "data/growth_CRISPR_correlation.csv",

    "data/drug_lmm_regressions_gexp.csv.gz",
    "data/drug_lmm_regressions_genomic.csv.gz",
    "data/drug_lmm_regressions_crispr.csv.gz",

    "data/drug_lmm_regressions_robust_genomic.csv.gz",
    "data/drug_lmm_regressions_robust_gexp.csv.gz",
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
