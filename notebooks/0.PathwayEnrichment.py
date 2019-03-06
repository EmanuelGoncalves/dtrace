# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import logging
from bsub import bsub
from dtrace import dpath, logger
from dtrace.Associations import Association
from dtrace.DTraceEnrichment import DTraceEnrichment


# Configurations

PYTHON = "python3.6.1"
SCRIPT = "dtrace/DTraceEnrichment.py"

CORES = 4
MEMORY = 8000
QUEUE = "normal"


# Import data-sets and associations

assoc = Association(dtype="ic50", load_associations=True)


# Gene-set files

gsea = DTraceEnrichment(
    gmts=["h.all.v6.2.symbols.gmt"],
    permutations=100,
    sig_min_len=5,
    padj_method="fdr_bh",
    verbose=1,
)


# Gene values to perform enrichment

gvalues = [
    ("GExp", assoc.gexp.T, f"{dpath}/gexp_gsea"),
    # ("CRISPR", self.crispr.T, f"{dpath}/crispr_gsea"),
    # ("Drug-CRISPR", betas, f"{dpath}/drug_lmm_regressions_{self.dtype}_crispr_gsea"),
]


#

for (dtype, df, efile) in gvalues:
    for gmt in gsea.gmts:
        logger.log(logging.INFO, f"{dtype}: GSEA pathway enrichment {gmt}")

        for dindex in df.iloc[:10].index:
            logger.log(logging.INFO, f"bsub ssGSEA {dindex} {gmt}")

            # Set job name
            jname = f"ssGSEA{dindex}{gmt}"

            # Define command
            j_cmd = (
                f"{PYTHON} {SCRIPT} -dtype '{dtype}' -dindex '{dindex}' -gmt '{gmt}'"
            )

            # Create bsub
            j = bsub(
                jname,
                R=f"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]",
                M=MEMORY,
                verbose=True,
                q=QUEUE,
                J=jname,
                o=jname,
                e=jname,
                n=CORES,
            )

            # Submit
            j(j_cmd)


# Copyright (C) 2019 Emanuel Goncalves
