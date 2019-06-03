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
from dtrace.DTraceUtils import dpath
from dtrace.Associations import Association
from dtrace.DTraceEnrichment import DTraceEnrichment, DTraceEnrichmentBSUB


# Configurations

PYTHON = "python"
SCRIPT = "dtrace/DTraceEnrichment.py"

CORES = 4
MEMORY = 8000
QUEUE = "long"

PERMUTATIONS = 10000
SIG_MIN_LEN = 5
PADJ_METHOD = "fdr_bh"

GMTS = ["h.all.v6.2.symbols.gmt", "c2.cp.kegg.v6.2.symbols.gmt"]


# Import data-sets and associations

assoc = Association(dtype="ic50")


# Gene-set files

gsea = DTraceEnrichment(
    gmts=GMTS, sig_min_len=5, padj_method="fdr_bh", verbose=1
)


# Gene values to perform enrichment

gvalues = [
    ("GExp", assoc.gexp.T),
]


# Submit ssGSEA jobs

for dtype, df in gvalues:

    for gmt in gsea.gmts:
        logging.getLogger("DTrace").info(f"{dtype}: GSEA pathway enrichment {gmt}")

        for dindex in df.index:
            logging.getLogger("DTrace").info(f"bsub ssGSEA {dindex} {gmt}")

            # Set job name
            jname = f"ssGSEA{dtype}{gmt}{dindex}"

            # Define command
            j_cmd = f"{PYTHON} {SCRIPT} -dtype '{dtype}' -dindex '{dindex}' -gmt '{gmt}' -permutations {PERMUTATIONS} -len {SIG_MIN_LEN} -padj {PADJ_METHOD}"

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


# Assemble unique data-frame containing all enrichments and adj. p-value across all tests

dtype, gmt = "GExp", "h.all.v6.2.symbols.gmt"
gsea_df = DTraceEnrichmentBSUB(dtype, gmt, load_datasets=False).assemble_single_file(verbose=1)
gsea_df.to_csv(f"{dpath}/ssgsea/{dtype}_{gmt}.csv.gz", compression="gzip", index=False)


# Copyright (C) 2019 Emanuel Goncalves
