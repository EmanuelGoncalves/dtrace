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

import pandas as pd
from dtrace import dpath
from dtrace.Associations import Association
from dtrace.DTraceEnrichment import DTraceEnrichment


# Import data-sets

assoc = Association(dtype="ic50", pval_method="fdr_bh", load_associations=True)


# Build betas matrix

betas = pd.pivot_table(
    assoc.lmm_drug_crispr,
    index=["DRUG_ID", "DRUG_NAME", "VERSION"],
    columns="GeneSymbol",
    values="beta",
)


# ## GSEA enrichment

min_len = 5
gsea = DTraceEnrichment()


# Enrichment of cancer hallmarks

gsea_h = pd.DataFrame(
    {
        d: gsea.gsea_enrichments(
            betas.loc[d], "h.all.v6.2.symbols.gmt", min_len=min_len
        )["e_score"]
        for d in betas.index
    }
).T

gsea_h.round(5).to_csv(
    f"{dpath}/drug_lmm_regressions_{assoc.dtype}_crispr_gsea_h.all.v6.2.csv.gz",
    index=False,
    compression="gzip",
)

# Enrichment of GO Biological Processes

gsea_bp = pd.DataFrame(
    {
        d: gsea.gsea_enrichments(
            betas.loc[d], "c5.bp.v6.2.symbols.gmt", min_len=min_len
        )["e_score"]
        for d in betas.index
    }
).T

gsea_bp.round(5).to_csv(
    f"{dpath}/drug_lmm_regressions_{assoc.dtype}_crispr_gsea_c5.bp.v6.2.csv.gz",
    index=False,
    compression="gzip",
)

# Copyright (C) 2019 Emanuel Goncalves
