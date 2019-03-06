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
import seaborn as sns
import matplotlib.pyplot as plt
from DTraceUtils import dpath, rpath
from dtrace.Associations import Association
from dtrace.TargetBenchmark import TargetBenchmark


# ### Import data-sets and associations

assoc = Association(dtype="ic50", load_associations=True)


# Import pathway enrichment

kegg_enr = pd.read_csv(f"{dpath}/drug_lmm_regressions_ic50_crispr_gsea_c2.cp.kegg.v6.2.symbols.gmt.csv.gz", index_col=0)




# Copyright (C) 2019 Emanuel Goncalves
