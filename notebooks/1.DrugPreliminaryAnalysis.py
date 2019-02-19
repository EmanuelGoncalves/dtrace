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
import numpy as np
import pandas as pd
from Associations import Association
from Preliminary import DrugPreliminary

# ## Import data-sets

datasets = Association()

# # Drug-response analysis notebook

# ### Drug-response measurements across cell lines

DrugPreliminary.histogram_drug(datasets.drespo.count(1))

# ### Drug-responses below 50% of the maximum screened concentration

num_resp = pd.Series(
    {
        drug: (
            datasets.drespo.loc[drug].dropna()
            < np.log(datasets.drespo_obj.maxconcentration[drug] * 0.5)
        ).sum()
        for drug in datasets.drespo.index
    }
).reset_index()
num_resp.columns = ["DRUG_ID", "DRUG_NAME", "VERSION", "n_resp"]


DrugPreliminary.histogram_drug_response(num_resp)


