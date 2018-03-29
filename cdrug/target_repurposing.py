#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.regression.linear import lr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def lm_drug_crispr(xs, ys, ws):
    print('CRISPR genes: %d, Drug: %d' % (len(set(xs.columns)), len(set(ys.columns))))

    # # Standardize xs
    # xs = pd.DataFrame(StandardScaler().fit_transform(xs), index=xs.index, columns=xs.columns)

    # Regression
    res = lr(xs, ys, ws)

    # - Export results
    res_df = pd.concat([res[i].unstack().rename(i) for i in res], axis=1).reset_index()

    res_df = res_df.assign(f_fdr=multipletests(res_df['f_pval'], method='fdr_bh')[1])
    res_df = res_df.assign(lr_fdr=multipletests(res_df['lr_pval'], method='fdr_bh')[1])

    return res_df


if __name__ == '__main__':
    d_sheet = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0)

    d_repur = pd.read_csv('data/pre_clinical_targets.txt', sep='\t').dropna(subset=['drug_id'])

    dep_bin = pd.read_csv('data/binaryDepScores.tsv', sep='\t', index_col=0).dropna().astype(int)

    # Samplesheet
    ss = pd.read_csv(cdrug.SAMPLESHEET_FILE, index_col=0).dropna(subset=['Cancer Type'])

    # Growth rate
    growth = pd.read_csv(cdrug.GROWTHRATE_FILE, index_col=0)

    # # CRISPR gene-level corrected fold-changes
    # crispr = pd.read_csv(cdrug.CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    # crispr_scaled = cdrug.scale_crispr(crispr)

    # Drug response
    d_response = pd.read_csv(cdrug.DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # - Overlap
    samples = list(set(d_response).intersection(dep_bin).intersection(ss.index).intersection(growth.index))
    d_response, crispr = d_response[samples], dep_bin[samples]
    print('Samples: %d' % len(samples))

    # - Filter
    d_response = d_response[[i[0] in set(d_repur['drug_id'].astype(int)) for i in d_response.index]]
    crispr_scaled = dep_bin.loc[list(set(d_repur['Target']))].dropna()

    # - Covariates
    covariates = pd.concat([
        pd.get_dummies(ss[['Cancer Type']]),
        growth['growth_rate_median']
    ], axis=1).loc[samples]
    covariates = covariates.loc[:, covariates.sum() != 0]

    # - Linear regression: drug ~ crispr + tissue
    lm_res_df = lm_drug_crispr(crispr_scaled[samples].T, d_response[samples].T, covariates.loc[samples])

    repurposing_associations = {(t, int(d)) for t, d in d_repur[['Target', 'drug_id']].values}
    repurposing_associations = lm_res_df[[(t, d) in repurposing_associations for t, d in lm_res_df[['GeneSymbol', 'DRUG_ID']].values]]
    print(repurposing_associations.sort_values('lr_fdr'))
