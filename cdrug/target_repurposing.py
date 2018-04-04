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
    # - Import sample data
    # Samplesheet
    ss = pd.read_csv(cdrug.SAMPLESHEET_FILE, index_col=0).dropna(subset=['Cancer Type'])

    # Growth rate
    growth = pd.read_csv(cdrug.GROWTHRATE_FILE, index_col=0)

    # - Import data
    d_sheet = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0)

    # - Import drug repurposing
    d_repur_map = pd.read_csv('data/repo_drug_map.txt', sep='\t', index_col=0).dropna()['Drug ID']

    d_repur = pd.read_csv('data/repo_list.txt', sep='\t')
    d_repur = pd.DataFrame([{
        'tissue': t, 'genomic': f, 'gene': g, 'drug': d, 'drug_id': d_repur_map[d]
    } for t, f, g, ds, _ in d_repur.values for d in ds.split('|') if d in d_repur_map.index])

    # CRISPR gene-level corrected fold-changes
    # crispr = pd.read_csv(cdrug.CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    # crispr_scaled = cdrug.scale_crispr(crispr)
    crispr_scaled = pd.read_csv(cdrug.CRISPR_GENE_BINARY, sep='\t', index_col=0).dropna().astype(int)

    # Drug response
    d_response = pd.read_csv(cdrug.DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # - Overlap
    samples = list(set(d_response).intersection(crispr_scaled).intersection(ss.index).intersection(growth.index))
    d_response, crispr_scaled = d_response[samples], crispr_scaled[samples]
    print('Samples: %d' % len(samples))

    # - Linear regression: drug ~ crispr + tissue
    lr_df = []
    for d, d_id, g, f, t in d_repur.values:
        if d_id in d_response.index and g in crispr_scaled.index:
            print(d, d_id, g, f, t)

            y = d_response.loc[d_id, samples].T.dropna()
            x = ((crispr_scaled.loc[g, y.index] == 1) & (ss.loc[y.index, 'Cancer Type'] == t)).astype(int).rename(g).to_frame()

            if x[g].sum() >= 3:
                lm_res = lr(x, y)

                for d_name, s_version in lm_res['beta'].columns:
                    res = {'tissue': t, 'drug_id': d_id, 'drug_name': d_name, 'gene': g, 'genomic': f, 'version': s_version}

                    for s in ['beta', 'f_pval', 'r2']:
                        res[s] = lm_res[s].loc[g, (d_name, s_version)]

                    lr_df.append(res)

    lr_df = pd.DataFrame(lr_df)
    lr_df = lr_df.assign(f_fdr=multipletests(lr_df['f_pval'], method='fdr_bh')[1])
    print(lr_df.sort_values('f_fdr'))
