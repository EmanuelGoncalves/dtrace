#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from sklearn.preprocessing import StandardScaler
from cdrug.associations import multipletests_per_drug
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.model_selection import ShuffleSplit, cross_val_score


THRES_FDR, THRES_BETA = .2, 0.5


def get_drug_signif(lr, drug, fdr_thres, beta_thres):
    return list(lr[
        (lr['lr_fdr'] < fdr_thres) & (lr['beta'].abs() > beta_thres) & (lr['DRUG_NAME'] == drug)
    ]['GeneSymbol'])


if __name__ == '__main__':
    # - Import linear regressions
    lr_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)
    lr_rnaseq = pd.read_csv(lr_files.LR_DRUG_RNASEQ)

    # - FDR correction
    lr_crispr = multipletests_per_drug(lr_crispr)
    lr_rnaseq = multipletests_per_drug(lr_rnaseq)

    # - Import data-sets
    drespo = cdrug.get_drugresponse()

    crispr = cdrug.get_crispr(dtype='logFC')
    crispr_scaled = cdrug.scale_crispr(crispr)

    rnaseq = pd.read_csv(cdrug.RNASEQ_VOOM, index_col=0)

    samples = list(set(rnaseq).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # -
    crispr_drug = {
        (i[0], i[1], i[2])
        for i in lr_crispr[(lr_crispr['lr_fdr'] < THRES_FDR) & (lr_crispr['beta'].abs() > THRES_BETA)][['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']].values
    }
    rnaseq_drug = {
        (i[0], i[1], i[2])
        for i in lr_rnaseq[(lr_rnaseq['lr_fdr'] < THRES_FDR) & (lr_rnaseq['beta'].abs() > THRES_BETA)][['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']].values
    }
    drug_overlap = crispr_drug.intersection(rnaseq_drug)
    print('CRISPR drugs: {}; RNA-Seq: {}; Overlap: {}'.format(len(crispr_drug), len(rnaseq_drug), len(drug_overlap)))

    # -
    res_df = []
    # d_id, d_name, d_screen = 1032, 'Afatinib', 'v17'
    for d_id, d_name, d_screen in drug_overlap:
        d_crispr_feat = get_drug_signif(lr_crispr, d_name, THRES_FDR, THRES_BETA)
        d_rnaseq_feat = get_drug_signif(lr_rnaseq, d_name, THRES_FDR, THRES_BETA)

        #
        y = drespo.loc[(d_id, d_name, d_screen), samples].dropna()

        cv = ShuffleSplit(n_splits=10, test_size=.3)
        lm = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1])

        cv_crispr = cross_val_score(lm, crispr_scaled.loc[d_crispr_feat, y.index].T, y, scoring='r2', cv=cv)
        cv_rnaseq = cross_val_score(lm, rnaseq.loc[d_rnaseq_feat, y.index].T, y, scoring='r2', cv=cv)

        res = pd.DataFrame({'crispr': cv_crispr, 'rnaseq': cv_rnaseq, 'DRUG_ID_lib': d_id, 'DRUG_NAME': d_name, 'VERSION': d_screen})
        # res = pd.melt(res, id_vars=['DRUG_ID_lib', 'DRUG_NAME', 'VERSION'], value_vars=['crispr', 'rnaseq'], value_name='r2')

        res_df.append(res)

    res_df = pd.concat(res_df)

    # -
    plot_df = res_df.groupby(['DRUG_ID_lib', 'DRUG_NAME', 'VERSION'])['crispr', 'rnaseq'].median().reset_index()
    plot_df = plot_df[(plot_df['crispr'] > 0) & (plot_df['rnaseq'] > 0)]

    plt.scatter(plot_df['crispr'], plot_df['rnaseq'], marker='o', color=cdrug.PAL_BIN[0], s=5)
    sns.kdeplot(plot_df['crispr'], plot_df['rnaseq'], zorder=0, linewidths=.5, cmap=sns.light_palette(cdrug.PAL_BIN[0], as_cmap=True))

    xy_min, xy_max = 0, 0.5
    plt.plot((xy_min, xy_max), (xy_min, xy_max), 'k--', lw=.3, alpha=.5)
    plt.xlim(xy_min, xy_max)
    plt.ylim(xy_min, xy_max)
    plt.xlabel('CRISPR-Cas9 (R2)')
    plt.ylabel('RNA-Seq (R2)')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/predicting_drug_response.pdf', bbox_inches='tight')
    plt.close('all')

