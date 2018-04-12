#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from statsmodels.stats.multitest import multipletests
from cdrug.associations.ppi_enrichment import ppi_annotation
from sklearn.metrics import f1_score, precision_score, recall_score


def get_metric(df, func=precision_score, true_set=None, thres_fdr=0.1, thres_beta=0.25):
    if true_set is None:
        true_set = ['Target', '1']

    y_pred = is_signif(df, thres_fdr, thres_beta)

    y_true = df['target_thres'].isin(true_set).astype(int)

    return func(y_true, y_pred)


def is_signif(df, thres_fdr=0.1, thres_beta=0.25):
    signif_fdr = (df['lr_fdr_crispr'] < thres_fdr) | (df['lr_fdr_rnaseq'] < thres_fdr)
    signif_beta = (df['beta_crispr'].abs() > thres_beta) | (df['beta_rnaseq'].abs() > thres_beta)
    return signif_fdr & signif_beta


if __name__ == '__main__':
    # - Import Drug ~ RNA-Seq and Drug ~ CRISPR associations
    lr_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)
    lr_rnaseq = pd.read_csv(lr_files.LR_DRUG_RNASEQ)

    # - Merge associations
    lr_regressions = pd.concat([
        lr_crispr.set_index(['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']).add_suffix('_crispr'),
        lr_rnaseq.set_index(['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']).add_suffix('_rnaseq')
    ], axis=1).dropna().reset_index()

    # - FDR correction
    lr_regressions = lr_regressions.assign(lr_fdr_crispr=multipletests(lr_regressions['lr_pval_crispr'])[1])
    lr_regressions = lr_regressions.assign(lr_fdr_rnaseq=multipletests(lr_regressions['lr_pval_rnaseq'])[1])

    # - PPI annotation
    lr_regressions = ppi_annotation(lr_regressions)

    # -
    thres_fdr = 0.3

    plot_df = lr_regressions.query('lr_fdr_crispr < {} | lr_fdr_rnaseq < {}'.format(thres_fdr, thres_fdr))

    sns.kdeplot(
        np.log2(plot_df['f_stat_rnaseq']), np.log2(plot_df['f_stat_crispr']), cmap=sns.light_palette(cdrug.PAL_SET2[7], as_cmap=True), shade_lowest=False
    )
    sns.kdeplot(
        np.log2(plot_df.query('target <= 1')['f_stat_rnaseq']), np.log2(plot_df.query('target <= 1')['f_stat_crispr']),
        cmap=sns.light_palette(cdrug.PAL_SET2[1], as_cmap=True), shade_lowest=False
    )
    plt.savefig('reports/ppi_enrichment_hex.pdf', bbox_inches='tight')
    plt.close('all')

    order = ['Target', '1', '2', '3', '>=4']
    order_color = [cdrug.PAL_SET2[1]] + sns.light_palette(cdrug.PAL_SET2[8], len(order) - 1, reverse=True).as_hex()
    order_pal = dict(zip(*(order, order_color)))

    for t in order:
        sns.distplot(
            np.log2(lr_regressions[lr_regressions['target_thres'] == t]['lr_crispr']), color=order_pal[t]
        )

    plt.savefig('reports/ppi_enrichment_hex.pdf', bbox_inches='tight')
    plt.close('all')

    # -
    trueset = ['Target', '1']
    thres_fdr, thres_beta = 0.1, .5

    count = is_signif(lr_regressions, thres_fdr, thres_beta).sum()
    precision = get_metric(lr_regressions, precision_score, trueset, thres_fdr, thres_beta)
    recall = get_metric(lr_regressions, recall_score, trueset, thres_fdr, thres_beta)
    f1score = get_metric(lr_regressions, f1_score, trueset, thres_fdr, thres_beta)

    print('Count: {}; Precision: {:.2f}; Recall: {:.2f}; F1-score: {:.2f}'.format(count, precision, recall, f1score))
