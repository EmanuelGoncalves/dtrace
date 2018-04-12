#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from limix.plot import qqplot
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from cdrug.associations.ppi_enrichment import ppi_annotation
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef


LR_STATUS = {
    lr_files.LR_DRUG_CRISPR: dict(scale='Yes', growth='Yes'),
    lr_files.LR_DRUG_CRISPR_NOGROWTH: dict(scale='Yes', growth='No'),
    lr_files.LR_DRUG_CRISPR_NOSCALE: dict(scale='No', growth='Yes'),
    lr_files.LR_DRUG_CRISPR_NOSCALE_NOGROWTH: dict(scale='No', growth='No')
}


def get_signif_regressions(df, thres_fdr=0.1, thres_beta=0.25):
    return df[(df['lr_fdr'] < thres_fdr) & (df['beta'].abs() > thres_beta)]


def get_metric(df, func=precision_score, true_set=None, thres_fdr=0.1, thres_beta=0.25):
    if true_set is None:
        true_set = ['Target', '1']

    y_pred = ((df['beta'].abs() > thres_beta) & (df['lr_fdr'] < thres_fdr)).astype(int)

    y_true = df['target_thres'].isin(true_set).astype(int)

    return func(y_true, y_pred)


if __name__ == '__main__':
    # - Import regressions
    regression_files = [
        lr_files.LR_DRUG_CRISPR, lr_files.LR_DRUG_CRISPR_NOGROWTH, lr_files.LR_DRUG_CRISPR_NOSCALE, lr_files.LR_DRUG_CRISPR_NOSCALE_NOGROWTH
    ]

    regression_files = {f: pd.read_csv(f) for f in regression_files}

    # - FDR correction
    regression_files = {f: regression_files[f].assign(lr_fdr=multipletests(regression_files[f]['lr_pval'])[1]) for f in regression_files}

    # - Annotate PPI
    regression_files = {f: ppi_annotation(regression_files[f]) for f in regression_files}

    # - Benchmark best threshold
    trueset = ['Target', '1', '2']
    thres_fdrs = [1e-4, .5e-3, 1e-3, .5e-2, 1e-2, .5e-1, 1e-1, .15, .2, .25, .3]
    thres_betas = np.arange(0, 1.1, .1)

    thres_f1score = pd.DataFrame([{
        'fdr': tf, 'beta': tb, 'score': get_metric(regression_files[lr_files.LR_DRUG_CRISPR], precision_score, trueset, tf, tb)
    } for tf in thres_fdrs for tb in thres_betas])
    print(thres_f1score.sort_values('score', ascending=False).head(60))

    # -
    thres_fdr, thres_beta = 0.05, .5

    plot_df = pd.DataFrame([{
        'file': f, 'count': get_signif_regressions(regression_files[f], thres_fdr, thres_beta).shape[0]
    } for f in regression_files])
    plot_df = plot_df.assign(scale=[LR_STATUS[f]['scale'] for f in plot_df['file']])
    plot_df = plot_df.assign(growth=[LR_STATUS[f]['growth'] for f in plot_df['file']])
    plot_df = plot_df.assign(precision=[get_metric(regression_files[f], precision_score, trueset, thres_fdr, thres_beta) for f in plot_df['file']])
    plot_df = plot_df.assign(recall=[get_metric(regression_files[f], recall_score, trueset, thres_fdr, thres_beta) for f in plot_df['file']])
    plot_df = plot_df.assign(f1score=[get_metric(regression_files[f], f1_score, trueset, thres_fdr, thres_beta) for f in plot_df['file']])

    #
    pal = sns.light_palette(cdrug.PAL_SET2[8], 3, reverse=True).as_hex()[:-1]

    df = pd.melt(plot_df, value_vars=['precision', 'recall', 'f1score'], id_vars=['file', 'scale', 'growth'])

    g = sns.FacetGrid(df, col='variable', aspect=.5, legend_out=True, despine=False, sharey=True)

    g = g.map(sns.barplot, 'scale', 'value', 'growth', palette=pal, ci=None)

    for ax in np.ravel(g.axes):
        ax.yaxis.grid(True, color=cdrug.PAL_SET2[7], linestyle='-', linewidth=.1, alpha=.5, zorder=0)

    g.add_legend(title='Growth\nCovariate')
    g.set_axis_labels('Is scaled?', 'Score')
    g.set_titles('{col_name}')

    plt.savefig('reports/benchmark_barplot.pdf', bbox_inches='tight')
    plt.close('all')
