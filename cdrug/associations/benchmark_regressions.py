#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from cdrug.associations import multipletests_per_drug, ppi_annotation
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


LR_STATUS = {
    lr_files.LR_DRUG_CRISPR: dict(scale='Yes', growth='Yes'),
    lr_files.LR_DRUG_CRISPR_NOGROWTH: dict(scale='Yes', growth='No'),
    lr_files.LR_DRUG_CRISPR_NOSCALE: dict(scale='No', growth='Yes'),
    lr_files.LR_DRUG_CRISPR_NOSCALE_NOGROWTH: dict(scale='No', growth='No')
}


def drugs_to_consider(df, thres_beta=.5):
    return set(df[df['beta'].abs() > thres_beta]['DRUG_ID_lib'])


def get_signif_regressions(df, thres_fdr=0.1, thres_beta=0.25):
    df = df[df['DRUG_ID_lib'].isin(drugs_to_consider(df))]
    return df[(df['lr_fdr'] < thres_fdr) & (df['beta'].abs() > thres_beta)]


def get_metric(df, func, true_set, thres_fdr=0.1, thres_beta=0.25):
    y_pred = ((df['beta'].abs() > thres_beta) & (df['lr_fdr'].abs() < thres_fdr)).astype(int)

    y_true = df['target_thres'].isin(true_set).astype(int)

    return func(y_true, y_pred)


if __name__ == '__main__':
    # - Import regressions
    regression_files = [
        lr_files.LR_DRUG_CRISPR, lr_files.LR_DRUG_CRISPR_NOGROWTH, lr_files.LR_DRUG_CRISPR_NOSCALE, lr_files.LR_DRUG_CRISPR_NOSCALE_NOGROWTH
    ]

    regression_files = {f: pd.read_csv(f) for f in regression_files}

    # - FDR correction
    regression_files = {f: multipletests_per_drug(regression_files[f]) for f in regression_files}

    # - Annotate PPI
    regression_files = {f: ppi_annotation(regression_files[f], exp_type=None, int_type={'physical'}) for f in regression_files}

    # - Benchmark best threshold
    trueset = ['Target', '1']
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
    value_vars, id_vars = ['precision', 'recall', 'f1score', 'count'], ['file', 'scale', 'growth']
    titles = dict(zip(*(value_vars, ['Precision', 'Recall', 'F1 score', 'Significant associations'])))

    df = pd.melt(plot_df, value_vars=value_vars, id_vars=id_vars)

    g = sns.FacetGrid(df, col='variable', aspect=.5, legend_out=True, despine=False, sharey=False)
    g = g.map(sns.barplot, 'scale', 'value', 'growth', palette=sns.light_palette(cdrug.PAL_SET2[8], 3, reverse=True).as_hex()[:-1], ci=None)

    for label, ax in zip(value_vars, np.ravel(g.axes)):
        ax.yaxis.grid(True, color=cdrug.PAL_SET2[7], linestyle='-', linewidth=.1, alpha=.5, zorder=0)

        ax.set_title(titles[label])

        if label != 'count':
            ax.set_ylim(0, 1)

    g.add_legend(title='Growth\nCovariate')
    g.set_axis_labels('Is scaled?', 'Score')
    # g.set_titles('{col_name}')

    plt.suptitle(
        'Enrichment for [{}] in significant associations (FDR < {:.0f}% and abs(beta) > {})'.format('; '.join(trueset), thres_fdr * 100, thres_beta), y=1.05
    )

    plt.savefig('reports/benchmark_barplot.pdf', bbox_inches='tight')
    plt.close('all')
