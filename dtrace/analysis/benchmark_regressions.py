#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dtrace.associations as lr_files
from dtrace.associations import multipletests_per_drug, ppi_annotation
from sklearn.metrics import roc_auc_score, average_precision_score


LR_STATUS = {
    lr_files.LR_DRUG_CRISPR: dict(scale='Yes', growth='Yes'),
    lr_files.LR_DRUG_CRISPR_NOGROWTH: dict(scale='Yes', growth='No'),
    lr_files.LR_DRUG_CRISPR_NOSCALE: dict(scale='No', growth='Yes'),
    lr_files.LR_DRUG_CRISPR_NOSCALE_NOGROWTH: dict(scale='No', growth='No')
}


THRES_FDR, THRES_BETA, TRUESET = .1, 0.25, {'Target'}


def evaluate_lr(lr_df, true_set, thres_fdr, thres_beta):
    # Drop drug targets not annotated
    df = lr_df.dropna(subset=['target'])

    # Significant associations
    df = df.assign(signif=((df['beta'].abs() > thres_beta) & (df['lr_fdr'].abs() < thres_fdr)).astype(int))

    # Annotate if association is in Trueset
    df = df.assign(trueset=df['target_thres'].isin(true_set).astype(int))

    res = {
        'count': int(df['signif'].sum()),
        'avg_precision': average_precision_score(df['trueset'], 1 - df['lr_fdr']),
        'aroc': roc_auc_score(df['trueset'], df['lr_fdr'])
    }

    return res


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
    # lr_df, true_set, thres_fdr, thres_beta = regression_files[f], TRUESET, THRES_FDR, THRES_BETA
    plot_df = pd.DataFrame({f: evaluate_lr(regression_files[f], TRUESET, THRES_FDR, THRES_BETA) for f in regression_files}).T
    plot_df = plot_df.assign(scale=[LR_STATUS[f]['scale'] for f in plot_df.index])
    plot_df = plot_df.assign(growth=[LR_STATUS[f]['growth'] for f in plot_df.index])

    #
    value_vars, id_vars = ['precision', 'recall', 'f1score', 'count'], ['file', 'scale', 'growth']
    titles = dict(zip(*(value_vars, ['Precision', 'Recall', 'F1 score', 'Significant associations'])))

    df = pd.melt(plot_df, value_vars=value_vars, id_vars=id_vars)

    g = sns.FacetGrid(df, col='variable', aspect=.5, legend_out=True, despine=False, sharey=False)
    g = g.map(sns.barplot, 'scale', 'value', 'growth', palette=sns.light_palette(dtrace.PAL_SET2[8], 3, reverse=True).as_hex()[:-1], ci=None)

    for label, ax in zip(value_vars, np.ravel(g.axes)):
        ax.yaxis.grid(True, color=dtrace.PAL_SET2[7], linestyle='-', linewidth=.1, alpha=.5, zorder=0)

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
