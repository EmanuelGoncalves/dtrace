#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from dtrace.analysis import PAL_DTRACE
from dtrace.associations import DRUG_INFO_COLUMNS


if __name__ == '__main__':
    # - Linear regressions
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS)

    # - Build betas matrix
    betas = pd.pivot_table(lmm_drug, index=DRUG_INFO_COLUMNS, columns='GeneSymbol', values='beta')

    #
    plot_df = betas.T.corr()

    hue = pd.Series({i: PAL_DTRACE[0] if i[2] == 'RS' else PAL_DTRACE[1] for i in plot_df.index})

    g = sns.clustermap(plot_df, cmap='RdGy_r', center=0, xticklabels=False, yticklabels=False, col_colors=hue, row_colors=hue)

    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')

    plt.gcf().set_size_inches(5, 5)
    plt.savefig(f'reports/betas_clustermap.png', bbox_inches='tight', transparent=False, dpi=300)
    plt.close('all')

    #
    for s in ['v17', 'RS']:
        plot_df = betas[[i[2] == s for i in betas.index]]
        plot_df.index = [f'{i[1]} ({i[0]})' for i in plot_df.index]
        plot_df = plot_df.T.corr()

        g = sns.clustermap(plot_df, cmap='RdGy_r', center=0, xticklabels=1, yticklabels=3)

        g.ax_heatmap.set_xlabel('')
        g.ax_heatmap.set_ylabel('')

        plt.gcf().set_size_inches(25, 15)
        plt.savefig(f'reports/betas_clustermap_{s}.png', bbox_inches='tight', transparent=False, dpi=300)
        plt.close('all')

    #
    pancore_ceres = set(pd.read_csv(dtrace.CERES_PANCANCER).iloc[:, 0].apply(lambda v: v.split(' ')[0]))

    plot_df = betas[[i[2] == 'RS' for i in betas.index]]
    plot_df = plot_df.loc[:, ~plot_df.columns.isin(pancore_ceres)]
    plot_df = plot_df.corr()

    g = sns.clustermap(plot_df, cmap='RdGy_r', center=0, xticklabels=False, yticklabels=False)

    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')

    plt.gcf().set_size_inches(5, 5)
    plt.savefig(f'reports/betas_genes_clustermap.png', bbox_inches='tight', transparent=False, dpi=300)
    plt.close('all')
