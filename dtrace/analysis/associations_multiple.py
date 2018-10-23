#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
from importer import CRISPR
from crispy.utils import Utils
from associations import Association

if __name__ == '__main__':
    # - Imports
    crispr = CRISPR().get_data(scale=False)

    assoc = Association()

    core_essential = set.union(Utils.get_adam_core_essential(), Utils.get_broad_core_essential())

    mlmm = pd.read_csv('data/drug_lmm_regressions_multiple_ic50.csv')
    mlmm = mlmm[~mlmm['GeneSymbol'].isin(core_essential)]
    mlmm = mlmm[~mlmm['covar'].isin(core_essential)]


    #
    d_id, d_name, d_screen = (1411, 'SN1041137233', 'v17')
    gene_covar, gene_feat = 'EGFR', 'ERBB2'

    dmax = np.log(assoc.drespo_obj.maxconcentration.loc[(d_id, d_name, d_screen)])

    plot_df = pd.concat([
        assoc.drespo.loc[(d_id, d_name, d_screen)].rename('drug'),
        crispr.loc[[gene_covar, gene_feat]].T
    ], axis=1, sort=False).dropna()
    plot_df['Response'] = (plot_df['drug'] < (0.5 * dmax)).astype(int).replace({0: 'No', 1: 'Yes'})

    # Order
    sns.scatterplot(
        gene_feat, gene_covar, style='Response', data=plot_df, color=Plot.PAL_DTRACE[1], legend='full',
        markers=['.', 'X'], linewidth=.1, style_order=['No', 'Yes']
    )

    plt.axhline(0, ls=':', lw=0.1, zorder=0)
    plt.axvline(0, ls=':', lw=0.1, zorder=0)

    plt.xlabel(f'{gene_feat} CRISPR-Cas9 logFC')
    plt.ylabel(f'{gene_covar} CRISPR-Cas9 logFC')

    plt.title(f'{d_name} ({d_id} {d_screen})\nModel: {gene_feat} + {gene_covar}')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig(f'reports/drug_lmm_regressions_mutiple.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
