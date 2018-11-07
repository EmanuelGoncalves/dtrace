#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from plot import Plot
from crispy import QCplot
from importer import CRISPR, Sample
from crispy.utils import Utils
from associations import Association


def combination_tostring(df, n_features=2):
    df_names = pd.concat([
        df.iloc[:, i].replace(1, df.columns[i]) for i in range(1, n_features + 1)
    ], axis=1)

    df_names = df_names.apply(lambda v: ' + '.join([i for i in v if i != 0]), axis=1)

    df_names = df_names.replace('', 'None')

    return df_names.rename('name')


if __name__ == '__main__':
    # - Imports
    wes = pd.read_csv('/Users/eg14/Projects/prioritize_crispr/data/WES_variants.csv')

    samples = Sample()
    crispr = CRISPR()
    assoc = Association(dtype_drug='ic50', dtype_crispr='binary_dep')

    core_essential = set.union(Utils.get_adam_core_essential(), Utils.get_broad_core_essential())

    #
    # d_id, d_name, d_screen = (1411, 'SN1041137233', 'v17')
    # genes = ['ERBB2', 'EGFR']
    d_id, d_name, d_screen = (1982, 'PLK4_0066', 'RS')
    genes = ['PLK1', 'PCM1']

    dmax = np.log(assoc.drespo_obj.maxconcentration.loc[(d_id, d_name, d_screen)])

    plot_df = pd.concat([
        assoc.drespo.loc[(d_id, d_name, d_screen)].rename('drug'),
        crispr.get_data(dtype='binary_dep').loc[genes].T,
        samples.samplesheet['Cancer Type']
    ], axis=1, sort=False).dropna().sort_values('drug')
    plot_df = pd.concat([plot_df, combination_tostring(plot_df)], axis=1)

    # Order
    order = list(plot_df.groupby('name')['drug'].mean().sort_values(ascending=False).index)
    pal = pd.Series(QCplot.get_palette_continuous(len(order), Plot.PAL_DTRACE[2]), index=order)

    sns.boxplot(
        y='name', x='drug', data=plot_df, orient='h', palette=pal.to_dict(), flierprops=QCplot.FLIERPROPS, saturation=1., showcaps=False, order=order
    )
    sns.swarmplot(y='name', x='drug', data=plot_df, orient='h', palette=pal.to_dict(), size=2, edgecolor='white', linewidth=.5, order=order)

    plt.axvline(dmax, ls='-', lw=.5, zorder=3, color=Plot.PAL_DTRACE[0])

    plt.xlabel(f'{d_name} ({d_id}, {d_screen})')
    plt.ylabel('')

    plt.gcf().set_size_inches(3., 1.5)
    plt.savefig(f"reports/drug_lmm_regressions_mutiple_{';'.join(genes)}.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')
