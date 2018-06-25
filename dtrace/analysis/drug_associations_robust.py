#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.analysis import PAL_DTRACE
from dtrace.associations import DRUG_INFO_COLUMNS
from dtrace.analysis.plot.corrplot import plot_corrplot_discrete


def count_signif_associations(lmm_drug_robust, fdr=0.05):
    columns = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol', 'Genetic']

    d_signif = {tuple(i) for i in lmm_drug_robust.query('fdr_drug < {}'.format(fdr))[columns].values}
    c_signif = {tuple(i) for i in lmm_drug_robust.query('fdr_crispr < {}'.format(fdr))[columns].values}
    dc_signif = d_signif.intersection(c_signif)

    # Build dataframe
    plot_df = pd.DataFrame(dict(
        names=['Drug', 'CRISPR', 'Intersection'],
        count=list(map(len, [d_signif, c_signif, dc_signif]))
    )).sort_values('count', ascending=True)
    plot_df = plot_df.assign(y=range(plot_df.shape[0]))

    # Plot
    plt.barh(plot_df['y'], plot_df['count'], color=PAL_DTRACE[2])

    sns.despine(right=True, top=True)

    for c, y in plot_df[['count', 'y']].values:
        plt.text(c + 3, y, str(c), va='center', fontsize=5, zorder=10, color=PAL_DTRACE[2])

    plt.yticks(plot_df['y'], plot_df['names'])
    plt.xlabel('Number of signifcant associations')
    plt.ylabel('')


if __name__ == '__main__':
    # - Import
    # Data-sets
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()
    crispr = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    d_maxc = pd.read_csv(dtrace.DRUG_RESPONSE_MAXC, index_col=[0, 1, 2])

    # Robust associations
    lmm_drug_robust = pd.read_csv(dtrace.LMM_ASSOCIATIONS_ROBUST)

    # - Count number of significant associations overall
    count_signif_associations(lmm_drug_robust)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/robust_count_signif.pdf', bbox_inches='tight')
    plt.close('all')

    # - Strongest significant association per drug
    associations = lmm_drug_robust.query('fdr_crispr < 0.05 & fdr_drug < 0.05')
    associations = associations.sort_values('fdr_crispr').groupby(DRUG_INFO_COLUMNS).head(1)

    # - Plot robust association
    indices = [75293, 29689, 11943, 70709, 22504, 481, 68343, 69599]

    for idx in indices:
        columns = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol', 'Genetic']
        association = lmm_drug_robust.loc[idx, columns]

        name = 'Drug={}, Gene={}, Genetic={} [{}, {}]'.format(association[1], association[3], association[4], association[0], association[2])

        plot_df = pd.concat([
            crispr.loc[association[3]],
            drespo.loc[tuple(association[:3])].rename('{} [{}]'.format(association[1], association[2])),
            mobems.loc[association[4]]
        ], axis=1).dropna()

        g = plot_corrplot_discrete(association[3], '{} [{}]'.format(association[1], association[2]), association[4], plot_df)

        g.ax_joint.axhline(np.log(d_maxc.loc[tuple(association[:3]), 'max_conc_micromolar']), lw=.1, color=PAL_DTRACE[2], ls='--')

        plt.gcf().set_size_inches(2, 2)
        plt.savefig('reports/lmm_robust_{}.pdf'.format(name), bbox_inches='tight')
        plt.close('all')
