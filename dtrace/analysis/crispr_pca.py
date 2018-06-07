#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.analysis import PAL_DTRACE
import dtrace.analysis.drug_pca as f_pca


def growth_correlation_histogram(c_corr):
    hist_kws = dict(alpha=.4)
    kde_kws = dict(cut=0, lw=1, alpha=.8)

    sns.distplot(c_corr['corr'], color=PAL_DTRACE[2], kde_kws=kde_kws, hist_kws=hist_kws, bins=15)

    sns.despine(right=True, top=True)

    plt.axvline(0, c=PAL_DTRACE[1], lw=.1, ls='-', zorder=0)

    plt.xlabel('Gene correlation with growth rate\n(Pearson\'s R)')
    plt.ylabel('Density')


def corrplot_pcs_essentiality(pca, crispr, pc):
    df = pd.concat([pca['row']['pcs'], crispr.sum(1).rename('count')], axis=1)

    marginal_kws, annot_kws = dict(kde=False), dict(stat='R', loc=2)

    scatter_kws, line_kws = dict(edgecolor='w', lw=.3, s=10, alpha=.6), dict(lw=1., color=PAL_DTRACE[0], alpha=1.)
    joint_kws = dict(lowess=True, scatter_kws=scatter_kws, line_kws=line_kws)

    g = sns.jointplot(
        'count', pc, data=df, kind='reg', space=0, color=PAL_DTRACE[2],
        marginal_kws=marginal_kws, annot_kws=annot_kws, joint_kws=joint_kws
    )

    g.ax_joint.axhline(0, ls='-', lw=0.1, c=PAL_DTRACE[1], zorder=0)
    g.ax_joint.axvline(0, ls='-', lw=0.1, c=PAL_DTRACE[1], zorder=0)

    vexp = pca['row']['vex'][pc]
    g.set_axis_labels('Gene significantly essential count', '{} ({:.1f}%)'.format(pc, vexp * 100))


if __name__ == '__main__':
    # - Import
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='both')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # Growth rate
    growth = pd.read_csv(dtrace.GROWTHRATE_FILE, index_col=0)
    growth = growth.reindex(samples)['growth_rate_median'].dropna()

    # - Filter
    crispr = dtrace.filter_crispr(crispr[samples])
    crispr_logfc = crispr_logfc.loc[crispr.index, samples]

    # - Perform PCA analysis on CRISPR-Cas9 fold-changes (across Gene and Cell lines)
    pca = f_pca.perform_pca(crispr_logfc)

    # - Growth ~ CRISPR-Cas9 fold-changes correlation
    c_corr = crispr_logfc[growth.index].T.corrwith(growth).sort_values().rename('corr').reset_index()

    # - PCA pairplot per gene
    f_pca.pairplot_pca_drug(pca, hue=None)
    plt.suptitle('PCA CRISPR-Cas9 (Genes)', y=1.05, fontsize=9)
    plt.savefig('reports/pca_pairplot_crispr.pdf', bbox_inches='tight')
    plt.close('all')

    # - PCA pairplot per cell line
    f_pca.pairplot_pca_samples(pca, growth)
    plt.suptitle('PCA CRISPR-Cas9 (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/pca_pairplot_cell_lines_crispr.pdf', bbox_inches='tight')
    plt.close('all')

    # - PCA pairplot cell lines - hue by cancer type
    f_pca.pairplot_pca_samples_cancertype(pca)
    plt.suptitle('PCA CRISPR-Cas9 (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/pca_pairplot_cell_lines_cancertype_crispr.pdf', bbox_inches='tight')
    plt.close('all')

    # - Growth ~ PC1 corrplot
    f_pca.corrplot_pcs_growth(pca, growth, 'PC1')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/pca_growth_corrplot_crispr.pdf', bbox_inches='tight')
    plt.close('all')

    # - Correlations with growth histogram
    growth_correlation_histogram(c_corr)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/pca_growth_corr_histogram_crispr.pdf', bbox_inches='tight')
    plt.close('all')

    # - PC correlation with number of significantly essential events
    corrplot_pcs_essentiality(pca, crispr, 'PC1')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/pca_essentiality_corrplot_crispr.pdf', bbox_inches='tight')
    plt.close('all')
