#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dtrace.analysis import PAL_DTRACE


def histogram_drug(drespo):
    df = drespo.count(1).rename('count').reset_index()

    pal = dict(v17=PAL_DTRACE[2], RS=PAL_DTRACE[0])

    for s in pal:
        sns.distplot(df[df['VERSION'] == s]['count'], color=pal[s], kde=False, bins=15, label=s, hist_kws={'alpha': 1.})
        sns.despine(top=True, right=True)

    plt.xlabel('Number of IC50s measurements')
    plt.ylabel('Number of drugs')
    plt.legend()


def histogram_sample(drespo):
    df = drespo.count(0).rename('count').reset_index()

    sns.distplot(df['count'], color=PAL_DTRACE[2], kde=False, bins=20, hist_kws={'alpha': 1.})
    sns.despine(top=True, right=True)

    plt.xlabel('Number of IC50s')
    plt.ylabel('Number of cell lines')


def perform_pca(drespo):
    df = drespo.fillna(drespo.mean())

    pca = dict()
    for by in ['drug', 'sample']:
        pca[by] = dict()

        if by == 'sample':
            df = df.T

        pca[by]['pca'] = PCA(n_components=10).fit(df)
        pca[by]['vex'] = pd.Series(pca[by]['pca'].explained_variance_ratio_, index=map(lambda v: 'PC{}'.format(v + 1), range(10)))
        pca[by]['pcs'] = pd.DataFrame(pca[by]['pca'].transform(df), index=df.index, columns=map(lambda v: 'PC{}'.format(v + 1), range(10)))

    return pca


def _pairplot_fix_labels(g, pca, by):
    for i, ax in enumerate(g.axes):
        vexp = pca[by]['vex']['PC{}'.format(i + 1)]
        ax[0].set_ylabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))

    for i, ax in enumerate(g.axes[2]):
        vexp = pca[by]['vex']['PC{}'.format(i + 1)]
        ax.set_xlabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))


def pairplot_pca_drug(pca):
    df = pca['drug']['pcs'].reset_index()

    pal = dict(v17=PAL_DTRACE[2], RS=PAL_DTRACE[0])

    g = sns.PairGrid(df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1, hue='VERSION', palette=pal)
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plt.scatter, s=3, edgecolor='white', lw=.1)
    g = g.add_legend()

    _pairplot_fix_labels(g, pca, by='drug')


def pairplot_pca_samples(pca, growth):
    df = pd.concat([pca['sample']['pcs'], growth], axis=1).dropna().sort_values('growth_rate_median')

    cmap = sns.light_palette(PAL_DTRACE[2], as_cmap=True)

    g = sns.PairGrid(df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1)
    g = g.map_diag(plt.hist, color=PAL_DTRACE[2])

    g = g.map_offdiag(plt.scatter, s=3, edgecolor='white', lw=.1, color=df['growth_rate_median'], cmap=cmap, alpha=.5)
    cax = g.fig.add_axes([.98, .4, .01, .2])
    plt.colorbar(cax=cax)

    _pairplot_fix_labels(g, pca, by='sample')


def corrplot_pcs_growth(pca, growth, pc):
    df = pd.concat([pca['sample']['pcs'], growth], axis=1).dropna().sort_values('growth_rate_median')

    marginal_kws, annot_kws = dict(kde=False), dict(stat='R')

    scatter_kws, line_kws = dict(edgecolor='w', lw=.3, s=10, alpha=.6), dict(lw=1., color=PAL_DTRACE[0], alpha=1.)
    joint_kws = dict(lowess=True, scatter_kws=scatter_kws, line_kws=line_kws)

    g = sns.jointplot(
        pc, 'growth_rate_median', data=df, kind='reg', space=0, color=PAL_DTRACE[2],
        marginal_kws=marginal_kws, annot_kws=annot_kws, joint_kws=joint_kws
    )

    g.ax_joint.axvline(0, ls='-', lw=0.1, c=PAL_DTRACE[1], zorder=0)

    vexp = pca['sample']['vex'][pc]
    g.set_axis_labels('{} ({:.1f}%)'.format(pc, vexp * 100), 'Growth rate\n(median day 1 / day 4)')


def growth_correlation_histogram(g_corr):
    pal = dict(v17=PAL_DTRACE[2], RS=PAL_DTRACE[0])

    for i, s in enumerate(pal):
        hist_kws = dict(alpha=.4, zorder=i+1)
        kde_kws = dict(cut=0, lw=1, zorder=i+1, alpha=.8)

        sns.distplot(g_corr[g_corr['VERSION'] == s]['corr'], color=pal[s], kde_kws=kde_kws, hist_kws=hist_kws, bins=15, label=s)

    sns.despine(right=True, top=True)

    plt.axvline(0, c=PAL_DTRACE[1], lw=.1, ls='-', zorder=0)

    plt.xlabel('Drug correlation with growth rate\n(Pearson\'s R)')
    plt.ylabel('Density')

    plt.legend(prop={'size': 6})


def growth_correlation_top_drugs(g_corr):
    sns.barplot('corr', 'DRUG_NAME', data=g_corr.head(20), color=PAL_DTRACE[2])
    sns.despine(top=True, right=True)

    plt.axvline(0, c=PAL_DTRACE[1], lw=.1, ls='-', zorder=0)

    plt.xlabel('Drug correlation with growth rate\n(Pearson\'s R)')
    plt.ylabel('')


if __name__ == '__main__':
    # - Imports
    # Drug response
    drespo = dtrace.get_drugresponse()
    crispr = dtrace.get_crispr(dtype='both')

    samples = list(set(drespo).intersection(crispr))

    # Growth rate
    growth = pd.read_csv(dtrace.GROWTHRATE_FILE, index_col=0)
    growth = growth.reindex(samples)['growth_rate_median'].dropna()

    # - Filter Drug response
    drespo = dtrace.filter_drugresponse(drespo[samples])

    # - Perform PCA analysis on Drug Response (across Drug and Cell lines)
    pca = perform_pca(drespo)

    # - Growth ~ Drug-response correlation
    g_corr = drespo[growth.index].T.corrwith(growth).sort_values().rename('corr').reset_index()

    # - Histogram IC50s per Drug
    histogram_drug(drespo)
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/histogram_drug.pdf', bbox_inches='tight')
    plt.close('all')

    # - Histogram IC50s per Cell line
    histogram_sample(drespo)
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/histogram_cell_lines.pdf', bbox_inches='tight')
    plt.close('all')

    # - PCA pairplot drug
    pairplot_pca_drug(pca)
    plt.savefig('reports/pca_pairplot_drug.pdf', bbox_inches='tight')
    plt.close('all')

    # - PCA pairplot cell lines
    pairplot_pca_samples(pca, growth)
    plt.savefig('reports/pca_pairplot_cell_lines.pdf', bbox_inches='tight')
    plt.close('all')

    # - Growth ~ PC1 corrplot
    corrplot_pcs_growth(pca, growth, 'PC1')
    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/pca_growth_corrplot.pdf', bbox_inches='tight')
    plt.close('all')

    # - Correlations with growth histogram
    growth_correlation_histogram(g_corr)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/pca_growth_corr_histogram.pdf', bbox_inches='tight')
    plt.close('all')

    # - Top correlated drugs with growth
    growth_correlation_top_drugs(g_corr)
    plt.gcf().set_size_inches(2, 4)
    plt.savefig('reports/pca_growth_corr_top.pdf', bbox_inches='tight')
    plt.close('all')
