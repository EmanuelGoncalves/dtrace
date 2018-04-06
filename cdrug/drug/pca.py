#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.decomposition import PCA


def drug_counts_hist(plot_df, by):
    if by == 'drug':
        for v, c in cdrug.PAL_DICT_VERSION.items():
            sns.distplot(plot_df.query("VERSION == '{}'".format(v))['count'], color=c, kde=False, bins=20, label=v, hist_kws={'alpha': .9})

            md_count = plot_df.query("VERSION == '{}'".format(v))['count'].median()
            plt.axvline(md_count, c='white', lw=.3, ls='--')
            plt.text(md_count - 3, 3, 'Median = {:.0f}'.format(md_count), fontdict=dict(color='white', fontsize=3), rotation=90, ha='right', va='bottom')

        plt.xlabel('Number of IC50s (per drug)')
        plt.legend()

    elif by == 'sample':
        sns.distplot(plot_df['count'], color=cdrug.PAL_20[0], kde=False, bins=20, hist_kws={'alpha': .9})

        md_count = plot_df['count'].median()
        plt.axvline(md_count, c='white', lw=.3, ls='--')
        plt.text(md_count - 3, 3, 'Median = {:.0f}'.format(md_count), fontdict=dict(color='white', fontsize=3), rotation=90, ha='right', va='bottom')

        plt.xlabel('Number of IC50s (per cell line)')

    plt.ylabel('Counts')
    plt.title('IC50s histogram per drug across cell lines')

    plt.gcf().set_size_inches(2.5, 1.5)
    plt.savefig('reports/pca_{}_count_hist.pdf'.format(by), bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    ss = pd.read_csv(cdrug.SAMPLESHEET_FILE, index_col=0)

    # Drug response
    d_response = pd.read_csv(cdrug.DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # Growth rate
    growth = pd.read_csv(cdrug.GROWTHRATE_FILE, index_col=0)

    # - Plot Histogram IC50s per Drug
    drug_counts_hist(d_response.count(1).rename('count').reset_index(), by='drug')

    # - Plot Histogram IC50s per Cell Line
    drug_counts_hist(d_response.count().rename('count').reset_index(), by='sample')

    # -
    plot_df = d_response[ss[ss['Rapid Screen'] == 'Yes'].index]
    plot_df = plot_df.loc[plot_df.count(1) > (.5 * plot_df.shape[1])]

    index = plot_df.count(1).rename('count').reset_index().sort_values(['VERSION', 'count'], ascending=False)
    index = index.set_index(['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']).index

    columns = plot_df.count().sort_values(ascending=False).index

    plot_df = plot_df.loc[index, columns]

    cmap = colors.ListedColormap([cdrug.PAL_20[0], cdrug.PAL_20[4]])

    g = sns.clustermap(
        plot_df.T.fillna(plot_df.mean(axis=1)).T, xticklabels=False, yticklabels=False, square=False, cmap='Spectral', center=0,
        row_colors=[cdrug.PAL_DICT_VERSION[i[2]] for i in plot_df.index], side_colors_ratio=0.05, mask=plot_df.apply(np.isnan)
    )

    # g.cax.set_visible(False)
    #
    # g.ax_heatmap.set_ylabel('Cell lines')
    # g.ax_heatmap.set_xlabel('Drugs')

    plt.gcf().set_size_inches(5, 5)
    plt.savefig('reports/drug_heatmap.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # - PCA
    pca = dict()
    for by in ['drug', 'sample']:
        pca[by] = dict()

        df = d_response[ss[ss['Rapid Screen'] == 'Yes'].index]
        df = df.loc[df.count(1) > (.5 * df.shape[1])]
        df = df.T.fillna(df.mean(axis=1)).T
        df = df.subtract(df.mean(1), axis=0)

        if by == 'sample':
            df = df.T

        pca[by]['pca'] = PCA(n_components=10).fit(df)
        pca[by]['vex'] = pd.Series(pca[by]['pca'].explained_variance_ratio_, index=map(lambda v: 'PC{}'.format(v + 1), range(10)))
        pca[by]['pcs'] = pd.DataFrame(pca[by]['pca'].transform(df), index=df.index, columns=map(lambda v: 'PC{}'.format(v + 1), range(10)))

    # - Principal component plot
    for by in ['drug', 'sample']:
        plot_df = pd.concat([pca[by]['pcs'], growth], axis=1).dropna()
        plot_df = plot_df.drop(['WIL2-NS']).sort_values('growth_rate_median')

        g = sns.PairGrid(
            plot_df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1, hue='VERSION' if by == 'drug' else None,
            palette=cdrug.PAL_DICT_VERSION if by == 'drug' else None
        )

        g = g.map_diag(plt.hist, color=cdrug.PAL_20[0] if by == 'sample' else None)

        if by == 'sample':
            cmap = sns.light_palette(cdrug.PAL_20[0], as_cmap=True)
            g = g.map_offdiag(plt.scatter, s=3, edgecolor='white', lw=.1, color=plot_df['growth_rate_median'], cmap=cmap, alpha=.5)
            cax = g.fig.add_axes([.98, .4, .01, .2])
            plt.colorbar(cax=cax)
            # cax.set_ylabel('Growth rate (median day 1 / day 4)', rotation=270)

        else:
            g = g.map_offdiag(plt.scatter, s=3, edgecolor='white', lw=.1)
            g = g.add_legend()


        for i, ax in enumerate(g.axes):
            vexp = pca[by]['vex']['PC{}'.format(i + 1)]
            ax[0].set_ylabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))

        for i, ax in enumerate(g.axes[2]):
            vexp = pca[by]['vex']['PC{}'.format(i + 1)]
            ax.set_xlabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))

        plt.savefig('reports/pca_{}_pairplot.pdf'.format(by), bbox_inches='tight')
        plt.close('all')

    # -
    plot_df = pd.concat([pca['sample']['pcs'], growth], axis=1)\
        .drop(['WIL2-NS']).dropna()

    pci = 1

    g = sns.jointplot(
        'PC{}'.format(pci), 'growth_rate_median', data=plot_df, kind='reg', space=0, color=cdrug.PAL_20[0],
        marginal_kws=dict(kde=False), annot_kws=dict(stat='r'),
        joint_kws=dict(scatter_kws=dict(edgecolor='w', lw=.3, s=10, alpha=.6), line_kws=dict(lw=1., color=cdrug.PAL_20[4], alpha=1.), order=2)
    )

    # g.ax_joint.axhline(0, ls='-', lw=0.1, c=cdrug.PAL_20[5], zorder=0)
    g.ax_joint.axvline(0, ls='-', lw=0.1, c=cdrug.PAL_20[5], zorder=0)

    vexp = pca[by]['vex']['PC{}'.format(pci)]
    g.set_axis_labels('PC{} ({:.1f}%)'.format(pci, vexp * 100), 'Growth rate\n(median day 1 / day 4)')

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/pca_sample_jointplot_pc1_growth.pdf', bbox_inches='tight')
    plt.close('all')
