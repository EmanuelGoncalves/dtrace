#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from dtrace.analysis import PAL_DTRACE
from sklearn.preprocessing import StandardScaler
from dtrace.Associations import DRUG_INFO_COLUMNS
from analysis.drug_associations import drug_beta_tsne, DRUG_TARGETS_HUE


def histogram_drug(drespo):
    df = drespo.count(1).rename('count').reset_index()

    pal = dict(v17=PAL_DTRACE[2], RS=PAL_DTRACE[0])

    hist_kws = dict(alpha=1., linewidth=0)

    for s in pal:
        sns.distplot(df[df['VERSION'] == s]['count'], color=pal[s], kde=False, bins=15, label=s, hist_kws=hist_kws)
        sns.despine(top=True, right=True)

    plt.xlabel('Number of IC50s measurements')
    plt.ylabel('Number of drugs')
    plt.legend(frameon=False)


def histogram_sample(drespo):
    df = drespo.count(0).rename('count').reset_index()

    hist_kws = dict(alpha=1., linewidth=0)

    sns.distplot(df['count'], color=PAL_DTRACE[2], kde=False, bins=20, hist_kws=hist_kws)
    sns.despine(top=True, right=True)

    plt.xlabel('Number of IC50s')
    plt.ylabel('Number of cell lines')


def perform_pca(drespo, n_components=10):
    df = drespo.T.fillna(drespo.T.mean()).T

    pca = dict()

    for by in ['row', 'column']:
        pca[by] = dict()

        if by == 'column':
            df_ = df.T
            df_ = df_.subtract(df_.mean())
        else:
            df_ = df.subtract(df.mean())

        pcs_labels = list(map(lambda v: f'PC{v + 1}', range(n_components)))

        pca[by]['pca'] = PCA(n_components=n_components).fit(df_)
        pca[by]['vex'] = pd.Series(pca[by]['pca'].explained_variance_ratio_, index=pcs_labels)
        pca[by]['pcs'] = pd.DataFrame(pca[by]['pca'].transform(df_), index=df_.index, columns=pcs_labels)

    return pca


def _pairplot_fix_labels(g, pca, by):
    for i, ax in enumerate(g.axes):
        vexp = pca[by]['vex']['PC{}'.format(i + 1)]
        ax[0].set_ylabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))

    for i, ax in enumerate(g.axes[2]):
        vexp = pca[by]['vex']['PC{}'.format(i + 1)]
        ax.set_xlabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))


def pairplot_pca_drug(pca, hue='VERSION'):
    df = pca['row']['pcs'].reset_index()

    pal = dict(v17=PAL_DTRACE[2], RS=PAL_DTRACE[0])

    g = sns.PairGrid(df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1, hue=hue, palette=None if hue is None else pal)
    g = g.map_diag(plt.hist, color=PAL_DTRACE[2] if hue is None else None, linewidth=0, alpha=.5)
    g = g.map_offdiag(plt.scatter, s=3, edgecolor='white', lw=.1, alpha=.5, color=PAL_DTRACE[2] if hue is None else None)
    g = g.add_legend()

    _pairplot_fix_labels(g, pca, by='row')


def pairplot_pca_samples(pca, growth):
    df = pd.concat([pca['column']['pcs'], growth], axis=1).dropna().sort_values('growth_rate_median')

    g = sns.PairGrid(df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1)
    g = g.map_diag(plt.hist, color=PAL_DTRACE[2], linewidth=0, alpha=.5)

    g = g.map_offdiag(plt.scatter, s=3, edgecolor='white', lw=.1, color=PAL_DTRACE[2])

    _pairplot_fix_labels(g, pca, by='column')


def pairplot_pca_samples_cancertype(pca, min_cell_lines=20):
    # Build data-frame
    ss = dtrace.get_samplesheet()
    df = pd.concat([pca['column']['pcs'], ss['Cancer Type']], axis=1).dropna()

    # Order
    order = df['Cancer Type'].value_counts()
    df = df.replace({'Cancer Type': {i: 'Other' for i in order[order < min_cell_lines].index}})

    order = ['Other'] + list(order[order >= min_cell_lines].index)
    pal = [PAL_DTRACE[1]] + sns.color_palette('tab20', n_colors=len(order) - 1).as_hex()
    pal = dict(zip(*(order, pal)))

    # Plot
    g = sns.PairGrid(df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1, hue='Cancer Type', palette=pal, hue_order=order)
    g = g.map_diag(sns.distplot, hist=False)
    g = g.map_offdiag(plt.scatter, s=4, edgecolor='white', lw=.1, alpha=.8)
    g = g.add_legend()

    _pairplot_fix_labels(g, pca, by='column')


def corrplot_pcs_growth(pca, growth, pc):
    df = pd.concat([pca['column']['pcs'], growth], axis=1, sort=False).dropna().sort_values('growth_rate_median')

    marginal_kws, annot_kws = dict(kde=False, hist_kws={'linewidth': 0}), dict(stat='R')

    scatter_kws, line_kws = dict(edgecolor='w', lw=.3, s=10, alpha=.6), dict(lw=1., color=PAL_DTRACE[0], alpha=1.)
    joint_kws = dict(lowess=True, scatter_kws=scatter_kws, line_kws=line_kws)

    g = sns.jointplot(
        pc, 'growth_rate_median', data=df, kind='reg', space=0, color=PAL_DTRACE[2],
        marginal_kws=marginal_kws, annot_kws=annot_kws, joint_kws=joint_kws
    )

    g.annotate(pearsonr, template='R={val:.2g}, p={p:.1e}', frameon=False)

    g.ax_joint.axvline(0, ls='-', lw=0.1, c=PAL_DTRACE[1], zorder=0)

    vexp = pca['column']['vex'][pc]
    g.set_axis_labels('{} ({:.1f}%)'.format(pc, vexp * 100), 'Growth rate\n(median day 1 / day 4)')


def growth_correlation_histogram(g_corr):
    pal = dict(v17=PAL_DTRACE[2], RS=PAL_DTRACE[0])

    for i, s in enumerate(pal):
        hist_kws = dict(alpha=.4, zorder=i+1, linewidth=0)
        kde_kws = dict(cut=0, lw=1, zorder=i+1, alpha=.8)

        sns.distplot(g_corr[g_corr['VERSION'] == s]['corr'], color=pal[s], kde_kws=kde_kws, hist_kws=hist_kws, bins=15, label=s)

    sns.despine(right=True, top=True)

    plt.axvline(0, c=PAL_DTRACE[1], lw=.1, ls='-', zorder=0)

    plt.xlabel('Drug correlation with growth rate\n(Pearson\'s R)')
    plt.ylabel('Density')

    plt.legend(prop={'size': 6}, frameon=False)


def growth_correlation_top_drugs(g_corr):
    sns.barplot('corr', 'DRUG_NAME', data=g_corr.head(20), color=PAL_DTRACE[2], linewidth=0)
    sns.despine(top=True, right=True)

    plt.axvline(0, c=PAL_DTRACE[1], lw=.1, ls='-', zorder=0)

    plt.xlabel('Drug correlation with growth rate\n(Pearson\'s R)')
    plt.ylabel('')


def drug_tsne(drespo, perplexity=15, learning_rate=250, n_iter=2000):
    d_targets = dtrace.get_drugtargets()

    # Drugs into
    drugs = {tuple(i) for i in drespo.index}
    drugs_annot = {tuple(i) for i in drugs if i[0] in d_targets}
    drugs_screen = {v: {d for d in drugs if d[2] == v} for v in ['v17', 'RS']}

    # Imput NaNs with mean
    df = drespo.T.fillna(drespo.T.mean()).T

    # TSNE
    tsnes = []
    for s in drugs_screen:
        tsne_df = df.loc[list(drugs_screen[s])]
        tsne_df = pd.DataFrame(StandardScaler().fit_transform(tsne_df.T).T, index=tsne_df.index, columns=tsne_df.columns)

        tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, init='pca').fit_transform(tsne_df)

        tsne = pd.DataFrame(tsne, index=tsne_df.index, columns=['P1', 'P2']).reset_index()
        tsne = tsne.assign(target=['Yes' if tuple(i) in drugs_annot else 'No' for i in tsne[DRUG_INFO_COLUMNS].values])

        tsnes.append(tsne)

    tsnes = pd.concat(tsnes)
    tsnes = tsnes.assign(name=[';'.join(map(str, i[1:])) for i in tsnes[DRUG_INFO_COLUMNS].values])

    # Annotate compound replicated
    rep_names = tsnes['name'].value_counts()
    rep_names = set(rep_names[rep_names > 1].index)
    tsnes = tsnes.assign(rep=[i if i in rep_names else 'NA' for i in tsnes['name']])

    # Annotate targets
    tsnes = tsnes.assign(targets=[';'.join(d_targets[i]) if i in d_targets else '' for i in tsnes[DRUG_INFO_COLUMNS[0]]])

    return tsnes


def drug_response_heatmap(drespo):
    plot_df = drespo.T.fillna(drespo.T.mean()).T

    row_colors = pd.Series({i: PAL_DTRACE[0] if i[2] == 'RS' else PAL_DTRACE[1] for i in plot_df.index}).rename('Screen')

    g = sns.clustermap(plot_df, mask=drespo.isna(), cmap='RdGy', center=0, yticklabels=False, xticklabels=False, row_colors=row_colors, cbar_kws={'label': 'ln IC50'})

    g.ax_heatmap.set_xlabel('Cell lines')
    g.ax_heatmap.set_ylabel('Drugs')


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

    # - Perform tSNE analysis
    tsne = drug_tsne(drespo)

    # - Growth ~ Drug-response correlation
    g_corr = drespo[growth.index].T.corrwith(growth).sort_values().rename('corr').reset_index()

    # - Drug responses IC50s heatmap
    drug_response_heatmap(drespo)
    plt.gcf().set_size_inches(4, 4)
    plt.savefig('reports/drug_response_heatmap.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - tSNE plot of drug IC50s
    drug_beta_tsne(tsne, hueby=DRUG_TARGETS_HUE)
    plt.suptitle('tSNE analysis of drug IC50s', y=1.05)
    plt.savefig('reports/drug_associations_ic50s_tsne_targets.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Histogram IC50s per Drug
    histogram_drug(drespo)
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/histogram_drug.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Histogram IC50s per Cell line
    histogram_sample(drespo)
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/histogram_cell_lines.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - PCA pairplot drug
    pairplot_pca_drug(pca)
    plt.suptitle('PCA drug response (Drugs)', y=1.05, fontsize=9)
    plt.savefig('reports/pca_pairplot_drug.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - PCA pairplot cell lines
    pairplot_pca_samples(pca, growth)
    plt.suptitle('PCA drug response (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/pca_pairplot_cell_lines.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - PCA pairplot cell lines - hue by cancer type
    pairplot_pca_samples_cancertype(pca)
    plt.suptitle('PCA drug response (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/pca_pairplot_cell_lines_cancertype.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Growth ~ PC1 corrplot
    corrplot_pcs_growth(pca, growth, 'PC1')
    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/pca_growth_corrplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Correlations with growth histogram
    growth_correlation_histogram(g_corr)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/pca_growth_corr_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Top correlated drugs with growth
    growth_correlation_top_drugs(g_corr)
    plt.gcf().set_size_inches(2, 4)
    plt.savefig('reports/pca_growth_corr_top.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
