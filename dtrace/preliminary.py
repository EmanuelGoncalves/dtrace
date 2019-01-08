#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
from scipy.stats import pearsonr
from DataImporter import DrugResponse
from associations import Association
from sklearn.decomposition import PCA


class Preliminary:
    @classmethod
    def perform_pca(cls, dataframe, n_components=10):
        df = dataframe.T.fillna(dataframe.T.mean()).T

        pca = dict()

        for by in ['row', 'column']:
            pca[by] = dict()

            df_by = df.T.copy() if by != 'row' else df.copy()

            df_by = df_by.subtract(df_by.mean())

            pcs_labels = list(map(lambda v: f'PC{v + 1}', range(n_components)))

            pca[by]['pca'] = PCA(n_components=n_components).fit(df_by)
            pca[by]['vex'] = pd.Series(pca[by]['pca'].explained_variance_ratio_, index=pcs_labels)
            pca[by]['pcs'] = pd.DataFrame(pca[by]['pca'].transform(df_by), index=df_by.index, columns=pcs_labels)

        return pca

    @classmethod
    def _pairplot_fix_labels(cls, g, pca, by):
        for i, ax in enumerate(g.axes):
            vexp = pca[by]['vex']['PC{}'.format(i + 1)]
            ax[0].set_ylabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))

        for i, ax in enumerate(g.axes[2]):
            vexp = pca[by]['vex']['PC{}'.format(i + 1)]
            ax.set_xlabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))

    @classmethod
    def pairplot_pca_by_rows(cls, pca, hue='VERSION'):
        df = pca['row']['pcs'].reset_index()

        pal = None if hue is None else dict(v17=Plot.PAL_DTRACE[2], RS=Plot.PAL_DTRACE[0])
        color = Plot.PAL_DTRACE[2] if hue is None else None

        g = sns.PairGrid(df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1.5, hue=hue, palette=pal)

        g = g.map_diag(plt.hist, color=color, linewidth=0, alpha=.5)

        g = g.map_offdiag(plt.scatter, s=8, edgecolor='white', lw=.1, alpha=.8, color=color)

        if hue is not None:
            g = g.add_legend()

        cls._pairplot_fix_labels(g, pca, by='row')

    @classmethod
    def pairplot_pca_by_columns(cls, pca, hue=None, hue_vars=None):
        df = pca['column']['pcs']

        if hue_vars is not None:
            df = pd.concat([df, hue_vars], axis=1, sort=False).dropna()

        pal = None if hue is None else dict(Broad=Plot.PAL_DTRACE[2], Sanger=Plot.PAL_DTRACE[0])
        color = Plot.PAL_DTRACE[2] if hue is None else None

        g = sns.PairGrid(df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1.5, hue=hue, palette=pal)

        g = g.map_diag(plt.hist, color=color, linewidth=0, alpha=.5)

        g = g.map_offdiag(plt.scatter, s=8, edgecolor='white', lw=.1, alpha=.8, color=color)

        if hue is not None:
            g = g.add_legend()

        cls._pairplot_fix_labels(g, pca, by='column')

    @classmethod
    def pairplot_pca_samples_cancertype(cls, pca, cancertypes, min_cell_lines=20):
        # Build data-frame
        df = pd.concat([
            pca['column']['pcs'],
            cancertypes.rename('Cancer Type')
        ], axis=1, sort=False).dropna()

        # Order
        order = df['Cancer Type'].value_counts()
        df = df.replace({'Cancer Type': {i: 'Other' for i in order[order < min_cell_lines].index}})

        order = ['Other'] + list(order[order >= min_cell_lines].index)
        pal = [Plot.PAL_DTRACE[1]] + sns.color_palette('tab20', n_colors=len(order) - 1).as_hex()
        pal = dict(zip(*(order, pal)))

        # Plot
        g = sns.PairGrid(
            df, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1.5, hue='Cancer Type', palette=pal, hue_order=order
        )

        g = g.map_diag(sns.distplot, hist=False)
        g = g.map_offdiag(plt.scatter, s=8, edgecolor='white', lw=.1, alpha=.8)
        g = g.add_legend()

        cls._pairplot_fix_labels(g, pca, by='column')

    @classmethod
    def corrplot_pcs_growth(cls, pca, growth, pc):
        df = pd.concat([
            pca['column']['pcs'],
            growth
        ], axis=1, sort=False).dropna()

        annot_kws = dict(stat='R')
        marginal_kws = dict(kde=False, hist_kws={'linewidth': 0})

        line_kws = dict(lw=1., color=Plot.PAL_DTRACE[0], alpha=1.)
        scatter_kws = dict(edgecolor='w', lw=.3, s=10, alpha=.6)
        joint_kws = dict(lowess=True, scatter_kws=scatter_kws, line_kws=line_kws)

        g = sns.jointplot(
            pc, 'growth', data=df, kind='reg', space=0, color=Plot.PAL_DTRACE[2],
            marginal_kws=marginal_kws, annot_kws=annot_kws, joint_kws=joint_kws
        )

        g.annotate(pearsonr, template='R={val:.2g}, p={p:.1e}', frameon=False)

        g.ax_joint.axvline(0, ls='-', lw=0.1, c=Plot.PAL_DTRACE[1], zorder=0)

        vexp = pca['column']['vex'][pc]
        g.set_axis_labels('{} ({:.1f}%)'.format(pc, vexp * 100), 'Growth rate\n(median day 1 / day 4)')


class DrugPreliminary(Preliminary):
    DRUG_PAL = dict(v17=Plot.PAL_DTRACE[2], RS=Plot.PAL_DTRACE[0])

    HIST_KDE_KWS = dict(cumulative=True, cut=0)

    @classmethod
    def histogram_drug(cls, drug_count):
        df = drug_count.rename('count').reset_index()

        for s in cls.DRUG_PAL:
            sns.distplot(
                df[df['VERSION'] == s]['count'], color=cls.DRUG_PAL[s], hist=False, label=s, kde_kws=cls.HIST_KDE_KWS
            )
            sns.despine(top=True, right=True)

        plt.xlabel('Number of cell lines screened')
        plt.ylabel(f'Fraction of {df.shape[0]} drugs')

        plt.title('Cumulative distribution of drug measurements')

        plt.legend(loc=4, frameon=False, prop={'size': 6})

    @classmethod
    def histogram_sample(cls, samples_count):
        df = samples_count.rename('count').reset_index()

        sns.distplot(df['count'], color=Plot.PAL_DTRACE[2], hist=False, kde_kws=cls.HIST_KDE_KWS, label=None)
        sns.despine(top=True, right=True)

        plt.xlabel('Number of drugs screened')
        plt.ylabel(f'Fraction of {df.shape[0]} cell lines')

        plt.title('Cumulative distribution of drug measurements')

        plt.legend().remove()

    @classmethod
    def growth_correlation_histogram(cls, g_corr):
        for i, s in enumerate(cls.DRUG_PAL):
            hist_kws = dict(alpha=.4, zorder=i + 1, linewidth=0)
            kde_kws = dict(cut=0, lw=1, zorder=i + 1, alpha=.8)

            sns.distplot(
                g_corr[g_corr['VERSION'] == s]['corr'], color=cls.DRUG_PAL[s], kde_kws=kde_kws, hist_kws=hist_kws,
                bins=15, label=s
            )

        sns.despine(right=True, top=True)

        plt.axvline(0, c=Plot.PAL_DTRACE[1], lw=.1, ls='-', zorder=0)

        plt.xlabel('Drug correlation with growth rate\n(Pearson\'s R)')
        plt.ylabel('Density')

        plt.legend(prop={'size': 6}, frameon=False)

    @classmethod
    def growth_correlation_top_drugs(cls, g_corr, n_features=20):
        sns.barplot('corr', 'DRUG_NAME', data=g_corr.head(n_features), color=Plot.PAL_DTRACE[2], linewidth=0)
        sns.despine(top=True, right=True)

        plt.axvline(0, c=Plot.PAL_DTRACE[1], lw=.1, ls='-', zorder=0)

        plt.xlabel('Drug correlation with growth rate\n(Pearson\'s R)')
        plt.ylabel('')


class CrisprPreliminary(Preliminary):
    @classmethod
    def corrplot_pcs_essentiality(cls, pca, crispr, pc):
        df = pd.concat([pca['row']['pcs'], (crispr < -0.5).sum(1).rename('count')], axis=1)

        annot_kws = dict(stat='R', loc=2)
        marginal_kws = dict(kde=False, hist_kws={'linewidth': 0})

        scatter_kws = dict(edgecolor='w', lw=.3, s=6, alpha=.3)
        line_kws = dict(lw=1., color=Plot.PAL_DTRACE[0], alpha=1.)
        joint_kws = dict(lowess=True, scatter_kws=scatter_kws, line_kws=line_kws)

        g = sns.jointplot(
            'count', pc, data=df, kind='reg', space=0, color=Plot.PAL_DTRACE[2],
            marginal_kws=marginal_kws, annot_kws=annot_kws, joint_kws=joint_kws
        )

        g.annotate(pearsonr, template='R={val:.2g}, p={p:.1e}', frameon=False)

        g.ax_joint.axhline(0, ls='-', lw=0.1, c=Plot.PAL_DTRACE[1], zorder=0)
        g.ax_joint.axvline(0, ls='-', lw=0.1, c=Plot.PAL_DTRACE[1], zorder=0)

        vexp = pca['row']['vex'][pc]
        g.set_axis_labels('Gene significantly essential count', '{} ({:.1f}%)'.format(pc, vexp * 100))


if __name__ == '__main__':
    # - Import data
    datasets = Association()

    # - Drug PCAs
    pca_drug = Preliminary.perform_pca(datasets.drespo)
    pca_crispr = Preliminary.perform_pca(datasets.crispr)

    for n, pcas in [('drug', pca_drug), ('crispr', pca_crispr)]:
        for by in pcas:
            pcas[by]['pcs'].round(5).to_csv(f'data/{n}_pca_{by}_pcs.csv')

    # - Growth ~ Drug-response correlation
    g_corr = DrugResponse.growth_corr(datasets.drespo, datasets.samplesheet.samplesheet['growth'])

    # - Plots
    # Drug response
    DrugPreliminary.histogram_drug(datasets.drespo.count(1))
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/preliminary_drug_histogram_drug.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    DrugPreliminary.histogram_sample(datasets.drespo.count(0))
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/preliminary_drug_histogram_samples.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    DrugPreliminary.pairplot_pca_by_rows(pca_drug)
    plt.suptitle('PCA drug response (Drugs)', y=1.05, fontsize=9)
    plt.savefig('reports/preliminary_drug_pca_pairplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    DrugPreliminary.pairplot_pca_by_columns(pca_drug)
    plt.suptitle('PCA drug response (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/preliminary_drug_pca_pairplot_samples.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    DrugPreliminary.pairplot_pca_samples_cancertype(pca_drug, datasets.samplesheet.samplesheet['cancer_type'])
    plt.suptitle('PCA drug response (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/preliminary_drug_pca_pairplot_cancertype.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    DrugPreliminary.corrplot_pcs_growth(pca_drug, datasets.samplesheet.samplesheet['growth'], 'PC1')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/preliminary_drug_pca_growth_corrplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    DrugPreliminary.growth_correlation_histogram(g_corr)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/preliminary_drug_pca_growth_corrplot_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    DrugPreliminary.growth_correlation_top_drugs(g_corr)
    plt.gcf().set_size_inches(2, 4)
    plt.savefig('reports/preliminary_drug_pca_growth_corrplot_top.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Crispr
    CrisprPreliminary.pairplot_pca_by_rows(pca_crispr, hue=None)
    plt.suptitle('PCA CRISPR-Cas9 (Genes)', y=1.05, fontsize=9)
    plt.savefig('reports/preliminary_crispr_pca_pairplot.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close('all')

    CrisprPreliminary.pairplot_pca_by_columns(
        pca_crispr, hue='institute', hue_vars=datasets.samplesheet.samplesheet['institute']
    )
    plt.suptitle('PCA CRISPR-Cas9 (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/preliminary_crispr_pca_pairplot_samples.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    CrisprPreliminary.pairplot_pca_samples_cancertype(pca_crispr, datasets.samplesheet.samplesheet['cancer_type'])
    plt.suptitle('PCA CRISPR-Cas9 (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/preliminary_crispr_pca_pairplot_cancertype.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    CrisprPreliminary.corrplot_pcs_growth(pca_crispr, datasets.samplesheet.samplesheet['growth'], 'PC4')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/preliminary_crispr_pca_growth_corrplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    CrisprPreliminary.corrplot_pcs_essentiality(pca_crispr, datasets.crispr, 'PC1')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/preliminary_crispr_pca_essentiality_corrplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
