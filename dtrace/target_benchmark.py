#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
from natsort import natsorted
from importer import DrugResponse, PPI
from crispy.qc_plot import QCplot
from scipy.stats import ttest_ind
from associations import Association
from scipy.stats.distributions import hypergeom
from sklearn.metrics import f1_score, recall_score, precision_score


class TargetBenchmark:
    def __init__(self, fdr=.1):
        self.fdr = fdr

        # Imports
        self.datasets = Association(dtype_drug='ic50')
        self.lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')

        # Define sets of drugs
        self.df_genes = set(self.lmm_drug['GeneSymbol'])
        self.d_targets = self.datasets.drespo_obj.get_drugtargets(by='Name')

        self.drugs_all = set(self.lmm_drug['DRUG_NAME'])
        self.drugs_annot = {d for d in self.drugs_all if d in self.d_targets}
        self.drugs_tested = {d for d in self.drugs_annot if len(self.d_targets[d].intersection(self.df_genes)) > 0}
        self.drugs_tested_signif = {
            d for d in self.lmm_drug.query(f'fdr < {self.fdr}')['DRUG_NAME'] if d in self.drugs_tested
        }
        self.drugs_tested_correct = {
            d for d in self.lmm_drug.query(f"fdr < {self.fdr} & target == 'T'")['DRUG_NAME'] if d in self.drugs_tested_signif
        }

    def beta_histogram(self):
        kde_kws = dict(cut=0, lw=1, zorder=1, alpha=.8)
        hist_kws = dict(alpha=.4, zorder=1, linewidth=0)

        sns.distplot(
            self.lmm_drug.query("target != 'T'")['beta'], color=Plot.PAL_DTRACE[2], kde_kws=kde_kws, hist_kws=hist_kws,
            label='All', bins=30
        )

        sns.distplot(
            self.lmm_drug.query("target == 'T'")['beta'], color=Plot.PAL_DTRACE[0], kde_kws=kde_kws, hist_kws=hist_kws,
            label='Target', bins=30
        )

        sns.despine(right=True, top=True)

        plt.axvline(0, c=Plot.PAL_DTRACE[1], lw=.3, ls='-', zorder=0)

        plt.xlabel('Association beta')
        plt.ylabel('Density')

        plt.legend(prop={'size': 6}, loc=2, frameon=False)

    def pval_histogram(self):
        hist_kws = dict(alpha=.5, zorder=1, linewidth=.3, density=True)

        sns.distplot(
            self.lmm_drug.query("target != 'T'")['pval'], hist_kws=hist_kws, bins=30, kde=False, label='All',
            color=Plot.PAL_DTRACE[2]
        )

        sns.distplot(
            self.lmm_drug.query("target == 'T'")['pval'], hist_kws=hist_kws, bins=30, kde=False, label='Target',
            color=Plot.PAL_DTRACE[0]
        )

        sns.despine()

        plt.xlabel('Association p-value')
        plt.ylabel('Density')

        plt.legend(prop={'size': 6}, frameon=False)

    def recapitulated_drug_targets_barplot(self):
        # Build dataframe
        plot_df = pd.DataFrame(dict(
            names=['All', 'w/Target', 'w/Tested target', 'w/Signif tested target', 'w/Correct target'],
            count=list(map(len, [
                self.drugs_all, self.drugs_annot, self.drugs_tested, self.drugs_tested_signif, self.drugs_tested_correct
            ]))
        )).sort_values('count', ascending=True)
        plot_df = plot_df.assign(y=range(plot_df.shape[0]))

        # Plot
        plt.barh(plot_df['y'], plot_df['count'], color=Plot.PAL_DTRACE[2], linewidth=0)

        sns.despine(right=True, top=True)

        for c, y in plot_df[['count', 'y']].values:
            plt.text(c + 3, y, str(c), va='center', fontsize=5, zorder=10, color=Plot.PAL_DTRACE[2])

        plt.yticks(plot_df['y'], plot_df['names'])
        plt.xlabel('Number of drugs')
        plt.ylabel('')

    def drugs_ppi(self):
        df = self.lmm_drug[self.lmm_drug['DRUG_NAME'].isin(self.drugs_tested)]

        order = ['T', '1', '2', '3', '4', '5+', '-']

        pal = dict(zip(*(
            order,
            [Plot.PAL_DTRACE[1]] + QCplot.get_palette_continuous(len(order) - 2, color=Plot.PAL_DTRACE[2]) + [Plot.PAL_DTRACE[3]]
        )))

        QCplot.bias_boxplot(
            df.query(f'fdr < {self.fdr}'), x='target', y='fdr', notch=False, add_n=True, n_text_offset=5e-3, palette=pal, order=order
        )

        sns.despine()

        plt.xlabel('Associated gene position in PPI')
        plt.ylabel('Bonferroni adj. p-value')

        plt.title('Significant associations\n(adj. p-value < 10%)')

    def top_associations_barplot(self, ntop=50, n_cols=10):
        # Filter for signif associations
        df = self.lmm_drug \
            .query('fdr < {}'.format(self.fdr)) \
            .sort_values('fdr') \
            .groupby(['DRUG_NAME', 'GeneSymbol']) \
            .first() \
            .sort_values('fdr') \
            .reset_index()
        df = df.assign(logpval=-np.log10(df['pval']).values)

        # Drug order
        order = list(df.groupby('DRUG_NAME')['fdr'].min().sort_values().index)[:ntop]

        # Build plot dataframe
        df_, xpos = [], 0
        for i, drug_name in enumerate(order):
            if i % n_cols == 0:
                xpos = 0

            df_drug = df[df['DRUG_NAME'] == drug_name]
            df_drug = df_drug.assign(xpos=np.arange(xpos, xpos + df_drug.shape[0]))
            df_drug = df_drug.assign(irow=int(np.floor(i / n_cols)))

            xpos += (df_drug.shape[0] + 2)

            df_.append(df_drug)

        df = pd.concat(df_).reset_index()

        # Plot
        f, axs = plt.subplots(int(np.ceil(ntop / n_cols)), 1, sharex='none', sharey='all', gridspec_kw=dict(hspace=.0))

        # Barplot
        for irow in set(df['irow']):
            df_irow = df[df['irow'] == irow]

            df_irow_ = df_irow.query("target != 'T'")
            axs[irow].bar(df_irow_['xpos'], df_irow_['logpval'], .8, color=Plot.PAL_DTRACE[2], align='center', zorder=5, linewidth=0)

            df_irow_ = df_irow.query("target == 'T'")
            axs[irow].bar(df_irow_['xpos'], df_irow_['logpval'], .8, color=Plot.PAL_DTRACE[0], align='center', zorder=5, linewidth=0)

            for k, v in df_irow.groupby('DRUG_NAME')['xpos'].min().sort_values().to_dict().items():
                axs[irow].text(v - 1.2, 0.1, textwrap.fill(k, 15), va='bottom', fontsize=7, zorder=10, rotation='vertical', color=Plot.PAL_DTRACE[2])

            for g, p in df_irow[['GeneSymbol', 'xpos']].values:
                axs[irow].text(p, 0.1, g, ha='center', va='bottom', fontsize=5, zorder=10, rotation='vertical', color='white')

            for x, y, t, b in df_irow[['xpos', 'logpval', 'target', 'beta']].values:
                c = Plot.PAL_DTRACE[0] if t == 'T' else Plot.PAL_DTRACE[2]

                axs[irow].text(x, y + 0.25, t, color=c, ha='center', fontsize=6, zorder=10)
                axs[irow].text(x, -3, f'{b:.1f}', color=c, ha='center', fontsize=6, rotation='vertical', zorder=10)

            sns.despine(ax=axs[irow], right=True, top=True)
            axs[irow].axes.get_xaxis().set_ticks([])
            axs[irow].set_ylabel('Drug association\n(-log10 p-value)')

        plt.gcf().set_size_inches(10, 8)

        plt.savefig('reports/target_benchmark_associations_barplot.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')


if __name__ == '__main__':
    # Import target benchmark
    trg = TargetBenchmark(fdr=.1)

    ppi = PPI().build_string_ppi(score_thres=900)
    ppi = PPI.ppi_corr(ppi, trg.datasets.crispr)

    #
    trg.recapitulated_drug_targets_barplot()
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/target_benchmark_count_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.beta_histogram()
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/target_benchmark_beta_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.pval_histogram()
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/target_benchmark_pval_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.drugs_ppi()
    plt.gcf().set_size_inches(2.5, 2.5)
    plt.savefig('reports/target_benchmark_ppi_distance.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Single feature examples
    dgs = [('Alpelisib', 'PIK3CA'), ('Nutlin-3a (-)', 'MDM2'), ('MCL1_1284', 'MCL1'), ('MCL1_1284', 'MARCH5')]

    # dg = ('Alpelisib', 'PIK3CA')
    for dg in dgs:
        assoc = trg.lmm_drug[(trg.lmm_drug['DRUG_NAME'] == dg[0]) & (trg.lmm_drug['GeneSymbol'] == dg[1])].iloc[0]

        drug = tuple(assoc[DrugResponse.DRUG_COLUMNS])

        dmax = np.log(trg.datasets.drespo_obj.maxconcentration[drug])
        annot_text = f"Beta={assoc['beta']:.2g}, FDR={assoc['fdr']:.1e}"

        plot_df = pd.concat([
            trg.datasets.drespo.loc[drug].rename('drug'),
            trg.datasets.crispr.loc[dg[1]].rename('crispr'),
            trg.datasets.crispr_obj.institute.rename('Institute'),
        ], axis=1, sort=False).dropna()

        g = Plot().plot_corrplot('crispr', 'drug', 'Institute', plot_df, add_hline=True, annot_text=annot_text)

        g.ax_joint.axhline(y=dmax, linewidth=.3, color=Plot.PAL_DTRACE[2], ls=':', zorder=0)

        g.set_axis_labels(f'{dg[1]} (scaled log2 FC)', f'{dg[0]} (ln IC50)')

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(f'reports/association_drug_scatter_{dg[0]}_{dg[1]}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # CRISPR gene pair corr
    for gene_x, gene_y in [('MARCH5', 'MCL1'), ('SHC1', 'EGFR')]:
        plot_df = pd.concat([
            trg.datasets.crispr.loc[gene_x].rename(gene_x),
            trg.datasets.crispr.loc[gene_y].rename(gene_y),
            trg.datasets.crispr_obj.institute.rename('Institute'),
        ], axis=1, sort=False).dropna()

        g = Plot().plot_corrplot(gene_x, gene_y, 'Institute', plot_df, add_hline=True)

        g.set_axis_labels(f'{gene_x} (scaled log2 FC)', f'{gene_y} (scaled log2 FC)')

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(f'reports/association_scatter_{gene_x}_{gene_y}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # PPI
    d, t, o = ('Linsitinib', .3, 1)
    for d, t, o in [('Nutlin-3a (-)', .5, 1), ('MCL1_1284', .2, 2)]:
        graph = PPI.plot_ppi(d, trg.lmm_drug, ppi, corr_thres=t, norder=o, fdr=0.05)
        graph.write_pdf(f'reports/ppi_{d}.pdf')
