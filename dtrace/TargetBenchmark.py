#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from DTracePlot import DTracePlot
from natsort import natsorted
from crispy.utils import Utils
from crispy.qc_plot import QCplot
from Associations import Association
from DataImporter import DrugResponse, PPI
from scipy.stats import ttest_ind, mannwhitneyu


class TargetBenchmark(DTracePlot):
    def __init__(self, fdr=.1, dtype='ic50'):
        self.fdr = fdr
        self.dtype = dtype

        # Imports
        self.datasets = Association(dtype_drug=dtype)

        self.lmm_drug = pd.read_csv(f'data/drug_lmm_regressions_{dtype}.csv.gz')
        self.lmm_drug_gexp = pd.read_csv(f'data/drug_lmm_regressions_{dtype}_gexp.csv.gz')

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

        super().__init__()

    def boxplot_kinobead(self):
        drugs_correct = set(self.lmm_drug.query(f"(fdr < {self.fdr}) & (target == 'T')")['DRUG_ID'].apply(str))

        catds = pd.read_csv('data/klaeger_et_al_catds_most_potent.csv')

        def __catds_id_match(ids):
            if str(ids).lower() == 'nan':
                return 'NA'
            elif len(set(ids.split(';')).intersection(drugs_correct)) == 0:
                return 'No'
            else:
                return 'Yes'

        catds['signif'] = catds['ids'].apply(__catds_id_match)

        t, p = ttest_ind(
            catds[catds['signif'] == 'No']['CATDS_most_potent'],
            catds[catds['signif'] == 'Yes']['CATDS_most_potent'],
            equal_var=False
        )
        print(p)

        # Plot
        ax = plt.gca()

        order = ['No', 'Yes', 'NA']
        pal = {'No': self.PAL_DTRACE[2], 'Yes': self.PAL_DTRACE[0], 'NA': self.PAL_DTRACE[1]}

        sns.boxplot(catds['signif'], catds['CATDS_most_potent'], notch=True, palette=pal, linewidth=.3, fliersize=1.5, order=order, ax=ax)
        sns.swarmplot(catds['signif'], catds['CATDS_most_potent'], palette=pal, linewidth=.3, size=2, order=order, ax=ax)
        ax.axhline(0.5, lw=.3, c=self.PAL_DTRACE[1], ls='-', alpha=.8, zorder=0)

        sns.despine(top=True, right=True, ax=ax)

        ax.set_ylim((-0.1, 1.1))

        ax.set_xlabel('Drug has a significant\nCRISPR-Cas9 association')
        ax.set_ylabel('Selectivity[$CATDS_{most\ potent}$]')

    def beta_histogram(self):
        kde_kws = dict(cut=0, lw=1, zorder=1, alpha=.8)
        hist_kws = dict(alpha=.4, zorder=1, linewidth=0)

        sns.distplot(
            self.lmm_drug.query("target != 'T'")['beta'], color=self.PAL_DTRACE[2], kde_kws=kde_kws, hist_kws=hist_kws,
            label='All', bins=30
        )

        sns.distplot(
            self.lmm_drug.query("target == 'T'")['beta'], color=self.PAL_DTRACE[0], kde_kws=kde_kws, hist_kws=hist_kws,
            label='Target', bins=30
        )

        mannwhitneyu(self.lmm_drug.query("target != 'T'")['beta'], self.lmm_drug.query("target == 'T'")['beta'])
        ttest_ind(self.lmm_drug.query("target != 'T'")['beta'], self.lmm_drug.query("target == 'T'")['beta'], equal_var=False)

        sns.despine(right=True, top=True)

        plt.axvline(0, c=self.PAL_DTRACE[1], lw=.3, ls='-', zorder=0)

        plt.xlabel('Association beta')
        plt.ylabel('Density')

        plt.legend(prop={'size': 6}, loc=2, frameon=False)

        plt.gcf().set_size_inches(2, 2)
        plt.savefig('reports/target_benchmark_beta_histogram.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    def pval_histogram(self):
        hist_kws = dict(alpha=.5, zorder=1, linewidth=.3, density=True)

        sns.distplot(
            self.lmm_drug.query("target != 'T'")['pval'], hist_kws=hist_kws, bins=30, kde=False, label='All',
            color=self.PAL_DTRACE[2]
        )

        sns.distplot(
            self.lmm_drug.query("target == 'T'")['pval'], hist_kws=hist_kws, bins=30, kde=False, label='Target',
            color=self.PAL_DTRACE[0]
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
        plt.barh(plot_df['y'], plot_df['count'], color=self.PAL_DTRACE[2], linewidth=0)

        sns.despine(right=True, top=True)

        for c, y in plot_df[['count', 'y']].values:
            plt.text(c + 3, y, str(c), va='center', fontsize=5, zorder=10, color=self.PAL_DTRACE[2])

        plt.yticks(plot_df['y'], plot_df['names'])
        plt.xlabel('Number of drugs')
        plt.ylabel('')

    def drugs_ppi(self, dtype='crispr'):
        if dtype == 'crispr':
            df = self.lmm_drug[self.lmm_drug['DRUG_NAME'].isin(self.drugs_tested)]

        elif dtype == 'gexp':
            df = self.lmm_drug_gexp[self.lmm_drug['DRUG_NAME'].isin(self.drugs_tested)]

        order = ['T', '1', '2', '3', '4', '5+', '-']

        pal = dict(zip(*(
            order,
            [self.PAL_DTRACE[0]] + QCplot.get_palette_continuous(len(order) - 2, color=self.PAL_DTRACE[2]) + [self.PAL_DTRACE[3]]
        )))

        QCplot.bias_boxplot(
            df.query(f'fdr < {self.fdr}'), x='target', y='fdr', notch=False, add_n=True, n_text_offset=5e-3, palette=pal, order=order
        )

        sns.despine()

        plt.xlabel('Associated gene position in PPI')
        plt.ylabel('Bonferroni adj. p-value')

        plt.title('Significant associations\n(adj. p-value < 10%)')

    def drugs_ppi_countplot(self, dtype='crispr'):
        if dtype == 'crispr':
            df = self.lmm_drug[self.lmm_drug['DRUG_NAME'].isin(self.drugs_tested)]

        elif dtype == 'gexp':
            df = self.lmm_drug_gexp[self.lmm_drug['DRUG_NAME'].isin(self.drugs_tested)]

        order = ['T', '1', '2', '3', '4', '5+', '-']

        pal = dict(zip(*(
            order,
            [self.PAL_DTRACE[0]] + QCplot.get_palette_continuous(len(order) - 2, color=self.PAL_DTRACE[2]) + [self.PAL_DTRACE[3]]
        )))

        plot_df = df.query(f'fdr < {self.fdr}')['target'].value_counts().rename('count').reset_index()

        sns.barplot('index', 'count', data=plot_df, order=order, palette=pal)

        sns.despine()

        plt.xlabel('Associated gene position in PPI')
        plt.ylabel('Number of associations')

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
            axs[irow].bar(df_irow_['xpos'], df_irow_['logpval'], .8, color=self.PAL_DTRACE[2], align='center', zorder=5, linewidth=0)

            df_irow_ = df_irow.query("target == 'T'")
            axs[irow].bar(df_irow_['xpos'], df_irow_['logpval'], .8, color=self.PAL_DTRACE[0], align='center', zorder=5, linewidth=0)

            for k, v in df_irow.groupby('DRUG_NAME')['xpos'].min().sort_values().to_dict().items():
                axs[irow].text(v - 1.2, 0.1, textwrap.fill(k, 15), va='bottom', fontsize=7, zorder=10, rotation='vertical', color=self.PAL_DTRACE[2])

            for g, p in df_irow[['GeneSymbol', 'xpos']].values:
                axs[irow].text(p, 0.1, g, ha='center', va='bottom', fontsize=5, zorder=10, rotation='vertical', color='white')

            for x, y, t, b in df_irow[['xpos', 'logpval', 'target', 'beta']].values:
                c = self.PAL_DTRACE[0] if t == 'T' else self.PAL_DTRACE[2]

                axs[irow].text(x, y + 0.25, t, color=c, ha='center', fontsize=6, zorder=10)
                axs[irow].text(x, -3, f'{b:.1f}', color=c, ha='center', fontsize=6, rotation='vertical', zorder=10)

            sns.despine(ax=axs[irow], right=True, top=True)
            axs[irow].axes.get_xaxis().set_ticks([])
            axs[irow].set_ylabel('Drug association\n(-log10 p-value)')

        plt.gcf().set_size_inches(10, 8)

    def manhattan_plot(self, n_genes=20):
        # Import gene genomic coordinates from CRISPR-Cas9 library
        crispr_lib = Utils.get_crispr_lib().groupby('gene').agg({'start': 'min', 'chr': 'first'})

        # Plot data-frame
        df = self.lmm_drug.copy()
        df = df.assign(pos=crispr_lib.loc[df['GeneSymbol'], 'start'].values)
        df = df.assign(chr=crispr_lib.loc[df['GeneSymbol'], 'chr'].apply(lambda v: v.replace('chr', '')).values)
        df = df.sort_values(['chr', 'pos'])

        # Most frequently associated genes
        top_genes = self.lmm_drug.groupby('GeneSymbol')['fdr'].min().sort_values().head(n_genes)
        top_genes_pal = dict(zip(*(top_genes.index, sns.color_palette('tab20', n_colors=n_genes).as_hex())))

        # Plot
        chrms = set(df['chr'])
        label_fdr = 'Significant'

        f, axs = plt.subplots(1, len(chrms), sharex='none', sharey='row', gridspec_kw=dict(wspace=.05))
        for i, name in enumerate(natsorted(chrms)):
            df_group = df[df['chr'] == name]

            # Plot non-significant
            df_nonsignif = df_group.query(f'fdr >= {self.fdr}')
            axs[i].scatter(df_nonsignif['pos'], -np.log10(df_nonsignif['pval']), c=self.PAL_DTRACE[(i % 2) + 1], s=2)

            # Plot significant
            df_signif = df_group.query(f'fdr < {self.fdr}')
            df_signif = df_signif[~df_signif['GeneSymbol'].isin(top_genes.index)]
            axs[i].scatter(
                df_signif['pos'], -np.log10(df_signif['pval']), c=self.PAL_DTRACE[0], s=2, zorder=3, label=label_fdr
            )

            # Plot significant associations of top frequent genes
            df_genes = df_group.query(f'fdr < {self.fdr}')
            df_genes = df_genes[df_genes['GeneSymbol'].isin(top_genes.index)]
            for pos, pval, gene in df_genes[['pos', 'pval', 'GeneSymbol']].values:
                axs[i].scatter(
                    pos, -np.log10(pval), c=top_genes_pal[gene], s=6, zorder=4, label=gene, marker='2', lw=.75
                )

            # Misc
            axs[i].axes.get_xaxis().set_ticks([])
            axs[i].set_xlabel(name)
            axs[i].set_ylim(0)

            if i != 0:
                sns.despine(ax=axs[i], left=True, right=True, top=True)
                axs[i].yaxis.set_ticks_position('none')

            else:
                sns.despine(ax=axs[i], right=True, top=True)
                axs[i].set_ylabel('Drug-gene association\n(-log10 p-value)')

        f.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.xlabel('Chromosome')

        # Legend
        order_legend = [label_fdr] + list(top_genes.index)
        by_label = {l: p for ax in axs for p, l in zip(*(ax.get_legend_handles_labels()))}
        by_label = [(l, by_label[l]) for l in order_legend]

        plt.legend(
            list(zip(*(by_label)))[1], list(zip(*(by_label)))[0], loc='center left', bbox_to_anchor=(1.01, 0.5),
            prop={'size': 5}, frameon=False
        )

    def drug_notarget_barplot(self, drug, genes):
        df = self.lmm_drug.query(f"DRUG_NAME == '{drug}'")
        df = df[df['GeneSymbol'].isin(genes)]
        df = df.groupby(['DRUG_NAME', 'GeneSymbol']).first()
        df = df.sort_values(['pval', 'fdr'], ascending=False).reset_index()

        ax = plt.gca()

        ax.barh(
            df.query("target != 'T'").index, -np.log10(df.query("target != 'T'")['pval']), .8,
            color=self.PAL_DTRACE[2],
            align='center', zorder=0, linewidth=0
        )

        ax.barh(
            df.query("target == 'T'").index, -np.log10(df.query("target == 'T'")['pval']), .8,
            color=self.PAL_DTRACE[0],
            align='center', zorder=0, linewidth=0
        )

        sns.despine(right=True, top=True, ax=ax)

        for i, (y, t, f) in df[['pval', 'target', 'fdr']].iterrows():
            ax.text(-np.log10(y) - 0.1, i, t, color='white', ha='right', va='center', fontsize=6, zorder=10)

            if f < self.fdr:
                ax.text(-np.log10(y) + 0.1, i, '*', color=self.PAL_DTRACE[2], ha='left', va='center', fontsize=6,
                        zorder=10)

        ax.set_yticks(df.index)
        ax.set_yticklabels(df['GeneSymbol'])

        ax.set_xlabel('Drug association (-log10 p-value)')
        ax.set_title(drug)


if __name__ == '__main__':
    # Import target benchmark
    trg = TargetBenchmark(fdr=.1)

    ppi = PPI().build_string_ppi(score_thres=900)
    ppi = PPI.ppi_corr(ppi, trg.datasets.crispr)

    #
    trg.top_associations_barplot()
    plt.savefig('reports/target_benchmark_associations_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.manhattan_plot(n_genes=20)
    plt.gcf().set_size_inches(5, 2)
    plt.savefig('reports/drug_associations_manhattan.png', bbox_inches='tight', transparent=True, dpi=600)
    plt.close('all')

    trg.recapitulated_drug_targets_barplot()
    plt.gcf().set_size_inches(2, 1)
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

    for dtype in ['crispr', 'gexp']:
        trg.drugs_ppi(dtype)
        plt.gcf().set_size_inches(2.5, 2.5)
        plt.savefig(f'reports/target_benchmark_ppi_distance_{dtype}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

        trg.drugs_ppi_countplot(dtype)
        plt.gcf().set_size_inches(2.5, 2.5)
        plt.savefig(f'reports/target_benchmark_ppi_distance_{dtype}_countplot.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    #
    drugs_notarget = [
        ('Olaparib', ['STAG1', 'LIG1', 'FLI1', 'PARP1']),
        ('Talazoparib', ['PCGF5', 'XRCC1', 'RHNO1', 'LIG1', 'PARP1', 'PARP2'])
    ]

    for drug, genes in drugs_notarget:
        print(drug, genes)

        trg.drug_notarget_barplot(drug, genes)

        plt.gcf().set_size_inches(1.5, 1)
        plt.savefig(f'reports/target_benchmark_drug_notarget_{drug}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # Single feature examples
    dgs = [('Alpelisib', 'PIK3CA'), ('Nutlin-3a (-)', 'MDM2'), ('MCL1_1284', 'MCL1'), ('MCL1_1284', 'MARCH5')]

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

        g = DTracePlot.plot_corrplot(
            'crispr', 'drug', 'Institute', plot_df, add_hline=False, add_vline=False, annot_text=annot_text
        )

        g.ax_joint.axhline(y=dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

        g.set_axis_labels(f'{dg[1]} (scaled log2 FC)', f'{dg[0]} (ln IC50)')

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(f'reports/association_drug_scatter_{dg[0]}_{dg[1]}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # Drug ~ Gexp
    dgs = [('Nutlin-3a (-)', 'MDM2'), ('Poziotinib', 'ERBB2'), ('Afatinib', 'ERBB2'), ('WEHI-539', 'BCL2L1')]
    for dg in dgs:
        assoc = trg.lmm_drug[(trg.lmm_drug['DRUG_NAME'] == dg[0]) & (trg.lmm_drug['GeneSymbol'] == dg[1])].iloc[0]

        drug = tuple(assoc[DrugResponse.DRUG_COLUMNS])

        dmax = np.log(trg.datasets.drespo_obj.maxconcentration[drug])
        annot_text = f"Beta={assoc['beta']:.2g}, FDR={assoc['fdr']:.1e}"

        plot_df = pd.concat([
            trg.datasets.drespo.loc[drug].rename('drug'),
            trg.datasets.gexp.loc[dg[1]].rename('crispr'),
        ], axis=1, sort=False).dropna()
        plot_df['Institute'] = 'Sanger'

        g = DTracePlot.plot_corrplot('crispr', 'drug', 'Institute', plot_df, add_hline=True, annot_text=annot_text)

        g.ax_joint.axhline(y=dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

        g.set_axis_labels(f'{dg[1]} (voom)', f'{dg[0]} (ln IC50)')

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(f'reports/association_drug_gexp_scatter_{dg[0]}_{dg[1]}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # CRISPR gene pair corr
    for gene_x, gene_y in [('MARCH5', 'MCL1'), ('SHC1', 'EGFR')]:
        plot_df = pd.concat([
            trg.datasets.crispr.loc[gene_x].rename(gene_x),
            trg.datasets.crispr.loc[gene_y].rename(gene_y),
            trg.datasets.crispr_obj.institute.rename('Institute'),
        ], axis=1, sort=False).dropna()

        g = DTracePlot().plot_corrplot(gene_x, gene_y, 'Institute', plot_df, add_hline=True)

        g.set_axis_labels(f'{gene_x} (scaled log2 FC)', f'{gene_y} (scaled log2 FC)')

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(f'reports/association_scatter_{gene_x}_{gene_y}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # PPI
    ppi_examples = [
        ('Nutlin-3a (-)', .4, 1, ['RPL37', 'UBE3B']),
        ('AZD3759', .3, 1, None)
    ]
    # d, t, o = ('Nutlin-3a (-)', .4, 1, ['RPL37', 'UBE3B'])
    for d, t, o, e in ppi_examples:
        graph = PPI.plot_ppi(d, trg.lmm_drug, ppi, corr_thres=t, norder=o, fdr=0.05, exclude_nodes=e)
        graph.write_pdf(f'reports/association_ppi_{d}.pdf')
