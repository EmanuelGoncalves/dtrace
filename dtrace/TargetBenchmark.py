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
from matplotlib.lines import Line2D
from Associations import Association
from DataImporter import DrugResponse, PPI
from scipy.stats import ttest_ind, mannwhitneyu, gmean


class TargetBenchmark(DTracePlot):
    DRUG_TARGETS_COLORS = {
        '#8ebadb': {'RAF1', 'BRAF'}, '#5f9ecc': {'MAPK1', 'MAPK3'}, '#3182bd': {'MAP2K1', 'MAP2K2'},
        '#f2a17a': {'PIK3CA', 'PIK3CB'}, '#ec7b43': {'AKT1', 'AKT2', 'AKT3'}, '#e6550d': {'MTOR'},
        '#6fc088': {'EGFR'}, '#31a354': {'IGF1R'},
        '#b2acd3': {'CHEK1', 'CHEK2'}, '#938bc2': {'ATR'}, '#756bb1': {'WEE1', 'TERT'},
        '#eeaad3': {'BTK'}, '#e78ac3': {'SYK'},
        '#66c2a5': {'PARP1'},
        '#fedf57': {'BCL2', 'BCL2L1'}, '#fefb57': {'MCL1'},
        '#636363': {'GLS'},
        '#dd9a00': {'AURKA', 'AURKB'},
        '#bc80bd': {'BRD2', 'BRD4', 'BRD3'},
        '#983539': {'JAK1', 'JAK2', 'JAK3'},
        '#ffffff': {'No target'}, '#e1e1e1': {'Other target'}, '#bbbbbb': {'Multiple targets'}
    }

    def __init__(self, fdr=.1, dtype='ic50', lmm_drug=None, lmm_drug_gexp=None, lmm_drug_genomic=None):
        self.fdr = fdr
        self.dtype = dtype

        # Imports
        self.datasets = Association(dtype_drug=dtype)

        # Associations
        if lmm_drug is None:
            self.lmm_drug = pd.read_csv(f'data/drug_lmm_regressions_{self.dtype}.csv.gz')
        else:
            self.lmm_drug = lmm_drug

        if lmm_drug_gexp is None:
            self.lmm_drug_gexp = pd.read_csv(f'data/drug_lmm_regressions_{self.dtype}_gexp.csv.gz')
        else:
            self.lmm_drug_gexp = lmm_drug_gexp

        if lmm_drug_genomic is None:
            self.lmm_drug_genomic = pd.read_csv(f'data/drug_lmm_regressions_{self.dtype}_genomic.csv.gz')
        else:
            self.lmm_drug_genomic = lmm_drug_genomic

        # Define sets of drugs
        self.df_genes = set(self.lmm_drug['GeneSymbol'])
        self.d_targets = self.datasets.drespo_obj.get_drugtargets(by='Name')
        self.d_targets_id = self.datasets.drespo_obj.get_drugtargets(by='id')

        self.drugs_all = set(self.lmm_drug['DRUG_NAME'])
        self.drugs_signif = {d for d in self.lmm_drug.query(f'fdr < {self.fdr}')['DRUG_NAME']}
        self.drugs_not_signif = {d for d in self.lmm_drug.query(f'fdr > {self.fdr}')['DRUG_NAME'] if d not in self.drugs_signif}

        self.drugs_annot = {d for d in self.drugs_all if d in self.d_targets}
        self.drugs_tested = {d for d in self.drugs_annot if len(self.d_targets[d].intersection(self.df_genes)) > 0}
        self.drugs_tested_signif = {d for d in self.lmm_drug.query(f'fdr < {self.fdr}')['DRUG_NAME'] if d in self.drugs_tested}
        self.drugs_tested_correct = {d for d in self.lmm_drug.query(f"fdr < {self.fdr} & target == 'T'")['DRUG_NAME'] if d in self.drugs_tested_signif}

        # -
        self.ppi_order = ['T', '1', '2', '3', '4', '5+', '-']

        df = self.lmm_drug.query(f'fdr < {self.fdr}')
        df = df[df['DRUG_NAME'].isin(self.drugs_tested_signif)]

        d_signif_ppi = []
        for d in self.drugs_tested_signif:
            df_ppi = df[df['DRUG_NAME'] == d].sort_values('fdr')
            df_ppi['target'] = pd.Categorical(df_ppi['target'], self.ppi_order)
            d_signif_ppi.append(df_ppi.sort_values('target').iloc[0])

        self.d_signif_ppi = pd.DataFrame(d_signif_ppi)
        self.d_signif_ppi['target'] = pd.Categorical(self.d_signif_ppi['target'], self.ppi_order)
        self.d_signif_ppi = self.d_signif_ppi.set_index('DRUG_NAME').sort_values('target')

        self.d_signif_ppi_count = self.d_signif_ppi['target'].value_counts()[self.ppi_order]

        super().__init__()

    def get_drug_target_color(self, drug_id):
        if drug_id not in self.d_targets_id:
            return '#ffffff'

        drug_targets = [
            c for c in self.DRUG_TARGETS_COLORS
            if len(self.d_targets_id[drug_id].intersection(self.DRUG_TARGETS_COLORS[c])) > 0
        ]

        if len(drug_targets) == 0:
            return '#e1e1e1'

        elif len(drug_targets) == 1:
            return drug_targets[0]

        else:
            return '#bbbbbb'

    def boxplot_kinobead(self):
        dmap = pd.read_csv('data/klaeger_et_al_catds_most_potent.csv').set_index('Drug')['name'].dropna()

        catds_m = pd.read_csv('data/klaeger_et_al_catds.csv', index_col=0)
        catds_m = catds_m[catds_m.index.isin(dmap.index)]
        catds_m.index = dmap[catds_m.index].values

        catds_m = catds_m.unstack().dropna().reset_index()
        catds_m.columns = ['target', 'drug', 'catds']

        catds_m['is_target'] = [int(t in self.d_targets[d]) for d, t in catds_m[['drug', 'target']].values]

        lmm_drug_subset = self.lmm_drug[self.lmm_drug['DRUG_NAME'].isin(catds_m['drug'])]
        catds_m['lmm_pval'] = lmm_drug_subset.groupby(['GeneSymbol', 'DRUG_NAME'])['fdr'].min()[catds_m.set_index(['target', 'drug']).index].values
        catds_m['lmm_signif'] = catds_m['lmm_pval'].apply(lambda v: 'Yes' if v < self.fdr else 'No')

        catds_m = catds_m.sort_values('catds').dropna()

        #
        t, p = mannwhitneyu(
            catds_m.query("lmm_signif == 'Yes'")['catds'], catds_m.query("lmm_signif == 'No'")['catds']
        )

        # Plot
        order = ['No', 'Yes']
        pal = {'No': self.PAL_DTRACE[1], 'Yes': self.PAL_DTRACE[0]}

        ax = sns.boxplot(
            catds_m['catds'], catds_m['lmm_signif'], palette=pal, linewidth=.3, fliersize=1.5, order=order,
            flierprops=self.FLIERPROPS, showcaps=False, orient='h'
        )

        ax.scatter(gmean(catds_m.query("lmm_signif == 'Yes'")['catds']), 1, marker='+', lw=.3, color='k', s=3)
        ax.scatter(gmean(catds_m.query("lmm_signif == 'No'")['catds']), 0, marker='+', lw=.3, color='k', s=3)

        ax.set_xscale('log')

        ax.set_title(f'Drug-Gene association\n(Mann-Whitney p-value={p:.2g})')
        ax.set_ylabel('Significant')
        ax.set_xlabel('Kinobeads selectivity (apparent pKd [nM])')

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

        plt.axvline(0, c=self.PAL_DTRACE[1], lw=.3, ls='-', zorder=0)

        plt.xlabel('Association beta')
        plt.ylabel('Density')

        plt.legend(prop={'size': 6}, loc=2, frameon=False)

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

        plt.xlabel('Association p-value')
        plt.ylabel('Density')

        plt.legend(prop={'size': 6}, frameon=False)

    def countplot_drugs(self):
        plot_df = pd.Series({
            'All': self.datasets.drespo.shape[0], 'Unique': len(self.drugs_all), 'Annotated': len(self.drugs_annot),
            'Target tested': len(self.drugs_tested),
        }).rename('count').reset_index()

        plt.barh(plot_df.index, plot_df['count'], color=self.PAL_DTRACE[2], linewidth=0)

        for y, c in enumerate(plot_df['count']):
            plt.text(c - 3, y, str(c), va='center', ha='right', fontsize=5, zorder=10, color='white')

        plt.grid(axis='x', lw=.3, color=self.PAL_DTRACE[1], zorder=0)

        plt.yticks(plot_df.index, plot_df['index'])
        plt.xlabel('Number of drugs')
        plt.ylabel('')
        plt.title('')

    def countplot_drugs_significant(self):
        plot_df = self.d_signif_ppi['target'].value_counts()[reversed(self.ppi_order)].reset_index()

        plt.barh(plot_df.index, plot_df['target'], color=self.PAL_DTRACE[2], linewidth=0)

        for y, c in enumerate(plot_df['target']):
            plt.text(c - 3, y, str(c), va='center', ha='right', fontsize=5, zorder=10, color='white')

        plt.grid(axis='x', lw=.3, color=self.PAL_DTRACE[1], zorder=0)

        plt.yticks(plot_df.index, plot_df['index'])
        plt.xlabel('Number of drugs')
        plt.ylabel('')
        plt.title('')

    def pichart_drugs_significant(self):
        plot_df = self.d_signif_ppi['target'].value_counts().to_dict()
        plot_df['X'] = len([d for d in self.drugs_tested if d not in self.drugs_tested_signif])
        plot_df = pd.Series(plot_df)[self.ppi_order + ['X']]

        colors = [self.PAL_DTRACE[0]] + DTracePlot.get_palette_continuous(5, self.PAL_DTRACE[2]) + [self.PAL_DTRACE[3],
                                                                                                    self.PAL_DTRACE[1]]
        explode = [0, 0, 0, 0, 0, 0, 0, .1]

        plt.pie(
            plot_df, labels=plot_df.index, explode=explode, colors=colors, autopct='%1.1f%%', shadow=False,
            startangle=90,
            textprops={'fontsize': 7}, wedgeprops=dict(linewidth=0)
        )

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

        plt.grid(axis='y', lw=.3, color=self.PAL_DTRACE[1], zorder=0)

        plt.xlabel('Associated gene position in PPI')
        plt.ylabel('Number of associations')
        plt.title('Significant associations\n(adj. p-value < 10%)')

    def drugs_ppi_countplot_background(self, dtype='crispr'):
        if dtype == 'crispr':
            df = self.lmm_drug[self.lmm_drug['DRUG_NAME'].isin(self.drugs_tested)]

        elif dtype == 'gexp':
            df = self.lmm_drug_gexp[self.lmm_drug['DRUG_NAME'].isin(self.drugs_tested)]

        order = ['T', '1', '2', '3', '4', '5+', '-']

        pal = dict(zip(*(
            order,
            [self.PAL_DTRACE[0]] + QCplot.get_palette_continuous(len(order) - 2, color=self.PAL_DTRACE[2]) + [self.PAL_DTRACE[3]]
        )))

        plot_df = df['target'].value_counts().rename('count').reset_index()

        sns.barplot('index', 'count', data=plot_df, order=order, palette=pal)

        plt.grid(axis='y', lw=.3, color=self.PAL_DTRACE[1], zorder=0)

        plt.xlabel('Associated gene position in PPI')
        plt.ylabel('Number of associations')
        plt.title('All associations')

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

            axs[irow].axes.get_xaxis().set_ticks([])
            axs[irow].set_ylabel('Drug association\n(-log10 p-value)')

        plt.gcf().set_size_inches(10, 8)

    def manhattan_plot(self, n_genes=20, is_gexp=False):
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

        f, axs = plt.subplots(1, len(chrms), sharex='none', sharey='row', gridspec_kw=dict(wspace=0))
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

            if i == 0:
                sns.despine(ax=axs[i], right=True, top=False)
                axs[i].set_ylabel('Drug-gene association (-log10 p-value)')

            elif i == (len(chrms) - 1):
                sns.despine(ax=axs[i], left=True, right=False, top=False)

            else:
                sns.despine(ax=axs[i], left=True, right=True, top=False)
                axs[i].yaxis.set_ticks_position('none')

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
            align='center', zorder=1, linewidth=0
        )

        ax.barh(
            df.query("target == 'T'").index, -np.log10(df.query("target == 'T'")['pval']), .8,
            color=self.PAL_DTRACE[0],
            align='center', zorder=1, linewidth=0
        )

        for i, (y, t, f) in df[['pval', 'target', 'fdr']].iterrows():
            ax.text(-np.log10(y) - 0.1, i, t, color='white', ha='right', va='center', fontsize=6, zorder=10)

            if f < self.fdr:
                ax.text(-np.log10(y) + 0.1, i, '*', color=self.PAL_DTRACE[2], ha='left', va='center', fontsize=6, zorder=10)

        ax.grid(axis='x', lw=.3, color=self.PAL_DTRACE[1], zorder=0)

        ax.set_yticks(df.index)
        ax.set_yticklabels(df['GeneSymbol'])

        ax.set_xlabel('Drug association (-log10 p-value)')
        ax.set_title(drug)

    def lmm_betas_clustermap(self, matrix_betas):
        matrix_betas_corr = matrix_betas.T.corr()

        row_cols = pd.Series({d: self.get_drug_target_color(d[0]) for d in matrix_betas_corr.index})
        col_cols = pd.Series({d: self.get_drug_target_color(d[0]) for d in matrix_betas_corr.columns})

        sns.clustermap(
            matrix_betas_corr, xticklabels=False, yticklabels=False, col_colors=col_cols, row_colors=row_cols,
            cmap='mako'
        )

    def lmm_betas_clustermap_legend(self):
        labels = {
            ';'.join(self.DRUG_TARGETS_COLORS[c]): Line2D([0], [0], color=c, lw=4) for c in self.DRUG_TARGETS_COLORS
        }

        plt.legend(labels.values(), labels.keys(), bbox_to_anchor=(.5, 1.), frameon=False)

    def signif_essential_heatmap(self):
        ess_genes = self.datasets.crispr_obj.import_sanger_essential_genes()

        df_ess = pd.DataFrame({d: {
            'Target correct': 'Yes' if d in self.drugs_tested_correct else 'No',
            'Essential gene': 'Yes' if len(self.d_targets[d].intersection(ess_genes)) > 0 else 'No'
        } for d in self.drugs_tested}).T

        df_ess = pd.pivot_table(
            df_ess.reset_index(), index='Target correct', columns='Essential gene', values='index', aggfunc='count'
        )

        sns.heatmap(df_ess, annot=True, cbar=False, fmt='.0f', cmap='Greys')

    def signif_per_screen(self):
        df = self.lmm_drug.groupby(self.datasets.drespo_obj.DRUG_COLUMNS).first().reset_index()
        df = df[df['DRUG_NAME'].isin(self.drugs_tested)]
        df['signif'] = (df['fdr'] < self.fdr).astype(int)
        df = df.groupby('VERSION')['signif'].agg(['count', 'sum']).reset_index()
        df['perc'] = df['sum'] / df['count'] * 100

        plt.bar(df.index, df['count'], color=self.PAL_DTRACE[1], label='All')
        plt.bar(df.index, df['sum'], color=self.PAL_DTRACE[2], label='Signif.')

        for x, (y, p) in enumerate(df[['sum', 'perc']].values):
            plt.text(x, y + 1, f'{p:.1f}%', va='bottom', ha='center', fontsize=5, zorder=10, color=self.PAL_DTRACE[2])

        plt.grid(axis='y', lw=.3, color=self.PAL_DTRACE[1], zorder=0)

        plt.xticks(df.index, df['VERSION'])
        plt.ylabel('Number of drugs')
        plt.legend(prop={'size': 4}, frameon=False)

    def signif_genomic_markers(self):
        plot_df = pd.concat([
            self.lmm_drug.groupby('DRUG_NAME')['fdr'].min().rename('crispr_fdr'),
            self.lmm_drug_genomic.groupby('DRUG_NAME')['fdr'].min().rename('genomic_fdr')
        ], axis=1).reset_index()
        plot_df = plot_df[plot_df['DRUG_NAME'].isin(self.drugs_tested)]

        plot_df['crispr_signif'] = (plot_df['crispr_fdr'] < self.fdr).astype(int).replace(1, 'Yes').replace(0, 'No')
        plot_df['genomic_signif'] = (plot_df['genomic_fdr'] < self.fdr).astype(int).replace(1, 'Yes').replace(0, 'No')

        plot_df = pd.pivot_table(
            plot_df.reset_index(), index='crispr_signif', columns='genomic_signif', values='DRUG_NAME', aggfunc='count'
        )

        g = sns.heatmap(plot_df, annot=True, cbar=False, fmt='.0f', cmap='Greys')

        g.set_xlabel('Genomic marker')
        g.set_ylabel('CRISPR association')
        g.set_title('Drug association')


if __name__ == '__main__':
    # Import target benchmark
    trg = TargetBenchmark(fdr=.1)

    ppi = PPI().build_string_ppi(score_thres=900)
    ppi = PPI.ppi_corr(ppi, trg.datasets.crispr)

    # - Non-significant associations description
    #
    trg.signif_essential_heatmap()
    plt.gcf().set_size_inches(1, 1)
    plt.savefig('reports/target_benchmark_signif_essential_heatmap.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    trg.signif_per_screen()
    plt.gcf().set_size_inches(0.75, 1.5)
    plt.savefig('reports/target_benchmark_significant_by_screen.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    trg.signif_genomic_markers()
    plt.gcf().set_size_inches(1, 1)
    plt.savefig('reports/target_benchmark_signif_genomic_heatmap.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # -
    trg.countplot_drugs()
    plt.gcf().set_size_inches(2, 0.75)
    plt.savefig('reports/target_benchmark_association_countplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.countplot_drugs_significant()
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/target_benchmark_association_signif_countplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.pichart_drugs_significant()
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/target_benchmark_association_signif_piechart.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.boxplot_kinobead()
    plt.gcf().set_size_inches(2.5, .75)
    plt.savefig(f'reports/target_benchmark_kinobeads.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.top_associations_barplot()
    plt.savefig('reports/target_benchmark_associations_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.manhattan_plot(n_genes=20)
    plt.gcf().set_size_inches(7, 3)
    plt.savefig('reports/drug_associations_manhattan.png', bbox_inches='tight', transparent=True, dpi=600)
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

    trg.drugs_ppi_countplot_background()
    plt.gcf().set_size_inches(2.5, 2.5)
    plt.savefig(f'reports/target_benchmark_ppi_distance_{dtype}_countplot_bkg.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Drug targets
    drugs_notarget = [
        ('Olaparib', ['STAG1', 'LIG1', 'FLI1', 'PARP1']),
        ('Talazoparib', ['PCGF5', 'XRCC1', 'RHNO1', 'LIG1', 'PARP1', 'PARP2'])
    ]

    for drug, genes in drugs_notarget:
        trg.drug_notarget_barplot(drug, genes)

        plt.gcf().set_size_inches(2, 1.5)
        plt.savefig(f'reports/target_benchmark_drug_notarget_{drug}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # - Single feature examples
    dgs = [
        ('Alpelisib', 'PIK3CA'), ('Nutlin-3a (-)', 'MDM2'), ('MCL1_1284', 'MCL1'), ('MCL1_1284', 'MARCH5'),
        ('Venetoclax', 'BCL2'), ('AZD4320', 'BCL2'), ('Volasertib', 'PLK1'), ('Rigosertib', 'PLK1')
    ]

    # dg = ('Rigosertib', 'PLK1')
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

        plt.gcf().set_size_inches(2, 2)
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

    # - Betas clustermap
    betas_crispr = pd.pivot_table(
        trg.lmm_drug.query("VERSION == 'RS'"), index=['DRUG_ID', 'DRUG_NAME'], columns='GeneSymbol', values='beta'
    )

    #
    trg.lmm_betas_clustermap(betas_crispr)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig('reports/target_benchmark_clustermap_betas_crispr.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    #
    trg.lmm_betas_clustermap_legend()
    plt.savefig('reports/target_benchmark_clustermap_betas_crispr_legend.pdf', bbox_inches='tight')
    plt.close('all')

    # - Export "all" drug scatter
    for d in trg.datasets.drespo.index:
        print(d)
        dassoc = trg.lmm_drug[
            (trg.lmm_drug['DRUG_NAME'] == d[1]) & (trg.lmm_drug['DRUG_ID'] == d[0]) & (trg.lmm_drug['VERSION'] == d[2])
        ]

        dselected = dassoc.iloc[:5]
        if 'T' in dassoc['target'].values:
            dselected = pd.concat([dselected, dassoc[dassoc['target'] == 'T']])

        dmax = np.log(trg.datasets.drespo_obj.maxconcentration[d])

        for i, assoc in dselected.iterrows():
            annot_text = f"Beta={assoc['beta']:.2g}, FDR={assoc['fdr']:.1e}"

            plot_df = pd.concat([
                trg.datasets.drespo.loc[d].rename('drug'),
                trg.datasets.crispr.loc[assoc['GeneSymbol']].rename('crispr'),
                trg.datasets.crispr_obj.institute.rename('Institute'),
            ], axis=1, sort=False).dropna()

            g = DTracePlot.plot_corrplot(
                'crispr', 'drug', 'Institute', plot_df, add_hline=False, add_vline=False, annot_text=annot_text
            )

            g.ax_joint.axhline(y=dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

            g.set_axis_labels(f"{assoc['GeneSymbol']} (scaled log2 FC)", f"{d[1]} (ln IC50, {d[0]}, {d[2]})")

            plot_name = "reports/association_drug_scatters/"
            plot_name += f"{d[1].replace('/', '')} {d[0]} {d[2]} FDR{(assoc['fdr']*100):.2f} {assoc['target']} {assoc['GeneSymbol']}.pdf"

            plt.gcf().set_size_inches(1.5, 1.5)
            plt.savefig(plot_name, bbox_inches='tight', transparent=True)
            plt.close('all')
