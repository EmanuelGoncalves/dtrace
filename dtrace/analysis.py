#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
from natsort import natsorted
from crispy.utils import Utils
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE
from importer import DrugResponse
from associations import Association
from sklearn.preprocessing import StandardScaler


class SingleLMMAnalysis(object):
    @classmethod
    def manhattan_plot(cls, lmm_drug, fdr=.1, n_genes=20):
        # Import gene genomic coordinates from CRISPR-Cas9 library
        crispr_lib = Utils.get_crispr_lib().groupby('gene').agg({'start': 'min', 'chr': 'first'})

        # Plot data-frame
        df = lmm_drug.copy()
        df = df.assign(pos=crispr_lib.loc[df['GeneSymbol'], 'start'].values)
        df = df.assign(chr=crispr_lib.loc[df['GeneSymbol'], 'chr'].apply(lambda v: v.replace('chr', '')).values)
        df = df.sort_values(['chr', 'pos'])

        # Most frequently associated genes
        top_genes = lmm_drug.groupby('GeneSymbol')['fdr'].min().sort_values().head(n_genes)
        top_genes_pal = dict(zip(*(top_genes.index, sns.color_palette('tab20', n_colors=n_genes).as_hex())))

        # Plot
        chrms = set(df['chr'])
        label_fdr = 'Significant'

        f, axs = plt.subplots(1, len(chrms), sharex='none', sharey='row', gridspec_kw=dict(wspace=.05))
        for i, name in enumerate(natsorted(chrms)):
            df_group = df[df['chr'] == name]

            # Plot non-significant
            df_nonsignif = df_group.query(f'fdr >= {fdr}')
            axs[i].scatter(df_nonsignif['pos'], -np.log10(df_nonsignif['pval']), c=Plot.PAL_DTRACE[(i % 2) + 1], s=2)

            # Plot significant
            df_signif = df_group.query(f'fdr < {fdr}')
            df_signif = df_signif[~df_signif['GeneSymbol'].isin(top_genes.index)]
            axs[i].scatter(df_signif['pos'], -np.log10(df_signif['pval']), c=Plot.PAL_DTRACE[0], s=2, zorder=3,
                           label=label_fdr)

            # Plot significant associations of top frequent genes
            df_genes = df_group.query(f'fdr < {fdr}')
            df_genes = df_genes[df_genes['GeneSymbol'].isin(top_genes.index)]
            for pos, pval, gene in df_genes[['pos', 'pval', 'GeneSymbol']].values:
                axs[i].scatter(pos, -np.log10(pval), c=top_genes_pal[gene], s=6, zorder=4, label=gene, marker='2',
                               lw=.75)

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
        plt.legend(list(zip(*(by_label)))[1], list(zip(*(by_label)))[0], loc='center left', bbox_to_anchor=(1.01, 0.5),
                   prop={'size': 5}, frameon=False)

    @classmethod
    def beta_histogram(cls, lmm_drug):
        kde_kws = dict(cut=0, lw=1, zorder=1, alpha=.8)
        hist_kws = dict(alpha=.4, zorder=1, linewidth=0)

        sns.distplot(
            lmm_drug.query("target != 'T'")['beta'], color=Plot.PAL_DTRACE[2], kde_kws=kde_kws, hist_kws=hist_kws,
            label='All', bins=30
        )

        sns.distplot(
            lmm_drug.query("target == 'T'")['beta'], color=Plot.PAL_DTRACE[0], kde_kws=kde_kws, hist_kws=hist_kws,
            label='Target', bins=30
        )

        sns.despine(right=True, top=True)

        plt.axvline(0, c=Plot.PAL_DTRACE[1], lw=.3, ls='-', zorder=0)

        plt.xlabel('Association beta')
        plt.ylabel('Density')

        plt.legend(prop={'size': 6}, loc=2, frameon=False)

    @classmethod
    def recapitulated_drug_targets_barplot(cls, lmm_drug, fdr=.1):
        dcols = DrugResponse.DRUG_COLUMNS
        d_targets, df_genes = DrugResponse().get_drugtargets(), set(lmm_drug['GeneSymbol'])

        # Count number of drugs
        d_all = {
            tuple(i) for i in lmm_drug[dcols].values
        }

        d_annot = {
            tuple(i) for i in d_all if i[0] in d_targets
        }

        d_tested = {
            tuple(i) for i in d_annot if len(d_targets[i[0]].intersection(df_genes)) > 0
        }

        d_tested_signif = {
            tuple(i) for i in lmm_drug.query(f'fdr < {fdr}')[dcols].values if tuple(i) in d_tested
        }

        d_tested_correct = {
            tuple(i) for i in lmm_drug.query(f"fdr < {fdr} & target == 'T'")[dcols].values if
        tuple(i) in d_tested_signif
        }

        # Build dataframe
        plot_df = pd.DataFrame(dict(
            names=['All', 'w/Target', 'w/Tested target', 'w/Signif tested target', 'w/Correct target'],
            count=list(map(len, [d_all, d_annot, d_tested, d_tested_signif, d_tested_correct]))
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

    @classmethod
    def top_associations_barplot(cls, lmm_drug, fdr_line=0.1, ntop=60, n_cols=16):
        # Filter for signif associations
        df = lmm_drug \
            .query('fdr < {}'.format(fdr_line)) \
            .sort_values('fdr') \
            .groupby(['DRUG_NAME', 'GeneSymbol']) \
            .first() \
            .sort_values('fdr') \
            .reset_index()
        df = df.assign(logpval=-np.log10(df['pval']).values)
        df = df.replace({'target': {'>=3': '3+'}})

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
            axs[irow].set_ylabel('Drug-gene association\n(-log10 p-value)')

        plt.gcf().set_size_inches(13, 6)

    @staticmethod
    def boxplot_kinobead(lmm_drug, fdr_thres=.1, ax=None):
        d_targets = DrugResponse.get_drugtargets()

        # Build data-frame
        drug_id_fdr = lmm_drug.groupby('DRUG_ID_lib')['fdr'].min()

        def from_ids_to_minfdr(ids):
            if str(ids).lower() == 'nan':
                return np.nan
            else:
                dids = list(map(int, ids.split(';')))
                dids = [i for i in dids if i in drug_id_fdr.index]

                if len(dids) == 0:
                    return np.nan
                else:
                    return drug_id_fdr.loc[dids].min()

        def targets_from_ids(ids):
            if str(ids).lower() == 'nan':
                return np.nan
            else:
                dids = list(map(int, ids.split(';')))
                dids = [i for i in dids if i in d_targets]

                if len(dids) == 0:
                    return np.nan
                else:
                    return '; '.join({g for i in dids for g in d_targets[i]})

        catds = pd.read_csv('data/klaeger_et_al_catds_most_potent.csv')
        catds = catds.assign(fdr=[from_ids_to_minfdr(i) for i in catds['ids']])
        catds = catds.assign(
            signif=[('NA' if np.isnan(i) else ('Yes' if i < fdr_thres else 'No')) for i in catds['fdr']])
        catds = catds.assign(targets=[targets_from_ids(i) for i in catds['ids']])

        t, p = ttest_ind(
            catds[catds['signif'] == 'No']['CATDS_most_potent'],
            catds[catds['signif'] == 'Yes']['CATDS_most_potent'],
            equal_var=False
        )
        print(p)

        # Plot
        if ax is None:
            ax = plt.gca()

        order = ['No', 'Yes', 'NA']
        pal = {'No': Plot.PAL_DTRACE[2], 'Yes': Plot.PAL_DTRACE[0], 'NA': Plot.PAL_DTRACE[1]}

        sns.boxplot(catds['signif'], catds['CATDS_most_potent'], notch=True, palette=pal, linewidth=.3, fliersize=1.5, order=order, ax=ax)
        sns.swarmplot(catds['signif'], catds['CATDS_most_potent'], palette=pal, linewidth=.3, size=2, order=order,
                      ax=ax)
        ax.axhline(0.5, lw=.3, c=Plot.PAL_DTRACE[1], ls='-', alpha=.8, zorder=0)

        sns.despine(top=True, right=True, ax=ax)

        ax.set_ylim((-0.1, 1.1))

        ax.set_xlabel('Drug has a significant\nCRISPR-Cas9 association')
        ax.set_ylabel('Selectivity[$CATDS_{most\ potent}$]')


class SingleLMMTSNE(object):
    DRUG_TARGETS_HUE = [
        ('#3182bd', [{'RAF1', 'BRAF'}, {'MAPK1', 'MAPK3'}, {'MAP2K1', 'MAP2K2'}]),
        ('#e6550d', [{'PIK3CA', 'PIK3CB'}, {'AKT1', 'AKT2', 'AKT3'}, {'MTOR'}]),
        ('#31a354', [{'EGFR'}, {'IGF1R'}]),
        ('#756bb1', [{'CHEK1', 'CHEK2'}, {'ATR'}, {'WEE1', 'TERT'}]),
        ('#e78ac3', [{'BTK'}, {'SYK'}]),
        ('#66c2a5', [{'PARP1'}]),
        ('#fdd10f', [{'BCL2', 'BCL2L1'}]),
        ('#636363', [{'GLS'}]),
        ('#92d2df', [{'MCL1'}]),
        ('#dd9a00', [{'AURKA', 'AURKB'}]),
        ('#bc80bd', [{'BRD2', 'BRD4', 'BRD3'}]),
        ('#983539', [{'JAK1', 'JAK2', 'JAK3'}]),
    ]

    DRUG_TARGETS_HUE = [
        (sns.light_palette(c, n_colors=len(s) + 1, reverse=True).as_hex()[:-1], s) for c, s in DRUG_TARGETS_HUE
    ]

    def __init__(self, lmm_dsingle, perplexity=15, learning_rate=250, n_iter=2000, fdr=.1):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.fdr = fdr

        self.dtargets = DrugResponse().get_drugtargets()

        self.tsnes = self.drug_betas_tsne(lmm_dsingle)

    def drug_betas_tsne(self, lmm_dsingle):
        # Drugs into
        drugs = {tuple(i) for i in lmm_dsingle[DrugResponse.DRUG_COLUMNS].values}
        drugs_annot = {tuple(i) for i in drugs if i[0] in self.dtargets}
        drugs_screen = {v: {d for d in drugs if d[2] == v} for v in ['v17', 'RS']}

        # Build drug association beta matrix
        betas = pd.pivot_table(lmm_drug, index=DrugResponse.DRUG_COLUMNS, columns='GeneSymbol', values='beta')
        betas = betas.loc[list(drugs)]

        # TSNE
        tsnes = []
        for s in drugs_screen:
            tsne_df = betas.loc[list(drugs_screen[s])]
            tsne_df = pd.DataFrame(StandardScaler().fit_transform(tsne_df.T).T, index=tsne_df.index,
                                   columns=tsne_df.columns)

            tsne = TSNE(
                perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter, init='pca'
            ).fit_transform(tsne_df)

            tsne = pd.DataFrame(tsne, index=tsne_df.index, columns=['P1', 'P2']).reset_index()
            tsne = tsne.assign(
                target=['Yes' if tuple(i) in drugs_annot else 'No' for i in tsne[DrugResponse.DRUG_COLUMNS].values])

            tsnes.append(tsne)

        tsnes = pd.concat(tsnes)
        tsnes = tsnes.assign(name=[';'.join(map(str, i[1:])) for i in tsnes[DrugResponse.DRUG_COLUMNS].values])

        # Annotate compound replicated
        rep_names = tsnes['name'].value_counts()
        rep_names = set(rep_names[rep_names > 1].index)
        tsnes = tsnes.assign(rep=[i if i in rep_names else 'NA' for i in tsnes['name']])

        # Annotate targets
        tsnes = tsnes.assign(
            targets=[';'.join(self.dtargets[i]) if i in self.dtargets else '' for i in tsnes[DrugResponse.DRUG_COLUMNS[0]]])

        # Annotate significant
        d_signif = {tuple(i) for i in lmm_drug.query(f'fdr < {self.fdr}')[DrugResponse.DRUG_COLUMNS].values}
        tsnes = tsnes.assign(has_signif=['Yes' if tuple(i) in d_signif else 'No' for i in tsnes[DrugResponse.DRUG_COLUMNS].values])

        return tsnes

    def drug_beta_tsne(self, hueby):
        if hueby == 'signif':
            pal = {'No': Plot.PAL_DTRACE[2], 'Yes': Plot.PAL_DTRACE[0]}

            g = sns.FacetGrid(
                self.tsnes, col='VERSION', hue='has_signif', palette=pal, hue_order=['No', 'Yes'], sharey=False,
                sharex=False, legend_out=True, despine=False, size=2, aspect=1
            )

            g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
            g.map(plt.axhline, y=0, ls='-', lw=0.3, c=Plot.PAL_DTRACE[1], alpha=.2, zorder=0)
            g.map(plt.axvline, x=0, ls='-', lw=0.3, c=Plot.PAL_DTRACE[1], alpha=.2, zorder=0)

            g.add_legend(title='Significant', prop=dict(size=4), frameon=False)

        elif hueby == 'target':
            pal = {'No': Plot.PAL_DTRACE[2], 'Yes': Plot.PAL_DTRACE[0]}

            g = sns.FacetGrid(
                self.tsnes, col='VERSION', hue='target', palette=pal, hue_order=['Yes', 'No'], sharey=False,
                sharex=False, legend_out=True, despine=False, size=2, aspect=1
            )

            g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
            g.map(plt.axhline, y=0, ls='-', lw=0.3, c=Plot.PAL_DTRACE[1], alpha=.2, zorder=0)
            g.map(plt.axvline, x=0, ls='-', lw=0.3, c=Plot.PAL_DTRACE[1], alpha=.2, zorder=0)

            g.add_legend(title='Known target?', prop=dict(size=4), frameon=False)

        elif hueby == 'replicates':
            rep_names = set(self.tsnes['rep'])

            pal_v17 = [n for n in rep_names if n.endswith(';v17')]
            pal_v17 = dict(zip(*(pal_v17, sns.color_palette('tab20', n_colors=len(pal_v17)).as_hex())))

            pal_rs = [n for n in rep_names if n.endswith(';RS')]
            pal_rs = dict(zip(*(pal_rs, sns.color_palette('tab20', n_colors=len(pal_rs)).as_hex())))

            pal = {**pal_v17, **pal_rs}
            pal['NA'] = Plot.PAL_DTRACE[1]

            g = sns.FacetGrid(
                self.tsnes, col='VERSION', hue='rep', palette=pal, sharey=False, sharex=False, legend_out=True,
                despine=False, size=2, aspect=1
            )

            g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
            g.map(plt.axhline, y=0, ls='-', lw=0.3, c=Plot.PAL_DTRACE[1], alpha=.2, zorder=0)
            g.map(plt.axvline, x=0, ls='-', lw=0.3, c=Plot.PAL_DTRACE[1], alpha=.2, zorder=0)

            g.add_legend(title='', prop=dict(size=4), frameon=False)

        elif type(hueby) == list:
            sets = [i for l in hueby for i in l[1]]
            labels = [';'.join(i) for l in hueby for i in l[1]]
            colors = [i for l in hueby for i in l[0]]

            pal = dict(zip(*(labels, colors)))
            pal['NA'] = Plot.PAL_DTRACE[1]

            df = self.tsnes.assign(hue=[[i for i, g in enumerate(sets) if g.intersection(t.split(';'))] for t in self.tsnes['targets']])
            df = self.tsnes.assign(hue=[labels[i[0]] if len(i) > 0 else 'NA' for i in df['hue']])

            g = sns.FacetGrid(
                df.query("target == 'Yes'"), col='VERSION', hue='hue', palette=pal, sharey=False, sharex=False,
                legend_out=True, despine=False, size=2, aspect=1
            )

            for i, s in enumerate(['v17', 'RS']):
                ax = g.axes.ravel()[i]
                df_plot = df.query("(target == 'No') & (VERSION == '{}')".format(s))
                ax.scatter(df_plot['P1'], df_plot['P2'], color=Plot.PAL_DTRACE[1], marker='x', lw=0.3, s=5, alpha=0.7,
                           label='No target info')

            g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
            g.map(plt.axhline, y=0, ls='-', lw=0.3, c=Plot.PAL_DTRACE[1], alpha=.2, zorder=0)
            g.map(plt.axvline, x=0, ls='-', lw=0.3, c=Plot.PAL_DTRACE[1], alpha=.2, zorder=0)

            g.add_legend(title='', prop=dict(size=4), label_order=labels + ['NA'] + ['No info'], frameon=False)

        g.set_titles('Screen = {col_name}')

    def drug_v17_id_batch(self):
        hue_order = ['[0, 200[', '[200, 1000[', '[1000, inf.[']

        plot_df = self.tsnes[self.tsnes['VERSION'] == 'v17']

        plot_df = plot_df.assign(
            id_discrete=pd.cut(
                plot_df['DRUG_ID'],
                bins=[0, 200, 1000, np.max(plot_df['DRUG_ID'])],
                labels=hue_order
            )
        )

        discrete_pal = pd.Series(Plot.PAL_DTRACE, index=set(plot_df['id_discrete'])).to_dict()

        g = Plot.plot_corrplot_discrete(
            'P1', 'P2', 'id_discrete', plot_df, discrete_pal=discrete_pal, legend_title='DRUG_ID discretised',
            hue_order=hue_order
        )

        g.set_axis_labels('', '')

        return g


class RobustLMMAnalysis(object):
    @staticmethod
    def count_signif_associations(lmm_robust, fdr=0.1):
        columns = ['DRUG_ID', 'DRUG_NAME', 'VERSION', 'GeneSymbol', 'Genetic']

        d_signif = {tuple(i) for i in lmm_robust.query(f'fdr_drug < {fdr}')[columns].values}
        c_signif = {tuple(i) for i in lmm_robust.query(f'fdr_crispr < {fdr}')[columns].values}
        dc_signif = d_signif.intersection(c_signif)

        # Build dataframe
        plot_df = pd.DataFrame(dict(
            names=['Drug', 'CRISPR', 'Intersection'],
            count=list(map(len, [d_signif, c_signif, dc_signif]))
        )).sort_values('count', ascending=True)
        plot_df = plot_df.assign(y=range(plot_df.shape[0]))

        # Plot
        plt.barh(plot_df['y'], plot_df['count'], color=Plot.PAL_DTRACE[2], linewidth=0)

        sns.despine(right=True, top=True)

        for c, y in plot_df[['count', 'y']].values:
            plt.text(c + 3, y, str(c), va='center', fontsize=5, zorder=10, color=Plot.PAL_DTRACE[2])

        plt.yticks(plot_df['y'], plot_df['names'])
        plt.xlabel('Number of signifcant associations')
        plt.ylabel('')

    @staticmethod
    def genomic_histogram(datasets, ntop=40):
        # Build dataframe
        plot_df = datasets.genomic.drop(['msi_status']).sum(1).rename('count').reset_index()

        plot_df['genes'] = [datasets.genomic_obj.mobem_feature_to_gene(i) for i in plot_df['index']]
        plot_df = plot_df[plot_df['genes'].apply(len) != 0]
        plot_df['genes'] = plot_df['genes'].apply(lambda v: ';'.join(v)).values

        plot_df['type'] = [datasets.genomic_obj.mobem_feature_type(i) for i in plot_df['index']]

        plot_df = plot_df.assign(name=['{} - {}'.format(t, g) for t, g in plot_df[['type', 'genes']].values])

        plot_df = plot_df.sort_values('count', ascending=False).head(ntop)

        # Plot
        order = ['Mutation', 'CN loss', 'CN gain']
        pal = pd.Series(Plot.PAL_DTRACE, index=order).to_dict()

        sns.barplot('count', 'name', 'type', data=plot_df, palette=pal, hue_order=order, dodge=False, saturation=1)

        plt.xlabel('Number of occurrences')
        plt.ylabel('')

        plt.legend(frameon=False)
        sns.despine()

        plt.gcf().set_size_inches(2, .15 * ntop)

    @staticmethod
    def top_robust_features(lmm_robust, ntop=40):
        f, axs = plt.subplots(1, 2, sharex='none', sharey='none', gridspec_kw=dict(wspace=.75))

        order = ['Mutation', 'CN loss', 'CN gain']
        pal = pd.Series(Plot.PAL_DTRACE, index=order).to_dict()

        for i, d in enumerate(['drug', 'crispr']):
            ax = axs[i]
            beta, pval, fdr = f'beta_{d}', f'pval_{d}', f'fdr_{d}'

            feature = 'DRUG_NAME' if d == 'drug' else 'GeneSymbol'

            # Dataframe
            plot_df = lmm_robust.query("Genetic != 'msi_status'")
            plot_df = plot_df.groupby([feature, 'Genetic'])[beta, pval, fdr].first().reset_index()
            plot_df = plot_df[[len(datasets.genomic_obj.mobem_feature_to_gene(i)) != 0 for i in plot_df['Genetic']]]
            plot_df = plot_df.sort_values([fdr, pval]).head(ntop)
            plot_df = plot_df.assign(type=[datasets.genomic_obj.mobem_feature_type(i) for i in plot_df['Genetic']])
            plot_df = plot_df.sort_values(beta, ascending=False)
            plot_df = plot_df.assign(y=range(plot_df.shape[0]))

            # Plot
            for t in order:
                df = plot_df.query(f"type == '{t}'")
                ax.scatter(df[beta], df['y'], c=pal[t], label=t)

            for fc, y, drug, genetic in plot_df[[beta, 'y', feature, 'Genetic']].values:
                g_genes = '; '.join(datasets.genomic_obj.mobem_feature_to_gene(genetic))

                xoffset = 0.075 if d == 'crispr' else 0.3

                ax.text(fc - xoffset, y, drug, va='center', fontsize=4, zorder=10, color='gray', ha='right')
                ax.text(fc + xoffset, y, g_genes, va='center', fontsize=3, zorder=10, color='gray', ha='left')

            ax.axvline(0, lw=.1, c=Plot.PAL_DTRACE[1])

            ax.set_xlabel('Effect size (beta)')
            ax.set_ylabel('')
            ax.set_title('{} associations'.format(d.capitalize() if d == 'drug' else d.upper()))
            ax.axes.get_yaxis().set_ticks([])

            sns.despine(left=True, ax=ax)

        plt.gcf().set_size_inches(2. * axs.shape[0], ntop * .12)

        plt.legend(title='Genetic event', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


if __name__ == '__main__':
    # - Import associations
    datasets = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_robust = pd.read_csv('data/drug_lmm_regressions_robust_ic50.csv.gz')

    # - Single feature linear regression
    SingleLMMAnalysis.manhattan_plot(lmm_drug, fdr=0.1)
    plt.gcf().set_size_inches(8, 2)
    plt.savefig('reports/drug_associations_manhattan.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close('all')

    SingleLMMAnalysis.beta_histogram(lmm_drug)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug_associations_beta_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    SingleLMMAnalysis.recapitulated_drug_targets_barplot(lmm_drug, fdr=.1)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/drug_associations_count_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    SingleLMMAnalysis.top_associations_barplot(lmm_drug, fdr_line=.1, ntop=60)
    plt.savefig('reports/drug_associations_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Drug betas TSNEs
    tsnes = SingleLMMTSNE(lmm_drug)

    tsnes.drug_v17_id_batch()
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug_tsne_v17_ids_discrete.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    tsnes.drug_beta_tsne(hueby='signif')
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    tsnes.drug_beta_tsne(hueby='target')
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    tsnes.drug_beta_tsne(hueby='replicates')
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne_replicates.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    tsnes.drug_beta_tsne(hueby=SingleLMMTSNE.DRUG_TARGETS_HUE)
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne_targets.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Robust linear regressions
    RobustLMMAnalysis.genomic_histogram(datasets, ntop=40)
    plt.savefig('reports/robust_mobems_countplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    RobustLMMAnalysis.count_signif_associations(lmm_robust)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/robust_count_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    RobustLMMAnalysis.top_robust_features(lmm_robust)
    plt.savefig('reports/robust_top_associations.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # -
    drug, gene_assoc, gene_extra = (1946, 'MCL1_5526', 'RS'), 'MARCH5', 'MCL1'

    dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

    plot_df = pd.concat([
        datasets.drespo.loc[drug].rename('drug'),
        datasets.crispr.loc[gene_assoc].rename(gene_assoc),
        datasets.crispr.loc[gene_extra].rename(gene_extra),
        datasets.crispr_obj.institute.rename('Institute')
    ], axis=1, sort=False).dropna()

    cbin = pd.concat([plot_df[g].apply(lambda v: g if v < -1 else '') for g in [gene_assoc, gene_extra]], axis=1)
    plot_df['essentiality'] = cbin.apply(lambda v: ' + '.join([i for i in v if i != '']), axis=1).replace('', 'None').values

    dg_lmm = datasets.get_association(lmm_drug, drug, gene_assoc)
    annot_text = f"Beta={dg_lmm.iloc[0]['beta']:.2g}, FDR={dg_lmm.iloc[0]['fdr']:.1e}"

    #
    g = Plot().plot_corrplot(gene_assoc, 'drug', 'Institute', plot_df, add_hline=True, annot_text=annot_text)

    g.ax_joint.axhline(y=dmax, linewidth=.3, color=Plot.PAL_DTRACE[2], ls=':', zorder=0)

    g.set_axis_labels(f'{gene_assoc} (scaled log2 FC)', f'{drug[1]} (ln IC50, {drug[2]})')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/association_drug_scatter.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    g = Plot().plot_multiple('drug', 'essentiality', 'Institute', plot_df)

    plt.axvline(dmax, linewidth=.3, color=Plot.PAL_DTRACE[2], ls=':', zorder=0)

    plt.xlabel(f'{drug[1]} (ln IC50, {drug[2]})')
    plt.ylabel('Essential genes')

    plt.gcf().set_size_inches(3., 1)
    plt.savefig(f"reports/association_multiple_scatter.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')
