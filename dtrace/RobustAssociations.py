#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from DTracePlot import DTracePlot
from natsort import natsorted
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE
from DataImporter import DrugResponse
from Associations import Association
from sklearn.preprocessing import StandardScaler


class SingleLMMTSNE:
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
            targets=
            [';'.join(self.dtargets[i]) if i in self.dtargets else '' for i in tsnes[DrugResponse.DRUG_COLUMNS[0]]]
        )

        # Annotate significant
        d_signif = {tuple(i) for i in lmm_drug.query(f'fdr < {self.fdr}')[DrugResponse.DRUG_COLUMNS].values}
        tsnes = tsnes.assign(has_signif=['Yes' if tuple(i) in d_signif else 'No' for i in tsnes[DrugResponse.DRUG_COLUMNS].values])

        return tsnes

    def drug_beta_tsne(self, hueby):
        if hueby == 'signif':
            pal = {'No': DTracePlot.PAL_DTRACE[2], 'Yes': DTracePlot.PAL_DTRACE[0]}

            g = sns.FacetGrid(
                self.tsnes, col='VERSION', hue='has_signif', palette=pal, hue_order=['No', 'Yes'], sharey=False,
                sharex=False, legend_out=True, despine=False, size=2, aspect=1
            )

            g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
            g.map(plt.axhline, y=0, ls='-', lw=0.3, c=DTracePlot.PAL_DTRACE[1], alpha=.2, zorder=0)
            g.map(plt.axvline, x=0, ls='-', lw=0.3, c=DTracePlot.PAL_DTRACE[1], alpha=.2, zorder=0)

            g.add_legend(title='Significant', prop=dict(size=4), frameon=False)

        elif hueby == 'target':
            pal = {'No': DTracePlot.PAL_DTRACE[2], 'Yes': DTracePlot.PAL_DTRACE[0]}

            g = sns.FacetGrid(
                self.tsnes, col='VERSION', hue='target', palette=pal, hue_order=['Yes', 'No'], sharey=False,
                sharex=False, legend_out=True, despine=False, size=2, aspect=1
            )

            g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
            g.map(plt.axhline, y=0, ls='-', lw=0.3, c=DTracePlot.PAL_DTRACE[1], alpha=.2, zorder=0)
            g.map(plt.axvline, x=0, ls='-', lw=0.3, c=DTracePlot.PAL_DTRACE[1], alpha=.2, zorder=0)

            g.add_legend(title='Known target?', prop=dict(size=4), frameon=False)

        elif hueby == 'replicates':
            rep_names = set(self.tsnes['rep'])

            pal_v17 = [n for n in rep_names if n.endswith(';v17')]
            pal_v17 = dict(zip(*(pal_v17, sns.color_palette('tab20', n_colors=len(pal_v17)).as_hex())))

            pal_rs = [n for n in rep_names if n.endswith(';RS')]
            pal_rs = dict(zip(*(pal_rs, sns.color_palette('tab20', n_colors=len(pal_rs)).as_hex())))

            pal = {**pal_v17, **pal_rs}
            pal['NA'] = DTracePlot.PAL_DTRACE[1]

            g = sns.FacetGrid(
                self.tsnes, col='VERSION', hue='rep', palette=pal, sharey=False, sharex=False, legend_out=True,
                despine=False, size=2, aspect=1
            )

            g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
            g.map(plt.axhline, y=0, ls='-', lw=0.3, c=DTracePlot.PAL_DTRACE[1], alpha=.2, zorder=0)
            g.map(plt.axvline, x=0, ls='-', lw=0.3, c=DTracePlot.PAL_DTRACE[1], alpha=.2, zorder=0)

            g.add_legend(title='', prop=dict(size=4), frameon=False)

        elif type(hueby) == list:
            sets = [i for l in hueby for i in l[1]]
            labels = [';'.join(i) for l in hueby for i in l[1]]
            colors = [i for l in hueby for i in l[0]]

            pal = dict(zip(*(labels, colors)))
            pal['NA'] = DTracePlot.PAL_DTRACE[1]

            df = self.tsnes.assign(hue=[[i for i, g in enumerate(sets) if g.intersection(t.split(';'))] for t in self.tsnes['targets']])
            df = self.tsnes.assign(hue=[labels[i[0]] if len(i) > 0 else 'NA' for i in df['hue']])

            g = sns.FacetGrid(
                df.query("target == 'Yes'"), col='VERSION', hue='hue', palette=pal, sharey=False, sharex=False,
                legend_out=True, despine=False, size=2, aspect=1
            )

            for i, s in enumerate(['v17', 'RS']):
                ax = g.axes.ravel()[i]
                df_plot = df.query("(target == 'No') & (VERSION == '{}')".format(s))
                ax.scatter(df_plot['P1'], df_plot['P2'], color=DTracePlot.PAL_DTRACE[1], marker='x', lw=0.3, s=5, alpha=0.7,
                           label='No target info')

            g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
            g.map(plt.axhline, y=0, ls='-', lw=0.3, c=DTracePlot.PAL_DTRACE[1], alpha=.2, zorder=0)
            g.map(plt.axvline, x=0, ls='-', lw=0.3, c=DTracePlot.PAL_DTRACE[1], alpha=.2, zorder=0)

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

        discrete_pal = pd.Series(DTracePlot.PAL_DTRACE[:3], index=set(plot_df['id_discrete'])).to_dict()

        g = DTracePlot.plot_corrplot_discrete(
            'P1', 'P2', 'id_discrete', plot_df, discrete_pal=discrete_pal, legend_title='DRUG_ID discretised',
            hue_order=hue_order
        )

        g.set_axis_labels('', '')

        return g


class RobustLMMAnalysis:
    @staticmethod
    def count_signif_associations(lmm_robust, lmm_robust_gexp, fdr=0.1):
        plot_df = []
        for f in ['fdr_crispr', 'fdr_drug', 'both']:
            for n, df in [('Mutation/Copy-number', lmm_robust), ('Gene-expression', lmm_robust_gexp)]:
                if f != 'both':
                    df_ = df.query(f'{f} < {fdr}')
                else:
                    df_ = df.query(f'(fdr_crispr < {fdr}) & (fdr_drug < {fdr})')

                df_ = df_.groupby('target')['DRUG_NAME'].agg('count').rename('count').to_frame().assign(
                    variable=f.split('_')[1] if f != 'both' else f).assign(genetic=n)

                plot_df.append(df_)

        plot_df = pd.concat(plot_df).reset_index()

        order = ['T', '1', '2', '3', '4', '5+', '-']
        pal = pd.Series(DTracePlot.get_palette_continuous(3, DTracePlot.PAL_DTRACE[2]),
                        index=['both', 'drug', 'crispr'])

        g = sns.catplot(
            x='count', y='target', hue='variable', row='genetic', data=plot_df, kind='bar', order=order, palette=pal,
            height=2.5, legend=False, legend_out=False
        )

        g.set_titles('{row_name}')

        g.add_legend(title='Association', frameon=False)

        for ax in g.axes[:, 0]:
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=100))
            ax.grid(lw=.3, alpha=.8, color=DTracePlot.PAL_DTRACE[1], ls='-', axis='x', zorder=0)

        g.set_axis_labels('Significant associations', 'Drug target ~ CRISPR\nPPI distance')

        plt.savefig('reports/robust_count_signif.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

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
        pal = pd.Series(DTracePlot.PAL_DTRACE[:3], index=order).to_dict()

        sns.barplot('count', 'name', 'type', data=plot_df, palette=pal, hue_order=order, dodge=False, saturation=1)

        plt.xlabel('Number of occurrences')
        plt.ylabel('')

        plt.legend(frameon=False)
        sns.despine()

        plt.gcf().set_size_inches(2, .15 * ntop)

    @staticmethod
    def top_robust_features(associations, ntop=30, dtype='genomic'):
        f, axs = plt.subplots(1, 2, sharex='none', sharey='none', gridspec_kw=dict(wspace=.75))

        for i, d in enumerate(['drug', 'crispr']):
            ax = axs[i]
            beta, pval, fdr = f'beta_{d}', f'pval_{d}', f'fdr_{d}'

            feature = 'DRUG_NAME' if d == 'drug' else 'GeneSymbol'

            # Dataframe
            plot_df = associations.query("Genetic != 'msi_status'")
            plot_df = plot_df.groupby([feature, 'Genetic'])[beta, pval, fdr].first().reset_index()

            if dtype == 'genomic':
                plot_df = plot_df[[len(datasets.genomic_obj.mobem_feature_to_gene(i)) != 0 for i in plot_df['Genetic']]]
                plot_df = plot_df.assign(type=[datasets.genomic_obj.mobem_feature_type(i) for i in plot_df['Genetic']])

            plot_df = plot_df.sort_values([fdr, pval]).head(ntop).sort_values(beta, ascending=False)
            plot_df = plot_df.assign(y=range(plot_df.shape[0]))

            # Scatter
            if dtype == 'genomic':
                order = ['Mutation', 'CN loss', 'CN gain']
                pal = pd.Series(DTracePlot.PAL_DTRACE[:3], index=order).to_dict()

                for t in order:
                    df = plot_df.query(f"type == '{t}'")
                    ax.scatter(df[beta], df['y'], c=pal[t], label=t)
            else:
                ax.scatter(plot_df[beta], plot_df['y'], c=DTracePlot.PAL_DTRACE[2])

            # Labels
            for fc, y, drug, genetic in plot_df[[beta, 'y', feature, 'Genetic']].values:
                if dtype == 'genomic':
                    g_genes = '; '.join(datasets.genomic_obj.mobem_feature_to_gene(genetic))

                else:
                    g_genes = genetic

                xoffset = plot_df[beta].abs().max() * .2

                ax.text(fc - xoffset, y, drug, va='center', fontsize=4, zorder=10, color='gray', ha='right')
                ax.text(fc + xoffset, y, g_genes, va='center', fontsize=3, zorder=10, color='gray', ha='left')

            # Misc
            ax.axvline(0, lw=.1, c=DTracePlot.PAL_DTRACE[1])

            ax.set_xlabel('Effect size (beta)')
            ax.set_ylabel('')
            ax.set_title('{} associations'.format(d.capitalize() if d == 'drug' else d.upper()))
            ax.axes.get_yaxis().set_ticks([])

            sns.despine(left=True, ax=ax)

        if dtype == 'genomic':
            plt.legend(title='Genetic event', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        plt.gcf().set_size_inches(2. * axs.shape[0], ntop * .12)


if __name__ == '__main__':
    # - Import associations
    datasets = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_drug_gexp = pd.read_csv('data/drug_lmm_regressions_ic50_gexp.csv.gz')

    lmm_robust = pd.read_csv('data/drug_lmm_regressions_robust_ic50.csv.gz')
    lmm_robust_gexp = pd.read_csv('data/drug_lmm_regressions_robust_gexp_ic50.csv.gz')

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

    RobustLMMAnalysis.count_signif_associations(lmm_robust, lmm_robust_gexp)
    plt.savefig('reports/robust_count_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    RobustLMMAnalysis.top_robust_features(lmm_robust, ntop=30)
    plt.savefig('reports/robust_top_associations.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    RobustLMMAnalysis.top_robust_features(lmm_robust_gexp, ntop=30, dtype='gene-expression')
    plt.savefig('reports/robust_top_associations_gexp.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    cols = ['DRUG_ID', 'DRUG_NAME', 'VERSION', 'Genetic']

    pd.concat([
        lmm_robust.query('(fdr_drug < .1)').groupby(cols)['GeneSymbol'].count(),
        lmm_robust.query('(fdr_drug < .1) & (fdr_crispr < .1)').groupby(cols)['GeneSymbol'].count()
    ], axis=1, sort=False).replace(np.nan, 0).astype(int)

    # Examples
    rassocs = [
        ('Olaparib', 'FLI1', 'EWSR1.FLI1_mut'),
        ('Dabrafenib', 'BRAF', 'BRAF_mut'),
        ('Nutlin-3a (-)', 'MDM2', 'TP53_mut'),
        ('Taselisib', 'PIK3CA', 'PIK3CA_mut')
    ]

    # d, c, g = ('Taselisib', 'PIK3CA', 'PIK3CA_mut')
    for d, c, g in rassocs:
        assoc = lmm_robust[
            (lmm_robust['DRUG_NAME'] == d) & (lmm_robust['GeneSymbol'] == c) & (lmm_robust['Genetic'] == g)
        ].iloc[0]

        drug = tuple(assoc[DrugResponse.DRUG_COLUMNS])

        dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

        plot_df = pd.concat([
            datasets.drespo.loc[drug].rename('drug'),
            datasets.crispr.loc[c].rename('crispr'),
            datasets.genomic.loc[g].rename('genetic'),
            datasets.crispr_obj.institute.rename('Institute'),
        ], axis=1, sort=False).dropna()

        grid = DTracePlot.plot_corrplot_discrete('crispr', 'drug', 'genetic', 'Institute', plot_df)

        grid.ax_joint.axhline(y=dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

        grid.set_axis_labels(f'{c} (scaled log2 FC)', f'{d} (ln IC50)')

        plt.suptitle(g, y=1.05, fontsize=8)

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(f'reports/robust_scatter_{d}_{c}_{g}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')
