#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.associations import DRUG_INFO_COLUMNS
from dtrace.analysis import PAL_DTRACE, MidpointNormalize
from dtrace.analysis.plot.corrplot import plot_corrplot_discrete


def count_signif_associations(lmm_drug_robust, fdr=0.05):
    columns = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol', 'Genetic']

    d_signif = {tuple(i) for i in lmm_drug_robust.query('fdr_drug < {}'.format(fdr))[columns].values}
    c_signif = {tuple(i) for i in lmm_drug_robust.query('fdr_crispr < {}'.format(fdr))[columns].values}
    dc_signif = d_signif.intersection(c_signif)

    # Build dataframe
    plot_df = pd.DataFrame(dict(
        names=['Drug', 'CRISPR', 'Intersection'],
        count=list(map(len, [d_signif, c_signif, dc_signif]))
    )).sort_values('count', ascending=True)
    plot_df = plot_df.assign(y=range(plot_df.shape[0]))

    # Plot
    plt.barh(plot_df['y'], plot_df['count'], color=PAL_DTRACE[2], linewidth=0)

    sns.despine(right=True, top=True)

    for c, y in plot_df[['count', 'y']].values:
        plt.text(c + 3, y, str(c), va='center', fontsize=5, zorder=10, color=PAL_DTRACE[2])

    plt.yticks(plot_df['y'], plot_df['names'])
    plt.xlabel('Number of signifcant associations')
    plt.ylabel('')


def genomic_histogram(mobems, ntop=40):
    # Build dataframe
    plot_df = mobems[samples].sum(1).rename('count').reset_index()
    plot_df = plot_df[[len(dtrace.mobem_feature_to_gene(i)) != 0 for i in plot_df['index']]]

    plot_df = plot_df.assign(genes=['; '.join(dtrace.mobem_feature_to_gene(i)) for i in plot_df['index']])
    plot_df = plot_df.assign(type=[dtrace.mobem_feature_type(i) for i in plot_df['index']])

    plot_df = plot_df.assign(name=['{} - {}'.format(t, g) for t, g in plot_df[['type', 'genes']].values])

    plot_df = plot_df.sort_values('count', ascending=False).head(ntop)

    # Plot
    order = ['Mutation', 'CN loss', 'CN gain']
    pal = pd.Series(PAL_DTRACE, index=order).to_dict()

    sns.barplot('count', 'name', 'type', data=plot_df, palette=pal, hue_order=order, dodge=False, saturation=1)

    plt.xlabel('Number of occurrences')
    plt.ylabel('')

    plt.legend(frameon=False)
    sns.despine()

    plt.gcf().set_size_inches(2, .15 * ntop)


def top_robust_features(lmm_drug_robust, ntop=40):
    f, axs = plt.subplots(1, 2, sharex=False, sharey=False, gridspec_kw=dict(wspace=.75))

    order = ['Mutation', 'CN loss', 'CN gain']
    pal = pd.Series(PAL_DTRACE, index=order).to_dict()

    for i, d in enumerate(['drug', 'crispr']):
        ax = axs[i]
        beta, pval, fdr = 'beta_{}'.format(d), 'pval_{}'.format(d), 'fdr_{}'.format(d)

        feature = 'DRUG_NAME' if d == 'drug' else 'GeneSymbol'

        # Dataframe
        plot_df = lmm_drug_robust.groupby([feature, 'Genetic'])[beta, pval, fdr].first().reset_index()
        plot_df = plot_df[[len(dtrace.mobem_feature_to_gene(i)) != 0 for i in plot_df['Genetic']]]
        plot_df = plot_df.sort_values([fdr, pval]).head(ntop)
        plot_df = plot_df.assign(type=[dtrace.mobem_feature_type(i) for i in plot_df['Genetic']])
        plot_df = plot_df.sort_values(beta, ascending=False)
        plot_df = plot_df.assign(y=range(plot_df.shape[0]))

        # Plot
        for t in order:
            df = plot_df.query("type == '{}'".format(t))
            ax.scatter(df[beta], df['y'], c=pal[t], label=t)

        for fc, y, drug, genetic in plot_df[[beta, 'y', feature, 'Genetic']].values:
            g_genes = '; '.join(dtrace.mobem_feature_to_gene(genetic))

            xoffset = 0.075 if d == 'crispr' else 0.3

            ax.text(fc - xoffset, y, drug, va='center', fontsize=4, zorder=10, color='gray', ha='right')
            ax.text(fc + xoffset, y, g_genes, va='center', fontsize=3, zorder=10, color='gray', ha='left')

        ax.axvline(0, lw=.1, c=PAL_DTRACE[1])

        ax.set_xlabel('Effect size (beta)')
        ax.set_ylabel('')
        ax.set_title('{} associations'.format(d.capitalize() if d == 'drug' else d.upper()))
        ax.axes.get_yaxis().set_ticks([])

        sns.despine(left=True, ax=ax)

    plt.gcf().set_size_inches(2. * axs.shape[0], ntop * .12)

    plt.legend(title='Genetic event', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


if __name__ == '__main__':
    # - Import
    # Data-sets
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()
    crispr = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    d_maxc = pd.read_csv(dtrace.DRUG_RESPONSE_MAXC, index_col=[0, 1, 2])

    # Robust associations
    lmm_drug_robust = pd.read_csv(dtrace.LMM_ASSOCIATIONS_ROBUST)

    # - Distribution of genomic events
    genomic_histogram(mobems, ntop=40)
    plt.savefig('reports/lmm_robust_mobems_countplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Count number of significant associations overall
    count_signif_associations(lmm_drug_robust)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/lmm_robust_count_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Top associatios
    top_robust_features(lmm_drug_robust)
    plt.savefig('reports/lmm_robust_top_associations.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Strongest significant association per drug
    associations = lmm_drug_robust.query('fdr_crispr < 0.05 & fdr_drug < 0.05')
    associations = associations.sort_values('fdr_crispr').groupby(DRUG_INFO_COLUMNS).head(1)

    # - Plot robust association
    indices = [15117, 36766, 60214, 52585, 37234, 33009]

    for idx in indices:
        columns = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol', 'Genetic']
        association = lmm_drug_robust.loc[idx, columns]

        name = 'Drug={}, Gene={}, Genetic={} [{}, {}]'.format(association[1], association[3], association[4], association[0], association[2])

        plot_df = pd.concat([
            crispr.loc[association[3]],
            drespo.loc[tuple(association[:3])].rename('{} [{}]'.format(association[1], association[2])),
            mobems.loc[association[4]]
        ], axis=1).dropna()

        g = plot_corrplot_discrete(association[3], '{} [{}]'.format(association[1], association[2]), association[4], plot_df)

        g.ax_joint.axhline(np.log(d_maxc.loc[tuple(association[:3]), 'max_conc_micromolar']), lw=.1, color=PAL_DTRACE[2], ls='--')

        plt.gcf().set_size_inches(2, 2)
        plt.savefig(f'reports/lmm_robust_{name}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')
