#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
from natsort import natsorted
from crispy.utils import Utils
from associations import Association
from importer import PPI, DrugResponse, CRISPR


def manhattan_plot(lmm_drug, fdr_line=.1, n_genes=13):
    # Import gene genomic coordinates from CRISPR-Cas9 library
    crispr_lib = Utils.get_crispr_lib().groupby('gene').agg({'start': 'min', 'chr': 'first'})

    # Plot data-frame
    df = lmm_drug.copy()
    df = df.assign(pos=crispr_lib.loc[df['GeneSymbol'], 'start'].values)
    df = df.assign(chr=crispr_lib.loc[df['GeneSymbol'], 'chr'].apply(lambda v: v.replace('chr', '')).values)
    df = df.sort_values(['chr', 'pos'])

    # Most frequently associated genes
    top_genes = df.query('fdr < 0.05')['GeneSymbol'].value_counts().head(n_genes)
    top_genes_pal = dict(zip(*(top_genes.index, sns.color_palette('tab20', n_colors=n_genes).as_hex())))

    # Plot
    chrms = set(df['chr'])
    label_fdr = 'Significant'.format(fdr_line*100)

    f, axs = plt.subplots(1, len(chrms), sharex='none', sharey='row', gridspec_kw=dict(wspace=.05))
    for i, name in enumerate(natsorted(chrms)):
        df_group = df[df['chr'] == name]

        # Plot all associations
        df_nonsignif = df_group.query('fdr >= {}'.format(fdr_line))
        axs[i].scatter(df_nonsignif['pos'], -np.log10(df_nonsignif['pval']), c=Plot.PAL_DTRACE[(i % 2) + 1], s=2)

        # Plot significant associationsdrug_associations_count.pdf
        df_signif = df_group.query('fdr < {}'.format(fdr_line))
        df_signif = df_signif[~df_signif['GeneSymbol'].isin(top_genes.index)]
        axs[i].scatter(df_signif['pos'], -np.log10(df_signif['pval']), c=Plot.PAL_DTRACE[0], s=2, zorder=3, label=label_fdr)

        # Plot significant associations of top frequent genes
        df_genes = df_group.query('fdr < {}'.format(fdr_line))
        df_genes = df_genes[df_genes['GeneSymbol'].isin(top_genes.index)]
        for pos, pval, gene in df_genes[['pos', 'pval', 'GeneSymbol']].values:
            axs[i].scatter(pos, -np.log10(pval), c=top_genes_pal[gene], s=6, zorder=4, label=gene, marker='2', lw=.75)

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
    plt.legend(list(zip(*(by_label)))[1], list(zip(*(by_label)))[0], loc='center left', bbox_to_anchor=(1.01, 0.5), prop={'size': 5}, frameon=False)


def beta_histogram(lmm_drug):
    kde_kws = dict(cut=0, lw=1, zorder=1, alpha=.8)
    hist_kws = dict(alpha=.4, zorder=1, linewidth=0)

    label_order = ['All', 'Target', 'Target + Significant']

    sns.distplot(
        lmm_drug['beta'], color=Plot.PAL_DTRACE[2], kde_kws=kde_kws, hist_kws=hist_kws, label=label_order[0], bins=30
    )

    sns.distplot(
        lmm_drug.query("target == 'T'")['beta'], color=Plot.PAL_DTRACE[0], kde_kws=kde_kws, hist_kws=hist_kws,
        label=label_order[1], bins=30
    )

    sns.despine(right=True, top=True)

    plt.axvline(0, c=Plot.PAL_DTRACE[1], lw=.3, ls='-', zorder=0)

    plt.xlabel('Association beta')
    plt.ylabel('Density')

    plt.legend(prop={'size': 6}, loc=2, frameon=False)


def recapitulated_drug_targets_barplot(lmm_drug, fdr=.1):
    # Count number of drugs
    df_genes = set(lmm_drug['GeneSymbol'])

    d_response = DrugResponse()

    d_targets = d_response.get_drugtargets()

    d_all = {tuple(i) for i in lmm_drug[d_response.DRUG_INFO_COLUMNS].values}
    d_annot = {tuple(i) for i in d_all if i[0] in d_targets}
    d_tested = {tuple(i) for i in d_annot if len(d_targets[i[0]].intersection(df_genes)) > 0}
    d_tested_signif = {tuple(i) for i in lmm_drug.query('fdr < {}'.format(fdr))[d_response.DRUG_INFO_COLUMNS].values if tuple(i) in d_tested}
    d_tested_correct = {tuple(i) for i in lmm_drug.query("fdr < {} & target == 'T'".format(fdr))[d_response.DRUG_INFO_COLUMNS].values if tuple(i) in d_tested_signif}
    d_tested_correct_neigh = {tuple(i) for i in lmm_drug.query(f"fdr < {fdr} & (target == 'T' | target == '1')")[d_response.DRUG_INFO_COLUMNS].values if tuple(i) in d_tested_signif}

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


if __name__ == '__main__':
    ppi = PPI()

    # - Import associations
    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv')
    lmm_drug = ppi.ppi_annotation(lmm_drug, ppi_type='string', ppi_kws=dict(score_thres=900), target_thres=3)

    crispr = CRISPR(foldchanges_file='crispr_crispy.csv', foldchanges_file_sep=',')
    drug = DrugResponse()

    # d, g = (1243, 'Piperlongumine', 'v17'), 'NFE2L2'
    d, g = (2235, 'AZD5991', 'RS'), 'MARCH5'

    plot_df = pd.concat([
        drug.get_data().loc[d].rename('drug'),
        crispr.get_data().loc[g].rename('crispr')
    ], axis=1, sort=False).dropna()

    ax = sns.jointplot('crispr', 'drug', data=plot_df)
    ax.ax_joint.axhline(y=np.log(drug.maxconcentration[d]), ls='-', color='k', lw=.5)

    plt.show()

    # - Drug associations manhattan plot
    manhattan_plot(lmm_drug, fdr_line=0.1)
    plt.gcf().set_size_inches(8, 2)
    plt.savefig('reports/drug_associations_manhattan.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close('all')

    # - Associations beta histogram
    beta_histogram(lmm_drug)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug_associations_beta_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Count number of significant associations overall
    recapitulated_drug_targets_barplot(lmm_drug, fdr=.1)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/drug_associations_count_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
