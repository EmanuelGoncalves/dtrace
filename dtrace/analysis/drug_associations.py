#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from natsort import natsorted
from sklearn.metrics import auc
from sklearn.manifold import TSNE
from dtrace.analysis import PAL_DTRACE
from sklearn.preprocessing import StandardScaler
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.associations import ppi_annotation, corr_drugtarget_gene, DRUG_INFO_COLUMNS


def manhattan_plot(lmm_drug, fdr_line=.05):
    crispr_lib = pd.read_csv(dtrace.CRISPR_LIB).groupby('GENES').agg({'STARTpos': 'min', 'CHRM': 'first'})

    df = lmm_drug.copy()

    # df = df[df['DRUG_NAME'] == 'GSK2276186C']
    df = df.assign(pos=crispr_lib.loc[df['GeneSymbol'], 'STARTpos'].values)
    df = df.assign(chr=crispr_lib.loc[df['GeneSymbol'], 'CHRM'].values)
    df = df.sort_values(['chr', 'pos'])

    chrms = set(df['chr'])

    f, axs = plt.subplots(1, len(chrms), sharex=False, sharey=True, gridspec_kw=dict(wspace=.05))

    for i, name in enumerate(natsorted(chrms)):
        df_group = df[df['chr'] == name]

        axs[i].scatter(df_group['pos'], -np.log10(df_group['pval']), c=PAL_DTRACE[(i % 2) + 1], s=2)

        axs[i].scatter(
            df_group.query('fdr < {}'.format(fdr_line))['pos'], -np.log10(df_group.query('fdr < {}'.format(fdr_line))['pval']), c=PAL_DTRACE[0], s=2, zorder=3
        )

        axs[i].axes.get_xaxis().set_ticks([])
        axs[i].set_xlabel(name)
        axs[i].set_ylim(0)

        if i != 0:
            sns.despine(ax=axs[i], left=True, right=True, top=True)
            axs[i].yaxis.set_ticks_position('none')

        else:
            sns.despine(ax=axs[i], right=True, top=True)
            axs[i].set_ylabel('Drug-gene association (-log10 p-value)')

    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel('Chromosome')


def top_associations_barplot(lmm_drug, fdr_line=0.05, ntop=40):
    # Filter for signif associations
    df = lmm_drug\
        .query('fdr < {}'.format(fdr_line))\
        .sort_values('fdr')\
        .groupby(['DRUG_NAME', 'GeneSymbol'])\
        .first() \
        .sort_values('fdr') \
        .reset_index()
    df = df.assign(logpval=-np.log10(df['pval']).values)

    # Drug order
    order = list(df.groupby('DRUG_NAME')['fdr'].min().sort_values().index)[:ntop]

    # Build plot dataframe
    df_, xpos = [], 0
    for i, drug_name in enumerate(order):
        if i % 10 == 0:
            xpos = 0

        df_drug = df[df['DRUG_NAME'] == drug_name]
        df_drug = df_drug.assign(xpos=np.arange(xpos, xpos + df_drug.shape[0]))
        df_drug = df_drug.assign(irow=int(np.floor(i / 10)))

        xpos += (df_drug.shape[0] + 2)

        df_.append(df_drug)

    df = pd.concat(df_).reset_index()

    # Plot
    f, axs = plt.subplots(int(np.ceil(ntop / 10)), 1, sharex=False, sharey=True, gridspec_kw=dict(hspace=.0))

    # Barplot
    for irow in set(df['irow']):
        df_irow = df[df['irow'] == irow]

        df_irow_ = df_irow.query("target != 'T'")
        axs[irow].bar(df_irow_['xpos'], df_irow_['logpval'], .8, color=PAL_DTRACE[2], align='center', zorder=5)

        df_irow_ = df_irow.query("target == 'T'")
        axs[irow].bar(df_irow_['xpos'], df_irow_['logpval'], .8, color=PAL_DTRACE[0], align='center', zorder=5)

        for k, v in df_irow.groupby('DRUG_NAME')['xpos'].min().sort_values().to_dict().items():
            axs[irow].text(v - 1.2, 0.1, textwrap.fill(k, 15), va='bottom', fontsize=7, zorder=10, rotation='vertical', color=PAL_DTRACE[2])

        for g, p in df_irow[['GeneSymbol', 'xpos']].values:
            axs[irow].text(p, 0.1, g, ha='center', va='bottom', fontsize=5, zorder=10, rotation='vertical', color='white')

        for x, y, t in df_irow[['xpos', 'logpval', 'target']].values:
            axs[irow].text(x, y + 0.25, t, color=PAL_DTRACE[0] if t == 'T' else PAL_DTRACE[2], ha='center', fontsize=6, zorder=10)

        sns.despine(ax=axs[irow], right=True, top=True)
        axs[irow].axes.get_xaxis().set_ticks([])
        axs[irow].set_ylabel('Drug-gene association\n(-log10 p-value)')


def plot_count_associations(lmm_drug, fdr_line=0.05, min_events=2):
    df = lmm_drug[lmm_drug['fdr'] < fdr_line]
    df = df.groupby(DRUG_INFO_COLUMNS)['fdr'].count().rename('count').sort_values(ascending=False).reset_index()
    df = df.assign(name=['{} ({}, {})'.format(n, i, v) for i, n, v in df[DRUG_INFO_COLUMNS].values])
    df = df.query('count >= {}'.format(min_events))

    sns.barplot('count', 'name', data=df, color=PAL_DTRACE[2], linewidth=.8, orient='h')
    sns.despine(right=True, top=True)

    plt.ylabel('')
    plt.xlabel('Count')
    plt.title('Significant associations FDR<{}%'.format(fdr_line * 100))


def recapitulated_drug_targets_barplot(lmm_drug, fdr=0.05):
    df_genes = set(lmm_drug['GeneSymbol'])

    d_targets = dtrace.get_drugtargets()

    d_all = {tuple(i) for i in lmm_drug[DRUG_INFO_COLUMNS].values}
    d_annot = {tuple(i) for i in d_all if i[0] in d_targets}
    d_tested = {tuple(i) for i in d_annot if len(d_targets[i[0]].intersection(df_genes)) > 0}
    d_tested_signif = {tuple(i) for i in lmm_drug.query('fdr < {}'.format(fdr))[DRUG_INFO_COLUMNS].values if tuple(i) in d_tested}
    d_tested_correct = {tuple(i) for i in lmm_drug.query("fdr < {} & target == 'T'".format(fdr))[DRUG_INFO_COLUMNS].values if tuple(i) in d_tested_signif}

    plot_df = pd.DataFrame(dict(
        names=['All', 'w/Target', 'w/Tested target', 'w/Signif tested target', 'w/Correct target'],
        count=list(map(len, [d_all, d_annot, d_tested, d_tested_signif, d_tested_correct]))
    ))

    sns.barplot('count', 'names', data=plot_df, color=PAL_DTRACE[2], orient='h')
    sns.despine(right=True, top=True)
    plt.xlabel('Number of drugs')
    plt.ylabel('')


def beta_histogram(lmm_drug):
    kde_kws = dict(cut=0, lw=1, zorder=1, alpha=.8)
    hist_kws = dict(alpha=.4, zorder=1)

    label_order = ['All', 'Target', 'Target + Significant']

    sns.distplot(lmm_drug['beta'], color=PAL_DTRACE[2], kde_kws=kde_kws, hist_kws=hist_kws, label=label_order[0], bins=30)
    sns.distplot(lmm_drug.query("target == 'T'")['beta'], color=PAL_DTRACE[0], kde_kws=kde_kws, hist_kws=hist_kws, label=label_order[1], bins=30)

    sns.despine(right=True, top=True)

    plt.axvline(0, c=PAL_DTRACE[1], lw=.3, ls='-', zorder=0)

    plt.xlabel('Association beta')
    plt.ylabel('Density')

    plt.legend(prop={'size': 6}, loc=2)


def beta_corr_boxplot(lmm_drug):
    # Build betas matrix
    betas = pd.pivot_table(lmm_drug, index=DRUG_INFO_COLUMNS, columns='GeneSymbol', values='beta')

    # Drug information
    drugsheet = dtrace.get_drugsheet()

    # Drug annotation
    def classify_drug_pairs(d1, d2):
        d1_id, d1_name, d1_screen = d1.split(' ; ')
        d2_id, d2_name, d2_screen = d2.split(' ; ')

        d1_id, d2_id = int(d1_id), int(d2_id)

        # Check if Drugs are the same
        if d1_id in drugsheet.index and d2_id in drugsheet.index:
            dsame = dtrace.is_same_drug(d1_id, d2_id, drugsheet=drugsheet)

        else:
            dsame = d1_name == d2_name

        # Check if Drugs were screened with same panel
        if dsame & (d1_screen == d2_screen):
            return 'Same Drug & Screen'
        elif dsame:
            return 'Same Drug'
        else:
            return 'Different'

    # Correlation matrix and upper triangle
    betas_corr = betas.T.corr()
    betas_corr = betas_corr.where(np.triu(np.ones(betas_corr.shape), 1).astype(np.bool))

    # Merge drug info
    betas_corr.index = [' ; '.join(map(str, i)) for i in betas_corr.index]
    betas_corr.columns = [' ; '.join(map(str, i)) for i in betas_corr.columns]

    # Unstack
    betas_corr = betas_corr.unstack().dropna()
    betas_corr = betas_corr.reset_index()
    betas_corr.columns = ['drug_1', 'drug_2', 'r']

    # Annotate if same drugs
    betas_corr = betas_corr.assign(
        dtype=[classify_drug_pairs(d1, d2) for d1, d2 in betas_corr[['drug_1', 'drug_2']].values]
    )

    # Plot
    order = ['Different', 'Same Drug', 'Same Drug & Screen']
    pal = dict(zip(*(order, sns.light_palette(PAL_DTRACE[2], len(order) + 1).as_hex()[1:])))

    sns.boxplot('r', 'dtype', data=betas_corr, orient='h', order=order, palette=pal, fliersize=1, notch=True, linewidth=.3)

    sns.despine(top=True, right=True)

    plt.axvline(0, lw=0.3, c=PAL_DTRACE[1], zorder=0)

    plt.xlabel('Pearson\'s R')
    plt.ylabel('')


def drug_beta_tsne(lmm_drug, fdr=0.05, perplexity=15, learning_rate=200, n_iter=2000):
    d_targets = dtrace.get_drugtargets()

    # Drugs into
    drugs = {tuple(i) for i in lmm_drug.query('fdr < {}'.format(fdr))[DRUG_INFO_COLUMNS].values}
    drugs_annot = {tuple(i) for i in drugs if i[0] in d_targets}
    drugs_screen = {v: {d for d in drugs if d[2] == v} for v in ['v17', 'RS']}

    # Build drug association beta matrix
    betas = pd.pivot_table(lmm_drug, index=DRUG_INFO_COLUMNS, columns='GeneSymbol', values='beta')
    betas = betas.loc[list(drugs)]

    # TSNE
    tsnes = []
    for s in drugs_screen:
        tsne_df = betas.loc[list(drugs_screen[s])]
        tsne_df = pd.DataFrame(StandardScaler().fit_transform(tsne_df.T).T, index=tsne_df.index, columns=tsne_df.columns)

        tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, init='pca').fit_transform(tsne_df)

        tsne = pd.DataFrame(tsne, index=tsne_df.index, columns=['P1', 'P2']).reset_index()
        tsne = tsne.assign(target=[int(tuple(i) in drugs_annot) for i in tsne[DRUG_INFO_COLUMNS].values])

        tsnes.append(tsne)

    tsnes = pd.concat(tsnes)

    # Plot
    pal = {0: PAL_DTRACE[2], 1: PAL_DTRACE[0]}

    g = sns.FacetGrid(
        tsnes, col='VERSION', hue='target', palette=pal, hue_order=[1, 0], sharey=False, sharex=False, legend_out=True, despine=False,
        size=2, aspect=1
    )

    g.map(plt.scatter, 'P1', 'P2', alpha=.7, lw=.3, edgecolor='white')
    g.set_titles('{col_name}')
    g.add_legend()


def drug_aurc(lmm_drug, fdr=0.05, corr=0.25, label='target', rank_label='pval', legend_size=6, title='', legend_title=''):
    # Subset to entries that have a target
    df = lmm_drug.query("{} != '-'".format(label))
    df = df[(df['target'] == 'T') | (df['corr'].abs() > corr)]

    drugs_signif = {tuple(i) for i in df.query('fdr < {}'.format(fdr))[DRUG_INFO_COLUMNS].values}
    df = df[[tuple(i) in drugs_signif for i in df[DRUG_INFO_COLUMNS].values]]

    # Define order and palette
    order = natsorted(set(df[label]))
    order.insert(0, order.pop(order.index('T')))

    pal = [PAL_DTRACE[0]] + sns.light_palette(PAL_DTRACE[2], n_colors=len(order)).as_hex()[1:]
    pal = pd.Series(pal, index=order)

    # Initial plot
    ax = plt.gca()

    # Build curve for each group
    for t in pal.index:
        index_set = set(df[df[label] == t].index)

        # Build data-frame
        x = df[rank_label].sort_values().dropna()

        # Observed cumsum
        y = x.index.isin(index_set)
        y = np.cumsum(y) / sum(y)

        # Rank fold-changes
        x = st.rankdata(x) / x.shape[0]

        # Calculate AUC
        f_auc = auc(x, y)

        # Plot
        auc_label = 'Target' if t == 'T' else 'Distance {}'.format(t)
        ax.plot(x, y, label='{} (AURC={:.2f})'.format(auc_label, f_auc), lw=1., c=pal[t])

    # Random
    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

    # Limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Labels
    ax.set_xlabel('Ranked')
    ax.set_ylabel('Recall')

    ax.legend(loc=4)

    ax.set_title(title)
    legend = ax.legend(loc=4, title=legend_title, prop={'size': legend_size})
    legend.get_title().set_fontsize('{}'.format(legend_size))


def boxplot_kinobead(lmm_drug):
    drug_id_fdr = lmm_drug.groupby('DRUG_ID_lib')['fdr'].min()

    def from_ids_to_minfdr(ids):
        if str(ids).lower() == 'nan':
            return np.nan
        else:
            return drug_id_fdr.reindex(list(map(int, ids.split(';')))).min()

    catds = pd.read_csv('data/klaeger_et_al_catds_most_potent.csv')
    catds = catds.assign(fdr=[from_ids_to_minfdr(i) for i in catds['ids']])
    catds = catds.assign(signif=[('NA' if np.isnan(i) else ('Yes' if i < 0.05 else 'No')) for i in catds['fdr']])

    t, p = ttest_ind(catds[catds['signif'] == 'No']['CATDS_most_potent'], catds[catds['signif'] == 'NA']['CATDS_most_potent'], equal_var=False)

    order = ['No', 'Yes', 'NA']
    pal = {'No': PAL_DTRACE[2], 'Yes': PAL_DTRACE[0], 'NA': PAL_DTRACE[1]}

    sns.boxplot(catds['signif'], catds['CATDS_most_potent'], notch=True, palette=pal, linewidth=.3, fliersize=1.5, order=order)
    sns.swarmplot(catds['signif'], catds['CATDS_most_potent'], palette=pal, linewidth=.3, size=2, order=order)
    plt.axhline(0.5, lw=.3, c=PAL_DTRACE[1], ls='-', alpha=.8, zorder=0)

    sns.despine(top=True, right=True)

    plt.ylim((-0.1, 1.1))

    plt.xlabel('Drug has a significant\nCRISPR-Cas9 association')
    plt.ylabel('Selectivity[$CATDS_{most\ potent}$]')


if __name__ == '__main__':
    # - Linear regressions
    lmm_drug = pd.read_csv(dtrace.DRUG_LMM)

    lmm_drug = ppi_annotation(
        lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3,
    )

    lmm_drug = corr_drugtarget_gene(lmm_drug)

    # - Drug associations manhattan plot
    manhattan_plot(lmm_drug)
    plt.gcf().set_size_inches(8, 2)
    plt.savefig('reports/drug_associations_manhattan.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Top drug associations
    top_associations_barplot(lmm_drug)
    plt.gcf().set_size_inches(8, 6)
    plt.savefig('reports/drug_associations_barplot.pdf', bbox_inches='tight')
    plt.close('all')

    # - Count number of significant associations per drug
    plot_count_associations(lmm_drug)
    plt.gcf().set_size_inches(2, 8)
    plt.savefig('reports/drug_associations_count.pdf', bbox_inches='tight')
    plt.close('all')

    # - Count number of significant associations overall
    recapitulated_drug_targets_barplot(lmm_drug)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/drug_associations_count_signif.pdf', bbox_inches='tight')
    plt.close('all')

    # - Associations beta histogram
    beta_histogram(lmm_drug)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug_associations_beta_histogram.pdf', bbox_inches='tight')
    plt.close('all')

    # - Drug betas correlation
    beta_corr_boxplot(lmm_drug)
    plt.gcf().set_size_inches(3, 1)
    plt.savefig('reports/drug_associations_beta_corr_boxplot.pdf', bbox_inches='tight')
    plt.close('all')

    # - Drug betas TSNE
    drug_beta_tsne(lmm_drug)
    plt.savefig('reports/drug_associations_beta_tsne.pdf', bbox_inches='tight')
    plt.close('all')

    # - Drug target/ppi enrichment curves
    drug_aurc(lmm_drug)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/drug_associations_aurc.pdf', bbox_inches='tight')
    plt.close('all')

    # - Drug target and PPI annotation AURCs
    drug_aurc(lmm_drug)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/drug_associations_aurc.pdf', bbox_inches='tight')
    plt.close('all')

    # - Drug kinobeads boxplot
    boxplot_kinobead(lmm_drug)
    plt.gcf().set_size_inches(1, 2)
    plt.savefig('reports/drug_associations_kinobeads.pdf', bbox_inches='tight')
    plt.close('all')
