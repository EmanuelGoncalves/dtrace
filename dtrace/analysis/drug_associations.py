#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from natsort import natsorted
from sklearn.metrics import auc
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE
from dtrace.analysis import PAL_DTRACE
from sklearn.preprocessing import StandardScaler
from analysis.plot.corrplot import plot_corrplot_discrete
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.associations import ppi_annotation, DRUG_INFO_COLUMNS, multipletests_per_drug

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

DRUG_TARGETS_HUE = [(sns.light_palette(c, n_colors=len(s) + 1, reverse=True).as_hex()[:-1], s) for c, s in DRUG_TARGETS_HUE]


def top_associations_barplot(lmm_drug, fdr_line=0.1, ntop=60, n_cols=16):
    # Filter for signif associations
    df = lmm_drug\
        .query('fdr < {}'.format(fdr_line))\
        .sort_values('fdr')\
        .groupby(['DRUG_NAME', 'GeneSymbol'])\
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
    f, axs = plt.subplots(int(np.ceil(ntop / n_cols)), 1, sharex=False, sharey=True, gridspec_kw=dict(hspace=.0))

    # Barplot
    for irow in set(df['irow']):
        df_irow = df[df['irow'] == irow]

        df_irow_ = df_irow.query("target != 'T'")
        axs[irow].bar(df_irow_['xpos'], df_irow_['logpval'], .8, color=PAL_DTRACE[2], align='center', zorder=5, linewidth=0)

        df_irow_ = df_irow.query("target == 'T'")
        axs[irow].bar(df_irow_['xpos'], df_irow_['logpval'], .8, color=PAL_DTRACE[0], align='center', zorder=5, linewidth=0)

        for k, v in df_irow.groupby('DRUG_NAME')['xpos'].min().sort_values().to_dict().items():
            axs[irow].text(v - 1.2, 0.1, textwrap.fill(k, 15), va='bottom', fontsize=7, zorder=10, rotation='vertical', color=PAL_DTRACE[2])

        for g, p in df_irow[['GeneSymbol', 'xpos']].values:
            axs[irow].text(p, 0.1, g, ha='center', va='bottom', fontsize=5, zorder=10, rotation='vertical', color='white')

        for x, y, t, b in df_irow[['xpos', 'logpval', 'target', 'beta']].values:
            c = PAL_DTRACE[0] if t == 'T' else PAL_DTRACE[2]
            axs[irow].text(x, y + 0.25, t, color=c, ha='center', fontsize=6, zorder=10)
            axs[irow].text(x, -1.2, f'{b:.1f}', color=c, ha='center', fontsize=6, rotation='vertical', zorder=10)

        sns.despine(ax=axs[irow], right=True, top=True)
        axs[irow].axes.get_xaxis().set_ticks([])
        axs[irow].set_ylabel('Drug-gene association\n(-log10 p-value)')

    plt.gcf().set_size_inches(10, 6)


def plot_count_associations(lmm_drug, fdr_line=0.05, min_events=2):
    df = lmm_drug[lmm_drug['fdr'] < fdr_line]
    df = df.groupby(DRUG_INFO_COLUMNS)['fdr'].count().rename('count').sort_values(ascending=False).reset_index()
    df = df.assign(name=['{} ({}, {})'.format(n, i, v) for i, n, v in df[DRUG_INFO_COLUMNS].values])
    df = df.query('count >= {}'.format(min_events))

    g = sns.barplot('count', 'name', data=df, color=PAL_DTRACE[2], linewidth=.8)
    sns.despine(right=True, top=True)

    # for item in g.get_xticklabels():
    #     item.set_rotation(90)

    # g.yaxis.set_major_locator(plticker.MultipleLocator(base=1))

    plt.ylabel('')
    plt.xlabel('Count')
    plt.title('Significant associations (adj p-value < {}%)'.format(fdr_line * 100))


def recapitulated_drug_targets_barplot_per_screen(lmm_drug, fdr=0.05):
    # Count number of drugs
    df_genes = set(lmm_drug['GeneSymbol'])

    d_targets = dtrace.get_drugtargets()

    d_all = {tuple(i) for i in lmm_drug[DRUG_INFO_COLUMNS].values}
    d_annot = {tuple(i) for i in d_all if i[0] in d_targets}
    d_tested = {tuple(i) for i in d_annot if len(d_targets[i[0]].intersection(df_genes)) > 0}
    d_tested_signif = {tuple(i) for i in lmm_drug.query('fdr < {}'.format(fdr))[DRUG_INFO_COLUMNS].values if tuple(i) in d_tested}
    d_tested_correct = {tuple(i) for i in lmm_drug.query("fdr < {} & target == 'T'".format(fdr))[DRUG_INFO_COLUMNS].values if tuple(i) in d_tested_signif}

    d_labels = ['All', 'w/Target', 'w/Tested target', 'w/Signif tested target', 'w/Correct target']
    d_lists = list(zip(*(d_labels, [d_all, d_annot, d_tested, d_tested_signif, d_tested_correct])))

    # Build dataframe
    screens = ['RS', 'v17']
    plot_df = pd.DataFrame([
        {'count': len([i for i in d_list if i[2] == s]), 'screen': s, 'names': d_label} for d_label, d_list in d_lists for s in screens
    ])

    pal = dict(RS=PAL_DTRACE[0], v17=PAL_DTRACE[2])

    sns.barplot('count', 'names', 'screen', data=plot_df, palette=pal, orient='h', linewidth=0, saturation=1)


def beta_corr_boxplot(betas_corr):
    # Dataframe
    df = betas_corr.copy()

    df = df.assign(drug1_screen=[i.split(' ; ')[2] for i in df['drug_1']])
    df = df.assign(drug2_screen=[i.split(' ; ')[2] for i in df['drug_2']])

    df = df[df['drug1_screen'] == df['drug2_screen']]

    df = df.replace({'dtype': {'Same Drug & Screen': 'Same Drug'}})

    # Plot
    order = ['Different', 'Same Target', 'Same Drug']
    pal = dict(v17=PAL_DTRACE[2], RS=PAL_DTRACE[0])

    sns.violinplot(
        'r', 'dtype', 'drug1_screen', data=df, order=order, palette=pal, orient='h',
        linewidth=.75, split=True, cut=0, inner='quartile', saturation=1
    )

    sns.despine(top=True, right=True)
    plt.axvline(0, lw=0.3, c=PAL_DTRACE[1], zorder=0)

    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), prop={'size': 5}, frameon=False)
    plt.xlabel('Correlation of drug pairs association profiles')
    plt.ylabel('Drug pairs')
    plt.title('Drug association profiles')


def drug_aurc(lmm_drug, fdr=.1, rank_label='pval', legend_size=4, legend_title='', ax=None):
    # Subset to entries that have a target
    df = lmm_drug.query("target != '-'")

    drugs_signif = {tuple(i) for i in df.query('fdr < {}'.format(fdr))[DRUG_INFO_COLUMNS].values}
    df = df[[tuple(i) in drugs_signif for i in df[DRUG_INFO_COLUMNS].values]]

    # Define order and palette
    order = natsorted(set(df['target']))
    order.insert(0, order.pop(order.index('T')))

    pal = [PAL_DTRACE[0]] + sns.light_palette(PAL_DTRACE[2], n_colors=len(order)).as_hex()[1:]
    pal = pd.Series(pal, index=order)

    # Initial plot
    if ax is None:
        ax = plt.gca()

    # Build curve for each group
    for t in pal.index:
        index_set = set(df[df['target'] == t].index)

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
        auc_label = 'Target' if t == 'T' else 'PPI distance {}'.format(t)
        ax.plot(x, y, label='{} (AURC={:.2f})'.format(auc_label, f_auc), lw=1., c=pal[t])

    # Random
    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

    # Limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Labels
    ax.set_xlabel('Association ranked by p-value')
    ax.set_ylabel('Recall')

    ax.legend(loc=4, frameon=False)

    legend = ax.legend(loc=4, title=legend_title, prop={'size': legend_size}, frameon=False)
    legend.get_title().set_fontsize('{}'.format(legend_size))

    ax.set_title('Drug ~ Gene associations\nnetwork interactions enrichment')

    return ax


def boxplot_kinobead(lmm_drug, fdr_thres=.1, ax=None):
    d_targets = dtrace.get_drugtargets()

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
    catds = catds.assign(signif=[('NA' if np.isnan(i) else ('Yes' if i < fdr_thres else 'No')) for i in catds['fdr']])
    catds = catds.assign(targets=[targets_from_ids(i) for i in catds['ids']])

    t, p = ttest_ind(catds[catds['signif'] == 'No']['CATDS_most_potent'], catds[catds['signif'] == 'NA']['CATDS_most_potent'], equal_var=False)

    # Plot
    if ax is None:
        ax = plt.gca()

    order = ['No', 'Yes', 'NA']
    pal = {'No': PAL_DTRACE[2], 'Yes': PAL_DTRACE[0], 'NA': PAL_DTRACE[1]}

    sns.boxplot(catds['signif'], catds['CATDS_most_potent'], notch=True, palette=pal, linewidth=.3, fliersize=1.5, order=order, ax=ax)
    sns.swarmplot(catds['signif'], catds['CATDS_most_potent'], palette=pal, linewidth=.3, size=2, order=order, ax=ax)
    ax.axhline(0.5, lw=.3, c=PAL_DTRACE[1], ls='-', alpha=.8, zorder=0)

    sns.despine(top=True, right=True, ax=ax)

    ax.set_ylim((-0.1, 1.1))

    ax.set_xlabel('Drug has a significant\nCRISPR-Cas9 association')
    ax.set_ylabel('Selectivity[$CATDS_{most\ potent}$]')


def drug_targets_count(lmm_drug, min_events=5, fdr=0.05):
    d_targets = dtrace.get_drugtargets()

    df = pd.DataFrame([
        {'drug': i, 'target': t} for i in set(lmm_drug[DRUG_INFO_COLUMNS[0]]) if i in d_targets for t in d_targets[i]
    ])

    df_signif = set(lmm_drug.query("(fdr < {}) & (target == 'T')".format(fdr))[DRUG_INFO_COLUMNS[0]])
    df_signif = pd.DataFrame([{'drug': i, 'target': t} for i in df_signif if i in d_targets for t in d_targets[i]])

    # Build data-frame
    plot_df = pd.concat([
        df['target'].value_counts().rename('all'),
        df_signif['target'].value_counts().rename('signif')
    ], axis=1).replace(np.nan, 0).sort_values('all', ascending=True).astype(int).reset_index()

    plot_df = plot_df.query('all >= {}'.format(min_events))

    plot_df = plot_df.assign(y=range(plot_df.shape[0]))

    # Plot
    plt.barh(plot_df['y'], plot_df['all'], color=PAL_DTRACE[2], label='Drugs targeting')
    plt.barh(plot_df['y'], plot_df['signif'], color=PAL_DTRACE[0], label='Drugs targeting - significant')

    sns.despine(right=True, top=True)

    for c, y in plot_df[['all', 'y']].values:
        plt.text(c + .25, y, str(c), va='center', ha='left', fontsize=5, zorder=10, color=PAL_DTRACE[2])

    for c, y in plot_df[['signif', 'y']].values:
        plt.text(c - 0.25, y, str(c), va='center', ha='right', fontsize=5, zorder=10, color='white')

    plt.yticks(plot_df['y'], plot_df['index'])
    plt.xlabel('Number of drugs')
    plt.ylabel('')

    plt.legend(prop={'size': 5}, frameon=False)


def drug_betas_corr(lmm_drug):
    # Build betas matrix
    betas = pd.pivot_table(lmm_drug, index=DRUG_INFO_COLUMNS, columns='GeneSymbol', values='beta')

    # Drug information
    drugsheet = dtrace.get_drugsheet()

    # Drug targets
    drugtargets = dtrace.get_drugtargets()

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

        # Check drug targets
        d1_targets = drugtargets[d1_id] if d1_id in drugtargets else set()
        d2_targets = drugtargets[d2_id] if d2_id in drugtargets else set()

        # Check if Drugs were screened with same panel
        if dsame & (d1_screen == d2_screen):
            return 'Same Drug & Screen'
        elif dsame:
            return 'Same Drug'
        elif len(d1_targets.intersection(d2_targets)) > 0:
            return 'Same Target'
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

    return betas_corr


def drug_betas_tsne(lmm_drug, perplexity, learning_rate, n_iter, fdr=.05):
    d_targets = dtrace.get_drugtargets()

    # Drugs into
    drugs = {tuple(i) for i in lmm_drug[DRUG_INFO_COLUMNS].values}
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

    # Annotate significant
    d_signif = {tuple(i) for i in lmm_drug.query('fdr < {}'.format(fdr))[DRUG_INFO_COLUMNS].values}
    tsnes = tsnes.assign(has_signif=['Yes' if tuple(i) in d_signif else 'No' for i in tsnes[DRUG_INFO_COLUMNS].values])

    return tsnes


def drug_beta_tsne(tsnes, hueby):
    if hueby == 'signif':
        pal = {'No': PAL_DTRACE[2], 'Yes': PAL_DTRACE[0]}

        g = sns.FacetGrid(
            tsnes, col='VERSION', hue='has_signif', palette=pal, hue_order=['No', 'Yes'], sharey=False,
            sharex=False, legend_out=True, despine=False, size=2, aspect=1
        )

        g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
        g.map(plt.axhline, y=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)
        g.map(plt.axvline, x=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)

        g.add_legend(title='Significant', prop=dict(size=4), frameon=False)

    elif hueby == 'pathway':
        ds = dtrace.get_drugsheet().dropna(subset=['Pathway'])

        df = tsnes.assign(pathway=ds.reindex(tsnes['DRUG_ID_lib'])['Pathway'].fillna('Other').values)

        path_count = df['pathway'].value_counts()
        df.loc[df['pathway'].isin(list(path_count[path_count <= 3].index) + ['Other, kinases']), 'pathway'] = 'Other'

        order = natsorted(set(df['pathway']).difference(['Other']))
        pal = pd.Series(sns.color_palette('tab20c', n_colors=len(order)).as_hex(), index=order).to_dict()

        g = sns.FacetGrid(
            df, col='VERSION', hue='pathway', palette=pal, hue_order=order, sharey=False, sharex=False, legend_out=True, despine=False, size=2, aspect=1
        )

        g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
        g.map(plt.axhline, y=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)
        g.map(plt.axvline, x=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)

        g.add_legend(title='Drug type', prop=dict(size=6), fontsize=6, frameon=False)

    elif hueby == 'type':
        ds = dtrace.get_drugsheet().dropna(subset=['Drug Type'])

        df = tsnes.assign(type=ds.reindex(tsnes['DRUG_ID_lib'])['Drug Type'].fillna('ND').values)

        order = ['cytotoxic', 'targetted']
        pal = pd.Series([PAL_DTRACE[0], PAL_DTRACE[2]], index=order).to_dict()

        g = sns.FacetGrid(
            df.query("type != 'ND'"), col='VERSION', hue='type', palette=pal, hue_order=order, sharey=False,
            sharex=False, legend_out=True, despine=False, size=2, aspect=1
        )

        for i, s in enumerate(['v17', 'RS']):
            ax = g.axes.ravel()[i]
            df_plot = df.query("(type == 'ND') & (VERSION == '{}')".format(s))
            ax.scatter(df_plot['P1'], df_plot['P2'], color=PAL_DTRACE[1], marker='x', lw=0.3, s=5, alpha=0.7, label='No drug info')

        g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
        g.map(plt.axhline, y=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)
        g.map(plt.axvline, x=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)

        g.add_legend(title='Drug type', prop=dict(size=6), fontsize=6, frameon=False)

    elif hueby == 'target':
        pal = {'No': PAL_DTRACE[2], 'Yes': PAL_DTRACE[0]}

        g = sns.FacetGrid(
            tsnes, col='VERSION', hue='target', palette=pal, hue_order=['Yes', 'No'], sharey=False,
            sharex=False, legend_out=True, despine=False, size=2, aspect=1
        )

        g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
        g.map(plt.axhline, y=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)
        g.map(plt.axvline, x=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)

        g.add_legend(title='Known target?', prop=dict(size=4), frameon=False)

    elif hueby == 'replicates':
        rep_names = set(tsnes['rep'])

        pal_v17 = [n for n in rep_names if n.endswith(';v17')]
        pal_v17 = dict(zip(*(pal_v17, sns.color_palette('tab20', n_colors=len(pal_v17)).as_hex())))

        pal_rs = [n for n in rep_names if n.endswith(';RS')]
        pal_rs = dict(zip(*(pal_rs, sns.color_palette('tab20', n_colors=len(pal_rs)).as_hex())))

        pal = {**pal_v17, **pal_rs}
        pal['NA'] = PAL_DTRACE[1]

        g = sns.FacetGrid(
            tsnes, col='VERSION', hue='rep', palette=pal, sharey=False, sharex=False, legend_out=True,
            despine=False, size=2, aspect=1
        )

        g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
        g.map(plt.axhline, y=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)
        g.map(plt.axvline, x=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)

        g.add_legend(title='', prop=dict(size=4), frameon=False)

    elif type(hueby) == list:
        sets = [i for l in hueby for i in l[1]]
        labels = [';'.join(i) for l in hueby for i in l[1]]
        colors = [i for l in hueby for i in l[0]]

        pal = dict(zip(*(labels, colors)))
        pal['NA'] = PAL_DTRACE[1]

        df = tsnes.assign(hue=[[i for i, g in enumerate(sets) if g.intersection(t.split(';'))] for t in tsnes['targets']])
        df = tsnes.assign(hue=[labels[i[0]] if len(i) > 0 else 'NA' for i in df['hue']])

        g = sns.FacetGrid(
            df.query("target == 'Yes'"), col='VERSION', hue='hue', palette=pal, sharey=False, sharex=False, legend_out=True, despine=False, size=2, aspect=1
        )

        for i, s in enumerate(['v17', 'RS']):
            ax = g.axes.ravel()[i]
            df_plot = df.query("(target == 'No') & (VERSION == '{}')".format(s))
            ax.scatter(df_plot['P1'], df_plot['P2'], color=PAL_DTRACE[1], marker='x', lw=0.3, s=5, alpha=0.7, label='No target info')

        g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
        g.map(plt.axhline, y=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)
        g.map(plt.axvline, x=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)

        g.add_legend(title='', prop=dict(size=4), label_order=labels + ['NA'] + ['No info'], frameon=False)

    g.set_titles('Screen = {col_name}')


def drug_targets_not_tested(lmm_drug, top_n=20):
    crispr = dtrace.get_crispr(dtype='both')

    # - Count number of drugs
    df_genes = set(lmm_drug['GeneSymbol'])

    d_targets = dtrace.get_drugtargets()

    d_all = {tuple(i) for i in lmm_drug[DRUG_INFO_COLUMNS].values}
    d_annot = {tuple(i) for i in d_all if i[0] in d_targets}

    d_not_tested = {tuple(i) for i in d_annot if (len(d_targets[i[0]].intersection(df_genes)) == 0) and (len(d_targets[i[0]].intersection(crispr.index)) > 0)}

    # - Plot
    plot_df = pd.Series([t for d in d_not_tested for t in d_targets[d[0]]])\
        .value_counts().rename('count').reset_index().head(top_n)

    sns.barplot('count', 'index', data=plot_df, color=PAL_DTRACE[2])

    plt.axes().xaxis.set_major_locator(plticker.MultipleLocator(base=1.))

    plt.title(f'Traget frequency of responding drugs\nwith non-essential targets (top {top_n})')
    plt.xlabel('Number of drugs')
    plt.ylabel('Drug targets')

    sns.despine()


def drug_v17_id_batch(tsnes):
    hue_order = ['[0, 200[', '[200, 1000[', '[1000, inf.[']

    plot_df = tsnes[tsnes['VERSION'] == 'v17']
    plot_df = plot_df.assign(
        id_discrete=pd.cut(
            plot_df['DRUG_ID_lib'],
            bins=[0, 200, 1000, np.max(plot_df['DRUG_ID_lib'])],
            labels=hue_order
        )
    )

    discrete_pal = pd.Series(PAL_DTRACE, index=set(plot_df['id_discrete'])).to_dict()

    g = plot_corrplot_discrete(
        'P1', 'P2', 'id_discrete', plot_df, discrete_pal=discrete_pal, legend_title='DRUG_ID discretised', hue_order=hue_order
    )

    g.set_axis_labels('', '')

    return g


if __name__ == '__main__':
    # - Linear regressions
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS)
    lmm_drug = lmm_drug[['+' not in i for i in lmm_drug['DRUG_NAME']]]
    lmm_drug = ppi_annotation(lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)

    # - Drug betas correlation
    betas_corr = drug_betas_corr(lmm_drug)
    betas_corr.sort_values('r').to_csv(dtrace.DRUG_BETAS_CORR, index=False)

    # - Drug betas tSNE
    tsnes = drug_betas_tsne(lmm_drug, perplexity=15, learning_rate=250, n_iter=2000)
    tsnes.to_csv(dtrace.DRUG_BETAS_TSNE, index=False)

    # - Drug targets not tested
    drug_targets_not_tested(lmm_drug)
    plt.gcf().set_size_inches(1.5, 2.5)
    plt.savefig('reports/drug_not_tested_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Drug betas TSNE
    # Drug screen v17 ID batch effect
    drug_v17_id_batch(tsnes)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug_tsne_v17_ids_discrete.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Has drug significant association
    drug_beta_tsne(tsnes, hueby='signif')
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Has drug targets annotation
    drug_beta_tsne(tsnes, hueby='target')
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Drug type
    drug_beta_tsne(tsnes, hueby='type')
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne_type.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Drug pathway
    drug_beta_tsne(tsnes, hueby='pathway')
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne_pathway.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Drug replicates within screen
    drug_beta_tsne(tsnes, hueby='replicates')
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne_replicates.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Drug targets
    drug_beta_tsne(tsnes, hueby=DRUG_TARGETS_HUE)
    plt.suptitle('tSNE analysis of drug associations', y=1.05)
    plt.savefig('reports/drug_associations_beta_tsne_targets.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Drug targets countplot
    drug_targets_count(lmm_drug, min_events=5, fdr=0.1)
    plt.title('Drug targets histogram')
    plt.gcf().set_size_inches(2, 6)
    plt.savefig('reports/drug_targets_count.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Top drug associations
    top_associations_barplot(lmm_drug, fdr_line=.1, ntop=60)
    plt.savefig('reports/drug_associations_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Count number of significant associations per drug
    plot_count_associations(lmm_drug, min_events=3, fdr_line=.1)
    plt.gcf().set_size_inches(2, 4)
    plt.savefig('reports/drug_associations_count.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Count number of significant associations overall
    recapitulated_drug_targets_barplot_per_screen(lmm_drug)
    plt.legend(frameon=False)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/drug_associations_count_signif_per_screen.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Drug betas correlation
    beta_corr_boxplot(betas_corr)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/drug_associations_beta_corr_boxplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Drug target and PPI annotation AURCs
    ax = drug_aurc(lmm_drug, fdr=.1)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug_associations_aurc.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Drug kinobeads boxplot
    boxplot_kinobead(lmm_drug)
    plt.gcf().set_size_inches(1, 2)
    plt.savefig('reports/drug_associations_kinobeads.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
