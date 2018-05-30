#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from dtrace.analysis import PAL_DTRACE
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.associations import ppi_annotation, DRUG_INFO_COLUMNS


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


def top_associations_barplot(lmm_drug, fdr_line=0.05, ntop=30, ppi_text_offset=0.075, drug_name_offset=1., ylim_offset=1.1):
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

        axs[irow].bar(df_irow.query('target != 0')['xpos'], df_irow.query('target != 0')['logpval'], .8, color=PAL_DTRACE[2], align='center', zorder=5)
        axs[irow].bar(df_irow.query('target == 0')['xpos'], df_irow.query('target == 0')['logpval'], .8, color=PAL_DTRACE[0], align='center', zorder=5)

        for k, v in df_irow.groupby('DRUG_NAME')['xpos'].min().sort_values().to_dict().items():
            axs[irow].text(v - 1.25, 0.1, textwrap.fill(k, 15), va='bottom', fontsize=7, zorder=10, rotation='vertical', color=PAL_DTRACE[2])

        for g, p in df_irow[['GeneSymbol', 'xpos']].values:
            axs[irow].text(p, 0.1, g, ha='center', va='bottom', fontsize=5, zorder=10, rotation='vertical', color='white')

        for x, y, t in df_irow[['xpos', 'logpval', 'target']].values:
            l = '-' if np.isnan(t) or np.isposinf(t) else ('T' if t == 0 else str(int(t)))
            axs[irow].text(x, y - (df['logpval'].max() * ppi_text_offset), l, color='white', ha='center', fontsize=6, zorder=10)

        sns.despine(ax=axs[irow], right=True, top=True)
        axs[irow].axes.get_xaxis().set_ticks([])
        axs[irow].set_ylabel('Drug-gene association\n(-log10 p-value)')


if __name__ == '__main__':
    # - Linear regressions
    lmm_drug = pd.read_csv(dtrace.DRUG_LMM)

    lmm_drug = ppi_annotation(
        lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3,
    )

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
