#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.analysis import PAL_DTRACE
from dtrace.Associations import DRUG_INFO_COLUMNS
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.analysis.plot.corrplot import plot_corrplot_discrete
from dtrace.Associations import ppi_annotation, corr_drugtarget_gene, DRUG_INFO_COLUMNS


def _get_id(d_name_ext):
    return int(d_name_ext.split(' ; ')[0])


if __name__ == '__main__':
    # - Import
    # Max concentration
    d_maxc = pd.read_csv(dtrace.DRUG_RESPONSE_MAXC, index_col=[0, 1, 2])

    # Betas correlation
    betas_corr = pd.read_csv(dtrace.DRUG_BETAS_CORR)

    # tSNEs compounents
    tsnes = pd.read_csv(dtrace.DRUG_BETAS_TSNE)

    # Drug targets
    drugtargets = dtrace.get_drugtargets()

    # Data-sets
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()
    crispr = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # Linear regressions
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS)
    lmm_drug = ppi_annotation(lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)
    lmm_drug = corr_drugtarget_gene(lmm_drug)

    # Robust associations
    lmm_drug_robust = pd.read_csv(dtrace.LMM_ASSOCIATIONS_ROBUST)

    # - FLI1 vignette
    idx_robust = 75293

    # Build dataframe
    columns = DRUG_INFO_COLUMNS + ['GeneSymbol', 'Genetic']
    association = lmm_drug_robust.loc[idx_robust, columns]

    name = 'Drug={}, Gene={}, Genetic={} [{}, {}]'.format(association[1], association[3], association[4], association[0], association[2])

    plot_df = pd.concat([
        crispr.loc[association[3]],
        drespo.loc[tuple(association[:3])].rename('{} [{}]'.format(association[1], association[2])),
        mobems.loc[association[4]]
    ], axis=1).dropna()

    # Corrplot
    g = plot_corrplot_discrete(association[3], '{} [{}]'.format(association[1], association[2]), association[4], plot_df)

    g.ax_joint.axhline(np.log(d_maxc.loc[tuple(association[:3]), 'max_conc_micromolar']), lw=.1, color=PAL_DTRACE[2], ls='--')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/lmm_robust_{}.pdf'.format(name), bbox_inches='tight')
    plt.close('all')

    # Top associated drugs
    d_name_exp = ' ; '.join(map(str, association[:3]))

    d_corr = betas_corr[(betas_corr['drug_1'] == d_name_exp) | (betas_corr['drug_2'] == d_name_exp)]
    d_corr = d_corr[[np.any([_get_id(d) in drugtargets for d in [d1, d2] if d != d_name_exp]) for d1, d2 in d_corr[['drug_1', 'drug_2']].values]]
    d_corr = d_corr.assign(targets=[[drugtargets[_get_id(d)] for d in [d1, d2] if d != d_name_exp] for d1, d2 in d_corr[['drug_1', 'drug_2']].values])
    d_corr = d_corr.assign(targets=['; '.join(i[0]) for i in d_corr['targets']])
    d_corr = d_corr.assign(name=[[d.split(' ; ')[1] for d in [d1, d2] if d != d_name_exp][0] for d1, d2 in d_corr[['drug_1', 'drug_2']].values])

    #
    plot_df = d_corr.sort_values('r', ascending=False).head(10)
    plot_df = plot_df.sort_values('r')
    plot_df = plot_df.assign(y=range(plot_df.shape[0]))

    plt.barh(plot_df['y'], plot_df['r'], color=PAL_DTRACE[2])
    sns.despine(right=True, top=True)

    for c, y, t in plot_df[['r', 'y', 'targets']].values:
        plt.text(c, y, t, va='center', fontsize=5, zorder=10, color='white', ha='right')

    plt.yticks(plot_df['y'], plot_df['name'])
    plt.xlabel('Correlation')
    plt.ylabel('Similar drug')
    plt.title('Correlation of drug association profiles')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/lmm_robust_{}_corr_barplot.pdf'.format(name), bbox_inches='tight')
    plt.close('all')

    #
    plot_df = tsnes.assign(selected=(tsnes[DRUG_INFO_COLUMNS[0]] == association[0]).astype(int).values)

    pal = {0: PAL_DTRACE[2], 1: PAL_DTRACE[0]}

    g = sns.FacetGrid(
        plot_df, col='VERSION', hue='selected', palette=pal, sharey=False, sharex=False, legend_out=True, despine=False, size=2, aspect=1
    )

    g.map(plt.scatter, 'P1', 'P2', alpha=1., lw=.3, edgecolor='white', s=10)
    g.map(plt.axhline, y=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)
    g.map(plt.axvline, x=0, ls='-', lw=0.3, c=PAL_DTRACE[1], alpha=.2, zorder=0)

    plt.savefig('reports/lmm_robust_{}_tsne.pdf'.format(name), bbox_inches='tight')
    plt.close('all')
