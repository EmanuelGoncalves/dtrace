#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from dtrace.analysis import PAL_DTRACE
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.associations import ppi_annotation, corr_drugtarget_gene, DRUG_INFO_COLUMNS


if __name__ == '__main__':
    # - Import data-set
    crispr = dtrace.get_crispr(dtype='both')

    # - Linear regressions
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS)
    lmm_drug = ppi_annotation(lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)
    lmm_drug = corr_drugtarget_gene(lmm_drug)

    # - Count number of drugs
    df_genes = set(lmm_drug['GeneSymbol'])

    d_targets = dtrace.get_drugtargets()

    d_all = {tuple(i) for i in lmm_drug[DRUG_INFO_COLUMNS].values}
    d_annot = {tuple(i) for i in d_all if i[0] in d_targets}

    d_tested = {tuple(i) for i in d_annot if len(d_targets[i[0]].intersection(df_genes)) > 0}
    d_not_tested = {tuple(i) for i in d_annot if (len(d_targets[i[0]].intersection(df_genes)) == 0) and (len(d_targets[i[0]].intersection(crispr.index)) > 0)}

    d_not_covered = {tuple(i) for i in d_annot if len(d_targets[i[0]].intersection(crispr.index)) == 0}

    # - Plot
    plot_df = pd.Series([t for d in d_not_tested for t in d_targets[d[0]]])\
        .value_counts().rename('count').reset_index().head(20)

    sns.barplot('count', 'index', data=plot_df, color=PAL_DTRACE[2])

    plt.axes().xaxis.set_major_locator(plticker.MultipleLocator(base=1.))

    plt.title('Traget frequency of responding drugs\nwith non-essential targets')
    plt.xlabel('Number of drugs')
    plt.ylabel('Drug targets')

    sns.despine()

    plt.gcf().set_size_inches(1.5, 2.5)
    plt.savefig('reports/drug_not_tested_barplot.pdf', bbox_inches='tight')
    plt.close('all')

