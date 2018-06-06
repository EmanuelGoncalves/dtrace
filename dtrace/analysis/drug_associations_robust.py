#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.analysis import PAL_DTRACE
from dtrace.associations import DRUG_INFO_COLUMNS
from dtrace.analysis.plot.corrplot import plot_corrplot_discrete


if __name__ == '__main__':
    # - Import
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()
    crispr = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    d_maxc = pd.read_csv(dtrace.DRUG_RESPONSE_MAXC, index_col=[0, 1, 2])

    # - Robust associations
    lmm_drug_robust = pd.read_csv(dtrace.LMM_ASSOCIATIONS_ROBUST)

    # - Strongest significant association per drug
    associations = lmm_drug_robust.query('fdr_crispr < 0.05 & fdr_drug < 0.05')
    associations = associations.sort_values('fdr_crispr').groupby(DRUG_INFO_COLUMNS).head(1)

    # -
    idx = 75293

    columns = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol', 'Genetic']
    association = lmm_drug_robust.loc[idx, columns]

    plot_df = pd.concat([
        crispr.loc[association[3]],
        drespo.loc[tuple(association[:3])].rename('{} [{}]'.format(association[1], association[2])),
        mobems.loc[association[4]]
    ], axis=1).dropna()

    g = plot_corrplot_discrete(association[3], '{} [{}]'.format(association[1], association[2]), association[4], plot_df)
    g.ax_joint.axhline(np.log(d_maxc.loc[tuple(association[:3]), 'max_conc_micromolar']), lw=.1, color=PAL_DTRACE[2], ls='--')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/corrplot_drug_crispr_genetic.pdf', bbox_inches='tight')
    plt.close('all')
