#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves


import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.analysis import PAL_DTRACE
from associations import multipletests_per_drug
from dtrace.associations import DRUG_INFO_COLUMNS
from analysis.plot.corrplot import plot_corrplot_discrete
from associations.lmm_drug import lmm_association as lmm_drug_regression
from associations.lmm_robust_associations import lmm_association as lmm_robust_regression


if __name__ == '__main__':
    # - Import
    ss = dtrace.get_samplesheet()
    ss = ss[ss['Cancer Type'] == 'Colorectal Carcinoma']

    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='both')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    d_maxc = pd.read_csv(dtrace.DRUG_RESPONSE_MAXC, index_col=[0, 1, 2])

    # - Overlap
    samples = list(set.intersection(set(mobems), set(drespo), set(crispr), set(ss.index)))
    print('#(Samples) = {}'.format(len(samples)))

    drespo, crispr, crispr_logfc, mobems = drespo[samples], crispr[samples], crispr_logfc[samples], mobems[samples]

    # - Filter
    drespo = dtrace.filter_drugresponse(drespo)
    crispr = dtrace.filter_crispr(crispr)
    crispr_logfc = crispr_logfc.loc[crispr.index]

    # - Linear regressions
    lmm_drug = pd.concat([lmm_drug_regression(d, drespo, crispr_logfc) for d in drespo.index])
    lmm_drug = multipletests_per_drug(lmm_drug, field='pval')
    print(lmm_drug.sort_values('pval').head(60))

    # - Robust regressions
    lmm_robust = pd.concat([
        lmm_robust_regression(a, drespo, crispr_logfc, mobems, 3) for a in lmm_drug.query('fdr < .1')[DRUG_INFO_COLUMNS + ['GeneSymbol']].values
    ])

    lmm_robust = multipletests_per_drug(lmm_robust, field='pval_drug', fdr_field='fdr_drug')
    lmm_robust = multipletests_per_drug(lmm_robust, field='pval_crispr', fdr_field='fdr_crispr')

    # -
    comb = lmm_robust.query('fdr_drug < 0.2')

    # -
    indices = [9376]

    for idx in indices:
        columns = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol', 'Genetic']
        a = comb.loc[idx, columns]

        name = f'Drug={a[1]}, Gene={a[3]}, Genetic={a[4]} [{a[0]}, {a[2]}]'

        plot_df = pd.concat([
            crispr_logfc.loc[a[3], samples],
            drespo.loc[tuple(a[:3]), samples].rename(f'{a[1]} [{a[2]}]'),
            mobems.loc[a[4], samples]
        ], axis=1).dropna()

        g = plot_corrplot_discrete(a[3], f'{a[1]} [{a[2]}]', a[4], plot_df)

        g.ax_joint.axhline(np.log(d_maxc.loc[tuple(a[:3]), 'max_conc_micromolar']), lw=.1, color=PAL_DTRACE[2], ls='--')

        plt.gcf().set_size_inches(2, 2)
        plt.savefig(f'reports/comb_robust_{name}.pdf', bbox_inches='tight')
        plt.close('all')
