#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from cdrug.plot.corrplot import plot_corrplot


if __name__ == '__main__':
    # WES
    wes = pd.read_csv(cdrug.WES_COUNT)

    # Samplesheet
    ss = cdrug.get_samplesheet()

    # Linear regressions
    lm_df_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)

    # Drug response
    drespo = cdrug.get_drugresponse()

    # CIRSPR CN corrected logFC
    crispr = cdrug.get_crispr(dtype='logFC')

    # MOBEM
    mobem = cdrug.get_mobem()

    samples = list(set(drespo).intersection(crispr).intersection(mobem))
    print(len(samples))

    # - Annotate regressions with Drug -> Target -> Protein (in PPI)
    lm_df_crispr = cdrug.ppi_annotation(lm_df_crispr, exp_type={'Affinity Capture-MS', 'Affinity Capture-Western'}, int_type={'physical'})

    # --
    idx = 6

    d_id, d_name, d_screen, gene = lm_df_crispr.loc[idx, ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']].values

    x = drespo.loc[(d_id, d_name, d_screen), samples].dropna()
    y = crispr.loc[lm_df_crispr.loc[idx, 'GeneSymbol'], x.index]

    lm_1 = sm.OLS(y, sm.add_constant(x)).fit()
    print(lm_1.summary())

    z = mobem[y.index].T.astype(float)
    z = z.loc[:, z.sum() >= 3]

    resid_sum = lm_1.resid.to_frame().T.dot(z).loc[0]
    resid_mean = resid_sum / z.sum()

    lm_2 = sm.OLS(lm_1.resid, sm.add_constant(z)).fit()
    print(lm_2.params.sort_values())

    # -
    x, y, z = '{}'.format(gene), '{} {}'.format(d_name, d_screen), '{}'.format('EGFR_mut')

    #
    plot_df = pd.concat([lm_1.resid.rename('residuals'), mobem.loc[z]], axis=1).dropna()

    sns.boxplot(z, 'residuals', data=plot_df, palette=cdrug.PAL_BIN, linewidth=.3, sym='')
    sns.swarmplot(z, 'residuals', data=plot_df, alpha=.8, edgecolor='white', linewidth=.3, size=2, palette=cdrug.PAL_BIN)
    plt.gcf().set_size_inches(1., 2.)
    plt.savefig('reports/residuals_boxplot.pdf', bbox_inches='tight')
    plt.close('all')

    #
    plot_df = pd.concat([
        crispr.loc[gene].rename(x),
        drespo.loc[(d_id, d_name, d_screen)].rename(y),
        mobem.loc[z].rename(z)
    ], axis=1).dropna()

    g = plot_corrplot(y, x, plot_df, add_hline=True, lowess=False)
    sns.regplot(y, x, data=plot_df[plot_df[z] == 1], ax=g.ax_joint, color=cdrug.PAL_SET2[1], fit_reg=False)

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/residuals_corrplot.pdf', bbox_inches='tight')
    plt.close('all')
