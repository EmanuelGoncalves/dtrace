#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import drispr
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import drispr.associations as lr_files
from drispr.plot.corrplot import plot_corrplot


if __name__ == '__main__':
    # WES
    wes = pd.read_csv(drispr.WES_COUNT)

    # Samplesheet
    ss = drispr.get_samplesheet()

    # Linear regressions
    lm_df_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)

    # Drug response
    drespo = drispr.get_drugresponse()

    # CIRSPR CN corrected logFC
    crispr = drispr.get_crispr(dtype='logFC')

    # RNA-seq
    gexp = drispr.get_geneexpression()

    samples = list(set(drespo).intersection(crispr).intersection(gexp))
    print(len(samples))

    # --
    idx = 6

    d_id, d_name, d_screen, gene = lm_df_crispr.loc[idx, ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']].values

    x = drespo.loc[(d_id, d_name, d_screen), samples].dropna()
    y = crispr.loc[lm_df_crispr.loc[idx, 'GeneSymbol'], x.index]

    lm_1 = sm.OLS(y, sm.add_constant(x)).fit()
    print(lm_1.summary())

    z = gexp[y.index].T.astype(float)
    z.corrwith(lm_1.resid).sort_values()

    # -
    x, y, z = '{}'.format(gene), '{} {}'.format(d_name, d_screen), '{}'.format('ZNF34')

    # #
    # plot_df = pd.concat([lm_1.resid.rename('residuals'), gexp.loc[z]], axis=1).dropna()
    #
    # sns.boxplot(z, 'residuals', data=plot_df, palette=drispr.PAL_BIN, linewidth=.3, sym='')
    # sns.swarmplot(z, 'residuals', data=plot_df, alpha=.8, edgecolor='white', linewidth=.3, size=2, palette=drispr.PAL_BIN)
    # plt.gcf().set_size_inches(1., 2.)
    # plt.savefig('reports/residuals_boxplot.pdf', bbox_inches='tight')
    # plt.close('all')

    #
    plot_df = pd.concat([
        crispr.loc[gene].rename(x),
        drespo.loc[(d_id, d_name, d_screen)].rename(y),
        gexp.loc[z].rename(z)
    ], axis=1).dropna()

    s = plt.scatter(plot_df[y], plot_df[x], c=plot_df[z], s=8, edgecolor='white', lw=.3, cmap='RdYlGn', alpha=.5)
    sns.regplot(plot_df[y], plot_df[x], scatter=False, line_kws=dict(lw=1., color=drispr.PAL_SET2[1]))

    plt.colorbar(s)

    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/residuals_corrplot.pdf', bbox_inches='tight')
    plt.close('all')
