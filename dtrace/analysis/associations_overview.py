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





if __name__ == '__main__':
    # - Import associations
    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv')

    crispr = CRISPR()
    drug = DrugResponse()

    # d, g = (1243, 'Piperlongumine', 'v17'), 'NFE2L2'
    d, g = (1991, 'Ibrutinib', 'RS'), 'BTK'

    plot_df = pd.concat([
        drug.get_data().loc[d].rename('drug'),
        crispr.get_data().loc[g].rename('crispr')
    ], axis=1, sort=False).dropna()

    dplot = Plot()
    dplot.plot_corrplot('crispr', 'drug', plot_df, add_hline=True, lowess=False)

    plt.axhline(np.log(drug.maxconcentration[d]), lw=.3, color=dplot.PAL_DTRACE[2], ls=':', zorder=0)

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug_associations_scatter.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Drug associations manhattan plot
    manhattan_plot(lmm_drug, fdr=0.1)
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
