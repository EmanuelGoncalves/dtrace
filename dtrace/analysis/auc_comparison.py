#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
from importer import PPI
from associations import Association

if __name__ == '__main__':
    # - Import
    headers = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']

    assoc_auc = pd.read_csv('data/drug_lmm_regressions_auc.csv').set_index(headers)
    assoc_ic50 = pd.read_csv('data/drug_lmm_regressions_ic50.csv').set_index(headers)

    # -
    plot_df = pd.concat([assoc_auc['beta'].rename('AUC'), assoc_ic50['beta'].rename('IC50')], axis=1)

    Plot().plot_corrplot('AUC', 'IC50', plot_df)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/associations_beta_corr.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close('all')

    # -
    lmm_drug_diff = plot_df.eval('AUC - IC50').sort_values().rename('beta').reset_index()
    lmm_drug_diff = PPI().ppi_annotation(lmm_drug_diff, ppi_type='string', ppi_kws=dict(score_thres=900), target_thres=3)

    # -
    crispr = Association().crispr

    crispr_corr = crispr.T.corr()
