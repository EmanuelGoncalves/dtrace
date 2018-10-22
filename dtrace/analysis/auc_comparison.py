#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.plot.corrplot import plot_corrplot

if __name__ == '__main__':
    # - Import
    headers = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']

    assoc_auc = pd.read_csv('data/drug_lmm_regressions_auc.csv').set_index(headers)
    assoc_ic50 = pd.read_csv('data/drug_lmm_regressions_ic50.csv').set_index(headers)

    # -
    plot_df = pd.concat([assoc_auc['beta'].rename('AUC'), assoc_ic50['beta'].rename('IC50')], axis=1)

    plot_corrplot('AUC', 'IC50', plot_df)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/associations_beta_corr.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close('all')
