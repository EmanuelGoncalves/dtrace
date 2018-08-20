#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import analysis.ctrp as ctrp
import matplotlib.pyplot as plt


def is_target(g, ts):
    return int(g in ts.split(';')) if str(ts) != 'nan' else np.nan


if __name__ == '__main__':
    # - Import
    drugs = ctrp.import_ctrp_compound_sheet()

    #
    lmm_res = pd.read_csv(ctrp.LMM_ASSOCIATIONS_CTRP)

    # -
    lmm_signif = lmm_res.query('fdr < .1')
    lmm_signif = lmm_signif.merge(drugs.reset_index(), left_on='DRUG_ID', right_on='index_cpd')
    lmm_signif['target'] = [is_target(g, ts) for g, ts in lmm_signif[['GeneSymbol', 'gene_symbol_of_protein_target']].values]