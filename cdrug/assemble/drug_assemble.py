#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd

INDEX_VALUE = 'IC50_nat_log'
INDEX_SAMPLE = ('COSMIC_ID', 'CELL_LINE_NAME')
INDEX_DRUG = ('DRUG_ID_lib', 'DRUG_NAME', 'VERSION')


def merge_drug_dfs(d_17, d_rs):
    # Build tables and exclude mismatching drugs
    drug_v17_matrix_m = pd.pivot_table(d_17, index=INDEX_DRUG, columns=INDEX_SAMPLE, values=INDEX_VALUE)
    drug_v17_matrix_m = drug_v17_matrix_m[[idx not in drug_id_remove for idx, dname, dversion in drug_v17_matrix_m.index]]

    drug_rs_matrix_m = pd.pivot_table(d_rs, index=INDEX_DRUG, columns=INDEX_SAMPLE, values=INDEX_VALUE)

    drug_matrix_m = pd.concat([drug_v17_matrix_m, drug_rs_matrix_m], axis=0)

    return drug_matrix_m


if __name__ == '__main__':
    # - Imports
    d_v17 = pd.read_csv(cdrug.DRUG_RESPONSE_V17)
    d_vrs = pd.read_csv(cdrug.DRUG_RESPONSE_VRS)

    # Build tables and exclude mismatching drugs
    d_v17_matrix = pd.pivot_table(d_v17, index=INDEX_DRUG, columns=INDEX_SAMPLE, values=INDEX_VALUE)

    d_vrs_matrix = pd.pivot_table(d_vrs, index=INDEX_DRUG, columns=INDEX_SAMPLE, values=INDEX_VALUE)

    # - Merge screens
    d_merged = merge_drug_dfs(d_v17, d_vrs)
    d_merged.to_csv('data/drug_ic50_merged_matrix.csv')
