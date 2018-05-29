#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import drispr
import pandas as pd

INDEX_VALUE = 'IC50_nat_log'
INDEX_SAMPLE = ('COSMIC_ID', 'CELL_LINE_NAME')
INDEX_DRUG = ('DRUG_ID_lib', 'DRUG_NAME', 'VERSION')


if __name__ == '__main__':
    # - Imports
    d_v17 = pd.read_csv(drispr.DRUG_RESPONSE_V17).assign(VERSION='v17')
    d_vrs = pd.read_csv(drispr.DRUG_RESPONSE_VRS).assign(VERSION='RS')

    # Build tables
    d_v17_matrix = pd.pivot_table(d_v17, index=INDEX_DRUG, columns=INDEX_SAMPLE, values=INDEX_VALUE)
    d_vrs_matrix = pd.pivot_table(d_vrs, index=INDEX_DRUG, columns=INDEX_SAMPLE, values=INDEX_VALUE)

    # - Merge screens
    d_merged = pd.concat([d_v17_matrix, d_vrs_matrix], axis=0)
    d_merged.to_csv(drispr.DRUG_RESPONSE_FILE)
