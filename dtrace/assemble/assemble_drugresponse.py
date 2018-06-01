#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
from dtrace.associations import DRUG_INFO_COLUMNS

INDEX_VALUE = 'IC50_nat_log'
INDEX_SAMPLE = ('COSMIC_ID', 'CELL_LINE_NAME')


if __name__ == '__main__':
    # - Imports
    d_v17 = pd.read_csv(dtrace.DRUG_RESPONSE_V17).assign(VERSION='v17')
    d_vrs = pd.read_csv(dtrace.DRUG_RESPONSE_VRS).assign(VERSION='RS')

    # Build tables
    d_v17_matrix = pd.pivot_table(d_v17, index=DRUG_INFO_COLUMNS, columns=INDEX_SAMPLE, values=INDEX_VALUE)
    d_vrs_matrix = pd.pivot_table(d_vrs, index=DRUG_INFO_COLUMNS, columns=INDEX_SAMPLE, values=INDEX_VALUE)

    # - Merge screens
    d_merged = pd.concat([d_v17_matrix, d_vrs_matrix], axis=0)
    d_merged.to_csv(dtrace.DRUG_RESPONSE_FILE)

    # -
    drug_maxconcentration = pd.concat([
        d_vrs.groupby(DRUG_INFO_COLUMNS)['max_conc_micromolar'].min(),
        d_v17.groupby(DRUG_INFO_COLUMNS)['max_conc_micromolar'].min()
    ]).sort_values().reset_index()
    drug_maxconcentration.to_csv(dtrace.DRUG_RESPONSE_MAXC, index=False)
