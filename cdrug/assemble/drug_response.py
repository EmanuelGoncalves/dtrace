#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Imports
    ss = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0)

    d_v17 = pd.read_csv(cdrug.DRUG_RESPONSE_V17)
    d_vrs = pd.read_csv(cdrug.DRUG_RESPONSE_VRS)

    len(set(d_v17['CELL_LINE_NAME']))
    len(set(d_vrs['CELL_LINE_NAME']))

    len(set(ss.loc[d_v17['DRUG_ID_lib'], 'Name']))
    len(set(ss.loc[d_vrs['DRUG_ID_lib'], 'Name']))

    len(set(ss.loc[d_v17['DRUG_ID_lib'], 'Name']).intersection(set(ss.loc[d_vrs['DRUG_ID_lib'], 'Name'])))

