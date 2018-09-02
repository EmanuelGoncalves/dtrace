#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import re
import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Import
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='both')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # - Drugs
    drugs_comb = [d for d in drespo.index if '+' in d[1]]

    drugs_sing = [int(re.search('\((.*?)\)', d_idx).group(1)) for dc in drugs_comb for d_idx in dc[1].split(' + ')]
    drugs_sing = [(d_idx, d[0], d[1]) for d_idx in drugs_sing for d in drespo.loc[d_idx].index]
