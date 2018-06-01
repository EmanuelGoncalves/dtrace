#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace import is_same_drug
from dtrace.analysis import PAL_DTRACE


def drugs_overlap():
    # Import drug-response data
    drespo = dtrace.get_drugresponse()

    # Import drug information
    dsheet = dtrace.get_drugsheet()

    # Check drugs overlap
    df_count = {}
    for t, f in [('All', False), ('Pub or Web', True)]:
        df_count[t] = {}

        # Subset drug list
        if t == 'Pub or Web':
            drug_pub = dsheet[(dsheet['Web Release'] == 'Y') | (dsheet['Suitable for publication'] == 'Y')]
            drugs = [tuple(i) for i in drespo.index if i[0] in drug_pub.index]
        else:
            drugs = [tuple(i) for i in drespo.index]

        # Count unique drugs in v17
        drugs_v17 = [i for i in drugs if i[2] == 'v17']
        drugs_v17_match = pd.DataFrame({d1: {d2: int(is_same_drug(d1[0], d2[0], dsheet)) for d2 in drugs_v17} for d1 in drugs_v17})
        df_count[t]['V17'] = drugs_v17_match.shape[0] - sum(sum(np.tril(drugs_v17_match, -1)))

        # Count unique drugs in RS
        drugs_rs = [i for i in drugs if i[2] == 'RS']
        drugs_rs_match = pd.DataFrame({d1: {d2: int(is_same_drug(d1[0], d2[0], dsheet)) for d2 in drugs_rs} for d1 in drugs_rs})
        df_count[t]['RS'] = drugs_rs_match.shape[0] - sum(sum(np.tril(drugs_rs_match, -1)))

        # Overlap
        d_match_overlap = pd.DataFrame({d1: {d2: int(is_same_drug(d1[0], d2[0], dsheet)) for d2 in drugs_v17} for d1 in drugs_rs})
        df_count[t]['Overlap'] = sum(sum(d_match_overlap.values))

        # Total
        df_count[t]['Total'] = df_count[t]['V17'] + df_count[t]['RS'] - df_count[t]['Overlap']

        print('V17: {}; RS: {}; Overlap: {}'.format(df_count[t]['V17'], df_count[t]['RS'], df_count[t]['Overlap']))

    df_count = pd.DataFrame(df_count)
    df_count = pd.melt(df_count.reset_index(), id_vars='index', value_vars=['All', 'Pub or Web'])

    return df_count


if __name__ == '__main__':
    # - Build data-frame of drugs overlaping
    drug_overlap = drugs_overlap()

    # - Plot
    order = ['V17', 'RS', 'Overlap', 'Total']
    pal = dict(zip(*(set(drug_overlap['variable']), [PAL_DTRACE[2], PAL_DTRACE[0]])))

    sns.barplot('index', 'value', 'variable', drug_overlap, palette=pal, order=order, saturation=1)

    sns.despine(top=True, right=True)

    plt.axes().yaxis.grid(True, color=PAL_DTRACE[1], linestyle='-', linewidth=.1, alpha=.5, zorder=0)

    plt.legend(title='Drugs')
    plt.ylabel('Number of unique drugs')
    plt.xlabel('')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/drug_overlap.pdf', bbox_inches='tight')
    plt.close('all')
