#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import trace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from trace import is_same_drug


if __name__ == '__main__':
    # - Imports
    d_v17 = pd.read_csv(trace.DRUG_RESPONSE_V17)
    d_vrs = pd.read_csv(trace.DRUG_RESPONSE_VRS)

    # - Check drugs overlap
    df_count = {}
    for t, f in [('All', False), ('Pub or Web', True)]:
        df_count[t] = {}

        ds = trace.import_drug_list(filter_web_pub=f)

        d_id_v17 = list(set(d_v17['DRUG_ID_lib']).intersection(ds.index))
        d_match_v17 = pd.DataFrame({d1: {d2: int(is_same_drug(d1, d2, ds)) for d2 in d_id_v17} for d1 in d_id_v17})
        df_count[t]['V17'] = d_match_v17.shape[0] - sum(sum(np.tril(d_match_v17, -1)))

        d_id_vrs = list(set(d_vrs['DRUG_ID_lib']).intersection(ds.index))
        d_match_vrs = pd.DataFrame({d1: {d2: int(is_same_drug(d1, d2, ds)) for d2 in d_id_vrs} for d1 in d_id_vrs})
        df_count[t]['RS'] = d_match_vrs.shape[0] - sum(sum(np.tril(d_match_vrs, -1)))

        #
        d_match_overlap = pd.DataFrame({d1: {d2: int(is_same_drug(d1, d2, ds)) for d2 in d_id_v17} for d1 in d_id_vrs})
        df_count[t]['Overlap'] = sum(sum(d_match_overlap.values))

        #
        df_count[t]['Total'] = df_count[t]['V17'] + df_count[t]['RS'] - df_count[t]['Overlap']

        print('V17: {}; RS: {}; Overlap: {}'.format(df_count[t]['V17'], df_count[t]['RS'], df_count[t]['Overlap']))

    df_count = pd.DataFrame(df_count)

    #
    plot_df = pd.melt(df_count.reset_index(), id_vars='index', value_vars=['All', 'Pub or Web'])

    order = ['V17', 'RS', 'Overlap', 'Total']
    pal = dict(zip(*(set(plot_df['variable']), [trace.PAL_SET2[8], trace.PAL_SET2[1]])))

    sns.barplot('index', 'value', 'variable', plot_df, palette=pal, order=order, saturation=1)

    plt.axes().yaxis.grid(True, color=trace.PAL_SET2[7], linestyle='-', linewidth=.1, alpha=.5, zorder=0)

    plt.legend(title='Drugs')
    plt.ylabel('Number of unique drugs')
    plt.xlabel('')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/drug_overlap.pdf', bbox_inches='tight')
    plt.close('all')
