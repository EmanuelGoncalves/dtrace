#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def is_same_drug(drug_id_1, drug_id_2, drug_list=None):
    drug_list = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0) if drug_list is None else drug_list

    for i, d in enumerate([drug_id_1, drug_id_2]):
        assert d in drug_list.index, 'Drug ID {} not in drug list'.format(i)

    drug_names = {d: get_drug_names(d, drug_list) for d in [drug_id_1, drug_id_2]}

    return len(drug_names[drug_id_1].intersection(drug_names[drug_id_2])) > 0


def get_drug_names(drug_id, drug_list=None):
    drug_list = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0) if drug_list is None else drug_list

    if drug_id not in drug_list.index:
        print('{} Drug ID not in drug list'.format(drug_id))
        return None

    drgu_name = [drug_list.loc[drug_id, 'Name']]

    drug_synonyms = drug_list.loc[drug_id, 'Synonyms']
    drug_synonyms = [] if str(drug_synonyms).lower() == 'nan' else drug_synonyms.split(', ')

    return set(drgu_name + drug_synonyms)


if __name__ == '__main__':
    # - Imports
    d_v17 = pd.read_csv(cdrug.DRUG_RESPONSE_V17)
    d_vrs = pd.read_csv(cdrug.DRUG_RESPONSE_VRS)

    #
    df_count = {}
    for t, f in [('All', False), ('Pub&Web', True)]:
        df_count[t] = {}

        ds = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0)

        if f:
            ds = ds[[w == 'Y' or p == 'Y' for w, p in ds[['Web Release', 'Suitable for publication']].values]]

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
    plot_df = pd.melt(df_count.reset_index(), id_vars='index', value_vars=['All', 'Pub&Web'])

    order = ['V17', 'RS', 'Overlap', 'Total']
    pal = dict(zip(*(set(plot_df['variable']), sns.light_palette(cdrug.bipal_dbgd[0], n_colors=3).as_hex()[1:])))

    sns.barplot('index', 'value', 'variable', plot_df, palette=pal, order=order)

    plt.axes().yaxis.grid(True, color=cdrug.bipal_dbgd[0], linestyle='-', linewidth=.1, alpha=.5)

    plt.legend(title='Drugs')
    plt.ylabel('Counts')
    plt.xlabel('')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/drug_overlap.png', bbox_inches='tight', dpi=600)
    plt.close('all')
