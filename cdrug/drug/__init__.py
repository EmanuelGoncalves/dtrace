#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import pandas as pd


def is_same_drug(drug_id_1, drug_id_2, drug_list=None):
    """
    From 2 drug IDs check if Drug ID 1 has Name or Synonyms in common with Drug ID 2.

    :param drug_id_1:
    :param drug_id_2:
    :param drug_list:
    :return: Bool
    """

    drug_list = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0) if drug_list is None else drug_list

    for i, d in enumerate([drug_id_1, drug_id_2]):
        assert d in drug_list.index, 'Drug ID {} not in drug list'.format(i)

    drug_names = {d: get_drug_names(d, drug_list) for d in [drug_id_1, drug_id_2]}

    return len(drug_names[drug_id_1].intersection(drug_names[drug_id_2])) > 0


def get_drug_names(drug_id, drug_list=None):
    """
    From a Drug ID get drug Name and Synonyms.

    :param drug_id:
    :param drug_list:
    :return:
    """

    drug_list = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0) if drug_list is None else drug_list

    if drug_id not in drug_list.index:
        print('{} Drug ID not in drug list'.format(drug_id))
        return None

    drgu_name = [drug_list.loc[drug_id, 'Name']]

    drug_synonyms = drug_list.loc[drug_id, 'Synonyms']
    drug_synonyms = [] if str(drug_synonyms).lower() == 'nan' else drug_synonyms.split(', ')

    return set(drgu_name + drug_synonyms)
