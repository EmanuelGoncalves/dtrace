#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CTRP_DIR = 'data/CTRPv2.2/'


LMM_ASSOCIATIONS_CTRP = 'data/drug_lmm_regressions_ctrp.csv'


def get_samplesheet():
    samplesheet = pd.read_csv(f'{CTRP_DIR}/sample_info.csv', index_col=1)
    return samplesheet


def get_ceres():
    samplesheet = get_samplesheet()

    ceres = pd.read_csv(f'{CTRP_DIR}/gene_effect.csv', index_col=0).T
    ceres = ceres.rename(columns=samplesheet['CCLE_name'])
    ceres.index = [i.split(' ')[0] for i in ceres.index]
    ceres.columns = [i.split('_')[0] for i in ceres.columns]

    return ceres


def import_ctrp_samplesheet():
    # Import GDSC samplesheet
    gdsc = dtrace.get_samplesheet().reset_index().dropna(subset=['CCLE ID'])
    gdsc = gdsc.assign(ccl_name=gdsc['CCLE ID'].apply(lambda v: v.split('_')[0]))

    # Import CTRP samplesheet
    ctrp_samples = pd.read_csv(f'{CTRP_DIR}/v22.meta.per_cell_line.txt', sep='\t')
    ctrp_samples = ctrp_samples.groupby('ccl_name').first().reset_index()

    # Map cell lines ids
    ctrp_samples['gdsc'] = ctrp_samples['ccl_name'].replace(gdsc.set_index('ccl_name')['Cell Line Name'].to_dict())

    return ctrp_samples


def import_ctrp_compound_sheet():
    return pd.read_csv(f'{CTRP_DIR}/v22.meta.per_compound.txt', sep='\t', index_col=0)


def import_ctrp_aucs():
    ctrp_samples = import_ctrp_samplesheet()

    ctrp_aucs = pd.read_csv(f'{CTRP_DIR}/v22.data.auc_sensitivities.txt', sep='\t')
    ctrp_aucs = pd.pivot_table(ctrp_aucs, index='index_cpd', columns='index_ccl', values='area_under_curve')
    ctrp_aucs = ctrp_aucs.rename(columns=ctrp_samples.set_index('index_ccl')['ccl_name'])

    return ctrp_aucs
