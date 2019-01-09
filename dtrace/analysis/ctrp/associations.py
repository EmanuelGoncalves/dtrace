#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
import analysis.ctrp as ctrp
from Associations import multipletests_per_drug
from Associations.lmm_drug import lmm_association

if __name__ == '__main__':
    # - Imports
    # Samplesheet
    ctrp_samples = ctrp.import_ctrp_samplesheet()

    # Compounds
    ctrp_compounds = ctrp.import_ctrp_compound_sheet()

    # Drug AUCs
    ctrp_aucs = ctrp.import_ctrp_aucs()

    # CRISPR
    gdsc_crispr = dtrace.get_crispr(dtype='both')
    gdsc_crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    # - Overlap
    samples = list(set(ctrp_aucs).intersection(gdsc_crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # - Filter
    ctrp_aucs = dtrace.filter_drugresponse(ctrp_aucs[samples], filter_max_concentration=False, filter_owner=False)

    gdsc_crispr = dtrace.filter_crispr(gdsc_crispr[samples])
    gdsc_crispr_logfc = gdsc_crispr_logfc.loc[gdsc_crispr.index, samples]
    print(f'#(Drugs) = {len(set(ctrp_aucs.index))}; #(Genes) = {len(set(gdsc_crispr.index))}')

    # - Linear Mixed Model
    lmm_res = pd.concat([lmm_association(d, ctrp_aucs, gdsc_crispr_logfc, expand_drug_id=False) for d in ctrp_aucs.index])
    lmm_res = multipletests_per_drug(lmm_res, field='pval', index_cols=['DRUG_ID'])
    print(lmm_res.sort_values('pval').head(60))

    # - Export
    lmm_res.to_csv(ctrp.LMM_ASSOCIATIONS_CTRP, index=False)
