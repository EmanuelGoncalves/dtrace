#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pydot
import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cdrug import get_drugtargets
from cdrug.assemble.assemble_ppi import build_biogrid_ppi, build_string_ppi
from cdrug.associations import LR_DRUG_CRISPR, multipletests_per_drug, ppi_corr


if __name__ == '__main__':
    exp_type, int_type = {'Affinity Capture-MS', 'Affinity Capture-Western'}, {'physical'}

    # Linear regressions
    lr = pd.read_csv(LR_DRUG_CRISPR)
    lr = multipletests_per_drug(lr)

    # CIRSPR CN corrected logFC
    crispr = cdrug.get_crispr(dtype='logFC')
    crispr_scaled = cdrug.scale_crispr(crispr)

    # PPI annotation
    # ppi = build_biogrid_ppi(int_type=int_type, exp_type=exp_type)
    ppi = build_string_ppi()

    # PPI correlation
    ppi = ppi_corr(ppi, crispr_scaled)

    # Drug target
    d_targets = get_drugtargets()

    # -
    drug_id = 1549

    drug_signif = lr[lr['DRUG_ID_lib'] == drug_id].query('lr_fdr < 0.05')

    drug_ppi = pd.DataFrame([{
        'source': ppi.vs[i.source]['name'],
        'target': ppi.vs[i.target]['name'],
        'r': i['corr']
    } for i in ppi.es if bool(set(drug_signif['GeneSymbol']) & set(ppi.vs[[i.source, i.target]]['name']))]).sort_values('r')

    #
    thres_corr = .3
    palette = pd.Series(
        [cdrug.PAL_BIN[1]] + sns.light_palette(cdrug.PAL_BIN[0], n_colors=3, reverse=True).as_hex()
    , index=range(4))

    graph = pydot.Dot(graph_type='graph', pagedir='TR')

    kws_nodes = dict(style='"rounded,filled"', shape='rect', color=palette[1], penwidth=2, fontcolor='white')
    kws_edges = dict()

    for s, t in drug_ppi[drug_ppi['r'].abs() > thres_corr][['source', 'target']].values:

        source = pydot.Node(
            s,
            fillcolor=palette[int(s not in d_targets[drug_id])],
            fontsize=15 if s in drug_signif['GeneSymbol'].values else 9,
            **kws_nodes
        )

        target = pydot.Node(
            t,
            fillcolor=palette[int(t not in d_targets[drug_id])],
            fontsize=15 if t in drug_signif['GeneSymbol'].values else 9,
            **kws_nodes
        )

        graph.add_node(source)
        graph.add_node(target)

        edge = pydot.Edge(source, target)
        graph.add_edge(edge)

    graph.write_pdf('reports/drug_target_ppi.pdf')
