#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pydot
import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
from dtrace import get_drugtargets
from dtrace.analysis import PAL_DTRACE
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.associations import ppi_annotation, corr_drugtarget_gene, ppi_corr


def get_edges(ppi, nodes, corr_thres):
    # Nodes incident edges
    incident_edges = {v for e in nodes for v in ppi.incident(e)}

    # Filter by correlation
    incident_edges = [e for e in incident_edges if abs(ppi.es[e]['corr']) >= corr_thres]

    # Build subgraph
    subgraph = ppi.subgraph_edges(incident_edges)

    # Build data-frame
    nodes_df = pd.DataFrame([{
        'source': subgraph.vs[e.source]['name'],
        'target': subgraph.vs[e.target]['name'],
        'r': e['corr']
    } for e in subgraph.es]).sort_values('r')

    return nodes_df


def plot_ppi(d_id, lmm_drug):
    # Build data-set
    d_signif = lmm_drug.query('DRUG_ID_lib == {} & fdr < 0.05'.format(d_id))
    d_ppi_df = get_edges(ppi, list(d_signif['GeneSymbol']), 0.2)

    # Build graph
    graph = pydot.Dot(graph_type='graph', pagedir='TR')

    kws_nodes = dict(style='"rounded,filled"', shape='rect', color=PAL_DTRACE[1], penwidth=2, fontcolor='white')
    kws_edges = dict(fontsize=9, fontcolor=PAL_DTRACE[2], color=PAL_DTRACE[2])

    for s, t, r in d_ppi_df[['source', 'target', 'r']].values:
        # Add source node
        fs = 15 if s in d_signif['GeneSymbol'].values else 9
        fc = PAL_DTRACE[0 if s in d_targets[d_id] else 2]

        source = pydot.Node(s, fillcolor=fc, fontsize=fs, **kws_nodes)
        graph.add_node(source)

        # Add target node
        fc = PAL_DTRACE[0 if t in d_targets[d_id] else 2]
        fs = 15 if t in d_signif['GeneSymbol'].values else 9

        target = pydot.Node(t, fillcolor=fc, fontsize=fs, **kws_nodes)
        graph.add_node(target)

        # Add edge
        edge = pydot.Edge(source, target, label='{:.2f}'.format(r), **kws_edges)
        graph.add_edge(edge)

    return graph


if __name__ == '__main__':
    # - Data-sets
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='both')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # - Linear regressions
    lmm_drug = pd.read_csv(dtrace.DRUG_LMM)
    lmm_drug = ppi_annotation(lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)
    lmm_drug = corr_drugtarget_gene(lmm_drug)

    # - Drug target
    d_targets = get_drugtargets()

    # - PPI
    ppi = build_string_ppi()
    ppi = ppi_corr(ppi, crispr_logfc)

    # - Plot network
    d_id = 1549
    graph = plot_ppi(d_id, lmm_drug)
    graph.write_pdf('reports/drug_target_ppi.pdf')
