#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pydot
import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace import get_drugtargets
from dtrace.analysis import PAL_DTRACE
from analysis.plot.corrplot import plot_corrplot
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.associations import ppi_annotation, corr_drugtarget_gene, ppi_corr


def get_edges(ppi, nodes, corr_thres):
    # Nodes that are contained in the network
    nodes = {v for v in nodes if v in ppi.vs['name']}

    assert len(nodes) > 0, 'None of the nodes is contained in the PPI'

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


def plot_ppi(d_id, lmm_drug, corr_thres=0.2, fdr_thres=0.05):
    # Build data-set
    d_signif = lmm_drug.query('DRUG_ID_lib == {} & fdr < {}'.format(d_id, fdr_thres))
    d_ppi_df = get_edges(ppi, list(d_signif['GeneSymbol']), corr_thres)

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
    # - Imports
    # Data-sets
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='both')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # Drug max screened concentration
    d_maxc = pd.read_csv(dtrace.DRUG_RESPONSE_MAXC, index_col=[0, 1, 2])

    # Linear regressions
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS)
    lmm_drug = ppi_annotation(lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)
    lmm_drug = corr_drugtarget_gene(lmm_drug)

    # Drug target
    d_targets = get_drugtargets()

    # PPI
    ppi = build_string_ppi(score_thres=900)
    ppi = ppi_corr(ppi, crispr_logfc)

    # - Top associations
    lmm_drug.sort_values('fdr')

    lmm_drug[lmm_drug['DRUG_NAME'] == 'Taselisib'].sort_values('fdr')

    # - Top correlation examples
    indices = [(934059, 0.2), (1048516, 0.2), (134251, 0.2), (232252, 0.2), (1020056, 0.4), (1502618, .2), (21812, 0.3)]

    for idx, cor_thres in indices:
        d_id, d_name, d_screen, d_gene = lmm_drug.loc[idx, ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']].values
        name = 'Drug={}, Gene={} [{}, {}]'.format(d_name, d_gene, d_id, d_screen)

        # Drug ~ CRISPR correlation
        x, y = '{}'.format(d_gene), '{}'.format(d_name)

        plot_df = pd.concat([
            crispr_logfc.loc[d_gene].rename(x), drespo.loc[(d_id, d_name, d_screen)].rename(y)
        ], axis=1).dropna().sort_values(x)

        plot_corrplot(x, y, plot_df, add_hline=True, lowess=False)
        plt.axhline(np.log(d_maxc.loc[(d_id, d_name, d_screen), 'max_conc_micromolar']), lw=.3, color=PAL_DTRACE[2], ls='--')

        plt.gcf().set_size_inches(2., 2.)
        plt.savefig('reports/lmm_association_corrplot_{}.pdf'.format(name), bbox_inches='tight')
        plt.close('all')

        # Drug network
        graph = plot_ppi(d_id, lmm_drug, corr_thres=cor_thres)
        graph.write_pdf('reports/lmm_association_ppi_{}.pdf'.format(name))
