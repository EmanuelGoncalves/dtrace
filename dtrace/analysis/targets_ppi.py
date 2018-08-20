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
from statsmodels.distributions.empirical_distribution import ECDF
from dtrace.associations import ppi_annotation, corr_drugtarget_gene, ppi_corr, DRUG_INFO_COLUMNS


def get_edges(ppi, nodes, corr_thres, norder):
    # Subset network
    ppi_sub = ppi.copy().subgraph_edges([e for e in ppi.es if abs(e['corr']) >= corr_thres])

    # Nodes that are contained in the network
    nodes = {v for v in nodes if v in ppi_sub.vs['name']}
    assert len(nodes) > 0, 'None of the nodes is contained in the PPI'

    # Nodes neighborhood
    neighbor_nodes = {v for n in nodes for v in ppi_sub.neighborhood(n, order=norder)}

    # Build subgraph
    subgraph = ppi_sub.subgraph(neighbor_nodes)

    # Build data-frame
    nodes_df = pd.DataFrame([{
        'source': subgraph.vs[e.source]['name'],
        'target': subgraph.vs[e.target]['name'],
        'r': e['corr']
    } for e in subgraph.es]).sort_values('r')

    return nodes_df


def plot_ppi(d_id, lmm_drug, corr_thres=0.2, fdr_thres=0.05, norder=1):
    # Build data-set
    d_signif = lmm_drug.query('DRUG_ID_lib == {} & fdr < {}'.format(d_id, fdr_thres))
    d_ppi_df = get_edges(ppi, list(d_signif['GeneSymbol']), corr_thres, norder)

    # Build graph
    graph = pydot.Dot(graph_type='graph', pagedir='TR')

    kws_nodes = dict(style='"rounded,filled"', shape='rect', color=PAL_DTRACE[1], penwidth=2, fontcolor='white')
    kws_edges = dict(fontsize=9, fontcolor=PAL_DTRACE[2], color=PAL_DTRACE[2])

    for s, t, r in d_ppi_df[['source', 'target', 'r']].values:
        # Add source node
        fs = 15 if s in d_signif['GeneSymbol'].values else 9
        fc = PAL_DTRACE[0 if d_id in d_targets and s in d_targets[d_id] else 2]

        source = pydot.Node(s, fillcolor=fc, fontsize=fs, **kws_nodes)
        graph.add_node(source)

        # Add target node
        fc = PAL_DTRACE[0 if d_id in d_targets and t in d_targets[d_id] else 2]
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

    ss = dtrace.get_samplesheet()

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
    # - Top correlation examples
    indices = [
        dict(d_name='MCL1_1284', g_name='MCL1', corr_thres=0.2, n_neighbors=1),
        dict(d_name='Nutlin-3a (-)', g_name='MDM4', corr_thres=0.4, n_neighbors=2),
        dict(d_name='Sapatinib', g_name='EGFR', corr_thres=0.3, n_neighbors=2),
        dict(d_name='Rigosertib', g_name='BUB1B', corr_thres=0.3, n_neighbors=2),
        dict(d_name='Volasertib', g_name='BUB1B', corr_thres=0.3, n_neighbors=2),
        dict(d_name='Vinblastine', g_name='BUB1B', corr_thres=0.3, n_neighbors=2),
    ]

    for assoc in indices:
        idx = lmm_drug.query(f"DRUG_NAME == '{assoc['d_name']}' & GeneSymbol == '{assoc['g_name']}'").index[0]

        d_id, d_screen = lmm_drug.loc[idx, ['DRUG_ID_lib', 'VERSION']].values
        name = f"Drug={assoc['d_name']}, Gene={assoc['g_name']} [{d_id}, {d_screen}]"

        # Data-frame
        x, y = f"{assoc['g_name']}", f"{assoc['d_name']}"

        plot_df = pd.concat([
            crispr_logfc.loc[assoc['g_name']].rename(x), drespo.loc[(d_id, assoc['d_name'], d_screen)].rename(y)
        ], axis=1, sort=False).dropna().sort_values(x)

        # Plot
        plot_corrplot(x, y, plot_df, add_hline=True, lowess=False)

        plt.axhline(np.log(d_maxc.loc[(d_id, assoc['d_name'], d_screen), 'max_conc_micromolar']), lw=.3, color=PAL_DTRACE[2], ls=':', zorder=0)

        plt.gcf().set_size_inches(2., 2.)
        plt.savefig(f'reports/lmm_association_corrplot_{name}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

        # Drug network
        graph = plot_ppi(d_id, lmm_drug, corr_thres=assoc['corr_thres'], norder=assoc['n_neighbors'])
        graph.write_pdf(f'reports/lmm_association_ppi_{name}.pdf')
