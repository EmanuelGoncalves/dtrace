# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtrace import rpath, dpath, logger
from dtrace.DTracePlot import DTracePlot
from dtrace.Associations import Association
from dtrace.DataImporter import KinobeadCATDS
from dtrace.TargetBenchmark import TargetBenchmark
from dtrace.DTraceEnrichment import DTraceEnrichment


# ### Import data-sets and associations

assoc = Association(dtype="ic50", load_associations=True, load_ppi=True)

target = TargetBenchmark(assoc=assoc, fdr=0.1)


# ## Drug-response and gene-essentiality associations

# Top associations between drug-response and gene-essentiality

assoc.lmm_drug_crispr.head(15)


# Top associations between drug-response and gene-expression

assoc.lmm_drug_gexp.head(15)


# Manhattan plot of the drug-gene associations, x-axis represents the gene genomic coordinate (obtained from the sgRNAs)
# and y-axis represent the -log10 association p-value.

# Spikes on the same genomic location represent associations with multiple inhibitors with the same target/
# mode-of-action.

target.manhattan_plot(n_genes=20)
plt.gcf().set_size_inches(10, 4.2)
plt.savefig(
    f"{rpath}/drug_associations_manhattan.png",
    bbox_inches="tight",
    transparent=True,
    dpi=300,
)


# Volcano plot of the significant associations.

plt.figure(figsize=(10, 20), dpi=300)
target.signif_volcano()
plt.gcf().set_size_inches(1.5, 3)
plt.savefig(
    f"{rpath}/target_benchmark_volcano.pdf", bbox_inches="tight", transparent=True
)


# Top 50 most strongly correlated drugs

target.top_associations_barplot()
plt.savefig(
    f"{rpath}/target_benchmark_associations_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Representative examples of the drug-gene associations.

dgs = [
    ("Alpelisib", "PIK3CA"),
    ("Nutlin-3a (-)", "MDM2"),
    ("MCL1_1284", "MCL1"),
    ("MCL1_1284", "MARCH5"),
    ("Venetoclax", "BCL2"),
    ("AZD4320", "BCL2"),
    ("Volasertib", "PLK1"),
    ("Rigosertib", "PLK1"),
    ('Linsitinib', 'CNPY2'),
]

for dg in dgs:
    pair = assoc.by(assoc.lmm_drug_crispr, drug_name=dg[0], gene_name=dg[1]).iloc[0]

    drug = tuple(pair[assoc.dcols])

    dmax = np.log(assoc.drespo_obj.maxconcentration[drug])
    annot_text = f"Beta={pair['beta']:.2g}, FDR={pair['fdr']:.1e}"

    plot_df = pd.concat(
        [
            assoc.drespo.loc[drug].rename("drug"),
            assoc.crispr.loc[dg[1]].rename("crispr"),
            assoc.crispr_obj.institute.rename("Institute"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    g = DTracePlot.plot_corrplot(
        "crispr",
        "drug",
        "Institute",
        plot_df,
        add_hline=False,
        add_vline=False,
        annot_text=annot_text,
    )
    g.ax_joint.axhline(
        y=dmax, linewidth=0.3, color=DTracePlot.PAL_DTRACE[2], ls=":", zorder=0
    )
    g.set_axis_labels(f"{dg[1]} (scaled log2 FC)", f"{dg[0]} (ln IC50)")
    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(
        f"{rpath}/association_drug_scatter_{dg[0]}_{dg[1]}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# Kinobeads drug-protein affinity measurements for 84 kinase inhibitors were obtained from an independent study [1] as
# aparent pKd (nM). These were ploted for the signifincant associations versus non-significant (Log-ratio test BH-FDR)
# found in our study.
#
#
# [1] Klaeger S, Heinzlmeir S, Wilhelm M, Polzer H, Vick B, Koenig P-A, Reinecke M, Ruprecht B, Petzoldt S, Meng C,
# Zecha J, Reiter K, Qiao H, Helm D, Koch H, Schoof M, Canevari G, Casale E, Depaolini SR, Feuchtinger A, et al. (2017)
# The target landscape of clinical kinase drugs. Science 358: eaan4368

target.boxplot_kinobead()
plt.gcf().set_size_inches(2.5, 0.75)
plt.savefig(
    f"{rpath}/target_benchmark_kinobeads.pdf", bbox_inches="tight", transparent=True
)


# Association effect sizes with between drugs and their know targets

target.beta_histogram()
plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/target_benchmark_beta_histogram.pdf", bbox_inches="tight", transparent=True
)


# P-value histogram of the Drug-Genes associations highlighting Drug-Target associations.

target.pval_histogram()
plt.gcf().set_size_inches(3, 2)
plt.savefig(
    f"{rpath}/target_benchmark_pval_histogram.pdf", bbox_inches="tight", transparent=True
)


# Distribution of the signifcant Drug-Gene associations across a protein-protein interaction network, with
# gene-essentiality and gene-expression.

for dtype in ["crispr", "gexp"]:

    fig, axs = plt.subplots(2, 1)

    # Boxplot
    target.drugs_ppi(dtype, ax=axs[0])

    axs[0].set_xlabel("Associated gene position in PPI")
    axs[0].set_ylabel("Bonferroni adj. p-value")
    axs[0].set_title("Significant associations\n(adj. p-value < 10%)")

    # Count plot
    target.drugs_ppi_countplot(dtype, ax=axs[1])

    axs[1].set_xlabel("Associated gene position in PPI")
    axs[1].set_ylabel("Number of associations")

    plt.gcf().set_size_inches(2.5, 5)
    plt.savefig(
        f"{rpath}/target_benchmark_ppi_distance_{dtype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


# Background distribution of all Drug-Gene associations tested.

target.drugs_ppi_countplot_background()
plt.gcf().set_size_inches(2.5, 2.5)
plt.savefig(
    f"{rpath}/target_benchmark_ppi_distance_{dtype}_countplot_bkg.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Breakdown numbers of (i) all the drugs screened, (ii) unique drugs, (iii) their annotation status, and (iv) those
# which at least one of the canonical targets were targeted with the CRISPR-Cas9 screen.

target.countplot_drugs()
plt.gcf().set_size_inches(2, 0.75)
plt.savefig(
    f"{rpath}/target_benchmark_association_countplot.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Histogram of drugs with at least one significant association across the protein-protein network

target.countplot_drugs_significant()
plt.gcf().set_size_inches(2, 1)
plt.savefig(
    f"{rpath}/target_benchmark_association_signif_countplot.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Pie chart of significant associations per unique durgs ordered by distance in the PPI

target.pichart_drugs_significant()
plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/target_benchmark_association_signif_piechart.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Heatmap counting the number of drugs which have a significant association with CRISPR and/or target a core-essential
# gene.

target.signif_essential_heatmap()
plt.gcf().set_size_inches(1, 1)
plt.savefig(
    f"{rpath}/target_benchmark_signif_essential_heatmap.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Number of significant associations found with drugs from the two different types of screening proceedures, i.e. RS
# and V17.

target.signif_per_screen()
plt.gcf().set_size_inches(0.75, 1.5)
plt.savefig(
    f"{rpath}/target_benchmark_significant_by_screen.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Heatmap counting the number of drugs which have a significant association with CRISPR and/or with a genomic marker

target.signif_genomic_markers()
plt.gcf().set_size_inches(1, 1)
plt.savefig(
    f"{rpath}/target_benchmark_signif_genomic_heatmap.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Significant associations p-value (y-axis) spread across the number of times a drug displayed an IC50 lower than the
# maximum screened concentration.

target.signif_maxconcentration_scatter()
plt.gcf().set_size_inches(2.5, 2.5)
plt.savefig(
    f"{rpath}/target_benchmark_signif_scatter_maxconcentration.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Drug-Gene CRISPR associations p-value (-log10) versus Drug-Genomic associations p-value (-log10).

target.signif_fdr_scatter()
plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/target_benchmark_signif_fdr_scatter.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Top associaitons 

drugs = [
    "SN1021632995",
    "AZD5582",
    "SN1043546339",
    "VE-821",
    "AZ20",
    "VE821",
    "VX-970",
    "AZD6738",
    "VE-822",
]

for d in drugs:
    target.drug_top_associations(d, fdr_thres=0.25)
    plt.gcf().set_size_inches(1.5, 2)
    plt.savefig(
        f"{rpath}/target_benchmark_{d}_top_associations_barplot.pdf",
        bbox_inches="tight",
        transparent=True,
        dpi=600,
    )
    plt.show()


# PARP inhibitors (olaparib and talazoparib) associations

genes = ["STAG1", "LIG1", "FLI1", "PARP1", "PARP2", "PARP3", "PCGF5", "XRCC1", "RHNO1"]

for drug in ["Olaparib", "Talazoparib"]:
    target.drug_notarget_barplot(drug, genes)

    plt.gcf().set_size_inches(2, 1.5)
    plt.savefig(
        f"{rpath}/target_benchmark_drug_notarget_{drug}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# Clustermap of drug association betas

betas_crispr = pd.pivot_table(
    assoc.lmm_drug_crispr.query("VERSION == 'RS'"),
    index=["DRUG_ID", "DRUG_NAME"],
    columns="GeneSymbol",
    values="beta",
)


target.lmm_betas_clustermap(betas_crispr)
plt.gcf().set_size_inches(8, 8)
plt.savefig(
    f"{rpath}/target_benchmark_clustermap_betas_crispr.png", bbox_inches="tight", dpi=300
)


target.lmm_betas_clustermap_legend()
plt.savefig(
    f"{rpath}/target_benchmark_clustermap_betas_crispr_legend.pdf", bbox_inches="tight"
)


# Drug association with gene-expression

dgs = [
    ("Nutlin-3a (-)", "MDM2"),
    ("Poziotinib", "ERBB2"),
    ("Afatinib", "ERBB2"),
    ("WEHI-539", "BCL2L1"),
]
for dg in dgs:
    pair = assoc.by(assoc.lmm_drug_crispr, drug_name=dg[0], gene_name=dg[1]).iloc[0]

    drug = tuple(pair[assoc.dcols])

    dmax = np.log(assoc.drespo_obj.maxconcentration[drug])
    annot_text = f"Beta={pair['beta']:.2g}, FDR={pair['fdr']:.1e}"

    plot_df = pd.concat(
        [assoc.drespo.loc[drug].rename("drug"), assoc.gexp.loc[dg[1]].rename("crispr")],
        axis=1,
        sort=False,
    ).dropna()
    plot_df["Institute"] = "Sanger"

    g = DTracePlot.plot_corrplot(
        "crispr", "drug", "Institute", plot_df, add_hline=False, add_vline=False, annot_text=annot_text
    )

    g.ax_joint.axhline(
        y=dmax, linewidth=0.3, color=DTracePlot.PAL_DTRACE[2], ls=":", zorder=0
    )

    g.set_axis_labels(f"{dg[1]} (voom)", f"{dg[0]} (ln IC50)")

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(
        f"{rpath}/association_drug_gexp_scatter_{dg[0]}_{dg[1]}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# CRISPR correlation profiles

for gene_x, gene_y in [("MARCH5", "MCL1"), ("SHC1", "EGFR")]:
    plot_df = pd.concat(
        [
            assoc.crispr.loc[gene_x].rename(gene_x),
            assoc.crispr.loc[gene_y].rename(gene_y),
            assoc.crispr_obj.institute.rename("Institute"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    g = DTracePlot().plot_corrplot(gene_x, gene_y, "Institute", plot_df, add_hline=True)

    g.set_axis_labels(f"{gene_x} (scaled log2 FC)", f"{gene_y} (scaled log2 FC)")

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(
        f"{rpath}/association_scatter_{gene_x}_{gene_y}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# PPI weighted network

ppi_examples = [
    ("Nutlin-3a (-)", 0.4, 1, ["RPL37", "UBE3B"]),
    ("AZD3759", 0.3, 1, None),
]
for d, t, o, e in ppi_examples:
    graph = assoc.ppi.plot_ppi(
        d,
        assoc.lmm_drug_crispr,
        assoc.ppi_string_corr,
        corr_thres=t,
        norder=o,
        fdr=0.05,
        exclude_nodes=e,
    )
    graph.write_pdf(f"{rpath}/association_ppi_{d}.pdf")


# Kinobeads

catds = KinobeadCATDS(assoc=assoc).get_data()
catds.sort_values("catds").head(15)


# E3 ligases enrichment within the significant associated gene-products

e3_ligases = pd.read_excel(f"{dpath}/ubq/Definite Ligase List.xlsx")
plot_df = assoc.lmm_drug_crispr.query(f"fdr < {target.fdr}")
plot_df["ligase"] = plot_df["GeneSymbol"].isin(e3_ligases["Gene Symbol"]).astype(int)

background = set(assoc.lmm_drug_crispr["GeneSymbol"])
signature = set(e3_ligases["Gene Symbol"]).intersection(background)
sublist = set(plot_df["GeneSymbol"]).intersection(background)

pval, int_len = DTraceEnrichment.hypergeom_test(
    signature=signature, background=background, sublist=sublist
)
logger.log(logging.INFO, f"E3 ligases hypergeom: {pval:.2e} (intersection={int_len})")
