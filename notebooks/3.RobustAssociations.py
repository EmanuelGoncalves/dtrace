# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dtrace.DTraceUtils import rpath
from dtrace.Associations import Association
from dtrace.RobustAssociations import RobustAssociations


# ### Import data-sets and associations

assoc = Association(load_associations=True, load_robust=True)

robust = RobustAssociations(assoc)


# ## Robust pharmacogenomic associations
#
# Robust pharmacogenomic associations represent pairs of Drug-Gene (drug response and gene essentiality) that are
# significantly correlated with each other and with a genomic feature (copy number/mutations) or a gene expression
# profile.

robust.assoc.lmm_robust_genomic.query("crispr_fdr < 0.1 & drug_fdr < 0.1").head(
    15
).sort_values("drug_fdr")

robust.assoc.lmm_robust_gexp.query("crispr_fdr < 0.1 & drug_fdr < 0.1").head(
    15
).sort_values("drug_fdr")

# Frequency of the genomic features across the cancer cell lines

robust.genomic_histogram()
plt.savefig(
    f"{rpath}/robust_mobems_countplot.pdf", bbox_inches="tight", transparent=True
)
plt.show()


# Top associations of drug and CRISPR wiht genomic features

robust.top_robust_features()
plt.savefig(
    f"{rpath}/robust_top_associations.pdf", bbox_inches="tight", transparent=True
)
plt.show()


# Top associations of drug and CRISPR with gene-expression

robust.top_robust_features(dtype="gene-expression")
plt.savefig(
    f"{rpath}/robust_top_associations_gexp.pdf", bbox_inches="tight", transparent=True
)
plt.show()


# Significant associations count

robust.robust_associations_barplot()
plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/robust_signif_association_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Significant associations count distributed by distance in the protein-protein interaction network

robust.robust_associations_barplot_ppi()
plt.savefig(
    f"{rpath}/robust_signif_association_barplot_ppi.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close('all')
plt.show()


# Representative examples of robust pharmacogenomic associations with copy-number and mutations

rassocs = [
    ("Olaparib", "FLI1", "EWSR1.FLI1_mut"),
    ("Dabrafenib", "BRAF", "BRAF_mut"),
    ("Nutlin-3a (-)", "MDM2", "TP53_mut"),
    ("Taselisib", "PIK3CA", "PIK3CA_mut"),
    ("MCL1_1284", "MCL1", "EZH2_mut"),
]

# d, c, g = ('Linifanib', 'STAT5B', 'XRN1_mut')
for d, c, g in rassocs:
    pair = robust.assoc.by(
        robust.assoc.lmm_robust_genomic, drug_name=d, gene_name=c, x_feature=g
    ).iloc[0]

    drug = tuple(pair[robust.assoc.dcols])

    dmax = np.log(robust.assoc.drespo_obj.maxconcentration[drug])

    plot_df = robust.assoc.build_df(
        drug=[drug], crispr=[c], genomic=[g], sinfo=["institute"]
    ).dropna()
    plot_df = plot_df.rename(columns={drug: "drug"})

    grid = robust.plot_corrplot_discrete(f"crispr_{c}", "drug", g, "institute", plot_df)

    grid.ax_joint.axhline(
        y=dmax, linewidth=0.3, color=robust.PAL_DTRACE[2], ls=":", zorder=0
    )

    grid.set_axis_labels(f"{c} (scaled log2 FC)", f"{d} (ln IC50)")

    plt.suptitle(g, y=1.05, fontsize=8)

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(
        f"{rpath}/robust_scatter_{d}_{c}_{g}.pdf", bbox_inches="tight", transparent=True
    )
    plt.show()

# Representative examples of robust pharmacogenomic associations with gene-expression

rassocs = [
    ("MCL1_1284", "MCL1", "BCL2L1"),
    ("Linsitinib", "IGF1R", "IGF1R"),
    ("EGFRM_5104", "ERBB2", "ERBB2"),
    ("Nutlin-3a (-)", "MDM2", "BAX"),
    ("Venetoclax", "BCL2", "CDC42BPA"),
    ("AZD5582", "MAP3K7", "TNF"),
    ("IAP_5620", "MAP3K7", "TNF"),
]

for d, c, g in rassocs:
    pair = robust.assoc.by(
        robust.assoc.lmm_robust_gexp, drug_name=d, gene_name=c, x_feature=g
    ).iloc[0]

    drug = tuple(pair[robust.assoc.dcols])
    dmax = np.log(robust.assoc.drespo_obj.maxconcentration[drug])

    plot_df = robust.assoc.build_df(
        drug=[drug], crispr=[c], gexp=[g], sinfo=["institute", "cancer_type"]
    ).dropna()
    plot_df = plot_df.rename(columns={drug: "drug"})

    #
    fig, axs = plt.subplots(1, 2, sharey="row", sharex="none", dpi=300)

    for i, dtype in enumerate(["crispr", "gexp"]):
        # Scatter
        for t, df in plot_df.groupby("institute"):
            axs[i].scatter(
                x=df[f"{dtype}_{c}" if dtype == "crispr" else f"{dtype}_{g}"],
                y=df["drug"],
                edgecolor="w",
                lw=0.05,
                s=10,
                color=robust.PAL_DTRACE[2],
                marker=robust.MARKERS[t],
                label=t,
                alpha=0.8,
            )

        # Reg
        sns.regplot(
            x=plot_df[f"{dtype}_{c}" if dtype == "crispr" else f"{dtype}_{g}"],
            y=plot_df["drug"],
            data=plot_df,
            color=robust.PAL_DTRACE[1],
            truncate=True,
            fit_reg=True,
            scatter=False,
            line_kws=dict(lw=1.0, color=robust.PAL_DTRACE[0]),
            ax=axs[i],
        )

        # Annotation
        cor, pval = pearsonr(
            plot_df[f"{dtype}_{c}" if dtype == "crispr" else f"{dtype}_{g}"],
            plot_df["drug"],
        )
        annot_text = f"R={cor:.2g}, p={pval:.1e}"

        axs[i].text(
            0.95, 0.05, annot_text, fontsize=4, transform=axs[i].transAxes, ha="right"
        )

        # Misc
        axs[i].axhline(
            y=dmax, linewidth=0.3, color=robust.PAL_DTRACE[2], ls=":", zorder=0
        )

        axs[i].set_ylabel(f"{d} (ln IC50)" if i == 0 else "")
        axs[i].set_xlabel(f"scaled log2 FC" if dtype == "crispr" else f"RNA-seq voom")
        axs[i].set_title(c if dtype == "crispr" else g)

        axs[i].grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        # Legend
        axs[i].legend(prop=dict(size=4), frameon=False, loc=2)

    plt.subplots_adjust(wspace=0.05)
    plt.gcf().set_size_inches(3, 1.5)
    plt.savefig(
        f"{rpath}/robust_scatter_gexp_{d}_{c}_{g}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# Synthetic lethal interaction between STAG1/2. STAG2 mutations lead to dependency in STAG1.

gene_gexp, gene_crispr, gene_mut = "STAG2", "STAG1", "STAG2_mut"

plot_df = robust.assoc.build_df(
    crispr=[gene_crispr], gexp=[gene_gexp], genomic=[gene_mut], sinfo=["institute"]
).dropna()

grid = RobustAssociations.plot_corrplot_discrete(
    f"crispr_{gene_crispr}", f"gexp_{gene_gexp}", gene_mut, "institute", plot_df
)
grid.set_axis_labels(f"{gene_crispr} (scaled log2 FC)", f"{gene_gexp} (RNA-seq voom)")
plt.suptitle(gene_mut, y=1.05, fontsize=8)
plt.gcf().set_size_inches(1.5, 1.5)
plt.savefig(
    f"{rpath}/robust_scatter_gexp_crispr_{gene_gexp}_{gene_crispr}_{gene_mut}.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()

plt.figure(figsize=(0.75, 1.5), dpi=300)
g = RobustAssociations.plot_boxplot_discrete(gene_mut, f"crispr_{gene_crispr}", plot_df)
plt.ylabel(f"{gene_crispr}\n(scaled log2 FC)")
plt.gcf().set_size_inches(0.75, 1.5)
plt.savefig(
    f"{rpath}/robust_genomic_boxplot_{gene_mut}.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()

# Copyright (C) 2019 Emanuel Goncalves
