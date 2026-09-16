import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import scanpy as sc
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.axes_grid1 import make_axes_locatable



def get_figs(ncols=4,x=6,y=6,vert=False):
    if vert:
        fig,axes = plt.subplots(nrows=ncols,figsize=(x,y*ncols))
    else:
        fig,axes = plt.subplots(ncols=ncols,figsize=(x*ncols,y))
    return(fig,axes)


def plot_boxplots(df, group_col, value_col,ax=None):
    if ax is None:
        fig,ax = get_figs(1)
    sns.boxplot(data=df, x=group_col, y=value_col,ax=ax)
    
    counts = df[group_col].value_counts()
    labels = [f"{label.get_text()} ({counts[label.get_text()]})" for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(value_col)

def space_plot(adata,color='annotation',ax=None,s=20,alpha=1,spatial_layer='spatial'):
    show=False
    if ax is None:
        fig,ax = get_figs(1)
        show=True
    sc.pl.embedding(adata,spatial_layer,color=color,ax=ax,s=s,alpha=alpha,show=show)

def space_plot_mask(adata,col,group,color=None,ax=None,sm=20,snm=5,alpha_sm=1,alpha_snm=.5,spatial_layer='spatial'):
    if ax is None:
        fig,ax = get_figs(1)
    if color is None:
        color=col
    mask = adata.obs[col]== group
    sc.pl.embedding(adata[~mask],spatial_layer,color=None,alpha=alpha_snm,s=snm,ax=ax,show=False)
    sc.pl.embedding(adata[mask],spatial_layer,color=color,alpha=alpha_sm,s=sm,ax=ax,show=False)
    # print(mask.sum())
    # ax.get_legend().remove()
    return(mask)


def umap_plot(adata,color=None,ax=None,s=20,umap_key='X_umap'):
    show=False
    if ax is None:
        fig,ax = get_figs(1)
        show=True
    
    umap = adata.obsm[umap_key]
    ref = 'X_umap' in adata.obsm
 
    if ref:
        old_umap = adata.obsm['X_umap']
 
    adata.obsm[f'X_umap'] = umap    
    sc.pl.umap(adata,color=color,ax=ax,s=s,show=show)
 
    if ref:
        adata.obsm['X_umap'] = old_umap














# Proportions 

def get_proportions(adata,col1='spatial_domains',col2='cell_type',normalize='index'):
    return(pd.crosstab(
        adata.obs[col1], 
        adata.obs[col2], 
        normalize=normalize  # normalize by row (i.e., each cluster sums to 1)
        ))
    

def plot_proportions(proportions: pd.DataFrame, ax=None, colors=None, horizontal=True):
    """
    proportions: DataFrame (rows=clusters, columns=cell_types)
    colors: Dictionary mapping column names (cell_types) to hex colors
    """
    if ax is None:
        fig, ax = get_figs(1)

    color_list = None
    if colors is not None:
        color_list = [colors[i] for i,col in enumerate(proportions.columns)]

    kind = 'barh' if horizontal else 'bar'         
    proportions.plot(kind=kind, stacked=True, ax=ax, color=color_list)

    if horizontal:  
        ax.invert_yaxis()  
        ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal', fontsize=15)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation='horizontal', fontsize=15)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    return ax


def heatmap_proportions(proportions,ax=None,scaled=True,dendro=True,col_order=None,colors=None):
    if ax is None:
        fig,ax = get_figs(1,15,10)


    if scaled:
        # --- standard_scale=1 equivalent (normalize each column to [0,1]) ---
        proportions = (proportions - proportions.min(axis=0)) / \
                            (proportions.max(axis=0) - proportions.min(axis=0))
        
    # --- Compute clustering on the scaled data ---
    row_linkage = linkage(proportions.values, method="ward")

    row_order = leaves_list(row_linkage)
    

    if col_order is None:
        proportions_ordered = proportions.iloc[row_order, :]
    else:
        proportions_ordered = proportions.iloc[row_order, col_order]

    if dendro:
        # Create space for a dendrogram above the heatmap
        divider = make_axes_locatable(ax)
        ax_dendro = divider.append_axes("left", size="15%", pad=0.05)

        # Plot column dendrogram
        dendrogram(row_linkage, ax=ax_dendro, no_labels=True, color_threshold=0,orientation='left')
        # ax_dendro.set_axis_off()
        ax_dendro.invert_yaxis()     # match heatmap row order (top to bottom)
        ax_dendro.set_axis_off()
    # --- Plot heatmap in axes[0] ---
    sns.heatmap(
        proportions_ordered,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.6,'anchor':(5.05,0.5)},
    )
    ax.set_title("Cell type composition (clustered)")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15) 
    ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical', fontsize=15) 
    if colors is not None:
        for t in ax.get_yticklabels():
            # print(t.get_text())
            cluster = t.get_text().split('_')[-1]
            cluster_idx = int(cluster)
            t.set_color(colors[cluster_idx])
    return(row_order,col_order)

def plot_region_distribution(proportions, colors=None, ax=None,fontsize=12,format_value=',',threshold=0.06,percentage=False):
    if ax is None:
        fig,ax=get_figs(1)
    totals = proportions.sum(axis=1)
    normed = proportions.div(totals, axis=0)
    regions = list(normed.columns)
    palette = colors or plt.cm.Set2(np.linspace(0, 1, len(regions)))
    y = np.arange(len(normed))
    left = np.zeros(len(normed))
    
    for i, reg in enumerate(regions):
        w = normed[reg].values
        c = normed[reg].values if percentage else proportions[reg].values 
        ax.barh(y, w, left=left, height=.9, color=palette[i], label=reg)
        for j, (wi, ci) in enumerate(zip(w, c)):
            if wi > threshold:
                text = f'{ci:{format_value}}' if wi < 0.10 else f'{ci:{format_value}}'
                ax.text(left[j] + wi/2, y[j], text,
                        ha='center', va='center', fontsize=fontsize,
                        color='black',
                        # path_effects=[pe.withStroke(linewidth=1.2, foreground='black')]
                        )
        left += w
    
    for j, t in enumerate(totals):
        ax.text(1.01, y[j], f'n={t:,}', va='center', ha='left', fontsize=15)
    
    ax.set_yticks(y)
    ax.set_yticklabels(normed.index)
    ax.set_xlim(0, 1.15)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 6)])
    ax.set_xlabel('Row-normalized proportion')
    ax.invert_yaxis()
    ax.legend(title='Region', bbox_to_anchor=(1.25, 1), loc='upper left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    return ax










def harmonize_target_colors(
    adata,
    col_ref="annotation",
    col_target="spatial_domains",
):
    """
    Match categories of `col_target` to those of `col_ref` by maximum overlap
    and recolor `col_target` so that matching spatial domains use the same
    color as in the reference.

    Modifies `adata.uns[f"{col_target}_colors"]` in place.
    """
    # Make sure both columns are treated as categorical
    for column in (col_ref, col_target):
        if not isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
            adata.obs[column] = adata.obs[column].astype("category")

    reference_categories = adata.obs[col_ref].cat.categories
    target_categories = adata.obs[col_target].cat.categories

    reference_colors_key = f"{col_ref}_colors"
    target_colors_key = f"{col_target}_colors"

    if reference_colors_key not in adata.uns:
        raise KeyError(
            f"Reference colors not found in adata.uns['{reference_colors_key}']. "
            "Plot the reference column once with scanpy to generate them."
        )

    reference_colors = list(adata.uns[reference_colors_key])

    # Keep any original target colors for clusters that cannot be matched
    if target_colors_key in adata.uns:
        original_target_colors = list(adata.uns[target_colors_key])
    else:
        original_target_colors = ["#888888"] * len(target_categories)

    # Contingency table: rows = reference domains, columns = target clusters
    overlap = pd.crosstab(adata.obs[col_ref], adata.obs[col_target])
    overlap = overlap.reindex(
        index=reference_categories,
        columns=target_categories,
        fill_value=0,
    )

    # One-to-one optimal matching that maximizes total overlap
    reference_indices, target_indices = linear_sum_assignment(
        overlap.values, maximize=True
    )  # \ue202turn1search1

    # Map each matched target category to its reference color
    target_color_by_category = {}
    for reference_index, target_index in zip(reference_indices, target_indices):
        target_label = target_categories[target_index]
        target_color_by_category[target_label] = reference_colors[reference_index]

    # Build the new color list in the original target-category order
    new_target_colors = []
    for target_label in target_categories:
        if target_label in target_color_by_category:
            new_target_colors.append(target_color_by_category[target_label])
        else:
            original_index = target_categories.get_loc(target_label)
            new_target_colors.append(original_target_colors[original_index])
    adata.uns[target_colors_key] = new_target_colors

    return adata







def _cmap(adata, col):
    vc = adata.obs[col]
    cats = vc.cat.categories if pd.api.types.is_categorical_dtype(vc) else vc.unique()
    return dict(zip(cats, adata.uns.get(f"{col}_colors", plt.cm.tab20(np.linspace(0, 1, len(cats))))))


def plot_ccc_score_per_group(adata, group_col, ccc_score_col, column_shown,sample_shown,ax=None,insersion=(.15,.25)):
    s = adata.obs.groupby(group_col)[ccc_score_col].mean().sort_values()
    order = s.index.tolist()
    # print(s)
    cmap = _cmap(adata, group_col)
    # cols = [cmap[c] for c in order]
    if isinstance(order[0],int):
        cols = [adata.uns[f'{group_col}_colors'][c] for i,c in enumerate(order)]
    else:
        cols = [adata.uns[f'{group_col}_colors'][i] for i,c in enumerate(order)]
    ax = ax or plt.subplots(figsize=(max(8, len(order)), 4))[1]

    ax.bar(range(len(order)), s.values, color=cols)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    # print(len(ax.get_xticklabels()), len(cols),len(order))
    for tl, c in zip(ax.get_xticklabels(), cols):
        # print(c)
        tl.set_color(c)
    ax.set(xlabel="Group", ylabel="CCC score (mean)")
    ax.grid(axis="y", ls="--", alpha=0.3)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0],ylim[1]+0.12*ylim[1])
    for i, c in enumerate(order):
        # print(c,cdict[c])
        x_ax, y_ax = ax.transAxes.inverted().transform(ax.transData.transform((i, s.values[i])))
        # ia = ax.inset_axes([x_ax - 0.04, y_ax + 0.01, 0.04, 0.15])
        ia = ax.inset_axes([x_ax - 0.04, y_ax + 0.01, insersion[0], insersion[1]])
        adatai = adata[adata.obs[column_shown]==sample_shown]
        space_plot_mask(adatai, group_col, c, ax=ia, snm=.1, sm=.8)
        ia.set(xticks=[], yticks=[])
        ia.set_title('')
        ia.set_xlabel('')
        ia.set_ylabel('')
        ia.get_legend().remove()
    return ax

def plot_ct_pair_across_samples(
    adatas, sender, receiver, groups="global_cell_type", direction="R",
    agg="mean", ax=None, genes=None, **heatmap_kw
):
    """
    Heatmap: samples (rows) × LR pairs (columns) for a fixed sender–receiver pair.
    
    Parameters
    ----------
    adatas : dict[str, AnnData]
        {sample_name: adata} with LR scores already computed.
    sender : str
        Cell type acting as sender.
    receiver : str
        Cell type acting as receiver (context source).
    groups : str
        obs column for cell type grouping.
    direction : str
        "E" (emitter) or "R" (receptor).
    agg : str
        Aggregation function ("mean", "sum", "median").
    ax : matplotlib Axes, optional
    **heatmap_kw : passed to sns.heatmap
    """
    pairs = None
    for ad in adatas.values():
        if pairs is None:
            pairs = list(ad.uns["context_LR"]["pairs"].index)
            break
    
    key_tpl = f"LR_{direction}_{groups}_c{sender}"

    rows = {}
    for sname, ad in adatas.items():
        mask = ad.obs[groups] == receiver
        if key_tpl in ad.obsm and mask.sum() > 0:
            S = ad.obsm[key_tpl][mask.values]
            rows[sname] = getattr(np, agg)(S, axis=0)
        else:
            rows[sname] = np.full(len(pairs), np.nan)
    
    df = pd.DataFrame(rows, index=pairs).T
    if genes is not None:
        df = df.T
        df = df.loc[list(genes)]
        df = df.T
        
    if len(df)>0:
        if ax is None:
            _, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.5), max(3, len(adatas) * 0.4)))
        
        kw = dict(cmap="viridis", linewidths=0.5,annot=True, fmt=".2f")
        kw.update(heatmap_kw)
        sns.heatmap(df.T, ax=ax, **kw)
        
        s = "emitter" if direction == "E" else "receptor"
        # title = f'{sender}→{receiver}' if direction == 'E' else f'{receiver}→{sender}'
        title = f'{sender} to {receiver}'
        ax.set_title(f"{title}")
        ax.set_ylabel("LR Pair")

        
        ax.set_xlabel("Sample")
        plt.xticks(rotation=90)
    return df



def pie_region(adata, value, region_col, col='annotation', ax=None, colors=None):
    p = adata.obs.loc[adata.obs[col] == value, region_col].value_counts()
    ax = ax or get_figs(1, 5, 5)[1]
    ax.pie(p.values, labels=[f'{k}\n{v:,}' for k, v in p.items()], autopct='%1.0f%%',
           colors=colors or adata.uns.get(f'{region_col}_colors'), startangle=90)
    ax.set_title(f'{value}  (n={p.sum():,})', fontsize=13)


# def plot_ct_pair_across_samples(
#     adatas, sender, receiver, groups="global_cell_type", direction="R",
#     agg="mean", ax=None, genes=None,
#     normalize=None,        # None | 'max' | 'sum' | 'zscore' | 'ref'
#     min_value=0.0,         # drop pairs whose max across samples is below this
#     annot_raw=True,        # annotate with raw values, colour with normalised
#     **heatmap_kw
# ):
#     """
#     Heatmap: samples (rows) × LR pairs (columns) for a fixed sender–receiver pair.
    
#     Parameters
#     ----------
#     adatas : dict[str, AnnData]
#         {sample_name: adata} with LR scores already computed.
#     sender : str
#         Cell type acting as sender.
#     receiver : str
#         Cell type acting as receiver (context source).
#     groups : str
#         obs column for cell type grouping.
#     direction : str
#         "E" (emitter) or "R" (receptor).
#     agg : str
#         Aggregation function ("mean", "sum", "median").
#     ax : matplotlib Axes, optional
#     **heatmap_kw : passed to sns.heatmap
#     """
#     pairs = None
#     for ad in adatas.values():
#         if pairs is None:
#             pairs = list(ad.uns["context_LR"]["pairs"].index)
#             break
    
#     key_tpl = f"LR_{direction}_{groups}_c{sender}"

#     rows = {}
#     for sname, ad in adatas.items():
#         mask = ad.obs[groups] == receiver
#         # if key_tpl in ad.obsm and mask.sum() > 0:
#         #     S = ad.obsm[key_tpl][mask.values]
#         #     if agg == "meanpos":
#         #         nz = (S != 0).sum(0)
#         #         v = np.asarray(S.sum(0)).ravel() / np.maximum(nz, 1)
#         #         v[nz < LR_MIN_SUPPORT] = np.nan      # same 100-cell rule
#         #         rows[sname] = v
#         #     else:
#         #         rows[sname] = getattr(np, agg)(S, axis=0)
#         if key_tpl in ad.obsm and mask.sum() > 0:
#             S = ad.obsm[key_tpl][mask.values]
#             rows[sname] = getattr(np, agg)(S, axis=0)
#         else:
#             rows[sname] = np.full(len(pairs), np.nan)
    
#     df = pd.DataFrame(rows, index=pairs).T

#     if genes is not None:
#         df = df.T.loc[list(genes)].T

#     # ---- pairs x samples: this is what is actually plotted -----------------
#     mat = df.T
#     raw = mat.copy()

#     if min_value:
#         keep = raw.max(axis=1) > min_value
#         mat, raw = mat.loc[keep], raw.loc[keep]

#     fmt_default, cbar_label, vlim = ".2f", "score", {}
#     if normalize == "max":            # % of the maximum of that pair
#         mat = 100 * mat.div(mat.max(axis=1).replace(0, np.nan), axis=0)
#         fmt_default, cbar_label, vlim = ".0f", "% of pair max", dict(vmin=0, vmax=100)
#     elif normalize == "sum":          # share of the pair total across samples
#         mat = 100 * mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
#         fmt_default, cbar_label, vlim = ".0f", "% of pair total", dict(vmin=0)
#     elif normalize == "ref":          # fold-change vs the first sample
#         mat = mat.div(mat.iloc[:, 0].replace(0, np.nan), axis=0)
#         fmt_default, cbar_label = ".2f", f"fold-change vs {mat.columns[0]}"
#     elif normalize == "zscore":       # centred/scaled across samples
#         mat = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1) + 1e-12, axis=0)
#         fmt_default, cbar_label = ".1f", "z-score across samples"
#         vlim = dict(cmap="RdBu_r", center=0)

#     if len(mat) > 0:
#         if ax is None:
#             _, ax = plt.subplots(figsize=(max(8, len(mat.columns) * 0.5),
#                                           max(3, len(mat) * 0.4)))
#         kw = dict(cmap="viridis", linewidths=0.5, annot=True,
#                   fmt=fmt_default, cbar_kws={"label": cbar_label}, **vlim)
#         if annot_raw and normalize:
#             kw["annot"], kw["fmt"] = raw.values, ".2g"
#         kw.update(heatmap_kw)
#         sns.heatmap(mat, ax=ax, **kw)

#         ax.set_title(f"{sender} to {receiver}")
#         ax.set_ylabel("LR Pair")
#         ax.set_xlabel("Sample")
#         plt.setp(ax.get_xticklabels(), rotation=90)
#     return mat            # was: df  (raw values still recoverable with normalize=None)

