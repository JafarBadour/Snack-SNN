import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np


def plt_them(
    df_dlvl,
    col,
    ax=None,
    tot=4,
    x_title=None,
    log=False,
    title="",
    batch_needed=False,
    suffix="",
    legend_place="upper right",
    categories=[]
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    max_avg_ct_ = 0
    min_avg_ct_ = np.inf
    unique_keys = df_dlvl["Type"].unique().tolist()
    colors = plt.cm.tab10.colors  # Assign colors per Type
    if len(categories) == 0:
        color_map = {
            category: colors[i] for i, category in enumerate(["Dense", "Snack", "JaxSparse", "SparseTorch", "SparseUT"])
        }
        idx_map = {category: i for i, category in enumerate(["Dense", "Snack", "JaxSparse", "SparseTorch", "SparseUT"])}
        color_map["Dense+Mask"] = color_map["Dense"]
        idx_map["Dense+Mask"] = idx_map["Dense"]
    else:
        color_map = {
            category: colors[i] for i, category in enumerate(categories)
        }
        idx_map = {category: i for i, category in enumerate(categories)}
        df_dlvl = df_dlvl[df_dlvl.Type.isin(categories)]



    for is_sparse, group_sparse in df_dlvl.groupby("Type"):
        for batch_size, group_batch in group_sparse.groupby("batch_size" if batch_needed else "Type"):
            avg_per_sparsity = group_batch.groupby("sparsity_level")[col].mean()
            # print(group_batch.groupby("sparsity_level")[col].agg(list))
            std_per_sparsity = group_batch.groupby("sparsity_level")[col].std() / 5
            sparsity_level = group_batch.groupby("sparsity_level")["sparsity_level"].first()
            max_avg_ct_ = max(max_avg_ct_, avg_per_sparsity.max())
            min_avg_ct_ = min(min_avg_ct_, avg_per_sparsity.min())
            ax.plot(
                sparsity_level,
                avg_per_sparsity,
                linestyle="-",
                marker="o",
                color=color_map[is_sparse],
                alpha=0.8,
            )
            ax.fill_between(
                sparsity_level,
                avg_per_sparsity - std_per_sparsity,
                avg_per_sparsity + std_per_sparsity,
                alpha=0.1,
                color=color_map[is_sparse],
            )

            # Annotate batch size on the middle of the line
            mid_idx = len(sparsity_level) // 2
            if batch_needed and idx_map[is_sparse] == 0 and 128 >= batch_size >= 2:
                continue

            ax.annotate(
                f"bz={batch_size}" if batch_needed else "",
                (
                    sparsity_level.iloc[idx_map[is_sparse]],
                    avg_per_sparsity.iloc[idx_map[is_sparse]],
                ),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=10,
                color=color_map[is_sparse],
                fontweight="bold",
            )

    ax.set_xticks(list(range(int(df_dlvl.sparsity_level.min()), 101, 10)))

    if log:
        ax.set_yscale("log")  # Set Y-axis to logarithmic scale
        yticks = np.geomspace(min_avg_ct_, max_avg_ct_, num=5)  # Log-spaced ticks
        ax.set_yticks(yticks)
        if "cuda" in col.lower():
            ax.set_yticklabels([f"{t:,.1f} μs" for t in yticks])  # Custom labels with time units
        else:
            ax.set_yticklabels([f"{t:0.1f}" for t in yticks])  # Custom labels with time units

    else:
        yticks = ax.set_yticks(np.linspace(0, max_avg_ct_, 10))

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Sparsity Level")
    ax.set_ylabel(x_title or "Cuda Time (in microseconds)")

    # Custom legend for Type categories
    handles = [
        plt.Line2D([0], [0], color=color_map[cat], lw=2, label=f"{cat} {suffix}")
        for cat in color_map.keys()
        if cat in unique_keys
    ]
    ax.legend(handles=handles, title="Type", loc=legend_place, fontsize=9)

    return ax


def create_multi_plot_for_batches(df, dense_level, BATCHES, log=False):
    fig, ax = plt.subplots(1, 3, figsize=(21, 5))

    df_dlvl = df[(df.dense_level == dense_level) & (df.batch_size.isin(BATCHES))]
    df_dlvl[col] = df_dlvl[col]
    plt_them(df_dlvl, ax=ax[0])

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Group by the 'category' column and plot each group
    df_dlvl = df_dlvl[df_dlvl.Type != "SparseTorch"]
    plt_them(df_dlvl, ax=ax[1])

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Group by the 'category' column and plot each group
    df_dlvl = df_dlvl[~df_dlvl.Type.isin(["SparseTorch", "JaxSparse"])]
    plt_them(df_dlvl, ax[2])
    plt.show()


def create_single_plot_for_batches(
    df,
    dense_level,
    col,
    BATCHES,
    figsize=(5, 5),
    ax=None,
    log=False,
    title="",
    x_label="",
    y_label="",
    batch_needed=False,
    suffix="",
    legend_place="upper right",
    categories=[]
):
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)

    df_dlvl = df[(df.dense_level == dense_level) & (df.batch_size.isin(BATCHES))]
    df_dlvl[col] = df_dlvl[col]  # .apply(lambda x: np.log2(x))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax = plt_them(
        df_dlvl,
        col,
        ax=ax,
        x_title=x_label,
        log=log,
        title=title,
        batch_needed=batch_needed,
        suffix=suffix,
        legend_place=legend_place,
        categories=categories
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig
