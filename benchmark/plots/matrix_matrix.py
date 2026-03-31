import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

# Default colors for paper-style legends (extended with tab20 for unknown types).
_PAPER_TYPE_ORDER = [
    "Dense",
    "Snack",
    "JaxSparse",
    "SparseTorch",
    "SparseUT",
    "Dense (Only)",
    "CuPy Sparse CSR",
    "Sputnik",
]


def _type_color_map(types_present):
    """Stable colors for known backends; assign distinct tab20 colors for any other Type."""
    tab20 = plt.cm.tab20.colors
    color_map = {}
    used = 0
    for name in _PAPER_TYPE_ORDER:
        if name in types_present:
            color_map[name] = tab20[used % len(tab20)]
            used += 1
    for t in sorted(types_present):
        if t not in color_map:
            color_map[t] = tab20[used % len(tab20)]
            used += 1
    return color_map


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
    if len(categories) == 0:
        color_map = _type_color_map(set(unique_keys))
        idx_map = {category: i for i, category in enumerate(sorted(color_map.keys()))}
        color_map["Dense+Mask"] = color_map.get("Dense+Mask", color_map.get("Dense", plt.cm.tab20.colors[0]))
        idx_map["Dense+Mask"] = idx_map.get("Dense+Mask", idx_map.get("Dense", 0))
    else:
        colors = plt.cm.tab20.colors
        color_map = {category: colors[i % len(colors)] for i, category in enumerate(categories)}
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

            # Annotate batch size on the middle of the sparsity sweep
            mid_idx = min(len(sparsity_level) // 2, len(sparsity_level) - 1)
            if batch_needed and idx_map.get(is_sparse, 0) == 0 and 128 >= batch_size >= 2:
                continue

            ax.annotate(
                f"bz={batch_size}" if batch_needed else "",
                (
                    sparsity_level.iloc[mid_idx],
                    avg_per_sparsity.iloc[mid_idx],
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


def dense_level_area(dense_level: str) -> int:
    a, b = dense_level.split("x")
    return int(a) * int(b)


def plot_matrix_size_sweep(
    df,
    col,
    sparsity_level,
    BATCHES,
    figsize=(7, 5),
    ax=None,
    log=False,
    title="",
    y_label="",
    categories=None,
):
    """
    Fixed sparsity: x = matrix area (layer_a * layer_b), y = timing.
    Use when CSVs only contain one sparsity level (line vs sparsity degenerates).
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.figure

    d = df[(df.sparsity_level == sparsity_level) & (df.batch_size.isin(BATCHES))].copy()
    if categories:
        d = d[d.Type.isin(categories)]
    d["_area"] = d["dense_level"].astype(str).map(dense_level_area)
    types_present = d["Type"].unique().tolist()
    color_map = _type_color_map(set(types_present))

    max_y, min_y = 0, np.inf
    for typ, g in d.groupby("Type"):
        curve = g.groupby("_area")[col].mean().sort_index()
        err = g.groupby("_area")[col].std().fillna(0) / np.sqrt(
            g.groupby("_area")[col].count().clip(lower=1)
        )
        max_y = max(max_y, curve.max())
        min_y = min(min_y, curve.min())
        ax.plot(
            curve.index,
            curve.values,
            linestyle="-",
            marker="o",
            color=color_map[typ],
            label=typ,
            alpha=0.9,
        )
        ax.fill_between(
            curve.index,
            curve.values - err,
            curve.values + err,
            alpha=0.12,
            color=color_map[typ],
        )

    ax.set_xlabel(r"Matrix size ($n \times m$ area)")
    ax.set_ylabel(y_label or col)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    if log:
        ax.set_yscale("log")
    ax.legend(title="Backend", loc="best", fontsize=9)
    return fig
