import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

def plt_them(df_dlvl, ax=None, tot=4, x_title=None, log=False, title=""):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    max_avg_ct_ = 0
    min_avg_ct_ = np.inf
    unique_keys = df_dlvl["SparseType"].unique().tolist()
    colors = plt.cm.tab10.colors  # Assign colors per SparseType
    color_map = {category: colors[i] for i, category in enumerate(['JaxSparse', 'SparseTorch', 'SparseUT', 'Dense', 'Sparse'])}
    idx_map = {category: i for i, category in enumerate(['JaxSparse', 'SparseTorch', 'SparseUT', 'Dense', 'Sparse'])}

    for is_sparse, group_sparse in df_dlvl.groupby('SparseType'):
        for batch_size, group_batch in group_sparse.groupby('batch_size'):
            avg_per_sparsity = group_batch.groupby('sparsity_level')['cuda_elapsed_time'].mean() * 10
            std_per_sparsity = group_batch.groupby('sparsity_level')['cuda_elapsed_time'].std() * 10
            sparsity_level = group_batch.groupby('sparsity_level')['sparsity_level'].first()
            max_avg_ct_ = max(max_avg_ct_, avg_per_sparsity.max())
            min_avg_ct_ = min(min_avg_ct_, avg_per_sparsity.min())
            ax.plot(sparsity_level, avg_per_sparsity, linestyle='-', marker='o',
                    color=color_map[is_sparse], alpha=0.8)
            ax.fill_between(sparsity_level, avg_per_sparsity - std_per_sparsity,
                            avg_per_sparsity + std_per_sparsity, alpha=0.1, color=color_map[is_sparse])

            # Annotate batch size on the middle of the line
            mid_idx = len(sparsity_level) // 2
            if is_sparse == "Dense" and 128 >= batch_size >= 2: 
                continue
            ax.annotate(f'bz={batch_size}',
                        (sparsity_level.iloc[idx_map[is_sparse]], avg_per_sparsity.iloc[idx_map[is_sparse]]),
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10,
                        color=color_map[is_sparse], fontweight='bold')

    ax.set_xticks(list(range(int(df_dlvl.sparsity_level.min()), 101, 5)))

    if log:
        ax.set_yscale("log")  # Set Y-axis to logarithmic scale
        yticks = np.geomspace(min_avg_ct_, max_avg_ct_, num=10)  # Log-spaced ticks
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(t)}μs" for t in yticks])  # Custom labels with time units
    else:
        ax.set_yticks(np.linspace(0, max_avg_ct_, 20))

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Sparsity Level')
    ax.set_ylabel(x_title or 'Cuda Time (in microseconds)')

    # Custom legend for SparseType categories
    handles = [plt.Line2D([0], [0], color=color_map[cat], lw=2, label=f'SparseType={cat}')
               for cat in color_map.keys() if cat in unique_keys]
    ax.legend(handles=handles, title='Multiplication Type', loc='upper right', fontsize=9)


    return ax
def create_multi_plot_for_batches(df, dense_level, BATCHES, log=False):
    fig, ax = plt.subplots(1,3,figsize=(21, 5));

    df_dlvl = df[(df.dense_level == dense_level) & (df.batch_size.isin(BATCHES))]
    df_dlvl['cuda_elapsed_time'] = df_dlvl['cuda_elapsed_time']
    plt_them(df_dlvl, ax=ax[0])


    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Group by the 'category' column and plot each group
    df_dlvl=df_dlvl[df_dlvl.SparseType != 'SparseTorch']
    plt_them(df_dlvl, ax=ax[1])



    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Group by the 'category' column and plot each group
    df_dlvl=df_dlvl[~df_dlvl.SparseType.isin(['SparseTorch', 'JaxSparse'])]
    plt_them(df_dlvl, ax[2])
    plt.show()
def create_single_plot_for_batches(df, dense_level, BATCHES, figsize=(5, 5), log=False, title=""):
    fig, ax = plt.subplots(1,figsize=figsize);

    df_dlvl = df[(df.dense_level == dense_level) & (df.batch_size.isin(BATCHES))]
    df_dlvl['cuda_elapsed_time'] = df_dlvl['cuda_elapsed_time']# .apply(lambda x: np.log2(x))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt_them(df_dlvl, ax=ax, x_title="log of cuda time", log=log, title=title)
    plt.show()