import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def cornerplot(data, n_dim, labels=None, norm=LogNorm(), figsize=None):
    """
    Create a corner plot (also known as a pair plot) for visualizing the distribution of data and the relationships between pairs of dimensions.

    Parameters:
    data (numpy.ndarray): The data to be plotted, expected to be a 2D array where each column represents a dimension.
    n_dim (int): The number of dimensions to plot.
    labels (list of str, optional): A list of labels for each dimension. If provided, these labels will be used for the axes.

    Returns:
    tuple: (figure, axes) - The matplotlib figure and axes objects

    Notes:
    - The diagonal plots (i == j) show histograms of each dimension.
    - The lower triangle plots (i > j) show 2D histograms of pairs of dimensions.
    - The upper triangle plots (i < j) are left empty.
    - If labels are provided, they are used to label the axes of the plots.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    if data.shape[1] < n_dim:
        raise ValueError(
            f"Insufficient dimensions in data. Expected {n_dim}, got {data.shape[1]}."
        )
    if figsize is None:
        figsize = (n_dim * 2, n_dim * 2)

    fig, axes = plt.subplots(n_dim, n_dim, figsize=figsize)

    # Remove all spacing between plots
    plt.subplots_adjust(
        wspace=0.0, hspace=0.0, left=0.1, right=0.9, bottom=0.1, top=0.9
    )

    for i in range(n_dim):
        for j in range(n_dim):
            if i == j:
                axes[i, j].hist(data[:, i], bins=100, histtype="step")
                if labels is not None and i == n_dim - 1:
                    axes[i, j].set_xlabel(labels[i])
                if labels is not None and j == 0:
                    axes[i, j].set_ylabel(labels[i])
            elif i > j:
                axes[i, j].hist2d(
                    data[:, j], data[:, i], bins=100, cmap="jet", norm=norm
                )
                if labels is not None and i == n_dim - 1:
                    axes[i, j].set_xlabel(labels[j])
                if labels is not None and j == 0:
                    axes[i, j].set_ylabel(labels[i])
            else:
                axes[i, j].axis("off")

            if i < n_dim - 1:
                axes[i, j].xaxis.set_visible(False)
            if j > 0:
                axes[i, j].yaxis.set_visible(False)

    return fig, axes
