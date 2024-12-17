import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def cornerplot(data, n_dim, labels=None, norm=LogNorm()):
    """
    Create a corner plot (also known as a pair plot) for visualizing the distribution of data and the relationships between pairs of dimensions.

    Parameters:
    data (numpy.ndarray): The data to be plotted, expected to be a 2D array where each column represents a dimension.
    n_dim (int): The number of dimensions to plot.
    labels (list of str, optional): A list of labels for each dimension. If provided, these labels will be used for the axes.

    Returns:
    None: This function does not return any value. It displays the plot using matplotlib.

    Notes:
    - The diagonal plots (i == j) show histograms of each dimension.
    - The lower triangle plots (i > j) show 2D histograms of pairs of dimensions.
    - The upper triangle plots (i < j) are left empty.
    - If labels are provided, they are used to label the axes of the plots.
    """
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(20, 20))
    for i in range(n_dim):
        for j in range(n_dim):
            if i == j:
                axes[i, j].hist(data[:, i], bins=100, color="black")
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
    plt.tight_layout()
    plt.show()
