import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import scipy.ndimage


def cornerplot(data, n_dim, labels=None, norm=LogNorm(), figsize=None, cmap="Greys"):
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
                    data[:, j], data[:, i], bins=100, cmap=cmap, norm=norm
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


def cornerplot_with_classes(
    data, n_dim, feature_labels=None, class_labels=None, norm=LogNorm(), figsize=None
):
    """
    Create a corner plot (also known as a pair plot) for visualizing the distribution of data
    and the relationships between pairs of dimensions.

    Parameters:
    data (numpy.ndarray): The data to be plotted, expected to be a 2D array.
    n_dim (int): The number of dimensions to plot.
    feature_labels (list of str, optional): Labels for each dimension. Used for the axes.
    class_labels (numpy.ndarray, optional): A 1D array of categorical labels matching the rows
                                            in `data`. If provided, colors distributions by class.
    norm (matplotlib.colors.Normalize, optional): Normalization for the 2D histogram.
    figsize (tuple, optional): Size of the figure.

    Returns:
    tuple: (figure, axes) - The matplotlib figure and axes objects
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    if data.shape[1] < n_dim:
        raise ValueError(
            f"Insufficient dimensions in data. Expected {n_dim}, got {data.shape[1]}."
        )
    if class_labels is not None and len(class_labels) != data.shape[0]:
        raise ValueError(
            "Length of class_labels must match the number of rows in data."
        )

    if figsize is None:
        figsize = (n_dim * 2, n_dim * 2)

    fig, axes = plt.subplots(n_dim, n_dim, figsize=figsize)

    # Remove all spacing between plots
    plt.subplots_adjust(
        wspace=0.0, hspace=0.0, left=0.1, right=0.9, bottom=0.1, top=0.9
    )

    # Extract unique classes if provided
    unique_classes = np.unique(class_labels) if class_labels is not None else [None]

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]

            # -----------------------------------------------------
            # DIAGONAL: 1D Distributions
            # -----------------------------------------------------
            if i == j:
                if class_labels is None:
                    ax.hist(data[:, i], bins=100, histtype="step", color="black")
                else:
                    for c in unique_classes:
                        # density=True ensures classes with different sample sizes are comparable
                        ax.hist(
                            data[class_labels == c, i],
                            bins=50,
                            histtype="step",
                            label=f"Class {c}",
                            density=True,
                        )

                    if i == 0:  # Place a legend on the top-left plot
                        ax.legend(loc="best", frameon=False)

                if feature_labels is not None and i == n_dim - 1:
                    ax.set_xlabel(feature_labels[i])
                if feature_labels is not None and j == 0 and i != 0:
                    ax.set_ylabel(feature_labels[i])

            # -----------------------------------------------------
            # LOWER TRIANGLE: 2D Relationships
            # -----------------------------------------------------
            elif i > j:
                if class_labels is None:
                    # Fallback to the original hist2d if no labels are provided
                    ax.hist2d(data[:, j], data[:, i], bins=100, cmap="jet", norm=norm)
                else:
                    # Switch to scatter plot for visualizing categorical separation
                    for c in unique_classes:
                        ax.scatter(
                            data[class_labels == c, j],
                            data[class_labels == c, i],
                            s=5,
                            alpha=0.5,
                            label=f"Class {c}",
                        )

                if feature_labels is not None and i == n_dim - 1:
                    ax.set_xlabel(feature_labels[j])
                if feature_labels is not None and j == 0:
                    ax.set_ylabel(feature_labels[i])

            # -----------------------------------------------------
            # UPPER TRIANGLE: Empty
            # -----------------------------------------------------
            else:
                ax.axis("off")

            # Handle Tick Visibility
            if i < n_dim - 1:
                ax.xaxis.set_visible(False)
            if j > 0:
                ax.yaxis.set_visible(False)

    return fig, axes


def cornerplot_classes_contours(
    data, n_dim, feature_labels=None, class_labels=None, bins=50, figsize=None
):
    """
    Create a corner plot with contour densities for visual clarity.

    Parameters:
    data (numpy.ndarray): The data to be plotted, expected to be a 2D array.
    n_dim (int): The number of dimensions to plot.
    feature_labels (list of str, optional): Labels for each dimension. Used for the axes.
    class_labels (numpy.ndarray, optional): A 1D array of categorical labels matching the rows
                                            in `data`. If provided, plots contours by class.
    bins (int, optional): The resolution of the 2D grid used to calculate the contours.
    figsize (tuple, optional): Size of the figure.

    Returns:
    tuple: (figure, axes) - The matplotlib figure and axes objects
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    if data.shape[1] < n_dim:
        raise ValueError(
            f"Insufficient dimensions. Expected {n_dim}, got {data.shape[1]}."
        )
    if class_labels is not None and len(class_labels) != data.shape[0]:
        raise ValueError(
            "Length of class_labels must match the number of rows in data."
        )

    if figsize is None:
        figsize = (n_dim * 2, n_dim * 2)

    fig, axes = plt.subplots(n_dim, n_dim, figsize=figsize)
    plt.subplots_adjust(
        wspace=0.0, hspace=0.0, left=0.1, right=0.9, bottom=0.1, top=0.9
    )

    unique_classes = np.unique(class_labels) if class_labels is not None else [None]
    # Grab standard matplotlib colors to keep classes consistent across subplots
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]

            # -----------------------------------------------------
            # DIAGONAL: 1D Distributions
            # -----------------------------------------------------
            if i == j:
                if class_labels is None:
                    ax.hist(data[:, i], bins=bins, histtype="step", color="black")
                else:
                    for idx, c in enumerate(unique_classes):
                        color = default_colors[idx % len(default_colors)]
                        ax.hist(
                            data[class_labels == c, i],
                            bins=bins,
                            histtype="step",
                            color=color,
                            label=f"Class {c}",
                            density=True,
                            linewidth=1.5,
                        )

                    if i == 0:
                        ax.legend(loc="best", frameon=False)

                if feature_labels is not None and i == n_dim - 1:
                    ax.set_xlabel(feature_labels[i])
                if feature_labels is not None and j == 0 and i != 0:
                    ax.set_ylabel(feature_labels[i])

            # -----------------------------------------------------
            # LOWER TRIANGLE: 2D Contour Densities
            # -----------------------------------------------------
            elif i > j:
                if class_labels is None:
                    # Calculate 2D density grid
                    counts, xedges, yedges = np.histogram2d(
                        data[:, j], data[:, i], bins=bins
                    )
                    xcenters = (xedges[:-1] + xedges[1:]) / 2
                    ycenters = (yedges[:-1] + yedges[1:]) / 2
                    X, Y = np.meshgrid(xcenters, ycenters)

                    # Plot default contours
                    ax.contour(X, Y, counts.T, levels=6, cmap="viridis", alpha=0.8)
                else:
                    for idx, c in enumerate(unique_classes):
                        x_data = data[class_labels == c, j]
                        y_data = data[class_labels == c, i]
                        color = default_colors[idx % len(default_colors)]

                        # Calculate 2D density grid for this specific class
                        counts, xedges, yedges = np.histogram2d(
                            x_data, y_data, bins=bins
                        )
                        xcenters = (xedges[:-1] + xedges[1:]) / 2
                        ycenters = (yedges[:-1] + yedges[1:]) / 2
                        X, Y = np.meshgrid(xcenters, ycenters)

                        # Plot contours (using single color for the class)
                        ax.contour(X, Y, counts.T, levels=4, colors=[color], alpha=0.7)

                if feature_labels is not None and i == n_dim - 1:
                    ax.set_xlabel(feature_labels[j])
                if feature_labels is not None and j == 0:
                    ax.set_ylabel(feature_labels[i])

            # -----------------------------------------------------
            # UPPER TRIANGLE: Empty
            # -----------------------------------------------------
            else:
                ax.axis("off")

            # Handle Tick Visibility
            if i < n_dim - 1:
                ax.xaxis.set_visible(False)
            if j > 0:
                ax.yaxis.set_visible(False)

    return fig, axes
