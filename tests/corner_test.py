import numpy as np
import matplotlib
from bettercorners.main import cornerplot

matplotlib.use("Agg")  # Set non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pytest

# Disable interactive plots during testing
plt.ioff()


@pytest.fixture
def sample_data():
    """Generate sample multivariate data."""
    np.random.seed(42)  # For reproducibility
    n_samples = 1000
    n_dim = 3
    mean = np.zeros(n_dim)
    cov = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 1.5]])
    return np.random.multivariate_normal(mean, cov, n_samples), n_dim


@pytest.mark.parametrize("n_dim, expected_shape", [(3, (3, 3)), (2, (2, 2))])
def test_return_types(sample_data, n_dim, expected_shape):
    """Test that the function returns a valid figure and axes."""
    data, _ = sample_data
    data = data[:, :n_dim]
    fig, axes = cornerplot(data, n_dim)

    assert isinstance(fig, plt.Figure), "cornerplot must return a Matplotlib figure."
    assert isinstance(axes, np.ndarray), "cornerplot must return axes as a NumPy array."
    assert (
        axes.shape == expected_shape
    ), f"Expected axes shape {expected_shape}, got {axes.shape}."


def test_labels(sample_data):
    """Test that axis labels are applied correctly."""
    data, n_dim = sample_data
    labels = ["X", "Y", "Z"]
    fig, axes = cornerplot(data, n_dim, labels=labels)

    # Check axis labels
    for i in range(n_dim):
        for j in range(n_dim):
            if i == n_dim - 1:  # Bottom row x-labels
                assert (
                    axes[i, j].get_xlabel() == labels[j]
                ), f"Expected xlabel {labels[j]}"
            if j == 0:  # First column y-labels
                assert (
                    axes[i, j].get_ylabel() == labels[i]
                ), f"Expected ylabel {labels[i]}"


def test_custom_figsize(sample_data):
    """Test that the figure size is applied correctly."""
    data, n_dim = sample_data
    figsize = (10, 10)
    fig, _ = cornerplot(data, n_dim, figsize=figsize)
    assert fig.get_size_inches().tolist() == list(
        figsize
    ), f"Expected figsize {figsize}"


def test_custom_norm(sample_data):
    """Test that custom norm is applied to 2D histograms."""
    data, n_dim = sample_data
    norm = LogNorm(vmin=1e-3, vmax=1)
    fig, axes = cornerplot(data, n_dim, norm=norm)

    # Check 2D histograms in the lower triangle
    for i in range(1, n_dim):
        for j in range(i):
            hist = axes[i, j].collections
            assert len(hist) > 0, "Expected 2D histogram in lower triangle."
            assert isinstance(
                hist[0].norm, LogNorm
            ), "Expected custom norm to be applied."


def test_empty_labels(sample_data):
    """Test behavior with empty labels."""
    data, n_dim = sample_data
    labels = [""] * n_dim
    fig, axes = cornerplot(data, n_dim, labels=labels)

    for i in range(n_dim):
        for j in range(n_dim):
            if i == n_dim - 1:
                assert axes[i, j].get_xlabel() == "", "Expected empty xlabel."
            if j == 0:
                assert axes[i, j].get_ylabel() == "", "Expected empty ylabel."


def test_axis_visibility(sample_data):
    """Test the visibility of axes ticks and labels."""
    data, n_dim = sample_data
    fig, axes = cornerplot(data, n_dim)

    for i in range(n_dim):
        for j in range(n_dim):
            if i < n_dim - 1:  # Not bottom row
                assert not axes[
                    i, j
                ].xaxis.get_visible(), "x-axis should not be visible."
            if j > 0:  # Not first column
                assert not axes[
                    i, j
                ].yaxis.get_visible(), "y-axis should not be visible."
