import numpy as np
import matplotlib

from bettercorners import cornerplot

matplotlib.use("Agg")  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pytest

# Disable showing plots during tests
plt.ioff()


@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Fixture to ensure no plots are shown during testing."""
    with plt.style.context("default"):
        yield
    plt.close("all")  # Close all figures after each test


@pytest.fixture
def sample_data():
    """Fixture to generate sample correlated data for testing."""
    np.random.seed(42)  # For reproducibility
    n_samples = 1000
    n_dim = 3
    mean = np.zeros(n_dim)
    cov = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 1.5]])
    return np.random.multivariate_normal(mean, cov, n_samples), n_dim


def test_basic_functionality(sample_data):
    """Test basic plot creation and return types."""
    data, n_dim = sample_data
    fig, axes = cornerplot(data, n_dim)

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (n_dim, n_dim)


def test_labels(sample_data):
    """Test proper handling and placement of axis labels."""
    data, n_dim = sample_data
    labels = ["X", "Y", "Z"]
    fig, axes = cornerplot(data, n_dim, labels=labels)

    # Check bottom row x-labels
    for j in range(n_dim):
        assert axes[n_dim - 1, j].get_xlabel() == labels[j]

    # Check first column y-labels
    for i in range(n_dim):
        assert axes[i, 0].get_ylabel() == labels[i]


def test_custom_figsize(sample_data):
    """Test custom figure size handling."""
    data, n_dim = sample_data
    custom_figsize = (8, 8)
    fig, axes = cornerplot(data, n_dim, figsize=custom_figsize)

    assert fig.get_size_inches().tolist() == list(custom_figsize)


def test_custom_norm(sample_data):
    """Test custom norm functionality."""
    data, n_dim = sample_data
    custom_norm = LogNorm(vmin=1e-3, vmax=1)
    fig, axes = cornerplot(data, n_dim, norm=custom_norm)

    # Check lower triangle plots have the custom norm
    for i in range(1, n_dim):
        for j in range(i):
            assert isinstance(axes[i, j].collections[0].norm, LogNorm)


def test_dimension_error(sample_data):
    """Test error handling for mismatched dimensions."""
    data, n_dim = sample_data
    with pytest.raises(IndexError):
        cornerplot(data, n_dim + 1)


def test_plot_structure(sample_data):
    """Test the structure of the corner plot."""
    data, n_dim = sample_data
    fig, axes = cornerplot(data, n_dim)

    for i in range(n_dim):
        for j in range(n_dim):
            if i < j:  # Upper triangle
                assert not axes[i, j].get_visible()
            elif i > j:  # Lower triangle
                assert len(axes[i, j].collections) > 0  # 2D histogram exists
            else:  # Diagonal
                assert len(axes[i, j].lines) > 0  # 1D histogram exists


def test_axis_visibility(sample_data):
    """Test proper axis visibility settings."""
    data, n_dim = sample_data
    fig, axes = cornerplot(data, n_dim)

    for i in range(n_dim):
        for j in range(n_dim):
            if i < n_dim - 1:  # All but bottom row
                assert not axes[i, j].xaxis.get_visible()
            if j > 0:  # All but first column
                assert not axes[i, j].yaxis.get_visible()
