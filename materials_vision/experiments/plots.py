"""
Updated plot_loss function to handle filtered test losses.
Add this to your materials_vision/experiments/plots.py file.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(epochs, losses, label, x_label='Epoch', show_markers=True):
    """
    Plot training or test losses over epochs.

    Parameters
    ----------
    epochs : int or array-like
        If int: number of epochs (will create range(epochs))
        If array-like: specific epoch numbers where losses were evaluated
    losses : array-like
        Loss values corresponding to epochs
    label : str
        Label for the plot (e.g., 'Train', 'Test')
    x_label : str
        Label for x-axis (default: 'Epoch')
    show_markers : bool
        Whether to show markers on the plot (useful for sparse data)

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(epochs, int):
        # Old behavior: epochs is just a count
        x = range(epochs)
        ax.plot(x, losses, label=label, linewidth=2, color='#1f77b4')
    else:
        # New behavior: epochs is an array of specific x-values
        x = np.array(epochs)

        # For sparse data (like test losses), show markers
        if show_markers or len(x) < 50:
            ax.plot(x, losses, label=label, linewidth=2,
                    marker='o', markersize=6, markerfacecolor='white',
                    markeredgewidth=2, color='#1f77b4')
        else:
            ax.plot(x, losses, label=label, linewidth=2, color='#1f77b4')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(
        f'{label} loss in {len(losses)} {"evaluations" if isinstance(
            epochs, (list, np.ndarray)) else "epochs"}',
        fontsize=14
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set sensible axis limits
    ax.set_xlim(left=-0.5)
    if len(losses) > 0:
        y_min, y_max = min(losses), max(losses)
        y_range = y_max - y_min
        ax.set_ylim(bottom=max(0, y_min - 0.1 * y_range),
                    top=y_max + 0.1 * y_range)

    plt.tight_layout()

    return fig
