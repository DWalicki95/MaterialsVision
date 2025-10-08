import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Sequence


def plot_loss(
    epochs: Sequence[int], losses: Sequence[float], label: str
) -> Figure:
    """
    Plot loss values over training epochs.

    Parameters
    ----------
    epochs : Sequence[int]
        Iterable containing epoch indices.
    losses : Sequence[float]
        Iterable of loss values corresponding to each epoch.
    label : str
        Label to display in the plot legend and title (e.g., 'Training' or
        'Validation').

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib Figure object containing the generated plot.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(range(1, epochs+1), losses, label=label)
    ax.legend()
    ax.grid()
    ax.set_title(f'{label} loss in {epochs} epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{label }Loss')
    return fig
