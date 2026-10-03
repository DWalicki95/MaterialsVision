"""
Paging through a set of images in a notebook, one comparison at a time.

This is for looking, not for measuring. Every figure in a title is what
that one image contributes, computed with the same matching as the
reported results; the F1 of a whole set pools the counts across its
images first, so it is not the mean of the per-image values shown here.

**Each image is segmented once.** Running the model takes seconds,
while going back to an image already seen should be instant, so a
prediction is kept for as long as the browser lives. A test subset of
about a hundred images fits in memory easily.

**The widgets library is imported only when the controls are built.**
Predictions and figures of a browser work without it, in a script as
well as in a notebook.
"""
import logging
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

from materials_vision.data.samples import (PreparedSample, SampleRecord,
                                           SampleSource)
from materials_vision.evaluation.watershed import (DEPLOYED_POSTPROCESSING,
                                                   WatershedParams)
from materials_vision.inspection.figures import comparison_figure
from materials_vision.inspection.model import predict

logger = logging.getLogger(__name__)

ALL_MATERIALS = "all"


def image_choices(
    records: Sequence[SampleRecord], material: str = ALL_MATERIALS
) -> list[tuple[str, int]]:
    """Entries of the image list, optionally narrowed to one material.

    Parameters
    ----------
    records : sequence of SampleRecord
    material : str, optional
        ``ALL_MATERIALS`` keeps every image.

    Returns
    -------
    list of tuple
        ``(label, index into the source)`` in source order.
    """
    return [
        (
            f"{record.image_id}  [{record.material}, "
            f"{record.formulation}, {record.scale_bin}]",
            record.index,
        )
        for record in records
        if material == ALL_MATERIALS or record.material == material
    ]


class PredictionBrowser:
    """Step through a source, one comparison figure at a time.

    Parameters
    ----------
    source : SampleSource
    segmenter : InstanceSegmentationWithDecoder
    settings : WatershedParams, optional
        Defaults to the post-processing the deployed model runs with.
    seed : int, optional
        Fixes the pair colours.
    """

    def __init__(
        self,
        source: SampleSource,
        segmenter: Any,
        settings: WatershedParams = DEPLOYED_POSTPROCESSING,
        seed: int = 0,
    ):
        self._source = source
        self._segmenter = segmenter
        self._settings = settings
        self._seed = seed
        self._predictions: dict[int, tuple[PreparedSample, np.ndarray]] = {}

    def prediction(self, index: int) -> tuple[PreparedSample, np.ndarray]:
        """The sample at ``index`` and the model's segmentation of it.

        Parameters
        ----------
        index : int
            Position in the source.

        Returns
        -------
        tuple
            ``(sample, pred_labels)``, segmented on first request only.
        """
        if index not in self._predictions:
            sample = self._source.load(index)
            self._predictions[index] = (
                sample, predict(self._segmenter, sample, self._settings)
            )
        return self._predictions[index]

    def figure(self, index: int) -> Figure:
        """The comparison figure of one image.

        Parameters
        ----------
        index : int

        Returns
        -------
        matplotlib.figure.Figure
        """
        sample, pred_labels = self.prediction(index)
        return comparison_figure(sample, pred_labels, seed=self._seed)

    def widget(self) -> Any:
        """Build the controls: material filter, image list, stepping.

        Returns
        -------
        ipywidgets.VBox
            Display it in a notebook cell. The first image is drawn
            straight away.
        """
        import ipywidgets as widgets
        from IPython.display import display

        records = self._source.records
        material_box = widgets.Dropdown(
            options=[ALL_MATERIALS] + sorted({r.material for r in records}),
            description="material",
        )
        image_box = widgets.Dropdown(
            options=image_choices(records), description="image",
            layout=widgets.Layout(width="70%"),
        )
        previous_button = widgets.Button(description="previous")
        next_button = widgets.Button(description="next")
        output = widgets.Output()

        def render(index: Optional[int]) -> None:
            # An emptied list reports its value as None.
            if index is None:
                return
            with output:
                output.clear_output(wait=True)
                display(self.figure(index))

        def step(offset: int) -> None:
            if image_box.value is None:
                return
            indices = [index for _, index in image_box.options]
            position = indices.index(image_box.value) + offset
            image_box.value = indices[position % len(indices)]

        material_box.observe(
            lambda change: setattr(
                image_box, "options", image_choices(records, change.new)
            ),
            names="value",
        )
        image_box.observe(lambda change: render(change.new), names="value")
        previous_button.on_click(lambda _: step(-1))
        next_button.on_click(lambda _: step(+1))

        render(image_box.value)
        return widgets.VBox([
            widgets.HBox([material_box, previous_button, next_button]),
            image_box,
            output,
        ])
