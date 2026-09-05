"""Tests for choosing the golden gallery of Phase 0."""
from pathlib import Path

import numpy as np
import pytest

from materials_vision.augmentation.config import (FAMILY_MASK_AWARE,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE)
from materials_vision.data.samples import PreparedSample, SampleRecord
from materials_vision.phase0.gallery import (ROLE_FALLBACK, ROLE_OUTLIER,
                                             GalleryError, GalleryImage,
                                             ImageAxes, _measure_one,
                                             assign_families, check_coverage,
                                             gallery_table, select_gallery)

QUOTAS = {("M1", "coarse"): 3, ("M2", "fine"): 2}


def make_axes(
    image_id: str,
    microscope: str = "M1",
    scale_bin: str = "coarse",
    *,
    walls: float = 5.0,
    contrast: float = 0.2,
    density: float = 4.0,
    diameter: float = 350.0,
) -> ImageAxes:
    """Build one candidate with the axes selection reads."""
    height_px = 960 if microscope == "M1" else 890
    return ImageAxes(
        image_id=image_id,
        formulation=image_id.split("_")[0],
        material="AS" if microscope == "M1" else "VAB",
        microscope=microscope,
        scale_bin=scale_bin,
        pixel_size_um=3.24 if scale_bin == "coarse" else 2.48,
        height_px=height_px,
        width_px=1280,
        n_instances=50,
        density_per_mm2=density,
        pore_diameter_median_um=diameter,
        wall_thickness_mean_px=walls,
        wall_thin_share=0.4,
        wall_contrast=contrast,
    )


def four_strata() -> list[ImageAxes]:
    """Candidates covering both microscopes and both scale bins."""
    candidates = []
    for index in range(4):
        candidates.append(make_axes(
            f"AS1_40_{index}", "M1", "coarse",
            walls=3.0 + index, contrast=0.30 - 0.1 * index,
            density=8.0 - index, diameter=200.0 + 50 * index,
        ))
        candidates.append(make_axes(
            f"VAB1_40_{index}", "M2", "fine",
            walls=4.0 + index, contrast=0.28 - 0.05 * index,
            density=7.0 - index, diameter=240.0 + 40 * index,
        ))
    return candidates


class TestStratifiedSelection:
    """Extremes are taken in a fixed order, hardest first."""

    def test_first_pick_of_a_stratum_is_the_thinnest_walled_image(
        self,
    ) -> None:
        gallery = select_gallery(
            four_strata(), quotas=QUOTAS, forced={},
            include_outlier=False,
        )
        first = next(
            entry for entry in gallery
            if entry.axes.stratum == ("M1", "coarse")
        )
        assert first.axes.image_id == "AS1_40_0"
        assert first.reason == "thinnest_walls"

    def test_quota_is_respected_per_stratum(self) -> None:
        gallery = select_gallery(
            four_strata(), quotas=QUOTAS, forced={},
            include_outlier=False,
        )
        picked = [entry.axes.stratum for entry in gallery]
        assert picked.count(("M1", "coarse")) == 3
        assert picked.count(("M2", "fine")) == 2

    def test_an_image_winning_two_extremes_is_picked_once(self) -> None:
        # The thinnest-walled image is also the densest, so the second
        # slot has to go to the runner-up rather than to it again.
        candidates = four_strata()
        gallery = select_gallery(
            candidates, quotas={("M1", "coarse"): 3}, forced={},
            include_outlier=False,
        )
        image_ids = [entry.axes.image_id for entry in gallery]
        assert len(image_ids) == len(set(image_ids))

    def test_selection_does_not_depend_on_candidate_order(self) -> None:
        forward = select_gallery(
            four_strata(), quotas=QUOTAS, forced={},
            include_outlier=False,
        )
        backward = select_gallery(
            list(reversed(four_strata())), quotas=QUOTAS, forced={},
            include_outlier=False,
        )
        assert [e.axes.image_id for e in forward] == [
            e.axes.image_id for e in backward
        ]

    def test_a_missing_measurement_never_wins_an_extreme(self) -> None:
        candidates = four_strata() + [make_axes(
            "AS1_40_9", "M1", "coarse", walls=float("nan"),
            contrast=float("nan"),
        )]
        gallery = select_gallery(
            candidates, quotas={("M1", "coarse"): 2}, forced={},
            include_outlier=False,
        )
        assert "AS1_40_9" not in {e.axes.image_id for e in gallery}

    def test_an_empty_stratum_is_an_error(self) -> None:
        with pytest.raises(GalleryError, match="no candidate image"):
            select_gallery(
                four_strata(), quotas={("M2", "coarse"): 1}, forced={},
                include_outlier=False,
            )


class TestForcedAndDiagnosticImages:
    """Images that must be in whatever the strata produce."""

    def test_a_forced_image_is_added_with_its_reason(self) -> None:
        gallery = select_gallery(
            four_strata(), quotas={("M1", "coarse"): 1},
            forced={"AS1_40_3": FAMILY_MASK_AWARE},
            include_outlier=False,
        )
        entry = next(
            e for e in gallery if e.axes.image_id == "AS1_40_3"
        )
        assert entry.role == ROLE_FALLBACK
        assert FAMILY_MASK_AWARE in entry.reason

    def test_a_forced_image_already_chosen_is_not_duplicated(
        self,
    ) -> None:
        gallery = select_gallery(
            four_strata(), quotas={("M1", "coarse"): 3},
            forced={"AS1_40_0": FAMILY_MASK_AWARE},
            include_outlier=False,
        )
        image_ids = [e.axes.image_id for e in gallery]
        assert image_ids.count("AS1_40_0") == 1
        assert image_ids[0] == "AS1_40_0"

    def test_a_forced_image_outside_the_candidates_is_an_error(
        self,
    ) -> None:
        with pytest.raises(GalleryError, match="not among the"):
            select_gallery(
                four_strata(), quotas=QUOTAS,
                forced={"AS9_40_1": FAMILY_MASK_AWARE},
                include_outlier=False,
            )

    def test_the_finest_close_up_is_added_as_a_diagnostic(self) -> None:
        candidates = four_strata() + [
            make_axes("VAB3_400_1", "M2", "outlier"),
            make_axes("AS4_500_1", "M1", "outlier"),
        ]
        candidates[-2] = ImageAxes(
            **{**vars(candidates[-2]), "pixel_size_um": 0.248}
        )
        candidates[-1] = ImageAxes(
            **{**vars(candidates[-1]), "pixel_size_um": 0.259}
        )
        gallery = select_gallery(
            candidates, quotas=QUOTAS, forced={}, include_outlier=True
        )
        outliers = [e for e in gallery if e.role == ROLE_OUTLIER]
        assert len(outliers) == 1
        assert outliers[0].axes.image_id == "VAB3_400_1"

    def test_close_ups_can_be_left_out(self) -> None:
        candidates = four_strata() + [
            make_axes("VAB3_400_1", "M2", "outlier")
        ]
        gallery = select_gallery(
            candidates, quotas=QUOTAS, forced={}, include_outlier=False
        )
        assert all(e.role != ROLE_OUTLIER for e in gallery)


class TestCoverage:
    """Conditions a reviewer cannot recover from afterwards."""

    def test_a_gallery_spanning_both_microscopes_passes(self) -> None:
        gallery = select_gallery(
            four_strata(), quotas=QUOTAS, forced={},
            include_outlier=False,
        )
        check_coverage(gallery)

    def test_one_microscope_only_is_refused(self) -> None:
        gallery = select_gallery(
            four_strata(), quotas={("M1", "coarse"): 3}, forced={},
            include_outlier=False,
        )
        with pytest.raises(GalleryError, match="microscope"):
            check_coverage(gallery)

    def test_one_scale_bin_only_is_refused(self) -> None:
        candidates = four_strata() + [
            make_axes("VAB1_30_1", "M2", "coarse")
        ]
        gallery = select_gallery(
            candidates,
            quotas={("M1", "coarse"): 2, ("M2", "coarse"): 1},
            forced={}, include_outlier=False,
        )
        with pytest.raises(GalleryError, match="scale bin"):
            check_coverage(gallery)


class TestFamilyAssignment:
    """Each family's subset, and what it has to keep spanning."""

    def build_gallery(self) -> tuple[GalleryImage, ...]:
        candidates = four_strata() + [
            make_axes("VAB3_400_1", "M2", "outlier")
        ]
        return select_gallery(
            candidates, quotas=QUOTAS, forced={}, include_outlier=True
        )

    def test_a_family_takes_the_number_of_images_it_asks_for(
        self,
    ) -> None:
        assignment = assign_families(
            self.build_gallery(), sizes={FAMILY_ORIENTATION: 4},
            forced={},
        )
        evaluated = [
            image_id
            for image_id in assignment[FAMILY_ORIENTATION]
            if not image_id.startswith("VAB3_400")
        ]
        assert len(evaluated) == 4

    def test_a_short_subset_still_spans_both_microscopes(self) -> None:
        assignment = assign_families(
            self.build_gallery(), sizes={FAMILY_ORIENTATION: 2},
            forced={},
        )
        picked = set(assignment[FAMILY_ORIENTATION])
        assert any(i.startswith("AS1_") for i in picked)
        assert any(i.startswith("VAB1_") for i in picked)

    def test_close_ups_are_appended_beyond_the_quota(self) -> None:
        assignment = assign_families(
            self.build_gallery(), sizes={FAMILY_ORIENTATION: 2},
            forced={},
        )
        picked = assignment[FAMILY_ORIENTATION]
        assert picked[-1] == "VAB3_400_1"
        assert len(picked) == 3

    def test_a_family_excluded_from_a_bin_never_sees_it(self) -> None:
        assignment = assign_families(
            self.build_gallery(), sizes={FAMILY_SCALE: 5}, forced={},
            excluded_bins={FAMILY_SCALE: frozenset({"fine",
                                                    "outlier"})},
        )
        assert all(
            image_id.startswith("AS1_")
            for image_id in assignment[FAMILY_SCALE]
        )

    def test_a_forced_image_leads_its_family_subset(self) -> None:
        assignment = assign_families(
            self.build_gallery(), sizes={FAMILY_MASK_AWARE: 2},
            forced={"VAB1_40_3": FAMILY_MASK_AWARE},
        )
        assert assignment[FAMILY_MASK_AWARE][0] == "VAB1_40_3"

    def test_a_family_missing_a_microscope_is_an_error(self) -> None:
        with pytest.raises(GalleryError, match="would be reviewed on"):
            assign_families(
                self.build_gallery(), sizes={FAMILY_ORIENTATION: 1},
                forced={},
            )


class TestGalleryTable:
    """The rows of the frozen artifact."""

    def test_every_family_reviewing_an_image_is_recorded_on_its_row(
        self,
    ) -> None:
        gallery = select_gallery(
            four_strata(), quotas=QUOTAS, forced={},
            include_outlier=False,
        )
        assignment = assign_families(
            gallery, sizes={FAMILY_ORIENTATION: 4}, forced={},
        )
        table = gallery_table(gallery, assignment)
        reviewed = table.set_index("image_id")["reviewed_by"]
        for image_id in assignment[FAMILY_ORIENTATION]:
            assert FAMILY_ORIENTATION in reviewed[image_id]

    def test_an_image_no_family_reviews_has_an_empty_cell(self) -> None:
        gallery = select_gallery(
            four_strata(), quotas=QUOTAS, forced={},
            include_outlier=False,
        )
        table = gallery_table(gallery, {})
        assert set(table["reviewed_by"]) == {""}


class TestMeasurement:
    """The axes are measured on the sample as training sees it."""

    def make_sample(self, labels: np.ndarray) -> PreparedSample:
        """Wrap a label image in the record the measurement reads."""
        image = np.where(labels > 0, 60, 200).astype(np.uint8)
        record = SampleRecord(
            index=0, image_id="AS1_40_1", formulation="AS1",
            material="AS", microscope="M1", scale_bin="coarse",
            pixel_size_um=2.0, q_max_i=1.0,
            source_path=Path("image.tif"), mask_path=Path("mask.tif"),
            crop_bbox=(0, 0, labels.shape[1], labels.shape[0]),
            n_instances_expected=int(labels.max()),
        )
        return PreparedSample(
            record=record, image=image, labels=labels,
            border_instance=np.zeros(int(labels.max()), dtype=bool),
            n_cut_by_crop=0, n_dropped_below_min_area=0,
            n_dropped_disconnected=0,
        )

    def two_pores(self) -> np.ndarray:
        """Two square pores separated by a two-pixel wall."""
        labels = np.zeros((40, 40), dtype=np.int32)
        labels[5:35, 5:19] = 1
        labels[5:35, 21:35] = 2
        return labels

    def test_density_is_reported_per_square_millimetre(self) -> None:
        axes = _measure_one(self.make_sample(self.two_pores()))
        area_mm2 = 40 * 40 * 2.0 ** 2 / 1e6
        assert axes.density_per_mm2 == pytest.approx(2 / area_mm2)

    def test_pore_diameter_is_the_equivalent_diameter_in_microns(
        self,
    ) -> None:
        axes = _measure_one(self.make_sample(self.two_pores()))
        expected = 2.0 * np.sqrt(30 * 14 / np.pi) * 2.0
        assert axes.pore_diameter_median_um == pytest.approx(
            expected, rel=1e-6
        )

    def test_the_wall_between_two_pores_is_measured(self) -> None:
        # The gap itself is two pixels wide, and most of the measured
        # ridge sits there. It runs on past the ends of the pores into
        # the frame's margin, where the two are still the nearest
        # instances but nothing separates them any more, so the mean
        # comes out a little above the nominal width.
        axes = _measure_one(self.make_sample(self.two_pores()))
        assert 2.0 <= axes.wall_thickness_mean_px <= 3.0
        assert axes.wall_thin_share > 0.5

    def test_an_image_with_one_pore_reports_no_wall(self) -> None:
        labels = np.zeros((40, 40), dtype=np.int32)
        labels[5:35, 5:35] = 1
        axes = _measure_one(self.make_sample(labels))
        assert np.isnan(axes.wall_thickness_mean_px)
        assert axes.n_instances == 1

    def test_walls_brighter_than_pores_give_a_positive_contrast(
        self,
    ) -> None:
        axes = _measure_one(self.make_sample(self.two_pores()))
        assert axes.wall_contrast > 0.0

    def test_the_record_travels_into_the_axes(self) -> None:
        axes = _measure_one(self.make_sample(self.two_pores()))
        assert axes.stratum == ("M1", "coarse")
        assert axes.formulation == "AS1"
        assert (axes.height_px, axes.width_px) == (40, 40)
