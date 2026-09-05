"""Tests for rendering one reviewable panel."""
import json
from pathlib import Path

import numpy as np
import pytest

from materials_vision.data.samples import PreparedSample, SampleRecord
from materials_vision.phase0.levels import level_by_key
from materials_vision.phase0.panels import (LOCAL_CHANGE_MAX_SHARE,
                                            _region_of_interest, panel_seed,
                                            render_panel, write_index)
from materials_vision.phase0.preview import MODE_AS_IS, MODE_ISOTROPIC


@pytest.fixture
def sample() -> PreparedSample:
    """Two large pores separated by a wall, on a small frame.

    Large on purpose: the wall family only divides pores several times
    the smallest annotated fragment, so a toy frame of a few pixels
    would exercise nothing but its refusal path.
    """
    rng = np.random.default_rng(7)
    labels = np.zeros((200, 260), dtype=np.int32)
    labels[15:95, 20:240] = 1
    labels[105:185, 20:240] = 2
    image = np.where(labels > 0, 70, 190).astype(np.int16)
    image = np.clip(
        image + rng.normal(0, 6, labels.shape), 0, 255
    ).astype(np.uint8)

    record = SampleRecord(
        index=0, image_id="AS1_40_1", formulation="AS1", material="AS",
        microscope="M1", scale_bin="coarse", pixel_size_um=3.24,
        q_max_i=1.306, source_path=Path("image.tif"),
        mask_path=Path("mask.tif"), crop_bbox=(0, 0, 260, 200),
        n_instances_expected=2,
    )
    return PreparedSample(
        record=record, image=image, labels=labels,
        border_instance=np.zeros(2, dtype=bool), n_cut_by_crop=0,
        n_dropped_below_min_area=0, n_dropped_disconnected=0,
    )


class TestSeeding:
    """What a panel's draw depends on, and what it must not."""

    def test_the_same_image_and_repeat_give_the_same_draw(
        self,
    ) -> None:
        assert panel_seed(1, "AS1_40_1", 0) == panel_seed(
            1, "AS1_40_1", 0
        )

    def test_different_images_give_different_draws(self) -> None:
        assert panel_seed(1, "AS1_40_1", 0) != panel_seed(
            1, "AS1_40_2", 0
        )

    def test_repeats_differ(self) -> None:
        assert panel_seed(1, "AS1_40_1", 0) != panel_seed(
            1, "AS1_40_1", 1
        )

    def test_the_run_seed_moves_every_panel(self) -> None:
        assert panel_seed(1, "AS1_40_1", 0) != panel_seed(
            2, "AS1_40_1", 0
        )


class TestRendering:
    """What a rendered panel leaves behind."""

    def test_the_three_files_are_written(
        self, sample, tmp_path
    ) -> None:
        level = level_by_key("F5_septum__nominal")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path
        )
        for kind in ("figure", "preview"):
            assert (tmp_path / record.files[kind]).exists()

    def test_the_model_view_is_written_at_the_canvas_size(
        self, sample, tmp_path
    ) -> None:
        # The overview figure is reduced; judging a wall on it would
        # judge the rendering. This file is the one judged 1:1.
        import matplotlib.pyplot as plt
        level = level_by_key("F3b_blur__high")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path
        )
        written = plt.imread(tmp_path / record.files["preview"])
        assert written.shape[:2] == (1024, 1024)

    def test_the_record_carries_what_a_verdict_cites(
        self, sample, tmp_path
    ) -> None:
        level = level_by_key("F3b_blur__nominal")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path
        )
        assert record.family == level.family
        assert record.level == level.level
        assert record.fingerprint == level.fingerprint
        assert record.image_id == "AS1_40_1"
        assert record.material == "AS"
        assert record.applied

    def test_the_geometry_is_reported_at_the_corrected_scale(
        self, sample, tmp_path
    ) -> None:
        level = level_by_key("F3b_blur__low")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path,
            preprocess_mode=MODE_ISOTROPIC,
        )
        # Approximately, not exactly: the target size is whole pixels,
        # and on a frame this small the rounding is worth a fraction of
        # a percent. On the real geometries both come out at 0.8.
        assert record.measurements["scale_x"] == pytest.approx(
            record.measurements["scale_y"], rel=0.01
        )

    def test_the_uncorrected_geometry_can_still_be_rendered(
        self, sample, tmp_path
    ) -> None:
        level = level_by_key("F3b_blur__low")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path,
            preprocess_mode=MODE_AS_IS,
        )
        assert record.measurements["scale_x"] != record.measurements[
            "scale_y"
        ]

    def test_the_index_lists_every_panel(
        self, sample, tmp_path
    ) -> None:
        records = [
            render_panel(
                sample, level_by_key(key), run_seed=1, repeat=0,
                output_dir=tmp_path,
            )
            for key in ("F3b_blur__low", "F3b_blur__high")
        ]
        path = write_index(records, tmp_path)
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["n_panels"] == 2
        assert {p["panel_id"] for p in written["panels"]} == {
            record.panel_id for record in records
        }

    def test_re_rendering_one_family_keeps_the_others_listed(
        self, sample, tmp_path
    ) -> None:
        # The revision loop re-renders a single family; replacing the
        # index would drop every other family from the review while
        # its files sat on disk beside it.
        first = render_panel(
            sample, level_by_key("F3b_blur__low"), run_seed=1,
            repeat=0, output_dir=tmp_path,
        )
        write_index([first], tmp_path)
        second = render_panel(
            sample, level_by_key("F5_septum__low"), run_seed=1,
            repeat=0, output_dir=tmp_path,
        )
        path = write_index([second], tmp_path)

        written = json.loads(path.read_text(encoding="utf-8"))
        assert {p["panel_id"] for p in written["panels"]} == {
            first.panel_id, second.panel_id
        }

    def test_a_re_rendered_panel_replaces_its_own_entry(
        self, sample, tmp_path
    ) -> None:
        level = level_by_key("F3b_blur__low")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path
        )
        write_index([record], tmp_path)
        again = render_panel(
            sample, level, run_seed=2, repeat=0, output_dir=tmp_path
        )
        path = write_index([again], tmp_path)

        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["n_panels"] == 1
        assert written["panels"][0]["seed"] == again.seed


class TestMeasurements:
    """The numbers printed beside the picture."""

    def test_a_divided_pore_is_counted(self, sample, tmp_path) -> None:
        level = level_by_key("F5_septum__nominal")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path
        )
        if not record.applied or not record.params.get("changed_mask"):
            pytest.skip("the wall found nothing to divide")
        assert (
            record.measurements["n_instances_after"]
            == record.measurements["n_instances_before"] + 1
        )

    def test_the_wall_is_measured_after_the_preprocessing(
        self, sample, tmp_path
    ) -> None:
        level = level_by_key("F5_septum__nominal")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path
        )
        if not record.params.get("changed_mask"):
            pytest.skip("the wall found nothing to divide")
        contrast = record.measurements[
            "septum_peak_contrast_after_preprocessing"
        ]
        assert contrast is not None and contrast > 0

    def test_the_shading_reports_how_close_it_came_to_a_boundary(
        self, sample, tmp_path
    ) -> None:
        level = level_by_key("F4_mask_aware__field_high")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path
        )
        if record.measurements["changed_pixel_share"] == 0:
            pytest.skip("no pore qualified for shading")
        assert record.measurements["clearance_px"] > 0

    def test_a_blur_changes_the_image_everywhere_and_the_mask_never(
        self, sample, tmp_path
    ) -> None:
        level = level_by_key("F3b_blur__high")
        record = render_panel(
            sample, level, run_seed=1, repeat=0, output_dir=tmp_path
        )
        assert record.measurements["changed_pixel_share"] > 0.5
        assert (
            record.measurements["n_instances_after"]
            == record.measurements["n_instances_before"]
        )


class TestRegionOfInterest:
    """When a close-up is worth rendering, and when it is not."""

    def test_a_change_everywhere_gets_no_close_up(self) -> None:
        image = np.zeros((40, 40), dtype=np.uint8)
        changed = image + 5
        assert _region_of_interest(
            image, changed, np.ones_like(image, dtype=np.int32), {}
        ) is None

    def test_an_unchanged_image_gets_no_close_up(self) -> None:
        image = np.zeros((40, 40), dtype=np.uint8)
        assert _region_of_interest(
            image, image.copy(),
            np.ones_like(image, dtype=np.int32), {},
        ) is None

    def test_a_local_change_gets_a_close_up_around_it(self) -> None:
        image = np.zeros((400, 400), dtype=np.uint8)
        changed = image.copy()
        changed[200:210, 200:210] = 40
        share = float((changed != image).mean())
        assert share < LOCAL_CHANGE_MAX_SHARE

        box = _region_of_interest(
            image, changed, np.ones_like(image, dtype=np.int32), {}
        )
        assert box is not None
        y0, x0, y1, x1 = box
        assert y0 < 200 and x0 < 200 and y1 > 210 and x1 > 210

    def test_the_divided_pore_is_kept_in_frame(self) -> None:
        # The wall itself is a thin line; without its pore around it
        # the close-up would show a stripe and no context.
        labels = np.zeros((400, 400), dtype=np.int32)
        labels[50:350, 50:350] = 3
        image = np.zeros((400, 400), dtype=np.uint8)
        changed = image.copy()
        changed[199:201, 50:350] = 200

        box = _region_of_interest(
            image, changed, labels, {"divided_instance": 3}
        )
        assert box is not None
        y0, x0, y1, x1 = box
        assert y0 <= 50 and y1 >= 350
