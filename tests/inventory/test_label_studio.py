"""Tests for data_prep.inventory.label_studio."""
import numpy as np
import pytest

from data_prep.inventory.issues import (AnnotationSelectionError,
                                        IssueCollector, IssueLevel,
                                        PolygonConversionError)
from data_prep.inventory.label_studio import (iter_polygon_results,
                                              polygon_to_pixels,
                                              select_annotation)


def _annotation(
    ann_id, updated_at, created_at, completed_by=1,
    was_cancelled=False, ground_truth=False, result=None,
):
    return {
        "id": ann_id,
        "completed_by": completed_by,
        "was_cancelled": was_cancelled,
        "ground_truth": ground_truth,
        "created_at": created_at,
        "updated_at": updated_at,
        "result": result if result is not None else [],
    }


class TestSelectAnnotation:
    def test_selects_by_updated_at(self):
        task = {
            "id": 1,
            "data": {"image": "img.jpg"},
            "annotations": [
                _annotation(10, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z"),
                _annotation(11, "2026-01-02T00:00:00Z",
                            "2026-01-01T00:00:00Z"),
            ],
        }
        collector = IssueCollector()
        selection = select_annotation(task, collector=collector)
        assert selection.mask_annotation_id == 11
        assert selection.n_annotations == 2

    def test_created_at_would_disagree_updated_at_wins(self):
        # created_at order picks id 10, but updated_at order (the rule)
        # must pick id 11.
        task = {
            "id": 2,
            "data": {"image": "img2.jpg"},
            "annotations": [
                _annotation(10, "2026-01-05T00:00:00Z",
                            "2026-01-02T00:00:00Z"),
                _annotation(11, "2026-01-06T00:00:00Z",
                            "2026-01-01T00:00:00Z"),
            ],
        }
        collector = IssueCollector()
        selection = select_annotation(task, collector=collector)
        assert selection.mask_annotation_id == 11

    def test_updated_at_tie_breaks_on_id(self):
        task = {
            "id": 3,
            "data": {"image": "img3.jpg"},
            "annotations": [
                _annotation(5, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z"),
                _annotation(9, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z"),
            ],
        }
        collector = IssueCollector()
        selection = select_annotation(task, collector=collector)
        assert selection.mask_annotation_id == 9

    def test_cancelled_annotation_filtered_out(self):
        task = {
            "id": 4,
            "data": {"image": "img4.jpg"},
            "annotations": [
                _annotation(1, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z"),
                _annotation(2, "2026-01-02T00:00:00Z",
                            "2026-01-01T00:00:00Z",
                            was_cancelled=True),
            ],
        }
        collector = IssueCollector()
        selection = select_annotation(task, collector=collector)
        assert selection.mask_annotation_id == 1
        assert selection.n_annotations == 1

    def test_no_candidates_raises(self):
        task = {"id": 5, "data": {"image": "img5.jpg"}, "annotations": []}
        collector = IssueCollector()
        with pytest.raises(AnnotationSelectionError):
            select_annotation(task, collector=collector)

    def test_all_cancelled_raises(self):
        task = {
            "id": 6,
            "data": {"image": "img6.jpg"},
            "annotations": [
                _annotation(1, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z", was_cancelled=True),
            ],
        }
        collector = IssueCollector()
        with pytest.raises(AnnotationSelectionError):
            select_annotation(task, collector=collector)

    def test_multiple_annotations_reported(self):
        task = {
            "id": 7,
            "data": {"image": "img7.jpg"},
            "annotations": [
                _annotation(1, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z"),
                _annotation(2, "2026-01-02T00:00:00Z",
                            "2026-01-01T00:00:00Z"),
            ],
        }
        collector = IssueCollector()
        select_annotation(task, collector=collector)
        codes = [i.code for i in collector.all()]
        assert "multiple_annotations" in codes

    def test_empty_annotation_reported(self):
        task = {
            "id": 8,
            "data": {"image": "img8.jpg"},
            "annotations": [
                _annotation(1, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z", result=[]),
            ],
        }
        collector = IssueCollector()
        select_annotation(task, collector=collector)
        codes = [i.code for i in collector.all()]
        assert "empty_annotation" in codes


class TestIterPolygonResults:
    def test_filters_non_polygon_types(self):
        annotation = {
            "result": [
                {"type": "polygonlabels", "id": "a"},
                {"type": "rectanglelabels", "id": "b"},
                {"type": "polygonlabels", "id": "c"},
            ]
        }
        collector = IssueCollector()
        results = list(
            iter_polygon_results(
                annotation, collector=collector, image_ref="img"
            )
        )
        assert [r["id"] for r in results] == ["a", "c"]
        codes = [i.code for i in collector.all()]
        assert codes == ["unexpected_result_type"]


class TestPolygonToPixels:
    def test_percent_to_pixel_conversion(self):
        result = {
            "original_width": 1280,
            "original_height": 960,
            "value": {"points": [[50.0, 50.0]]},
        }
        collector = IssueCollector()
        px = polygon_to_pixels(
            result, width_px=1280, height_px=960,
            collector=collector, image_ref="img",
        )
        assert np.allclose(px, [[640.0, 480.0]])
        assert collector.all() == []

    def test_out_of_frame_point_clipped(self):
        result = {
            "original_width": 1280,
            "original_height": 960,
            "value": {"points": [[50.0, 105.48]]},
        }
        collector = IssueCollector()
        px = polygon_to_pixels(
            result, width_px=1280, height_px=960,
            collector=collector, image_ref="img",
        )
        assert px[0, 1] == 959  # height_px - 1

    def test_negative_point_clipped(self):
        result = {
            "original_width": 1280,
            "original_height": 960,
            "value": {"points": [[-0.22, 50.0]]},
        }
        collector = IssueCollector()
        px = polygon_to_pixels(
            result, width_px=1280, height_px=960,
            collector=collector, image_ref="img",
        )
        assert px[0, 0] == 0

    def test_dims_mismatch_reported(self):
        result = {
            "original_width": 1280,
            "original_height": 959,
            "value": {"points": [[50.0, 50.0]]},
        }
        collector = IssueCollector()
        px = polygon_to_pixels(
            result, width_px=1280, height_px=960,
            collector=collector, image_ref="img_959",
        )
        # conversion uses original_height (959), not the file's 960
        assert np.isclose(px[0, 1], 50.0 / 100.0 * 959)
        issues = collector.all()
        assert len(issues) == 1
        assert issues[0].level == IssueLevel.WARNING
        assert issues[0].code == "ls_dims_mismatch"

    def test_missing_points_raises(self):
        result = {
            "original_width": 1280,
            "original_height": 960,
            "value": {},
        }
        collector = IssueCollector()
        with pytest.raises(PolygonConversionError):
            polygon_to_pixels(
                result, width_px=1280, height_px=960,
                collector=collector, image_ref="img",
            )

    def test_missing_original_dims_raises(self):
        result = {"value": {"points": [[1.0, 1.0]]}}
        collector = IssueCollector()
        with pytest.raises(PolygonConversionError):
            polygon_to_pixels(
                result, width_px=1280, height_px=960,
                collector=collector, image_ref="img",
            )
