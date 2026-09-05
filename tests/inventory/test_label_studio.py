"""Tests for data_prep.inventory.label_studio."""
import numpy as np
import pytest

from data_prep.inventory.issues import (AnnotationSelectionError,
                                        IssueCollector, IssueLevel,
                                        PolygonConversionError,
                                        PolygonLabelError)
from data_prep.inventory.label_studio import (NODE_LABEL, PORE_LABEL,
                                              count_excluded_polygons,
                                              iter_polygon_results,
                                              polygon_label, polygon_to_pixels,
                                              select_annotation)


def _polygon(result_id, label=PORE_LABEL, points=None):
    value = {"points": points if points is not None else [[0.0, 0.0]]}
    if label is not None:
        value["polygonlabels"] = (
            label if isinstance(label, list) else [label]
        )
    return {"type": "polygonlabels", "id": result_id, "value": value}


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

    def test_avoid_annotators_falls_back_to_latest_non_avoided(self):
        task = {
            "id": 9,
            "data": {"image": "img9.jpg"},
            "annotations": [
                _annotation(1, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z", completed_by=2),
                _annotation(2, "2026-01-02T00:00:00Z",
                            "2026-01-01T00:00:00Z", completed_by=1),
            ],
        }
        collector = IssueCollector()
        selection = select_annotation(
            task, collector=collector, avoid_annotators=(1,),
        )
        assert selection.mask_annotation_id == 1
        assert selection.mask_annotator == 2
        assert selection.selection_rule == (
            "latest_updated_at_then_max_id+annotator_fallback"
        )
        codes = [i.code for i in collector.all()]
        assert "annotator_fallback_applied" in codes

    def test_avoid_annotators_no_alternative_keeps_latest(self):
        task = {
            "id": 10,
            "data": {"image": "img10.jpg"},
            "annotations": [
                _annotation(1, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z", completed_by=1),
                _annotation(2, "2026-01-02T00:00:00Z",
                            "2026-01-01T00:00:00Z", completed_by=1),
            ],
        }
        collector = IssueCollector()
        selection = select_annotation(
            task, collector=collector, avoid_annotators=(1,),
        )
        assert selection.mask_annotation_id == 2
        assert selection.selection_rule == (
            "latest_updated_at_then_max_id"
        )
        codes = [i.code for i in collector.all()]
        assert "annotator_fallback_unavailable" in codes

    def test_avoid_annotators_not_triggered_when_latest_ok(self):
        task = {
            "id": 11,
            "data": {"image": "img11.jpg"},
            "annotations": [
                _annotation(1, "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z", completed_by=1),
                _annotation(2, "2026-01-02T00:00:00Z",
                            "2026-01-01T00:00:00Z", completed_by=2),
            ],
        }
        collector = IssueCollector()
        selection = select_annotation(
            task, collector=collector, avoid_annotators=(1,),
        )
        assert selection.mask_annotation_id == 2
        assert selection.selection_rule == (
            "latest_updated_at_then_max_id"
        )
        codes = [i.code for i in collector.all()]
        assert "annotator_fallback_applied" not in codes
        assert "annotator_fallback_unavailable" not in codes


class TestPolygonLabel:
    def test_reads_the_single_class(self):
        assert polygon_label(_polygon("a"), "img") == PORE_LABEL

    @pytest.mark.parametrize("label", [None, [], [PORE_LABEL, NODE_LABEL]])
    def test_rejects_anything_but_one_class(self, label):
        with pytest.raises(PolygonLabelError):
            polygon_label(_polygon("a", label=label), "img")

    def test_rejects_an_unknown_class(self):
        with pytest.raises(PolygonLabelError):
            polygon_label(_polygon("a", label="Sciana"), "img")


class TestIterPolygonResults:
    def test_filters_non_polygon_types(self):
        annotation = {
            "result": [
                _polygon("a"),
                {"type": "rectanglelabels", "id": "b"},
                _polygon("c"),
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

    def test_drops_node_polygons(self):
        annotation = {
            "result": [
                _polygon("a"),
                _polygon("b", label=NODE_LABEL),
                _polygon("c"),
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
        assert codes == ["polygons_excluded_by_class"]

    def test_records_nothing_when_every_polygon_is_a_pore(self):
        annotation = {"result": [_polygon("a"), _polygon("b")]}
        collector = IssueCollector()
        list(
            iter_polygon_results(
                annotation, collector=collector, image_ref="img"
            )
        )
        assert collector.all() == []

    def test_an_all_node_annotation_yields_nothing(self):
        annotation = {"result": [_polygon("a", label=NODE_LABEL)]}
        collector = IssueCollector()
        assert list(
            iter_polygon_results(
                annotation, collector=collector, image_ref="img"
            )
        ) == []

    def test_the_exclusion_is_recorded_before_the_caller_iterates(self):
        annotation = {"result": [_polygon("a", label=NODE_LABEL)]}
        collector = IssueCollector()
        iter_polygon_results(
            annotation, collector=collector, image_ref="img"
        )
        codes = [i.code for i in collector.all()]
        assert codes == ["polygons_excluded_by_class"]

    def test_an_unreadable_class_stops_the_read(self):
        annotation = {"result": [_polygon("a", label="Sciana")]}
        collector = IssueCollector()
        with pytest.raises(PolygonLabelError):
            iter_polygon_results(
                annotation, collector=collector, image_ref="img"
            )


class TestCountExcludedPolygons:
    def test_counts_only_node_polygons(self):
        annotation = {
            "result": [
                _polygon("a"),
                _polygon("b", label=NODE_LABEL),
                _polygon("c", label=NODE_LABEL),
                {"type": "rectanglelabels", "id": "d"},
            ]
        }
        assert count_excluded_polygons(annotation) == 2

    def test_zero_without_nodes(self):
        assert count_excluded_polygons({"result": [_polygon("a")]}) == 0


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
