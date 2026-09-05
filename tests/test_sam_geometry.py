"""Tests for the geometry the model receives an image at."""
import pytest
import torch
from segment_anything.utils.transforms import ResizeLongestSide

from materials_vision.sam_geometry import (CONTENT_GEOMETRIES, PATCH_MARKER,
                                           SamGeometryError,
                                           expected_content_shape,
                                           isotropic_apply_image_torch,
                                           patch_resize_longest_side,
                                           resize_isotropic,
                                           resize_upstream_defect,
                                           verify_preprocess_geometry)


@pytest.fixture
def unpatched_library():
    """Run a test against the library as it ships, then restore.

    The correction is a replacement of a library method and therefore
    outlives the test that installed it. Every test here either
    installs it inside this fixture or asserts on functions that do not
    touch the library at all.
    """
    original = ResizeLongestSide.apply_image_torch
    ResizeLongestSide.apply_image_torch = _shipped_apply_image_torch
    yield
    ResizeLongestSide.apply_image_torch = original


def _shipped_apply_image_torch(self, image):
    """The library's own implementation, reproduced for the fixture."""
    target = ResizeLongestSide.get_preprocess_shape(
        image.shape[0], image.shape[1], self.target_length
    )
    return torch.nn.functional.interpolate(
        image, target, mode="bilinear", align_corners=False,
        antialias=True,
    )


class TestTheDefect:
    """What the uncorrected resize does, stated as running code."""

    def test_a_three_channel_image_is_squeezed_to_341_by_1024(
        self,
    ) -> None:
        out = resize_upstream_defect(torch.zeros((1, 3, 960, 1280)))
        assert tuple(out.shape[-2:]) == (341, 1024)

    def test_the_result_ignores_the_image_geometry(self) -> None:
        shapes = [(1, 3, 960, 1280), (1, 3, 890, 1280),
                  (1, 3, 1280, 960), (1, 3, 512, 512)]
        contents = {
            tuple(resize_upstream_defect(torch.zeros(s)).shape[-2:])
            for s in shapes
        }
        assert contents == {(341, 1024)}

    def test_the_result_changes_with_the_batch_size(self) -> None:
        # The clearest evidence that this is a defect: an image's
        # geometry cannot depend on how many images travel beside it.
        one = resize_upstream_defect(torch.zeros((1, 3, 960, 1280)))
        two = resize_upstream_defect(torch.zeros((2, 3, 960, 1280)))
        assert tuple(one.shape[-2:]) != tuple(two.shape[-2:])

    def test_a_single_channel_square_patch_escapes_it(self) -> None:
        # Which is why the defect goes unnoticed upstream: the loaders
        # there pad to square single-channel patches.
        out = resize_upstream_defect(torch.zeros((1, 1, 512, 512)))
        assert tuple(out.shape[-2:]) == (1024, 1024)


class TestTheCorrection:
    """The geometry the plan freezes, produced by the replacement."""

    @pytest.mark.parametrize(
        "geometry,expected",
        [
            ((960, 1280), (768, 1024)),
            ((890, 1280), (712, 1024)),
            ((1280, 960), (1024, 768)),
            ((1280, 890), (1024, 712)),
        ],
    )
    def test_the_longest_side_fills_the_canvas(
        self, geometry, expected
    ) -> None:
        out = resize_isotropic(torch.zeros((1, 3, *geometry)))
        assert tuple(out.shape[-2:]) == expected

    def test_proportions_are_preserved(self) -> None:
        out = resize_isotropic(torch.zeros((1, 3, 960, 1280)))
        height, width = out.shape[-2:]
        assert height / 960 == pytest.approx(width / 1280)

    def test_the_scale_is_the_08_the_plan_freezes(self) -> None:
        out = resize_isotropic(torch.zeros((1, 3, 960, 1280)))
        assert out.shape[-1] / 1280 == pytest.approx(0.8)

    def test_the_batch_size_does_not_reach_the_geometry(self) -> None:
        one = resize_isotropic(torch.zeros((1, 3, 960, 1280)))
        four = resize_isotropic(torch.zeros((4, 3, 960, 1280)))
        assert tuple(one.shape[-2:]) == tuple(four.shape[-2:])

    def test_a_quarter_turn_keeps_the_working_scale(self) -> None:
        # D4 has to remain a symmetry: the eight orientations are eight
        # views of one microstructure only if each is scaled the same.
        upright = resize_isotropic(torch.zeros((1, 3, 960, 1280)))
        turned = resize_isotropic(torch.zeros((1, 3, 1280, 960)))
        assert sorted(upright.shape[-2:]) == sorted(turned.shape[-2:])

    def test_expected_content_shape_matches_what_is_produced(
        self,
    ) -> None:
        for height_px, width_px in CONTENT_GEOMETRIES:
            out = resize_isotropic(
                torch.zeros((1, 3, height_px, width_px))
            )
            assert tuple(out.shape[-2:]) == expected_content_shape(
                height_px, width_px
            )


class TestInstallingIt:
    """Replacing a library method, and noticing that it took."""

    def test_the_library_is_corrected_after_patching(
        self, unpatched_library
    ) -> None:
        patch_resize_longest_side()
        transform = ResizeLongestSide(1024)
        out = transform.apply_image_torch(
            torch.zeros((1, 3, 960, 1280))
        )
        assert tuple(out.shape[-2:]) == (768, 1024)

    def test_patching_twice_changes_nothing(
        self, unpatched_library
    ) -> None:
        assert patch_resize_longest_side() is True
        assert patch_resize_longest_side() is False

    def test_a_library_that_no_longer_needs_it_is_left_alone(
        self, unpatched_library
    ) -> None:
        # If the defect is ever fixed upstream, this module has to
        # retire rather than layer a correction on a correct function.
        def fixed_upstream(self, image):
            target = ResizeLongestSide.get_preprocess_shape(
                image.shape[-2], image.shape[-1], self.target_length
            )
            return torch.nn.functional.interpolate(
                image, target, mode="bilinear", align_corners=False,
                antialias=True,
            )

        ResizeLongestSide.apply_image_torch = fixed_upstream
        assert patch_resize_longest_side() is False
        assert ResizeLongestSide.apply_image_torch is fixed_upstream

    def test_the_replacement_carries_its_marker(self) -> None:
        assert getattr(
            isotropic_apply_image_torch, PATCH_MARKER, False
        )

    def test_the_target_length_of_the_instance_is_honoured(
        self, unpatched_library
    ) -> None:
        patch_resize_longest_side()
        out = ResizeLongestSide(512).apply_image_torch(
            torch.zeros((1, 3, 960, 1280))
        )
        assert tuple(out.shape[-2:]) == (384, 512)


class TestVerification:
    """The check a training run makes before it trusts anything."""

    def test_a_corrected_library_passes(
        self, unpatched_library
    ) -> None:
        patch_resize_longest_side()
        measured = verify_preprocess_geometry()
        assert measured[(960, 1280)] == (768, 1024)
        assert measured[(890, 1280)] == (712, 1024)

    def test_an_uncorrected_library_is_refused(
        self, unpatched_library
    ) -> None:
        with pytest.raises(SamGeometryError, match="341x1024"):
            verify_preprocess_geometry()

    def test_the_message_names_every_geometry_that_is_wrong(
        self, unpatched_library
    ) -> None:
        with pytest.raises(SamGeometryError) as error:
            verify_preprocess_geometry([(960, 1280), (890, 1280)])
        assert "960x1280" in str(error.value)
        assert "890x1280" in str(error.value)
