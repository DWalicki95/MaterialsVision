"""Tests for the strength settings each family is reviewed at."""
import pytest

from materials_vision.augmentation.config import (FAMILY_BLUR,
                                                  FAMILY_MASK_AWARE,
                                                  FAMILY_ORIENTATION,
                                                  FAMILY_SCALE, FAMILY_SEPTUM,
                                                  FAMILY_TONAL)
from materials_vision.phase0.levels import (KIND_DIAGNOSTIC, KIND_GATE,
                                            ReviewLevel, level_by_key,
                                            levels_for, review_levels)


def gate_levels(family: str) -> list[ReviewLevel]:
    """Settings of one family that decide its admission."""
    return [
        level for level in levels_for(family)
        if level.kind == KIND_GATE
    ]


class TestTheSetOfLevels:
    """Properties every setting has to have."""

    def test_every_family_is_reviewed(self) -> None:
        families = {level.family for level in review_levels()}
        assert families == {
            FAMILY_ORIENTATION, FAMILY_SCALE, FAMILY_TONAL,
            FAMILY_BLUR, FAMILY_MASK_AWARE, FAMILY_SEPTUM,
        }

    def test_keys_are_unique(self) -> None:
        keys = [level.key for level in review_levels()]
        assert len(keys) == len(set(keys))

    def test_a_level_enables_its_family_and_nothing_else(self) -> None:
        # A panel isolates one family: a promise like "the mask is
        # untouched" says nothing once another family changed it.
        for level in review_levels():
            assert level.config.families == (level.family,)

    def test_every_level_fires(self) -> None:
        # Otherwise a panel could show the identity and be reviewed as
        # though it showed the transformation.
        for level in review_levels():
            config = getattr(
                level.config,
                {
                    FAMILY_ORIENTATION: "orientation",
                    FAMILY_SCALE: "scale",
                    FAMILY_TONAL: "tonal",
                    FAMILY_BLUR: "blur",
                    FAMILY_MASK_AWARE: "mask_aware",
                    FAMILY_SEPTUM: "septum",
                }[level.family],
            )
            assert config.p == 1.0

    def test_each_family_has_a_weak_and_a_strong_gate(self) -> None:
        for family in (FAMILY_SCALE, FAMILY_BLUR, FAMILY_SEPTUM):
            names = {level.level for level in gate_levels(family)}
            assert {"low", "nominal", "high"} <= names

    def test_lookup_by_key(self) -> None:
        level = level_by_key("F3b_blur__high")
        assert level is not None
        assert level.family == FAMILY_BLUR
        assert level_by_key("F3b_blur__nonsense") is None


class TestTheNumbers:
    """The settings the plan names, pinned rather than drawn."""

    def test_the_crop_is_shown_at_the_three_named_magnifications(
        self,
    ) -> None:
        drawn = {
            level.level: level.config.scale.bands[0][1]
            for level in gate_levels(FAMILY_SCALE)
        }
        assert drawn == {"low": 1.00, "nominal": 1.15, "high": 1.30}

    def test_each_crop_level_draws_one_magnification_only(
        self,
    ) -> None:
        for level in levels_for(FAMILY_SCALE):
            (weight, low, high), = level.config.scale.bands
            assert (weight, low) == (1.0, high)

    def test_the_blur_widths_bracket_the_frozen_range(self) -> None:
        drawn = {
            level.level: level.config.blur.sigma_px
            for level in gate_levels(FAMILY_BLUR)
        }
        assert drawn["low"] == (0.2, 0.2)
        assert drawn["high"] == (0.8, 0.8)

    def test_the_wall_widths_bracket_the_measured_range(self) -> None:
        drawn = {
            level.level: level.config.septum.thickness_px
            for level in gate_levels(FAMILY_SEPTUM)
        }
        assert drawn["low"] == (2.0, 2.0)
        assert drawn["high"] == (4.0, 4.0)

    def test_the_faint_wall_uses_the_lowest_measured_contrast(
        self,
    ) -> None:
        faint = level_by_key("F5_septum__faint")
        assert faint is not None
        assert faint.kind == KIND_DIAGNOSTIC
        assert faint.config.septum.contrast < 0.2034

    def test_the_stressed_patch_lies_outside_the_frozen_range(
        self,
    ) -> None:
        stress = level_by_key("F4_mask_aware__patch_stress")
        assert stress is not None
        assert stress.kind == KIND_DIAGNOSTIC
        assert stress.config.mask_aware.darkening_factor[0] < 0.60

    def test_diagnostics_never_gate_a_family(self) -> None:
        for level in review_levels():
            if level.kind == KIND_DIAGNOSTIC:
                assert level.level not in {"low", "nominal", "high"}


class TestPinnedMembers:
    """A container of alternatives shows one of them at a time."""

    def test_each_tonal_level_pins_one_member(self) -> None:
        for level in levels_for(FAMILY_TONAL):
            assert len(level.config.tonal.members) == 1

    def test_both_tonal_members_are_reviewed(self) -> None:
        members = {
            level.config.tonal.members[0]
            for level in levels_for(FAMILY_TONAL)
        }
        assert members == {"brightness_contrast", "gamma"}

    def test_each_mask_aware_level_pins_one_member(self) -> None:
        for level in levels_for(FAMILY_MASK_AWARE):
            assert len(level.config.mask_aware.members) == 1

    def test_the_two_members_split_the_images_between_them(
        self,
    ) -> None:
        # Pinning a member and keeping every image would double an
        # already long review without covering anything new.
        images = tuple(f"image_{i}" for i in range(8))
        field = level_by_key("F4_mask_aware__field_low")
        patch = level_by_key("F4_mask_aware__patch_low")
        assert field is not None and patch is not None
        assert not set(field.images(images)) & set(patch.images(images))
        assert set(field.images(images)) | set(
            patch.images(images)
        ) == set(images)


class TestFingerprints:
    """What a verdict is attached to."""

    def test_the_same_level_hashes_the_same_way(self) -> None:
        assert (
            level_by_key("F3b_blur__low").fingerprint
            == level_by_key("F3b_blur__low").fingerprint
        )

    def test_different_settings_hash_differently(self) -> None:
        prints = {
            level.key: level.fingerprint for level in review_levels()
        }
        assert len(set(prints.values())) == len(prints)

    def test_changing_a_parameter_changes_the_fingerprint(
        self,
    ) -> None:
        # This is the whole mechanism: widen a range, and the verdict
        # made against the old numbers stops applying.
        original = level_by_key("F3b_blur__high")
        assert original is not None
        widened = ReviewLevel(
            family=original.family,
            level=original.level,
            kind=original.kind,
            config=original.config.__class__(
                blur=original.config.blur.__class__(
                    sigma_px=(0.8, 1.2), p=1.0
                )
            ),
            note=original.note,
        )
        assert widened.fingerprint != original.fingerprint

    def test_the_fingerprint_covers_every_parameter(self) -> None:
        level = level_by_key("F5_septum__nominal")
        assert level is not None
        assert level.parameters["thickness_px"] == (3.0, 3.0)
        assert "contrast" in level.parameters


class TestImageShares:
    """Which of a family's images a setting uses."""

    def test_the_default_share_is_all_of_them(self) -> None:
        images = tuple(f"image_{i}" for i in range(5))
        level = level_by_key("F5_septum__low")
        assert level is not None
        assert level.images(images) == images

    def test_a_diagnostic_takes_a_quarter(self) -> None:
        images = tuple(f"image_{i}" for i in range(16))
        level = level_by_key("F5_septum__faint")
        assert level is not None
        assert len(level.images(images)) == 4

    def test_the_orientation_family_is_drawn_twice_per_image(
        self,
    ) -> None:
        level = level_by_key("F1_orientation__nominal")
        assert level is not None
        assert level.repeats == 2

    @pytest.mark.parametrize("family", [FAMILY_SCALE, FAMILY_BLUR])
    def test_a_single_member_family_draws_once_per_image(
        self, family
    ) -> None:
        for level in levels_for(family):
            assert level.repeats == 1
