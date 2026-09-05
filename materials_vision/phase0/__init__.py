"""
Phase 0: judging every augmentation by eye before any training.

No augmentation family enters the experiment on the strength of its
unit tests. Tests can show that a mask stayed exact and that no
instance came apart; they cannot show that the image is still a
photograph of foam, that its pores could still be annotated by hand, or
that the wall between two of them survived being drawn at 0.8 of its
size. Those are the questions the whole comparison rests on, and they
are answered by a person looking at the result.

This package prepares that judgement and records it:

* ``gallery`` chooses the fixed set of training images every family is
  judged on, by rule rather than by hand, so the choice is auditable
  and repeatable;
* ``levels`` turns each family into a weak, a nominal and a strong
  setting, plus the deliberately punishing variants used to find where
  recognizability breaks;
* ``preview`` reproduces the downscaling the model performs inside
  itself, which is the resolution the thin-wall question has to be
  answered at;
* ``panels`` renders one reviewable figure per image, family and
  level, with the drawn parameters beside it;
* ``review`` collects the verdicts and reduces them to one status per
  family and level.

A verdict is tied to the hash of the parameters it was given, never to
the name of the image it was seen on. Widen a range afterwards and the
old verdict does not silently carry over to the new numbers - the panel
returns to the queue as undecided.
"""
from materials_vision.phase0.gallery import (FAMILY_EXCLUDED_BINS,
                                             FAMILY_SIZES, FORCED_IMAGES,
                                             GALLERY_RULES_VERSION,
                                             SELECTION_REASONS, STRATUM_QUOTAS,
                                             GalleryError, GalleryImage,
                                             ImageAxes, assign_families,
                                             check_coverage, gallery_table,
                                             measure_axes, select_gallery)

__all__ = [
    "FAMILY_EXCLUDED_BINS",
    "FAMILY_SIZES",
    "FORCED_IMAGES",
    "GALLERY_RULES_VERSION",
    "SELECTION_REASONS",
    "STRATUM_QUOTAS",
    "GalleryError",
    "GalleryImage",
    "ImageAxes",
    "assign_families",
    "check_coverage",
    "gallery_table",
    "measure_axes",
    "select_gallery",
]
