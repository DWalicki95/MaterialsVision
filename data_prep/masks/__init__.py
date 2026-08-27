"""
Build step that turns Label Studio polygons into instance mask files.

Training reads masks as image files rather than re-deriving them from
the annotation export on every run: the derivation is deterministic,
so doing it once keeps run startup free and makes the masks something
you can open and look at.
"""
from data_prep.masks.rasterize import RasterizedMask, rasterize_instances

__all__ = ["RasterizedMask", "rasterize_instances"]
