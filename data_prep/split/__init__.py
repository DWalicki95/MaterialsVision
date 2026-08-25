"""Grouped, stratified TRAIN/VALIDATION/TEST split of the SEM dataset.

The split grouping unit is the formulation (never the image), so that
images from one synthesis can never straddle two sets: they are
strongly correlated, and dividing them would let the model be scored
on material it effectively trained on.
"""
