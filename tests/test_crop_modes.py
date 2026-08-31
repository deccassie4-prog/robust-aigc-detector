# -*- coding: utf-8 -*-
"""Crop-strategy tests: grid3x3 legacy-vs-new pixel equivalence, constant block counts,
start-position mapping, argparse compatibility.

Pure algorithm tests: no GPU, no GUI, all images generated programmatically.
Also runs directly: `python tests/test_crop_modes.py`.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.predict import CROP, N_CROPS, _crop_padded, get_args, make_crops


def _old_grid_starts(length, n=3):
    """Legacy grid3x3 starts: [0, cx, length-CROP] clamped value by value."""
    cx = int(round((length - CROP) / 2.0))
    return [min(max(s, 0), max(length - CROP, 0)) for s in [0, cx, length - CROP]]


def test_grid3x3_pixel_equivalence():
    """The generalized grid branch matches the legacy grid3x3 formula pixel-for-pixel across 7 sizes."""
    for w, h in [(100, 80), (224, 224), (225, 300), (512, 512),
                 (939, 512), (1024, 1024), (4000, 6000)]:
        img = Image.new('RGB', (w, h))
        new = make_crops(img, 'grid3x3')
        assert len(new) == 9
        old_crops = [_crop_padded(img, x, y)
                     for y in _old_grid_starts(h) for x in _old_grid_starts(w)]
        assert len(old_crops) == 9
        for a, b in zip(old_crops, new):
            assert np.array_equal(np.array(a), np.array(b)), (w, h)


def test_grid_block_counts():
    """grid4x4=16 / grid5x5=25 crops always (small images included via padding), all 224x224."""
    for w, h in [(1024, 1024), (100, 80)]:
        img = Image.new('RGB', (w, h))
        for m, k in [('grid4x4', 16), ('grid5x5', 25)]:
            cs = make_crops(img, m)
            assert len(cs) == k == N_CROPS[m], (m, w, h, len(cs))
            assert all(c.size == (224, 224) for c in cs)


def test_grid4x4_start_positions():
    """At width 1024 the grid4x4 step is (1024-224)/3 -> the marker pixel must land in crop #6 (row 1, col 1)."""
    img = Image.new('RGB', (1024, 1024), 'white')
    img.putpixel((267 + 5, 267 + 5), (255, 0, 0))  # inside the row-1/col-1 crop
    cs = make_crops(img, 'grid4x4')
    found = [i for i, c in enumerate(cs) if c.getpixel((5, 5)) == (255, 0, 0)]
    assert found == [5], found


def test_argparse_accepts_grid5x5():
    sys.argv = ['x', '--data_dir', 'x', '--crop_mode', 'grid5x5']
    a = get_args()
    assert a.crop_mode == 'grid5x5'


if __name__ == '__main__':
    test_grid3x3_pixel_equivalence()
    print('1 grid3x3 legacy/new pixel equivalence OK (7 sizes)')
    test_grid_block_counts()
    print('2 constant block counts OK (incl. small-image padding)')
    test_grid4x4_start_positions()
    print('3 grid4x4 start step/positioning OK')
    test_argparse_accepts_grid5x5()
    print('4 argparse accepts grid5x5 OK')
    print('\nAll crop tests passed')
