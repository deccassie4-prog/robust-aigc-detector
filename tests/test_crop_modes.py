# -*- coding: utf-8 -*-
"""裁剪策略测试：grid3x3 新旧公式逐像素等价、块数恒定、起点定位、argparse 兼容。

纯算法测试：无 GPU、无 GUI、图像全部程序生成。可直接 `python tests/test_crop_modes.py` 运行。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.predict import CROP, N_CROPS, _crop_padded, get_args, make_crops


def _old_grid_starts(length, n=3):
    """旧版 grid3x3 起点：[0, cx, length-CROP] 逐值 clamp。"""
    cx = int(round((length - CROP) / 2.0))
    return [min(max(s, 0), max(length - CROP, 0)) for s in [0, cx, length - CROP]]


def test_grid3x3_pixel_equivalence():
    """新泛化 grid 分支与旧版 grid3x3 公式在 7 种尺寸下逐像素等价。"""
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
    """grid4x4=16 / grid5x5=25 块数恒定（含小图 padding），尺寸均为 224。"""
    for w, h in [(1024, 1024), (100, 80)]:
        img = Image.new('RGB', (w, h))
        for m, k in [('grid4x4', 16), ('grid5x5', 25)]:
            cs = make_crops(img, m)
            assert len(cs) == k == N_CROPS[m], (m, w, h, len(cs))
            assert all(c.size == (224, 224) for c in cs)


def test_grid4x4_start_positions():
    """1024 宽下 grid4x4 步长 (1024-224)/3 → 标记像素应落在第 6 块（行1列1）。"""
    img = Image.new('RGB', (1024, 1024), 'white')
    img.putpixel((267 + 5, 267 + 5), (255, 0, 0))  # 第二行第二列块内部
    cs = make_crops(img, 'grid4x4')
    found = [i for i, c in enumerate(cs) if c.getpixel((5, 5)) == (255, 0, 0)]
    assert found == [5], found


def test_argparse_accepts_grid5x5():
    sys.argv = ['x', '--data_dir', 'x', '--crop_mode', 'grid5x5']
    a = get_args()
    assert a.crop_mode == 'grid5x5'


if __name__ == '__main__':
    test_grid3x3_pixel_equivalence()
    print('1 grid3x3 新旧公式逐像素等价 OK（7 种尺寸）')
    test_grid_block_counts()
    print('2 grid4x4=16 / grid5x5=25 块数恒定 OK（含小图 padding）')
    test_grid4x4_start_positions()
    print('3 grid4x4 起点步长/定位 OK')
    test_argparse_accepts_grid5x5()
    print('4 argparse 接受 grid5x5 OK')
    print('\n全部裁剪测试通过')
