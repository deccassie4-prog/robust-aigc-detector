# -*- coding: utf-8 -*-
"""Unit tests for predict.py / predict_gui.py pure functions: command building (multi-dir),
progress parsing (with batch sections), duplicate-name detection.

Headless: no Tk, no GPU, no subprocess. All temp files and sample images are created with
tempfile at runtime.
Also runs directly: `python tests/test_predict_units.py`.
"""
import json
import os
import sys
import tempfile

try:
    import pytest
except ImportError:  # pytest optional: direct-run via the __main__ entry at the bottom
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.predict import out_json_name, write_results
from scripts.predict_gui import (build_command, count_images, find_name_conflicts,
                                 find_out_conflicts, parse_line)


def test_parse_line_single_dir():
    """Parser: real run output fed line by line (single directory)."""
    state = {}
    lines = [
        "    60 images x 5 crops; batch_size=32 blocks/batch -> 6 imgs/batch, 10 batches; "
        "est. allocated VRAM ~7.6 GiB (+~3GB CUDA context/workspace)",
        "    VRAM peak 7.6 GiB (first batch)",
        "    [3/10 batches] 18/60 images, 12 s",
        "[done] 60 images -> out.json (1 failed as pred=-1), 18 s",
        "    P(fake) stats: min=0.0681 mean=0.6452 max=0.9385, >0.5 -> fake: 44",
    ]
    for l in lines:
        parse_line(l, state)
    assert state['total'] == 10 and state['batch'] == 3 and state['vram'] == '7.6'
    assert state['done_n'] == '60' and state['fake_n'] == '44'
    assert state['folders'] == [{'name': '', 'n': '60', 'fake': '44'}], state['folders']


def test_parse_line_batch_sections():
    """Batch section parsing: sections reset progress, accumulate folders, count failed folders."""
    state = {}
    lines = [
        "=== [1/3] real ===",
        "    66 images x 1 crop; 3 batches",
        "    [3/3 batches] 66/66 images, 30 s",
        "[done] 66 images -> real.json (0 failed as pred=-1), 30 s",
        "    P(fake) stats: ..., >0.5 -> fake: 2",
        "=== [2/3] diffusion ===",
        "    [1/2 batches] 40/46 images, 10 s",
        "[skip] no images found in: bad",
        "=== [3/3] real ===",
        "[done] 46 images -> parent__real.json (0 failed as pred=-1), 5 s",
        "    P(fake) stats: ..., >0.5 -> fake: 40",
        "[batch done] 2/3 folders ok, failed: [('bad', '...')]",
    ]
    for l in lines:
        parse_line(l, state)
    assert state['sec_n'] == 3 and state['sec_k'] == 3 and state['sec_name'] == 'real'
    assert state['folder_fails'] == 1
    assert [f['name'] for f in state['folders']] == ['real', 'real']
    assert state['folders'][0]['fake'] == '2' and state['folders'][1]['fake'] == '40'
    assert state['total'] == 0 and state['batch'] == 0  # section header reset batch progress


def test_build_command_multi_dir():
    """build_command: multi-dir expansion + EXTRA flags; the registry flag --json_format appears exactly once."""
    params = [
        {"flag": "--data_dir", "widget": "dir", "default": ""},
        {"flag": "--output_json", "widget": "savefile", "default": ""},
        {"flag": "--json_format", "widget": "choice", "default": "detailed",
         "choices": ["detailed", "competition"]},
        {"flag": "--batch_size", "widget": "int", "default": 32},
        {"flag": "--use_amp", "widget": "bool", "default": False},
    ]
    v = {"--data_dir": ["X:\\a\\real", "X:\\b\\diffusion"], "--output_json": "",
         "--json_format": "detailed", "--batch_size": "32", "--use_amp": True,
         "--output_dir": "X:\\out"}
    cmd = build_command(v, params)
    assert cmd.count("--data_dir") == 1, "argparse nargs='+' silently overrides repeated flags!"
    assert cmd.count("--json_format") == 1, "--json_format passed twice!"
    i = cmd.index("--data_dir")
    assert cmd[i + 1:i + 3] == ["X:\\a\\real", "X:\\b\\diffusion"]
    assert cmd[cmd.index("--output_dir") + 1] == "X:\\out"
    assert cmd[cmd.index("--json_format") + 1] == "detailed"
    assert "--output_json" not in cmd  # empty values are skipped in batch mode
    v1 = {"--data_dir": "X:\\one", "--output_json": "x.json", "--batch_size": "32",
          "--use_amp": False}
    cmd1 = build_command(v1, params)
    assert cmd1.count("--data_dir") == 1 and cmd1[cmd1.index("--data_dir") + 1] == "X:\\one"


def test_name_conflicts_and_out_json_name():
    """Duplicate-name detection and renaming (path assertions rely on Windows separator semantics)."""
    conf = find_name_conflicts(["X:\\a\\real", "X:\\b\\real", "X:\\c\\fake"])
    assert set(conf) == {'real'} and len(conf['real']) == 2
    assert out_json_name("X:\\a\\real", {'real'}) == 'a__real'
    assert out_json_name("X:\\a\\real", set()) == 'real'
    assert out_json_name("X:\\a\\real\\", {'x'}) == 'real'  # trailing slash normalization


def test_out_conflicts():
    """Cross-sub-batch output folder conflict detection: case-insensitive, first path per group."""
    if os.name != 'nt':
        # normcase case-folding is Windows-specific semantics
        if pytest is not None:
            pytest.skip('non-Windows')
        return
    conf = find_out_conflicts(["X:\\out\\res", "X:\\OUT\\RES", "X:\\other"])
    assert conf == ["X:\\out\\res"], conf
    assert find_out_conflicts(["X:\\a", "X:\\b"]) == []


def test_write_results_schemas():
    """write_results: detailed schema + competition compatibility (same shape as the equivalence gate)."""
    tmp = tempfile.mkdtemp(prefix='_units_')
    try:
        results = [{"image_path": "a.png", "pred": 0.934},
                   {"image_path": "b.png", "pred": 0.021},
                   {"image_path": "c.png", "pred": -1.0}]
        meta = {"format_version": 2, "source_dir": "X", "threshold": 0.5, "n_images": 3}
        p_det = os.path.join(tmp, 'detailed.json')
        p_comp = os.path.join(tmp, 'comp.json')
        write_results(p_det, results, meta, 'detailed')
        write_results(p_comp, results, meta, 'competition')
        d = json.load(open(p_det, encoding='utf-8'))
        assert set(d) == {'meta', 'images'} and d['meta']['threshold'] == 0.5
        assert d['images'][0]['is_fake'] is True and d['images'][1]['is_fake'] is False
        assert d['images'][2]['is_fake'] is None and d['images'][2]['pred'] == -1.0
        c = json.load(open(p_comp, encoding='utf-8'))
        assert isinstance(c, list) and c[0] == {"image_path": "a.png", "pred": 0.934}
        # competition output matches the equivalence-gate file shape (top-level list, image_path/pred fields)
        assert set(c[0]) == {'image_path', 'pred'}
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_count_images():
    """count_images regression: recursive counting by extension only. The original suite relied on a local
    test_images dir with 7 files; an equivalent directory is created here instead."""
    tmp = tempfile.mkdtemp(prefix='_units_img_')
    try:
        sub = os.path.join(tmp, 'sub')
        os.makedirs(sub)
        names = ['a.png', 'b.jpg', 'c.jpeg', 'd.bmp', 'e.webp',
                 os.path.join('sub', 'f.png'), os.path.join('sub', 'g.JPG')]
        for n in names:
            open(os.path.join(tmp, n), 'wb').close()
        open(os.path.join(tmp, 'ignored.txt'), 'wb').close()
        assert count_images(tmp) == 7
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    fns = [test_parse_line_single_dir, test_parse_line_batch_sections,
           test_build_command_multi_dir, test_name_conflicts_and_out_json_name,
           test_out_conflicts, test_write_results_schemas, test_count_images]
    for i, fn in enumerate(fns, 1):
        fn()
        print(f'{i} {fn.__name__} OK')
    print('\nAll unit tests passed')
