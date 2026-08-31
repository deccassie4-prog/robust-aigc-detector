# -*- coding: utf-8 -*-
"""Headless regression tests for pred_filter_gui: no GPU, idempotent (repeatable).

Coverage: dual-format loading with invalid entries / closed interval and -1 exclusion /
pred file-name formatting / collision suffixes (case-insensitive) / name index resolution
(unique, multiple, miss) / output folder name prefixes / tab title dedup / real small-file
copy end-to-end (inline worker + start_run full flow) / config persistence.

Stale paths in the JSONs are faked with forward slashes (G:/stale/...) so basename
resolution assertions behave identically on Windows and POSIX.
Also runs directly: `python tests/test_pred_filter_gui.py`.
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import types
import tkinter as tk

try:
    import pytest
except ImportError:  # pytest optional: direct-run via the __main__ entry at the bottom
    pytest = types.SimpleNamespace(
        fixture=lambda *a, **k: (lambda f: f),
        skip=lambda msg='': (_ for _ in ()).throw(RuntimeError(msg)),
    )

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import pred_filter_gui as pf


# ---------- fixtures ----------

@pytest.fixture(scope='module')
def workdir():
    tmp = tempfile.mkdtemp(prefix='_pf_test_')
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope='module')
def config_guard():
    """Back up / restore pred_filter_config.json so the test is idempotent and never pollutes the real config."""
    bak = open(pf.CONFIG_PATH, 'rb').read() if os.path.exists(pf.CONFIG_PATH) else None
    yield
    for extra in (pf.CONFIG_PATH + '.bak', pf.CONFIG_PATH + '.tmp'):
        if os.path.exists(extra):
            os.remove(extra)
    if bak is None:
        if os.path.exists(pf.CONFIG_PATH):
            os.remove(pf.CONFIG_PATH)
    else:
        with open(pf.CONFIG_PATH, 'wb') as f:
            f.write(bak)


def _build_env(tmp):
    """Two predict output JSONs + a small recursive dataset."""
    comp = [{'image_path': f'X:/a/{i}.png', 'pred': v}
            for i, v in enumerate([0.9, 0.5, -1.0, 0.951053, 0.951053])]
    det = {'meta': {'x': 1}, 'images': [
        {'image_path': 'X:/b/z.jpg', 'pred': 0.75},
        {'image_path': 'X:/b/w.jpg', 'pred': 0.1},
        {'image_path': 'X:/b/bad1.jpg'},
        {'image_path': 'X:/b/bad2.jpg', 'pred': 'oops'}]}
    p1 = os.path.join(tmp, 'beach.json')
    with open(p1, 'w') as f:
        json.dump(comp, f)
    p2 = os.path.join(tmp, 'det.json')
    with open(p2, 'w') as f:
        json.dump(det, f)
    src = os.path.join(tmp, 'dataset')
    for d in ('beach', 'other', 'dup1', 'dup2'):
        os.makedirs(os.path.join(src, d))
    for rel in (('beach', 'x1.png'), ('beach', 'x2.png'), ('other', 'x3.jpg'),
                ('other', 'x4.jpg'), ('other', 'x5.png'),
                ('dup1', 'same.png'), ('dup2', 'same.png')):
        with open(os.path.join(src, *rel), 'wb') as f:
            f.write(b'data-' + '/'.join(rel).encode())
    return {'tmp': tmp, 'p1': p1, 'p2': p2, 'src': src}


@pytest.fixture(scope='module')
def env(workdir):
    return _build_env(workdir)


@pytest.fixture(scope='module')
def gui_app(config_guard):
    """A withdrawn pred_filter App shared by groups 7-10 (same state order as the original regression)."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('tkinter has no usable display')
    root.withdraw()
    app = pf.App(root)
    root.update()
    yield root, app
    root.destroy()


def _pump(root, secs=0.5):
    end = time.time() + secs
    while time.time() < end:
        root.update()
        time.sleep(0.02)


# ---------- groups 1-6: pure functions ----------

def test_load_entries_both_formats(env):
    e1, f1, b1 = pf.load_entries(env['p1'])
    assert f1 == 'competition' and len(e1) == 5 and b1 == 0, (f1, len(e1), b1)
    e2, f2, b2 = pf.load_entries(env['p2'])
    assert f2 == 'detailed' and len(e2) == 2 and b2 == 2, (f2, len(e2), b2)


def test_in_range_closed_interval():
    assert pf.in_range(0.5, 0, 1) and pf.in_range(0, 0, 1) and pf.in_range(1, 0, 1)
    assert not pf.in_range(-1.0, 0, 1) and not pf.in_range(1.1, 0, 1)
    assert pf.in_range(0.5, None, 0.6) and pf.in_range(0.7, 0.6, None)
    assert pf.in_range(-1.0, -1, 1)  # explicitly relaxing the lower bound admits failed items


def test_pred_label_formatting():
    assert pf.pred_label(0.951053) == '0.951053'
    assert pf.pred_label(0.5) == '0.5'
    assert pf.pred_label(1.0) == '1' and pf.pred_label(0.0) == '0'
    assert pf.pred_label(0.000051) == '0.000051'  # no scientific notation


def test_unique_name_collision_suffix():
    taken = set()
    assert pf.unique_name('0.5', '.png', taken) == '0.5.png'
    assert pf.unique_name('0.5', '.PNG', taken) == '0.5_2.PNG'
    assert pf.unique_name('0.5', '.png', taken) == '0.5_3.png'


def test_build_index_and_resolve(env):
    src = env['src']
    idx = pf.build_index(src)
    r, n = pf.resolve_by_name('G:/stale/beach/x1.png', idx)
    assert n == 1 and r.endswith('x1.png'), (r, n)
    r, n = pf.resolve_by_name('X:/any/same.png', idx)  # multiple hits -> first in sort order
    assert n == 2 and 'dup1' in r, (r, n)
    r, n = pf.resolve_by_name('X:/any/nope.png', idx)
    assert r is None and n == 0


def test_dedup_folder_names():
    pa, pb, pc = 'D:/x/FAKE_JPEG/beach.json', 'D:/x/FAKE_COLOR/beach.json', 'D:/y/REAL.json'
    names = pf.dedup_folder_names([pa, pb, pc])
    assert names[pa] == 'FAKE_JPEG__beach', names
    assert names[pb] == 'FAKE_COLOR__beach', names
    assert names[pc] == 'REAL', names


# ---------- groups 7-10: GUI full flow (shared state built by the ordered tests above) ----------

def test_inline_copy_end_to_end(env, gui_app):
    root, app = gui_app
    app.gmin_var.set('0.4')
    app.gmax_var.set('1.0')

    # the JSON records stale paths (G:/stale/...); basenames match the env dataset -> all must be found
    pick = [{'image_path': f'G:/stale/beach/{n}', 'pred': v}
            for n, v in [('x1.png', 0.9), ('x2.png', 0.5), ('x0.png', -1.0),
                         ('x3.jpg', 0.951053), ('x4.jpg', 0.951053)]]
    p1b = os.path.join(env['tmp'], 'pick', 'beach.json')
    os.makedirs(os.path.dirname(p1b), exist_ok=True)
    with open(p1b, 'w') as f:
        json.dump(pick, f)
    app.add_json_tabs([p1b])
    tab = app._tabs[0]
    assert app.notebook.tab(tab.frame, 'text') == 'beach'
    tab.dir_var.set(env['src'])
    tab.build_index_now()
    _pump(root, 0.4)  # trigger the debounced stats
    lo, hi, err = tab.effective_range()
    assert err is None, err
    assert len(tab._matched) == 4, len(tab._matched)   # -1 excluded
    assert tab.stat_var.get().startswith('4 matched · found 4 files'), tab.stat_var.get()

    out1 = os.path.join(env['tmp'], 'out1')
    app.out_var.set(out1)
    rows = tab.resolved()
    assert all(s is not None for _, _, s in rows)
    app._copy_worker([(tab, rows)], pf.dedup_folder_names([p1b]), out1, len(rows))
    _pump(root, 0.4)
    folder = os.path.join(out1, 'beach')
    got = sorted(os.listdir(folder))
    assert got == ['0.5.png', '0.9.png', '0.951053.jpg', '0.951053_2.jpg',
                   'manifest.csv'], got
    with open(os.path.join(folder, 'manifest.csv'), encoding='utf-8-sig',
              newline='') as f:
        mrows = list(csv.reader(f))
    assert mrows[0] == ['new_name', 'pred', 'json_path', 'copied_path']
    assert len(mrows) == 5 and mrows[4][0] == '0.951053_2.jpg'
    assert mrows[4][3].endswith('x4.jpg'), mrows[4]    # the renamed copy maps back to the right source image


def test_start_run_full_flow(env, gui_app):
    root, app = gui_app
    src2 = os.path.join(env['tmp'], 'dataset2')
    os.makedirs(os.path.join(src2, 'beach2'))
    for nm in ('x6.png', 'x7.png'):
        with open(os.path.join(src2, 'beach2', nm), 'wb') as f:
            f.write(b'img-' + nm.encode())
    # rewrite det.json: basenames match dataset2 (including 2 invalid entries)
    det = {'meta': {'x': 1}, 'images': [
        {'image_path': 'G:/moved/beach2/x6.png', 'pred': 0.75},
        {'image_path': 'G:/moved/beach2/x7.png', 'pred': 0.1},
        {'image_path': 'X:/b/bad1.jpg'},
        {'image_path': 'X:/b/bad2.jpg', 'pred': 'oops'}]}
    with open(env['p2'], 'w') as f:
        json.dump(det, f)
    app.add_json_tabs([env['p2']])              # detailed format
    tab2 = app._tabs[1]
    assert app.notebook.tab(tab2.frame, 'text') == 'det'
    tab2.dir_var.set(src2)
    tab2.build_index_now()
    out2 = os.path.join(env['tmp'], 'out2')
    app.out_var.set(out2)
    app.start_run()                             # preflight should pass directly, no dialogs
    end = time.time() + 8
    while app.running and time.time() < end:
        root.update()
        time.sleep(0.02)
    assert not app.running, 'start_run did not finish within the timeout'
    assert os.path.exists(os.path.join(out2, 'det', '0.75.png'))
    assert not os.path.exists(os.path.join(out2, 'det', '0.1.png'))  # below the global lower bound
    assert os.path.exists(os.path.join(out2, 'det', 'manifest.csv'))


def test_tab_title_dedup_and_reimport(env, gui_app):
    root, app = gui_app
    p3 = os.path.join(env['tmp'], 'sub')
    os.makedirs(p3, exist_ok=True)
    p3 = os.path.join(p3, 'det.json')
    shutil.copyfile(env['p2'], p3)              # same content as p2, different directory
    app.add_json_tabs([env['p2']])              # duplicate import -> skipped
    assert len(app._tabs) == 2
    app.add_json_tabs([p3])                     # same name, different dir -> title (2)
    assert app.notebook.tab(app._tabs[2].frame, 'text') == 'det (2)'
    names3 = pf.dedup_folder_names([env['p2'], p3])
    # both sides of a conflict group get the parent-dir prefix: p2's parent is the dynamic temp dir name, p3's is sub
    assert names3[p3] == 'sub__det', names3
    assert names3[env['p2']].endswith('__det') and names3[env['p2']] != names3[p3], names3


def test_config_persistence(env, gui_app):
    app = gui_app[1]
    app._save_config()
    with open(pf.CONFIG_PATH, encoding='utf-8') as f:
        saved = json.load(f)
    out2 = os.path.join(env['tmp'], 'out2')
    assert saved['min'] == '0.4' and saved['max'] == '1.0' and \
        saved['out_dir'].lower() == out2.lower(), saved


if __name__ == '__main__':
    # pytest-free direct-run entry: execute all 10 groups in order
    tmp = tempfile.mkdtemp(prefix='_pf_test_')
    bak = open(pf.CONFIG_PATH, 'rb').read() if os.path.exists(pf.CONFIG_PATH) else None
    root = None
    try:
        env_d = _build_env(tmp)
        root = tk.Tk()
        root.withdraw()
        app = pf.App(root)
        root.update()
        g = (root, app)
        test_load_entries_both_formats(env_d)
        print('1 dual-format loading + invalid entries OK')
        test_in_range_closed_interval()
        print('2 closed-interval boundaries OK')
        test_pred_label_formatting()
        print('3 pred file-name formatting OK')
        test_unique_name_collision_suffix()
        print('4 collision suffixes OK')
        test_build_index_and_resolve(env_d)
        print('5 index resolution (unique/multiple/miss) OK')
        test_dedup_folder_names()
        print('6 duplicate folder-name prefixes OK')
        test_inline_copy_end_to_end(env_d, g)
        print('7 inline copy end-to-end (rename + suffixes + manifest) OK')
        test_start_run_full_flow(env_d, g)
        print('8 start_run full flow (detailed format + global range) OK')
        test_tab_title_dedup_and_reimport(env_d, g)
        print('9 title dedup + duplicate-import skip OK')
        test_config_persistence(env_d, g)
        print('10 config persistence OK')
        root.destroy()
        print('\nAll 10 check groups passed')
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        for extra in (pf.CONFIG_PATH + '.bak', pf.CONFIG_PATH + '.tmp'):
            if os.path.exists(extra):
                os.remove(extra)
        if bak is None:
            if os.path.exists(pf.CONFIG_PATH):
                os.remove(pf.CONFIG_PATH)
        else:
            with open(pf.CONFIG_PATH, 'wb') as f:
                f.write(bak)
        shutil.rmtree(tmp, ignore_errors=True)
