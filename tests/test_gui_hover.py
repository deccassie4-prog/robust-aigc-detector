# -*- coding: utf-8 -*-
"""Hover-flicker regression tests: when hint text changes, every control in the parameter
form must keep its exact position.

This is the regression gate for the predict_gui layout red line (the hint label must never
be gridded into the parameter table). Headless (withdrawn window), no GPU, no subprocess.
Also runs directly: `python tests/test_gui_hover.py`.
"""
import os
import sys
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
from scripts import predict_gui as pg


@pytest.fixture()
def gui():
    """A withdrawn App; skipped when there is no display (Linux CI)."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('tkinter has no usable display')
    root.withdraw()
    app = pg.App(root)
    root.update()
    yield root, app
    root.destroy()


def _snapshot(root, app):
    root.update_idletasks()
    out = []
    for parent in (app.form, app.adv):
        for w in parent.winfo_children():
            out.append((str(w), w.winfo_x(), w.winfo_y(),
                        w.winfo_width(), w.winfo_height()))
    return out


def test_hover_zero_displacement(gui):
    """Show every parameter hint in turn (including the longest crop_mode / batch_size help), then clear: zero control displacement."""
    root, app = gui
    before = _snapshot(root, app)
    for p in app.all_params:
        app._show_hint(p)
        root.update()
    app._clear_hint()
    root.update()
    after = _snapshot(root, app)
    assert len(before) == len(after)
    moved = [(b[0], b[1:], a[1:]) for b, a in zip(before, after) if b[1:] != a[1:]]
    assert not moved, f"hover moved {len(moved)} controls (flicker root cause not fixed): {moved[:3]}"


def test_hint_label_not_in_form(gui):
    """The hint label must not live inside the parameter table (it must not participate in grid column-width calculation)."""
    root, app = gui
    hint_owner = app.hint_var

    def is_hint_widget(w):
        try:
            return str(w.cget('textvariable')) == str(hint_owner)
        except Exception:
            return False

    in_form = any(is_hint_widget(w) for w in app.form.winfo_children())
    assert not in_form, 'hint label is still gridded inside the parameter table'


def test_hint_label_height_constant(gui):
    """The fixed two-line height holds: switching hint texts of different lengths never changes the hint label height."""
    root, app = gui
    hint_owner = app.hint_var
    hint_lbl = [w for w in root.winfo_children()
                if w.winfo_class() == 'Label'
                and str(w.cget('textvariable')) == str(hint_owner)]
    assert hint_lbl, 'hint label not found'
    # show one hint so the label completes its geometry, then switch to a longer text and compare heights
    app._show_hint(app.all_params[0])
    root.update()
    h1 = hint_lbl[0].winfo_height()
    app._show_hint(app.all_params[1])
    root.update()
    h2 = hint_lbl[0].winfo_height()
    assert h1 == h2, f'hint label height changes with text: {h1} -> {h2}'


if __name__ == '__main__':
    # minimal manual fixture driver (pytest not required)
    root = tk.Tk()
    root.withdraw()
    app = pg.App(root)
    root.update()
    for fn in (test_hover_zero_displacement, test_hint_label_not_in_form,
               test_hint_label_height_constant):
        fn((root, app))
    print(f'Passed: hint label independent of the parameter grid, constant height '
          f'({len(app.all_params)} parameter hints polled with zero displacement)')
    root.destroy()
