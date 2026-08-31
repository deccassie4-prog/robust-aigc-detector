# -*- coding: utf-8 -*-
"""悬停闪动回归测试：提示文字变化时，参数表单所有控件坐标必须不变。

这是 predict_gui 布局红线（提示标签不得 grid 进参数表格）的回归门禁。
headless 运行（withdraw 窗口），无 GPU、无子进程。
可直接 `python tests/test_gui_hover.py` 运行。
"""
import os
import sys
import types
import tkinter as tk

try:
    import pytest
except ImportError:  # 允许无 pytest 直接运行（走文件尾的 __main__ 入口）
    pytest = types.SimpleNamespace(
        fixture=lambda *a, **k: (lambda f: f),
        skip=lambda msg='': (_ for _ in ()).throw(RuntimeError(msg)),
    )

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import predict_gui as pg


@pytest.fixture()
def gui():
    """一个 withdraw 的 App；无显示环境（Linux CI）时跳过。"""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('tkinter 无可用显示环境')
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
    """依次显示所有参数提示（含最长的 crop_mode / batch_size 帮助）再清空，控件零位移。"""
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
    assert not moved, f"悬停导致 {len(moved)} 个控件位移（闪动根因未修）：{moved[:3]}"


def test_hint_label_not_in_form(gui):
    """提示标签必须不在参数表格里（不能参与 grid 列宽计算）。"""
    root, app = gui
    hint_owner = app.hint_var

    def is_hint_widget(w):
        try:
            return str(w.cget('textvariable')) == str(hint_owner)
        except Exception:
            return False

    in_form = any(is_hint_widget(w) for w in app.form.winfo_children())
    assert not in_form, '提示标签仍 grid 在参数表格内'


def test_hint_label_height_constant(gui):
    """固定两行高度生效：切换不同长度的提示文字，提示标签自身高度不变。"""
    root, app = gui
    hint_owner = app.hint_var
    hint_lbl = [w for w in root.winfo_children()
                if w.winfo_class() == 'Label'
                and str(w.cget('textvariable')) == str(hint_owner)]
    assert hint_lbl, '未找到提示标签'
    # 先显示一条提示让标签完成几何布局，再切换到更长的文字比较高度
    app._show_hint(app.all_params[0])
    root.update()
    h1 = hint_lbl[0].winfo_height()
    app._show_hint(app.all_params[1])
    root.update()
    h2 = hint_lbl[0].winfo_height()
    assert h1 == h2, f'提示标签高度随文字变化: {h1} -> {h2}'


if __name__ == '__main__':
    # 简易 fixture 手动驱动（不依赖 pytest）
    root = tk.Tk()
    root.withdraw()
    app = pg.App(root)
    root.update()
    for fn in (test_hover_zero_displacement, test_hint_label_not_in_form,
               test_hint_label_height_constant):
        fn((root, app))
    print(f'回归通过：提示标签独立于参数表格，高度恒定（{len(app.all_params)} 个参数提示轮询无位移）')
    root.destroy()
