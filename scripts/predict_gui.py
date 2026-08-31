"""predict_gui.py — predict.py 的 tkinter 前端（子批次选项卡版）

架构（稳定优先）：GUI 不加载模型，只构造命令行 → 启动子进程 → 逐行读其
flush 输出渲染进度。推理崩溃只死子进程，GUI 永远存活、日志完整。
子批次 = 一个选项卡（独立输入目录列表 + 独立输出文件夹）= 一次 predict.py
子进程调用；子批次顺序执行、相互完全隔离（权重重读、CUDA 上下文全新），
单个子批次失败（含 device assert）不影响后续批次。
参数全局共享，运行开始时快照。
零新依赖（tkinter/json/subprocess/threading/queue 均为标准库）。
参数插件化：表单由 gui_config.json 动态生成，增删参数只改配置文件。

用法：python predict_gui.py             打开界面（需在 venv 内）
      python predict_gui.py --dry-run   按保存的子批次打印各自命令后退出
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, 'gui_config.json')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

BATCH_RE = re.compile(r'\[(\d+)/(\d+) 批\]')
VRAM_RE = re.compile(r'显存峰值 ([\d.]+) GiB')
DONE_RE = re.compile(r'\[完成\] (\d+) 张')
FAKE_RE = re.compile(r'>0\.5 判 fake: (\d+) 张')
SECTION_RE = re.compile(r'=== \[(\d+)/(\d+)\] (.+?) ===')
# 不在 gui_config.json 注册表里、由 GUI 按子批次构建命令时附加的标志
EXTRA_FLAGS = ('--output_dir', '--output_json')


# ---------- 纯函数（可脱离 GUI 单测） ----------

def count_images(root):
    n = 0
    for rd, _, fs in os.walk(root):
        n += sum(1 for f in fs if f.lower().endswith(IMG_EXTS))
    return n


def find_name_conflicts(dirs):
    """重名检测：返回 {文件夹名: [同名的目录路径...]}，仅含冲突项。"""
    by = {}
    for d in dirs:
        by.setdefault(os.path.basename(os.path.normpath(d)), []).append(d)
    return {b: ds for b, ds in by.items() if len(ds) > 1}


def find_out_conflicts(paths):
    """跨子批次输出文件夹冲突（大小写不敏感归一化）。"""
    by = {}
    for p in paths:
        if p:
            by.setdefault(os.path.normcase(os.path.normpath(p)), []).append(p)
    return [v[0] for v in by.values() if len(v) > 1]


def build_command(values, params):
    """values: {flag: str|int|bool|[bool,str]|list}；params: 注册表列表。返回参数列表。
    --data_dir 为列表时展开为一个标志 + 多路径（argparse nargs='+' 语义，
    注意不能重复写 --data_dir——重复出现时后者覆盖前者）。"""
    cmd = [sys.executable, os.path.join(BASE, 'predict.py')]
    for p in params:
        flag, w = p['flag'], p['widget']
        v = values.get(flag, p.get('default', ''))
        if w == 'bool':
            if v:
                cmd.append(flag)
        elif w == 'bool_path':
            if v and isinstance(v, list) and v[0]:
                cmd += [flag, str(v[1])]
        elif w == 'int':
            if v not in ('', None):
                cmd += [flag, str(int(v))]
        elif flag == '--data_dir':
            if isinstance(v, (list, tuple)):
                if v:
                    cmd += [flag] + [str(x) for x in v]
            elif v not in ('', None):
                cmd += [flag, str(v)]
        else:  # dir / file / savefile / choice / text
            if v not in ('', None):
                cmd += [flag, str(v)]
    for flag in EXTRA_FLAGS:  # 子批次输出（--output_dir 优先于 --output_json）
        v = values.get(flag)
        if v:
            cmd += [flag, str(v)]
    return cmd


def parse_line(line, state):
    """从 predict.py 的输出行提取进度信息，写入 state dict。
    批处理感知：=== [k/N] name === 分节头重置进度并记录当前文件夹；
    每个文件夹的 [完成] 与 P(fake) 行累积进 state['folders']。"""
    state.setdefault('folders', [])
    state.setdefault('folder_fails', 0)
    m = SECTION_RE.search(line)
    if m:
        state['sec_k'], state['sec_n'] = int(m.group(1)), int(m.group(2))
        state['sec_name'] = m.group(3)
        state['total'] = 0   # 新文件夹：重置批进度
        state['batch'] = 0
    m = BATCH_RE.search(line)
    if m:
        n, t = int(m.group(1)), int(m.group(2))
        state['total'] = max(state.get('total') or 0, t)
        state['batch'] = n
    m = VRAM_RE.search(line)
    if m:
        state['vram'] = m.group(1)
    m = DONE_RE.search(line)
    if m:
        state['done_n'] = m.group(1)
        state['done_sec'] = line.split('用时')[-1].strip(' 秒\n')
        state['folders'].append({'name': state.get('sec_name', ''),
                                 'n': m.group(1), 'fake': None})
    m = FAKE_RE.search(line)
    if m:
        state['fake_n'] = m.group(1)
        if state['folders']:
            state['folders'][-1]['fake'] = m.group(1)
    if line.startswith(('[跳过]', '[失败]')):
        state['folder_fails'] += 1
    if '[警告]' in line:
        state['warn'] = state.get('warn', 0) + 1


# ---------- 配置 ----------
# 设计取舍：注册表只以 gui_config.json 为唯一来源；每次保存前先备份上一份
# 为 .bak，损坏时可手动恢复；保存只写回 values/tabs/ui，注册表以磁盘最新为准。

def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = json.load(f)
    assert isinstance(cfg.get('params'), list) and cfg['params'], 'params 缺失'
    for p in cfg['params'] + cfg.get('advanced', []):
        assert 'flag' in p and 'widget' in p and 'label' in p, f'参数记录不完整: {p}'
    cfg.setdefault('advanced', [])
    cfg.setdefault('values', {})
    cfg.setdefault('ui', {})
    return cfg


def migrate_tabs(cfg):
    """旧配置（values 里的 data_dir/output_json）迁移为 tabs 结构；幂等。"""
    if isinstance(cfg.get('tabs'), list) and cfg['tabs']:
        return cfg['tabs']
    dd = cfg.get('values', {}).get('--data_dir', '')
    if isinstance(dd, str):
        dd = [dd] if dd else []
    dd = [d for d in dd if d]
    out = cfg.get('values', {}).get('--output_json', '')
    if out and not os.path.isdir(out) and not out.endswith(('/', '\\')):
        pass  # 旧的单文件输出路径作废，新语义输出为文件夹，留空自动命名
    else:
        out = out if out else ''
    return [{'dirs': dd, 'out': '' if (out and os.path.isfile(out)) else out}] if dd or out \
        else [{'dirs': [], 'out': ''}]


# ---------- 子批次标签页 ----------

class SubBatchTab:
    """一个子批次：独立输入目录列表 + 独立输出文件夹（的 UI 与数据）。"""

    def __init__(self, app, notebook, index, dirs=None, out=''):
        self.app = app
        self.dirs = []
        self.out_var = tk.StringVar(value=str(out or ''))
        self.frame = ttk.Frame(notebook)
        self.buttons = []  # 运行期间统一禁用
        self._build(notebook)
        for d in (dirs or []):
            self.add_dir(d)
        self.update_count()

    def _build(self, notebook):
        pad = dict(padx=6, pady=4)
        f = self.frame
        ttk.Label(f, text='输入目录').grid(row=0, column=0, sticky='ne', **pad)
        self.listbox = tk.Listbox(f, height=4, exportselection=False)
        self.listbox.grid(row=0, column=1, sticky='we', **pad)
        btns = ttk.Frame(f)
        btns.grid(row=0, column=2, sticky='w', **pad)
        for text, cmd in (('添加目录…', self.pick_dir), ('移除选中', self.remove_selected),
                          ('清空', self.clear)):
            b = ttk.Button(btns, text=text, command=cmd)
            b.pack(fill='x')
            self.buttons.append(b)
        self.count_var = tk.StringVar(value='（未添加）')
        ttk.Label(f, textvariable=self.count_var, foreground='#666',
                  wraplength=110, justify='left').grid(row=0, column=3, sticky='nw', **pad)
        ttk.Label(f, text='输出文件夹').grid(row=1, column=0, sticky='e', **pad)
        self.out_entry = ttk.Entry(f, textvariable=self.out_var)
        self.out_entry.grid(row=1, column=1, sticky='we', **pad)
        b = ttk.Button(f, text='浏览…', command=self.pick_out)
        b.grid(row=1, column=2, sticky='w', **pad)
        self.buttons.append(b)
        f.columnconfigure(1, weight=1)
        self.out_var.trace_add('write', lambda *a: self.app._refresh_cmd())

    # -- 目录管理 --
    def add_dir(self, d):
        d = os.path.normpath(d)
        if d not in self.dirs:
            self.dirs.append(d)
            self.listbox.insert('end', d)
            self.update_count()
            self.app.sync_title(self)
            return True
        return False

    def add_many(self, paths):
        added = skipped = 0
        for p in paths:
            if self.add_dir(p):
                added += 1
            else:
                skipped += 1
        return added, skipped

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            del self.dirs[i]
            self.listbox.delete(i)
        self.update_count()
        self.app.sync_title(self)
        self.app._refresh_cmd()

    def clear(self):
        self.dirs.clear()
        self.listbox.delete(0, 'end')
        self.update_count()
        self.app.sync_title(self)
        self.app._refresh_cmd()

    def pick_dir(self):
        d = filedialog.askdirectory(initialdir=self.app.cfg['ui'].get('last_dir') or BASE)
        if not d:
            return
        self.app.cfg['ui']['last_dir'] = d
        subs = self.app._list_subdirs(d)
        if subs:
            self.app._pick_from_subs(self, d, subs)
        elif self.add_dir(d):
            self.app.status_var.set(f'已添加 {os.path.basename(d)}')
        self.update_count()
        self.app._refresh_cmd()

    def pick_out(self):
        d = filedialog.askdirectory(initialdir=self.out_var.get() or BASE)
        if d:
            self.out_var.set(d)

    def update_count(self):
        if not self.dirs:
            self.count_var.set('（未添加）')
            return
        n = sum(count_images(d) for d in self.dirs if os.path.isdir(d))
        self.count_var.set(f'{len(self.dirs)} 个目录\n共 {n} 张图')

    def set_locked(self, locked):
        state = 'disabled' if locked else 'normal'
        for b in self.buttons:
            b.config(state=state)
        self.out_entry.config(state=state)
        self.listbox.config(state='disabled' if locked else 'normal')

    def title(self, index):
        return os.path.basename(self.dirs[0]) if self.dirs else f'批次{index + 1}'


# ---------- GUI 主窗口 ----------

class App:
    def __init__(self, root):
        self.root = root
        root.title('MIRROR 推理前端 · predict.py')
        root.geometry('880x760')
        try:
            self.cfg = load_config()
        except Exception as e:
            bak = os.path.join(BASE, 'gui_config.json.bak')
            tip = '可把 gui_config.json.bak 改名恢复后重启。' if os.path.exists(bak) else ''
            messagebox.showerror('配置错误', f'gui_config.json 读取失败：{e}\n{tip}')
            root.destroy()
            return
        self.all_params = self.cfg['params'] + self.cfg['advanced']
        self.vars = {}
        self.q = queue.Queue()
        self.proc = None
        self.t0 = None
        self.state = {}
        self.running = False
        self.stop_requested = False
        self.run_queue = []
        self.snap_params = {}
        self.tab_results = []
        self.last_out_dir = ''
        self._build()
        self._restore_values()
        self._refresh_cmd()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.after(150, self._poll)

    # ---- 界面构建 ----
    def _build(self):
        pad = dict(padx=6, pady=4)

        # 子批次选项卡区
        batchf = ttk.LabelFrame(self.root, text='子批次（每个选项卡独立输入/输出，顺序隔离执行）')
        batchf.pack(fill='x', **pad)
        tabbar = ttk.Frame(batchf)
        tabbar.pack(fill='x', **pad)
        self.notebook = ttk.Notebook(tabbar)
        self.notebook.pack(side='left', fill='x', expand=True)
        self.notebook.bind('<<NotebookTabChanged>>', lambda e: self._refresh_cmd())
        addb = ttk.Button(tabbar, text='＋ 添加批次', command=self.add_tab)
        addb.pack(side='left', padx=4)
        closeb = ttk.Button(tabbar, text='✕ 关闭当前', command=self.close_tab)
        closeb.pack(side='left', padx=4)
        self.tab_buttons = [addb, closeb]

        # 全局参数区（悬停提示行不进表格，防列宽撑动闪动）
        self.form = ttk.LabelFrame(self.root, text='参数（全局，由 gui_config.json 动态生成）')
        self.form.pack(fill='x', **pad)
        self.hint_var = tk.StringVar(value='把鼠标停在参数上查看说明')
        tk.Label(self.root, textvariable=self.hint_var, fg='#0a6',
                 wraplength=840, height=2, anchor='nw', justify='left').pack(
            fill='x', padx=12)
        self._build_group(self.cfg['params'], start_row=0)
        self.adv = ttk.LabelFrame(self.root, text='高级（模型路径 / 设备）')
        self.adv.pack(fill='x', **pad)
        self._build_group(self.cfg['advanced'], start_row=0, parent=self.adv)

        ttk.Label(self.root, text='等效命令行（当前标签页）：').pack(anchor='w', padx=10)
        self.cmd_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.cmd_var, foreground='#333',
                  wraplength=850, font=('Consolas', 8)).pack(anchor='w', padx=12)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill='x', **pad)
        self.btn_start = ttk.Button(ctrl, text='开始推理', command=self.start_run)
        self.btn_start.pack(side='left', padx=6)
        self.btn_stop = ttk.Button(ctrl, text='停止', command=self.stop_run, state='disabled')
        self.btn_stop.pack(side='left', padx=6)
        self.progress = ttk.Progressbar(ctrl, length=280, maximum=1, value=0)
        self.progress.pack(side='left', padx=10)
        self.btn_open = ttk.Button(ctrl, text='打开结果文件夹', command=self.open_result, state='disabled')
        self.btn_open.pack(side='right', padx=6)
        self.btn_reset = ttk.Button(ctrl, text='恢复默认参数', command=self.reset_defaults)
        self.btn_reset.pack(side='right', padx=6)

        self.status_var = tk.StringVar(value='就绪')
        ttk.Label(self.root, textvariable=self.status_var, foreground='#036').pack(anchor='w', padx=10)

        logf = ttk.LabelFrame(self.root, text='日志（子进程原样输出）')
        logf.pack(fill='both', expand=True, **pad)
        self.logbox = tk.Text(logf, height=12, font=('Consolas', 9), state='disabled',
                              wrap='none')
        self.logbox.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(logf, command=self.logbox.yview)
        sb.pack(side='right', fill='y')
        self.logbox.config(yscrollcommand=sb.set)

    # ---- 悬停提示 ----
    def _bind_hint(self, widget, p):
        """Enter/Leave 分别绑定（勿用 e.type=='10' 判断——EventType 枚举与字符串比较恒 False）。"""
        widget.bind('<Enter>', lambda e, p=p: self._show_hint(p))
        widget.bind('<Leave>', lambda e: self._clear_hint())

    def _show_hint(self, p):
        self.hint_var.set(f"{p['label']}（{p['flag']}）：{p.get('help', '')}")

    def _clear_hint(self):
        self.hint_var.set('把鼠标停在参数上查看说明')

    def _build_group(self, params, start_row=0, parent=None):
        parent = parent or self.form
        n = 0  # 紧凑排布计数（--data_dir/--output_json 由标签页管理，不进表单）
        for p in params:
            if p['flag'] in ('--data_dir', '--output_json'):
                continue
            r, c = divmod(n, 2)
            n += 1
            row, col = start_row + r, c * 2
            lab = ttk.Label(parent, text=p['label'])
            lab.grid(row=row, column=col, sticky='e', padx=6, pady=3)
            self._bind_hint(lab, p)
            w = p['widget']
            if w == 'bool':
                var = tk.BooleanVar(value=bool(p.get('default')))
                cb = ttk.Checkbutton(parent, variable=var, command=self._refresh_cmd)
                cb.grid(row=row, column=col + 1, sticky='w', padx=4)
                self._bind_hint(cb, p)
            elif w == 'bool_path':
                en = tk.BooleanVar(value=bool(p.get('default')))
                pv = tk.StringVar(value=str(p.get('sub_default', '')))
                fr = ttk.Frame(parent)
                fr.grid(row=row, column=col + 1, sticky='we', padx=4)
                ttk.Checkbutton(fr, variable=en, command=self._refresh_cmd).pack(side='left')
                ttk.Entry(fr, textvariable=pv, width=18).pack(side='left', padx=4, fill='x', expand=True)
                self._bind_hint(fr, p)
                self.vars[p['flag']] = (en, pv)
                en.trace_add('write', lambda *a, pv=pv, en=en: self._auto_csv(en, pv))
                continue
            elif w == 'int':
                var = tk.StringVar(value=str(p.get('default', '')))
                sp = ttk.Spinbox(parent, textvariable=var, from_=p.get('min', 0),
                                 to=p.get('max', 99999), width=9,
                                 command=self._refresh_cmd)
                sp.grid(row=row, column=col + 1, sticky='w', padx=4)
                self._bind_hint(sp, p)
            elif w == 'choice':
                var = tk.StringVar(value=str(p.get('default', '')))
                cbo = ttk.Combobox(parent, textvariable=var, values=p.get('choices', []),
                                   state='readonly', width=12)
                cbo.grid(row=row, column=col + 1, sticky='w', padx=4)
                self._bind_hint(cbo, p)
            elif w in ('file', 'dir'):
                var = tk.StringVar(value=str(p.get('default', '')))
                fr = ttk.Frame(parent)
                fr.grid(row=row, column=col + 1, sticky='we', padx=4)
                ttk.Entry(fr, textvariable=var, width=24).pack(side='left', fill='x', expand=True)
                ttk.Button(fr, text='…', width=3,
                           command=lambda v=var, w=w: self._browse(v, w)).pack(side='left', padx=2)
                self._bind_hint(fr, p)
            else:  # savefile / text
                var = tk.StringVar(value=str(p.get('default', '')))
                ent = ttk.Entry(parent, textvariable=var, width=28)
                ent.grid(row=row, column=col + 1, sticky='w', padx=4)
                self._bind_hint(ent, p)
            var.trace_add('write', lambda *a: self._refresh_cmd())
            self.vars[p['flag']] = var

    # ---- 标签页管理 ----
    def add_tab(self, dirs=None, out='', silent=False):
        tab = SubBatchTab(self, self.notebook, len(self.tabs()), dirs, out)
        self.tabs().append(tab)
        self.notebook.add(tab.frame, text=tab.title(len(self.tabs()) - 1))
        self.notebook.select(tab.frame)
        if not silent:
            self._refresh_cmd()
        return tab

    def close_tab(self):
        if self.running:
            return
        if len(self.tabs()) <= 1:
            self.status_var.set('至少保留一个子批次')
            return
        tab = self.current_tab()
        if (tab.dirs or tab.out_var.get()) and not messagebox.askyesno(
                '关闭子批次', f'关闭"{tab.title(0)}"？其目录列表与输出设置将丢弃。'):
            return
        i = self.tabs().index(tab)
        self.notebook.forget(tab.frame)
        self.tabs().remove(tab)
        for j, t in enumerate(self.tabs()):
            self.notebook.tab(t.frame, text=t.title(j))
        self._refresh_cmd()

    def tabs(self):
        return self._tabs

    def current_tab(self):
        i = max(0, self.notebook.index(self.notebook.select()))
        return self._tabs[i]

    def sync_title(self, tab):
        if tab not in self._tabs:
            return  # add_tab 构造过程中 tab 尚未注册，标题由 add_tab 收尾时设置
        i = self._tabs.index(tab)
        self.notebook.tab(tab.frame, text=tab.title(i))

    # ---- 值读写 ----
    def _restore_values(self):
        saved = self.cfg.get('values', {})
        for flag, v in saved.items():
            if flag not in self.vars:
                continue
            cur = self.vars[flag]
            if isinstance(cur, tuple):  # bool_path
                if isinstance(v, list) and len(v) == 2:
                    cur[0].set(bool(v[0]))
                    cur[1].set(str(v[1]))
            elif isinstance(cur, tk.BooleanVar):
                cur.set(bool(v))
            else:
                cur.set(str(v))
        self._tabs = []
        for spec in migrate_tabs(self.cfg):
            self.add_tab(spec.get('dirs'), spec.get('out', ''), silent=True)
        self._refresh_cmd()

    def _collect_form_values(self):
        vals = {}
        for p in self.all_params:
            flag, w = p['flag'], p['widget']
            if flag in ('--data_dir', '--output_json'):
                continue  # 标签页数据
            v = self.vars[flag]
            if w == 'bool_path':
                vals[flag] = [bool(v[0].get()), v[1].get()]
            elif isinstance(v, tk.BooleanVar):
                vals[flag] = bool(v.get())
            else:
                vals[flag] = v.get()
        return vals

    def _vals_for_tab(self, tab, snap=None):
        """组装一个子批次的命令值：表单参数（或快照）+ 该页的目录与输出。
        输出永远为文件夹（predict.py 的 --output_dir 对任意目录数生效）；
        留空 → results_gui\\batch_<启动时刻>\\批次i。"""
        vals = dict(snap) if snap else self._collect_form_values()
        vals['--data_dir'] = list(tab.dirs)
        vals['--output_json'] = ''
        if not tab.out_var.get().strip():
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            vals['--output_dir'] = os.path.join(BASE, 'results_gui',
                                                f'batch_{ts}', f'批次{self._tabs.index(tab) + 1}')
        else:
            vals['--output_dir'] = tab.out_var.get().strip()
        d = vals.get('--dump_scores')
        if isinstance(d, list) and d[0] and not d[1]:
            d[1] = os.path.join(BASE, 'results_gui', 'per_crop.csv')
        return vals

    def get_value(self, p):
        return self._collect_form_values()[p['flag']]

    def _auto_csv(self, en, pv):
        if en.get() and not pv.get():
            pv.set(str(next(p for p in self.all_params
                            if p['flag'] == '--dump_scores').get('sub_default', 'results_gui\\per_crop.csv')))
        self._refresh_cmd()

    def _refresh_cmd(self, *_):
        if not getattr(self, '_tabs', None):
            return
        vals = self._vals_for_tab(self.current_tab())
        self.cmd_var.set(' '.join(build_command(vals, self.all_params)))

    # ---- 目录多选对话框（供标签页调用） ----
    @staticmethod
    def _list_subdirs(d):
        return [os.path.join(d, n) for n in sorted(os.listdir(d))
                if os.path.isdir(os.path.join(d, n))]

    def _pick_from_subs(self, tab, parent, subs):
        win = tk.Toplevel(self.root)
        win.title('选择要添加的文件夹')
        win.transient(self.root)
        win.grab_set()
        ttk.Label(win, text=f'"{parent}" 下有 {len(subs)} 个子文件夹。\n'
                            '按住 Ctrl（点选）/ Shift（连选）可多选。').pack(
            anchor='w', padx=10, pady=(10, 4))
        frame = ttk.Frame(win)
        frame.pack(fill='both', expand=True, padx=10)
        lb = tk.Listbox(frame, selectmode='extended', height=min(16, max(6, len(subs))))
        for s in subs:
            lb.insert('end', s)
        lb.pack(side='left', fill='both', expand=True)
        lb.selection_set(0, 'end')
        sb = ttk.Scrollbar(frame, command=lb.yview)
        sb.pack(side='right', fill='y')
        lb.config(yscrollcommand=sb.set)

        def confirm():
            sel = [subs[i] for i in lb.curselection()]
            win.destroy()
            if sel:
                added, skipped = tab.add_many(sel)
                self.status_var.set(f'已添加 {added} 个（跳过重复 {skipped}）')
                self._refresh_cmd()

        btns = ttk.Frame(win)
        btns.pack(fill='x', padx=10, pady=10)
        ttk.Button(btns, text='添加所选子文件夹', command=confirm).pack(side='left', padx=(0, 6))
        ttk.Button(btns, text='取消', command=win.destroy).pack(side='right')
        win.wait_window()

    def _browse(self, var, kind):
        f = (filedialog.askopenfilename if kind == 'file' else filedialog.askdirectory)()
        if f:
            var.set(f)

    # ---- 预检与运行 ----
    def preflight_tab(self, tab):
        probs = []
        if not tab.dirs:
            probs.append('（空子批次，将跳过）')
        for d in tab.dirs:
            if not os.path.isdir(d):
                probs.append(f'目录不存在：{d}')
            elif count_images(d) == 0:
                probs.append(f'目录里没有可识别的图片（png/jpg/jpeg/bmp/webp）：{d}')
        conf = find_name_conflicts(tab.dirs)
        if conf:
            lines = '\n'.join(f'· {b}（{len(ds)} 个同名文件夹）' for b, ds in conf.items())
            if not messagebox.askyesno(
                    '文件夹重名警告',
                    f'子批次"{tab.title(0)}"内以下文件夹名相同，输出 JSON 会自动加'
                    f'父目录前缀区分（如 父目录__文件夹名.json）：\n\n' + lines + '\n\n是否继续？'):
                probs.append('已取消（文件夹重名）')
        out = tab.out_var.get().strip()
        if out and os.path.isfile(out):
            probs.append(f'输出文件夹指向的是一个文件：{out}')
        return probs

    def start_run(self):
        if self.running:
            return
        all_probs = []
        for tab in self._tabs:
            probs = self.preflight_tab(tab)
            if probs and probs != ['（空子批次，将跳过）']:
                all_probs.append(f'[{tab.title(0)}] ' + '；'.join(probs))
        outs = [t.out_var.get().strip() for t in self._tabs if t.dirs]
        dups = find_out_conflicts(outs)
        if dups:
            if not messagebox.askyesno(
                    '输出文件夹冲突',
                    '以下输出文件夹被多个子批次使用，后运行的会覆盖先运行的 JSON：\n\n'
                    + '\n'.join(dups) + '\n\n是否继续？'):
                return
        if all_probs:
            messagebox.showerror('预检未通过', '\n'.join(all_probs))
            return
        if not any(t.dirs for t in self._tabs):
            messagebox.showerror('无法开始', '所有子批次都是空的。')
            return
        self.snap_params = self._collect_form_values()  # 参数快照：运行期间表单修改不影响
        self.run_queue = [t for t in self._tabs if t.dirs]
        self.tab_results = []
        self.stop_requested = False
        self.running = True
        self.t0 = time.time()
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.btn_open.config(state='disabled')
        for b in self.tab_buttons:
            b.config(state='disabled')
        for t in self._tabs:
            t.set_locked(True)
        self._start_next_tab()

    def _start_next_tab(self):
        if self.stop_requested or not self.run_queue:
            self._finish_run()
            return
        self.cur_tab = self.run_queue.pop(0)
        self.cur_index = len(self.tab_results) + 1
        self.n_tabs = len(self.tab_results) + len(self.run_queue) + 1
        self.state = {}
        vals = self._vals_for_tab(self.cur_tab, snap=self.snap_params)
        self.cur_out = vals['--output_dir']
        os.makedirs(self.cur_out, exist_ok=True)
        self.last_out_dir = self.cur_out
        cmd = build_command(vals, self.all_params)
        self._log(f"\n### 子批次 {self.cur_index}/{self.n_tabs}: "
                  f"{self.cur_tab.title(self._tabs.index(self.cur_tab))}"
                  f"（{len(self.cur_tab.dirs)} 个目录 -> {self.cur_out}）\n")
        self._log(f'$ {" ".join(cmd)}\n')
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception as e:
            self._log(f'[GUI] 启动失败：{e}\n')
            self.tab_results.append((self.cur_tab.title(0), '启动失败'))
            self.root.after(300, self._start_next_tab)
            return
        self.status_var.set(f'子批次 {self.cur_index}/{self.n_tabs} 已启动…')
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(('line', line.rstrip('\n')))
        self.q.put(('done', proc.wait()))

    def stop_run(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno(
                    '停止', '停止当前子批次并取消剩余子批次？\n（已完成子批次的 JSON 已保存）'):
                return
            self.stop_requested = True
            self.proc.terminate()
            self._log('[GUI] 已请求停止…\n')
            self.root.after(3000, self._ensure_killed)

    def _ensure_killed(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
                self._log('[GUI] 进程未响应，已强制终止\n')
            except Exception:
                pass

    def open_result(self):
        target = self.last_out_dir
        if target and os.path.exists(target):
            os.startfile(target)

    def reset_defaults(self):
        if self.running:
            return
        if messagebox.askyesno('恢复默认', '将放弃当前参数修改（子批次与模型路径保留），继续？'):
            for p in self.all_params:
                if p['flag'] in ('--data_dir', '--output_json', '--model_path',
                                 '--memory_path', '--backbone_path'):
                    continue
                v = self.vars[p['flag']]
                w = p['widget']
                if w == 'bool_path':
                    v[0].set(bool(p.get('default')))
                    v[1].set(str(p.get('sub_default', '')))
                elif isinstance(v, tk.BooleanVar):
                    v.set(bool(p.get('default')))
                else:
                    v.set(str(p.get('default', '')))
            self._refresh_cmd()

    # ---- 运行时轮询（tkinter 单线程规则：子线程只入队） ----
    def after(self, ms, fn):
        self.root.after(ms, fn)

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == 'line':
                    self._on_line(payload)
                else:
                    self._on_proc_done(payload)
        except queue.Empty:
            pass
        if self.proc is not None and self.proc.poll() is None:
            self.status_var.set(self._status_text())
        self.root.after(150, self._poll)

    def _status_text(self):
        s = self.state
        parts = [f'子批次 {self.cur_index}/{self.n_tabs} {self.cur_tab.title(0)}']
        if s.get('sec_n'):
            parts.append(f'文件夹 {s.get("sec_k", "?")}/{s["sec_n"]}')
        if s.get('total'):
            parts.append(f'批次 {s.get("batch", 0)}/{s["total"]}')
        parts.append(f'已用 {int(time.time() - self.t0)} 秒')
        return ' · '.join(parts)

    def _on_line(self, line):
        self._log(line + '\n')
        parse_line(line, self.state)
        if self.state.get('total'):
            self.progress.config(maximum=self.state['total'],
                                 value=self.state.get('batch', 0))
        else:
            self.progress.config(value=0)
        if '[警告]' in line:
            self.status_var.set('有读取警告，详见日志 · ' + self._status_text())

    def _on_proc_done(self, rc):
        s = self.state
        title = self.cur_tab.title(0)
        if rc == 0:
            folders = s.get('folders', [])
            total_imgs = sum(int(f['n']) for f in folders)
            total_fake = sum(int(f['fake'] or 0) for f in folders)
            msg = (f'成功 {len(folders)} 文件夹/{total_imgs} 张/判 fake {total_fake}')
            if s.get('folder_fails'):
                msg += f'/文件夹失败 {s["folder_fails"]}'
            self.tab_results.append((title, msg))
            self._log(f'[GUI] 子批次"{title}"完成：{msg}，返回码 0\n')
        else:
            self.tab_results.append((title, f'失败（返回码 {rc}）'))
            self._log(f'[GUI] 子批次"{title}"失败，返回码 {rc}（后续子批次将在新进程中继续）\n')
        self.proc = None
        self.root.after(400, self._start_next_tab)  # 稍候再开新进程，给驱动喘息

    def _finish_run(self):
        sec = int(time.time() - self.t0)
        ok = sum(1 for _, m in self.tab_results if not m.startswith('失败'))
        lines = [f'[{t}] {m}' for t, m in self.tab_results]
        self._log('[GUI] ===== 运行总结 =====\n' + '\n'.join(lines) + '\n')
        self.status_var.set(f'运行结束：子批次成功 {ok}/{len(self.tab_results)}'
                            f' · 用时 {sec} 秒（明细见日志）')
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        for b in self.tab_buttons:
            b.config(state='normal')
        for t in self._tabs:
            t.set_locked(False)
        if os.path.isdir(self.last_out_dir):
            self.btn_open.config(state='normal')
        self.running = False
        self._save_values()

    def _log(self, text):
        self.logbox.config(state='normal')
        self.logbox.insert('end', text)
        self.logbox.see('end')
        self.logbox.config(state='disabled')

    # ---- 退出与持久化 ----
    def _save_values(self):
        """只写回 values（表单参数）/tabs/ui；注册表以磁盘最新为准（防覆盖手工编辑）。"""
        vals = {f: v for f, v in self._collect_form_values().items()
                if f not in ('--data_dir', '--output_json')}
        try:
            with open(CONFIG_PATH, encoding='utf-8') as f:
                fresh = json.load(f)
            fresh['values'] = vals
            fresh['tabs'] = [{'dirs': list(t.dirs), 'out': t.out_var.get()} for t in self._tabs]
            fresh['ui'] = self.cfg['ui']
            self.cfg = fresh
        except Exception:
            pass  # 磁盘文件不可用时退回内存版本
        tmp = CONFIG_PATH + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.cfg, f, ensure_ascii=False, indent=1)
            if os.path.exists(CONFIG_PATH):
                os.replace(CONFIG_PATH, CONFIG_PATH + '.bak')  # 上一份留档，损坏可手动恢复
            os.replace(tmp, CONFIG_PATH)
        except Exception as e:
            self._log(f'[GUI] 配置写回失败：{e}\n')
            try:
                os.remove(tmp)
            except OSError:
                pass

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno('退出', '推理仍在运行，停止并退出？\n（已完成子批次的 JSON 已保存）'):
                return
            self.stop_requested = True
            self.proc.terminate()
            self.root.after(2000, self._ensure_killed)
        self._save_values()
        self.root.destroy()


def main():
    if '--dry-run' in sys.argv:
        cfg = load_config()
        tabs = migrate_tabs(cfg)
        saved = cfg.get('values', {})
        form = {p['flag']: saved.get(p['flag'], p.get('default', ''))
                for p in cfg['params'] + cfg['advanced']
                if p['flag'] not in ('--data_dir', '--output_json')}
        for i, spec in enumerate(tabs):
            tab = SubBatchTab.__new__(SubBatchTab)  # 无 UI 的轻量构造，仅用 _vals 逻辑
            tab.dirs = list(spec.get('dirs', []))
            tab.out_var = type('V', (), {'get': lambda s, v=spec.get('out', ''): v})()
            tab.title = lambda idx, i=i: (os.path.basename(spec['dirs'][0])
                                          if spec.get('dirs') else f'批次{i + 1}')

            class _T:
                pass
            app_like = _T()
            app_like._tabs = tabs
            vals = dict(form)
            vals['--data_dir'] = list(tab.dirs)
            vals['--output_json'] = ''
            out = spec.get('out', '')
            vals['--output_dir'] = out.strip() if out.strip() else os.path.join(
                BASE, 'results_gui', f'batch dry-run tab{i + 1}')
            print(f'# 子批次 {i + 1}/{len(tabs)}: {tab.title(i)}')
            print(' '.join(build_command(vals, cfg['params'] + cfg['advanced'])))
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
