"""predict_gui.py — tkinter frontend for predict.py (sub-batch tab edition)

Architecture (stability first): the GUI never loads the model; it builds a command
line, starts a subprocess and renders its flushed output line by line. A crash in
inference only kills the subprocess - the GUI survives with the full log.
Sub-batch = one tab (independent input dir list + output folder) = one predict.py
subprocess call; sub-batches run sequentially and are fully isolated (weights reloaded,
fresh CUDA context), so one failing sub-batch (even a device assert) does not affect
the following ones.
Parameters are global and snapshotted when a run starts.
Zero extra dependencies (tkinter/json/subprocess/threading/queue are stdlib).
Plugin-style parameters: the form is generated from gui_config.json - to add or remove
parameters, edit that file only.

Usage: python predict_gui.py             open the UI (inside a venv)
       python predict_gui.py --dry-run   print each saved sub-batch's command and exit
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

BATCH_RE = re.compile(r'\[(\d+)/(\d+) batches\]')
VRAM_RE = re.compile(r'VRAM peak ([\d.]+) GiB')
DONE_RE = re.compile(r'\[done\] (\d+) images')
FAKE_RE = re.compile(r'>0\.5 -> fake: (\d+)')
SECTION_RE = re.compile(r'=== \[(\d+)/(\d+)\] (.+?) ===')
# flags not in the gui_config.json registry, appended by the GUI when building each sub-batch command
EXTRA_FLAGS = ('--output_dir', '--output_json')


# ---------- pure functions (unit-testable without the GUI) ----------

def count_images(root):
    n = 0
    for rd, _, fs in os.walk(root):
        n += sum(1 for f in fs if f.lower().endswith(IMG_EXTS))
    return n


def find_name_conflicts(dirs):
    """Duplicate-name detection: returns {folder name: [dirs sharing that name], ...}, conflicts only."""
    by = {}
    for d in dirs:
        by.setdefault(os.path.basename(os.path.normpath(d)), []).append(d)
    return {b: ds for b, ds in by.items() if len(ds) > 1}


def find_out_conflicts(paths):
    """Output-folder conflicts across sub-batches (case-insensitive normalization)."""
    by = {}
    for p in paths:
        if p:
            by.setdefault(os.path.normcase(os.path.normpath(p)), []).append(p)
    return [v[0] for v in by.values() if len(v) > 1]


def build_command(values, params):
    """values: {flag: str|int|bool|[bool,str]|list}; params: registry list. Returns an argv list.
    A list --data_dir expands into one flag + multiple paths (argparse nargs='+' semantics;
    never repeat the --data_dir flag - argparse silently keeps the last occurrence)."""
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
    for flag in EXTRA_FLAGS:  # sub-batch output (--output_dir wins over --output_json)
        v = values.get(flag)
        if v:
            cmd += [flag, str(v)]
    return cmd


def parse_line(line, state):
    """Extract progress info from a predict.py output line into the state dict.
    Batch-aware: '=== [k/N] name ===' section headers reset progress and record the folder;
    each folder's [done] and P(fake) stats lines accumulate into state['folders']."""
    state.setdefault('folders', [])
    state.setdefault('folder_fails', 0)
    m = SECTION_RE.search(line)
    if m:
        state['sec_k'], state['sec_n'] = int(m.group(1)), int(m.group(2))
        state['sec_name'] = m.group(3)
        state['total'] = 0   # new folder: reset batch progress
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
        state['done_sec'] = line.rsplit(', ', 1)[-1].strip(' s\n')
        state['folders'].append({'name': state.get('sec_name', ''),
                                 'n': m.group(1), 'fake': None})
    m = FAKE_RE.search(line)
    if m:
        state['fake_n'] = m.group(1)
        if state['folders']:
            state['folders'][-1]['fake'] = m.group(1)
    if line.startswith(('[skip]', '[fail]')):
        state['folder_fails'] += 1
    if '[warn]' in line:
        state['warn'] = state.get('warn', 0) + 1


# ---------- config ----------
# design choice: the registry's single source of truth is gui_config.json; each save first
# rotates the previous file to .bak (manually recoverable), and only values/tabs/ui are
# written back while the registry is re-read from disk (prevents clobbering manual edits).

def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = json.load(f)
    assert isinstance(cfg.get('params'), list) and cfg['params'], "'params' missing"
    for p in cfg['params'] + cfg.get('advanced', []):
        assert 'flag' in p and 'widget' in p and 'label' in p, f'incomplete parameter spec: {p}'
    cfg.setdefault('advanced', [])
    cfg.setdefault('values', {})
    cfg.setdefault('ui', {})
    return cfg


def migrate_tabs(cfg):
    """Migrate the legacy config (data_dir/output_json inside values) to the tabs structure; idempotent."""
    if isinstance(cfg.get('tabs'), list) and cfg['tabs']:
        return cfg['tabs']
    dd = cfg.get('values', {}).get('--data_dir', '')
    if isinstance(dd, str):
        dd = [dd] if dd else []
    dd = [d for d in dd if d]
    out = cfg.get('values', {}).get('--output_json', '')
    if out and not os.path.isdir(out) and not out.endswith(('/', '\\')):
        pass  # legacy single-file output paths are obsolete; the new semantics is a folder (empty = auto name)
    else:
        out = out if out else ''
    return [{'dirs': dd, 'out': '' if (out and os.path.isfile(out)) else out}] if dd or out \
        else [{'dirs': [], 'out': ''}]


# ---------- sub-batch tabs ----------

class SubBatchTab:
    """One sub-batch: independent input directory list + output folder (UI and data)."""

    def __init__(self, app, notebook, index, dirs=None, out=''):
        self.app = app
        self.dirs = []
        self.out_var = tk.StringVar(value=str(out or ''))
        self.frame = ttk.Frame(notebook)
        self.buttons = []  # uniformly disabled while running
        self._build(notebook)
        for d in (dirs or []):
            self.add_dir(d)
        self.update_count()

    def _build(self, notebook):
        pad = dict(padx=6, pady=4)
        f = self.frame
        ttk.Label(f, text='Input dirs').grid(row=0, column=0, sticky='ne', **pad)
        self.listbox = tk.Listbox(f, height=4, exportselection=False)
        self.listbox.grid(row=0, column=1, sticky='we', **pad)
        btns = ttk.Frame(f)
        btns.grid(row=0, column=2, sticky='w', **pad)
        for text, cmd in (('Add dir...', self.pick_dir), ('Remove selected', self.remove_selected),
                          ('Clear', self.clear)):
            b = ttk.Button(btns, text=text, command=cmd)
            b.pack(fill='x')
            self.buttons.append(b)
        self.count_var = tk.StringVar(value='(empty)')
        ttk.Label(f, textvariable=self.count_var, foreground='#666',
                  wraplength=110, justify='left').grid(row=0, column=3, sticky='nw', **pad)
        ttk.Label(f, text='Output folder').grid(row=1, column=0, sticky='e', **pad)
        self.out_entry = ttk.Entry(f, textvariable=self.out_var)
        self.out_entry.grid(row=1, column=1, sticky='we', **pad)
        b = ttk.Button(f, text='Browse...', command=self.pick_out)
        b.grid(row=1, column=2, sticky='w', **pad)
        self.buttons.append(b)
        f.columnconfigure(1, weight=1)
        self.out_var.trace_add('write', lambda *a: self.app._refresh_cmd())

    # -- directory management --
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
            self.app.status_var.set(f'Added {os.path.basename(d)}')
        self.update_count()
        self.app._refresh_cmd()

    def pick_out(self):
        d = filedialog.askdirectory(initialdir=self.out_var.get() or BASE)
        if d:
            self.out_var.set(d)

    def update_count(self):
        if not self.dirs:
            self.count_var.set('(empty)')
            return
        n = sum(count_images(d) for d in self.dirs if os.path.isdir(d))
        self.count_var.set(f'{len(self.dirs)} dirs\n{n} images')

    def set_locked(self, locked):
        state = 'disabled' if locked else 'normal'
        for b in self.buttons:
            b.config(state=state)
        self.out_entry.config(state=state)
        self.listbox.config(state='disabled' if locked else 'normal')

    def title(self, index):
        return os.path.basename(self.dirs[0]) if self.dirs else f'Batch {index + 1}'


# ---------- main window ----------

class App:
    def __init__(self, root):
        self.root = root
        root.title('MIRROR Inference Frontend - predict.py')
        root.geometry('880x760')
        try:
            self.cfg = load_config()
        except Exception as e:
            bak = os.path.join(BASE, 'gui_config.json.bak')
            tip = 'Rename gui_config.json.bak back to gui_config.json and restart to recover.' \
                if os.path.exists(bak) else ''
            messagebox.showerror('Config error', f'Failed to read gui_config.json: {e}\n{tip}')
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

    # ---- UI construction ----
    def _build(self):
        pad = dict(padx=6, pady=4)

        # sub-batch tab area
        batchf = ttk.LabelFrame(self.root, text='Sub-batches (each tab has its own input/output; runs sequentially, isolated)')
        batchf.pack(fill='x', **pad)
        tabbar = ttk.Frame(batchf)
        tabbar.pack(fill='x', **pad)
        self.notebook = ttk.Notebook(tabbar)
        self.notebook.pack(side='left', fill='x', expand=True)
        self.notebook.bind('<<NotebookTabChanged>>', lambda e: self._refresh_cmd())
        addb = ttk.Button(tabbar, text='+ Add batch', command=self.add_tab)
        addb.pack(side='left', padx=4)
        closeb = ttk.Button(tabbar, text='x Close current tab', command=self.close_tab)
        closeb.pack(side='left', padx=4)
        self.tab_buttons = [addb, closeb]

        # global parameter area (the hover-hint row stays out of the grid to prevent column-width flicker)
        self.form = ttk.LabelFrame(self.root, text='Parameters (global, generated from gui_config.json)')
        self.form.pack(fill='x', **pad)
        self.hint_var = tk.StringVar(value='Hover over a parameter to see its description')
        tk.Label(self.root, textvariable=self.hint_var, fg='#0a6',
                 wraplength=840, height=2, anchor='nw', justify='left').pack(
            fill='x', padx=12)
        self._build_group(self.cfg['params'], start_row=0)
        self.adv = ttk.LabelFrame(self.root, text='Advanced (model paths / device)')
        self.adv.pack(fill='x', **pad)
        self._build_group(self.cfg['advanced'], start_row=0, parent=self.adv)

        ttk.Label(self.root, text='Equivalent command line (current tab):').pack(anchor='w', padx=10)
        self.cmd_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.cmd_var, foreground='#333',
                  wraplength=850, font=('Consolas', 8)).pack(anchor='w', padx=12)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill='x', **pad)
        self.btn_start = ttk.Button(ctrl, text='Start inference', command=self.start_run)
        self.btn_start.pack(side='left', padx=6)
        self.btn_stop = ttk.Button(ctrl, text='Stop', command=self.stop_run, state='disabled')
        self.btn_stop.pack(side='left', padx=6)
        self.progress = ttk.Progressbar(ctrl, length=280, maximum=1, value=0)
        self.progress.pack(side='left', padx=10)
        self.btn_open = ttk.Button(ctrl, text='Open output folder', command=self.open_result, state='disabled')
        self.btn_open.pack(side='right', padx=6)
        self.btn_reset = ttk.Button(ctrl, text='Reset defaults', command=self.reset_defaults)
        self.btn_reset.pack(side='right', padx=6)

        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(self.root, textvariable=self.status_var, foreground='#036').pack(anchor='w', padx=10)

        logf = ttk.LabelFrame(self.root, text='Log (raw subprocess output)')
        logf.pack(fill='both', expand=True, **pad)
        self.logbox = tk.Text(logf, height=12, font=('Consolas', 9), state='disabled',
                              wrap='none')
        self.logbox.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(logf, command=self.logbox.yview)
        sb.pack(side='right', fill='y')
        self.logbox.config(yscrollcommand=sb.set)

    # ---- hover hints ----
    def _bind_hint(self, widget, p):
        """Bind Enter/Leave separately (never test e.type=='10' - comparing the EventType enum with a string is always False)."""
        widget.bind('<Enter>', lambda e, p=p: self._show_hint(p))
        widget.bind('<Leave>', lambda e: self._clear_hint())

    def _show_hint(self, p):
        self.hint_var.set(f"{p['label']} ({p['flag']}): {p.get('help', '')}")

    def _clear_hint(self):
        self.hint_var.set('Hover over a parameter to see its description')

    def _build_group(self, params, start_row=0, parent=None):
        parent = parent or self.form
        n = 0  # compact layout counter (--data_dir/--output_json are managed by tabs, not in the form)
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

    # ---- tab management ----
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
            self.status_var.set('At least one sub-batch must remain')
            return
        tab = self.current_tab()
        if (tab.dirs or tab.out_var.get()) and not messagebox.askyesno(
                'Close sub-batch', f'Close "{tab.title(0)}"? Its directory list and output settings will be discarded.'):
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
            return  # tab not registered yet during add_tab construction; the title is set when add_tab finishes
        i = self._tabs.index(tab)
        self.notebook.tab(tab.frame, text=tab.title(i))

    # ---- value load/save ----
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
                continue  # tab data (not a form field)
            v = self.vars[flag]
            if w == 'bool_path':
                vals[flag] = [bool(v[0].get()), v[1].get()]
            elif isinstance(v, tk.BooleanVar):
                vals[flag] = bool(v.get())
            else:
                vals[flag] = v.get()
        return vals

    def _vals_for_tab(self, tab, snap=None):
        """Assemble one sub-batch's command values: form parameters (or snapshot) + this tab's dirs and output.
        Output is always a folder (predict.py's --output_dir works for any number of dirs);
        empty -> results_gui/batch_<start time>/batch_i."""
        vals = dict(snap) if snap else self._collect_form_values()
        vals['--data_dir'] = list(tab.dirs)
        vals['--output_json'] = ''
        if not tab.out_var.get().strip():
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            vals['--output_dir'] = os.path.join(BASE, 'results_gui',
                                                f'batch_{ts}', f'batch_{self._tabs.index(tab) + 1}')
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

    # ---- multi-select directory dialog (used by tabs) ----
    @staticmethod
    def _list_subdirs(d):
        return [os.path.join(d, n) for n in sorted(os.listdir(d))
                if os.path.isdir(os.path.join(d, n))]

    def _pick_from_subs(self, tab, parent, subs):
        win = tk.Toplevel(self.root)
        win.title('Select folders to add')
        win.transient(self.root)
        win.grab_set()
        ttk.Label(win, text=f'"{parent}" has {len(subs)} subfolders.\n'
                            'Hold Ctrl (click) / Shift (range) to multi-select.').pack(
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
                self.status_var.set(f'Added {added} (skipped {skipped} duplicates)')
                self._refresh_cmd()

        btns = ttk.Frame(win)
        btns.pack(fill='x', padx=10, pady=10)
        ttk.Button(btns, text='Add selected subfolders', command=confirm).pack(side='left', padx=(0, 6))
        ttk.Button(btns, text='Cancel', command=win.destroy).pack(side='right')
        win.wait_window()

    def _browse(self, var, kind):
        f = (filedialog.askopenfilename if kind == 'file' else filedialog.askdirectory)()
        if f:
            var.set(f)

    # ---- preflight and run ----
    def preflight_tab(self, tab):
        probs = []
        if not tab.dirs:
            probs.append('(empty sub-batch, will be skipped)')
        for d in tab.dirs:
            if not os.path.isdir(d):
                probs.append(f'directory not found: {d}')
            elif count_images(d) == 0:
                probs.append(f'no recognizable images (png/jpg/jpeg/bmp/webp) in: {d}')
        conf = find_name_conflicts(tab.dirs)
        if conf:
            lines = '\n'.join(f'- {b} ({len(ds)} folders with the same name)' for b, ds in conf.items())
            if not messagebox.askyesno(
                    'Duplicate folder names',
                    f'Sub-batch "{tab.title(0)}" contains folders with identical names; output JSONs will get'
                    f' parent-directory prefixes to stay unique (as parent__folder.json):\n\n' + lines + '\n\nContinue?'):
                probs.append('cancelled (duplicate folder names)')
        out = tab.out_var.get().strip()
        if out and os.path.isfile(out):
            probs.append(f'output folder points to a file: {out}')
        return probs

    def start_run(self):
        if self.running:
            return
        all_probs = []
        for tab in self._tabs:
            probs = self.preflight_tab(tab)
            if probs and probs != ['(empty sub-batch, will be skipped)']:
                all_probs.append(f'[{tab.title(0)}] ' + '; '.join(probs))
        outs = [t.out_var.get().strip() for t in self._tabs if t.dirs]
        dups = find_out_conflicts(outs)
        if dups:
            if not messagebox.askyesno(
                    'Output folder conflict',
                    'The following output folders are used by multiple sub-batches; later runs overwrite'
                    ' the earlier runs JSONs:\n\n'
                    + '\n'.join(dups) + '\n\nContinue?'):
                return
        if all_probs:
            messagebox.showerror('Preflight failed', '\n'.join(all_probs))
            return
        if not any(t.dirs for t in self._tabs):
            messagebox.showerror('Cannot start', 'All sub-batches are empty.')
            return
        self.snap_params = self._collect_form_values()  # snapshot: form edits during a run have no effect
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
        self._log(f"\n### Sub-batch {self.cur_index}/{self.n_tabs}: "
                  f"{self.cur_tab.title(self._tabs.index(self.cur_tab))}"
                  f" ({len(self.cur_tab.dirs)} dirs -> {self.cur_out})\n")
        self._log(f'$ {" ".join(cmd)}\n')
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception as e:
            self._log(f'[GUI] failed to start: {e}\n')
            self.tab_results.append((self.cur_tab.title(0), 'start failed'))
            self.root.after(300, self._start_next_tab)
            return
        self.status_var.set(f'Sub-batch {self.cur_index}/{self.n_tabs} started...')
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(('line', line.rstrip('\n')))
        self.q.put(('done', proc.wait()))

    def stop_run(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno(
                    'Stop', 'Stop the current sub-batch and cancel the remaining ones?\n'
                            '(JSONs of finished sub-batches are already saved)'):
                return
            self.stop_requested = True
            self.proc.terminate()
            self._log('[GUI] stop requested...\n')
            self.root.after(3000, self._ensure_killed)

    def _ensure_killed(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
                self._log('[GUI] process unresponsive, force-killed\n')
            except Exception:
                pass

    def open_result(self):
        target = self.last_out_dir
        if target and os.path.exists(target):
            os.startfile(target)

    def reset_defaults(self):
        if self.running:
            return
        if messagebox.askyesno('Reset defaults', 'Discard current parameter changes (sub-batches and model paths are kept)?'):
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

    # ---- runtime polling (tkinter single-thread rule: worker threads only enqueue) ----
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
        parts = [f'Sub-batch {self.cur_index}/{self.n_tabs} {self.cur_tab.title(0)}']
        if s.get('sec_n'):
            parts.append(f'folder {s.get("sec_k", "?")}/{s["sec_n"]}')
        if s.get('total'):
            parts.append(f'batch {s.get("batch", 0)}/{s["total"]}')
        parts.append(f'elapsed {int(time.time() - self.t0)} s')
        return ' · '.join(parts)

    def _on_line(self, line):
        self._log(line + '\n')
        parse_line(line, self.state)
        if self.state.get('total'):
            self.progress.config(maximum=self.state['total'],
                                 value=self.state.get('batch', 0))
        else:
            self.progress.config(value=0)
        if '[warn]' in line:
            self.status_var.set('read warnings, see log · ' + self._status_text())

    def _on_proc_done(self, rc):
        s = self.state
        title = self.cur_tab.title(0)
        if rc == 0:
            folders = s.get('folders', [])
            total_imgs = sum(int(f['n']) for f in folders)
            total_fake = sum(int(f['fake'] or 0) for f in folders)
            msg = (f'{len(folders)} folders / {total_imgs} images / fake {total_fake}')
            if s.get('folder_fails'):
                msg += f' / {s["folder_fails"]} folder failures'
            self.tab_results.append((title, msg))
            self._log(f'[GUI] sub-batch "{title}" done: {msg}, return code 0\n')
        else:
            self.tab_results.append((title, f'failed (return code {rc})'))
            self._log(f'[GUI] sub-batch "{title}" failed, return code {rc}'
                      f' (remaining sub-batches continue in a fresh process)\n')
        self.proc = None
        self.root.after(400, self._start_next_tab)  # brief pause before the next process, giving the driver room

    def _finish_run(self):
        sec = int(time.time() - self.t0)
        ok = sum(1 for _, m in self.tab_results if not m.startswith('failed'))
        lines = [f'[{t}] {m}' for t, m in self.tab_results]
        self._log('[GUI] ===== Run summary =====\n' + '\n'.join(lines) + '\n')
        self.status_var.set(f'Finished: {ok}/{len(self.tab_results)} sub-batches ok'
                            f' · {sec} s (details in log)')
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

    # ---- quit and persistence ----
    def _save_values(self):
        """Write back only values (form parameters) / tabs / ui; the registry is re-read from disk (prevents clobbering manual edits)."""
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
            pass  # fall back to the in-memory version when the disk file is unavailable
        tmp = CONFIG_PATH + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.cfg, f, ensure_ascii=False, indent=1)
            if os.path.exists(CONFIG_PATH):
                os.replace(CONFIG_PATH, CONFIG_PATH + '.bak')  # keep the previous version for manual recovery
            os.replace(tmp, CONFIG_PATH)
        except Exception as e:
            self._log(f'[GUI] failed to write config: {e}\n')
            try:
                os.remove(tmp)
            except OSError:
                pass

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno('Quit', 'Inference still running. Stop and quit?\n'
                                              '(JSONs of finished sub-batches are already saved)'):
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
            tab = SubBatchTab.__new__(SubBatchTab)  # UI-less lightweight construction, only _vals logic
            tab.dirs = list(spec.get('dirs', []))
            tab.out_var = type('V', (), {'get': lambda s, v=spec.get('out', ''): v})()
            tab.title = lambda idx, i=i: (os.path.basename(spec['dirs'][0])
                                          if spec.get('dirs') else f'Batch {i + 1}')

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
            print(f'# Sub-batch {i + 1}/{len(tabs)}: {tab.title(i)}')
            print(' '.join(build_command(vals, cfg['params'] + cfg['advanced'])))
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
