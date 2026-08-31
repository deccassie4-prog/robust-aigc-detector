"""pred_filter_gui.py - collect images by pred range into an archive folder
(error-analysis picker tool)

Usage: pick one or more predict.py output JSONs (competition / detailed formats are
auto-detected); each JSON gets a tab where you set the image directory and an optional
score-range override; the status line live-updates "N matched - M files found".
"Start copy" creates a folder per JSON under the output directory, copies the in-range
images into it renamed to their pred value (duplicate names get _2/_3 suffixes), and
writes manifest.csv (new name, pred, json path, copied path).

Lookup semantics: matching is by FILE NAME only, never by path - the image_path in the
JSON is reduced to its basename and located via a recursive index of the chosen image
directory, so images are still found after the dataset moved (another disk / machine).
If a name hits several files, the first in sort order is taken and reported.

Zero third-party dependencies (tkinter/json/threading/queue/csv are stdlib).
Copying runs in a worker thread; the UI stays responsive. Stop = finish the current
file, already-copied files are kept.
Global settings persist in pred_filter_config.json (atomic write, .bak backup).
"""
import csv
import json
import os
import queue
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, 'pred_filter_config.json')


# ---------- pure functions (unit-testable without the GUI) ----------

def load_entries(path):
    """Read a JSON -> (entries, fmt, n_bad). entries=[(image_path, pred), ...].
    competition = top-level list [{"image_path","pred"}]; detailed = {"images":[...]}.
    Entries with missing fields / non-numeric pred are skipped and counted."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get('images'), list):
        raw, fmt = data['images'], 'detailed'
    elif isinstance(data, list):
        raw, fmt = data, 'competition'
    else:
        raise ValueError('unrecognized format (expected a list or a dict with an images key)')
    entries, bad = [], 0
    for it in raw:
        try:
            p = str(it['image_path'])
            pred = float(it['pred'])
        except (TypeError, KeyError, ValueError):
            bad += 1
            continue
        entries.append((p, pred))
    return entries, fmt, bad


def in_range(pred, lo, hi):
    """Closed interval [lo, hi]; None on either side = unbounded. Failed items (pred=-1) are naturally excluded when min >= 0."""
    if lo is not None and pred < lo:
        return False
    if hi is not None and pred > hi:
        return False
    return True


def pred_label(pred):
    """pred -> file-name stem: fixed 6 decimals with trailing zeros stripped (0.951053->'0.951053',
    0.5->'0.5', 1.0->'1'); avoids str() scientific notation for tiny values ('1e-05' cannot be a file stem)."""
    s = f'{pred:.6f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-') else '0'


def unique_name(stem, ext, taken):
    """Return a non-conflicting file name. taken = set of used names (lowercase); collisions append _2/_3..."""
    cand = stem + ext
    i = 2
    while cand.lower() in taken:
        cand = f'{stem}_{i}{ext}'
        i += 1
    taken.add(cand.lower())
    return cand


def build_index(root):
    """Index an image directory recursively: {lower-case name: [full paths...]}. Basis of the name-only lookup."""
    idx = {}
    for rd, _, fs in os.walk(root):
        for f in fs:
            idx.setdefault(f.lower(), []).append(os.path.join(rd, f))
    return idx


def resolve_by_name(img_path, index):
    """Resolve by file name -> (actual path or None, hit count). Multiple hits: first in sort order."""
    hits = index.get(os.path.basename(img_path).lower())
    if not hits:
        return None, 0
    if len(hits) == 1:
        return hits[0], 1
    return sorted(hits)[0], len(hits)


def dedup_folder_names(paths):
    """JSON path -> output folder name. Same names get a parent-dir prefix (walks up if parents collide too)."""
    base = {p: os.path.splitext(os.path.basename(p))[0] for p in paths}
    levels = {p: 0 for p in paths}

    def name_for(p, k):
        cur = os.path.dirname(os.path.normpath(p))
        parts = []
        for _ in range(k):
            parts.insert(0, os.path.basename(cur) or '?')
            cur = os.path.dirname(cur)
        return '__'.join(parts + [base[p]])

    names = {p: base[p] for p in paths}
    for _ in range(6):
        by = {}
        for p, n in names.items():
            by.setdefault(n.lower(), []).append(p)
        groups = [g for g in by.values() if len(g) > 1]
        if not groups:
            break
        for g in groups:
            for p in g:
                levels[p] += 1
                names[p] = name_for(p, levels[p])
    return names


# ---------- tab ----------

class FilterTab:
    """One JSON tab: image directory + optional range override (empty side = global) + live stats."""

    def __init__(self, app, notebook, json_path, entries, fmt, n_bad):
        self.app = app
        self.json_path = json_path
        self.entries = entries
        self.fmt = fmt
        self.n_bad = n_bad
        self.index = None          # {lower-case name: [full paths...]}
        self._index_dirty = True
        self._idx_after = None
        self.found_cache = {}      # name -> hit count (0=not found), reused across range tweaks
        self._matched = []
        self.dir_var = tk.StringVar()
        self.min_var = tk.StringVar()
        self.max_var = tk.StringVar()
        self.stat_var = tk.StringVar(value='')
        self.frame = ttk.Frame(notebook)
        self.widgets = []          # uniformly disabled while running
        self._build(notebook)
        self.dir_var.trace_add('write', lambda *a: self._on_dir_change())
        self.min_var.trace_add('write', lambda *a: self.app.refresh_all())
        self.max_var.trace_add('write', lambda *a: self.app.refresh_all())

    @property
    def title(self):
        return os.path.splitext(os.path.basename(self.json_path))[0]

    def _build(self, notebook):
        pad = dict(padx=6, pady=4)
        f = self.frame
        ttk.Label(f, text='Image dir').grid(row=0, column=0, sticky='e', **pad)
        fr = ttk.Frame(f)
        fr.grid(row=0, column=1, sticky='we', **pad)
        de = ttk.Entry(fr, textvariable=self.dir_var)
        de.pack(side='left', fill='x', expand=True)
        db = ttk.Button(fr, text='Browse...', command=self.pick_dir)
        db.pack(side='left', padx=4)
        self.widgets += [de, db]
        info = f'{os.path.basename(self.json_path)} - {len(self.entries):,} entries ({self.fmt})'
        if self.n_bad:
            info += f' - {self.n_bad} invalid skipped'
        ttk.Label(f, text=info, foreground='#666').grid(row=0, column=2, sticky='w', **pad)

        ttk.Label(f, text='Range override').grid(row=1, column=0, sticky='e', **pad)
        rf = ttk.Frame(f)
        rf.grid(row=1, column=1, sticky='w', **pad)
        ttk.Label(rf, text='min').pack(side='left')
        me = ttk.Entry(rf, textvariable=self.min_var, width=9)
        me.pack(side='left', padx=(2, 10))
        ttk.Label(rf, text='max').pack(side='left')
        xe = ttk.Entry(rf, textvariable=self.max_var, width=9)
        xe.pack(side='left', padx=2)
        self.widgets += [me, xe]
        ttk.Label(f, text='empty = use global value', foreground='#999').grid(
            row=1, column=2, sticky='w', **pad)

        st = tk.Label(f, textvariable=self.stat_var, fg='#0a6', wraplength=560,
                      anchor='w', justify='left')
        st.grid(row=2, column=0, columnspan=3, sticky='we', padx=10, pady=(0, 6))
        f.columnconfigure(1, weight=1)

    # -- directory --
    def pick_dir(self):
        d = filedialog.askdirectory(
            initialdir=self.app.cfg.get('last_dir') or BASE)
        if d:
            self.app.cfg['last_dir'] = d
            self.dir_var.set(os.path.normpath(d))

    def _on_dir_change(self):
        self._index_dirty = True
        self.index = None
        self.found_cache.clear()
        if self._idx_after is not None:
            self.app.after_cancel(self._idx_after)
            self._idx_after = None
        self.app.refresh_all()

    def build_index_now(self):
        d = self.dir_var.get().strip()
        self.index = build_index(d) if d and os.path.isdir(d) else {}
        self._index_dirty = False

    def _build_and_count(self):
        self._idx_after = None
        self.build_index_now()
        self.update_m()

    # -- range and stats --
    def effective_range(self):
        """(lo, hi, err): an empty side falls back to the global value; invalid text or min>max returns err."""
        try:
            s = self.min_var.get().strip()
            lo = float(s) if s else None
        except ValueError:
            return None, None, 'override min is not a number'
        try:
            s = self.max_var.get().strip()
            hi = float(s) if s else None
        except ValueError:
            return None, None, 'override max is not a number'
        gmin, gmax, gerr = self.app.glob_range()
        if gerr:
            return None, None, gerr
        lo = gmin if lo is None else lo
        hi = gmax if hi is None else hi
        if lo is not None and hi is not None and lo > hi:
            return None, None, 'min is greater than max'
        return lo, hi, None

    def matched(self, lo, hi):
        return [(p, pr) for p, pr in self.entries if in_range(pr, lo, hi)]

    def update_n(self):
        lo, hi, err = self.effective_range()
        if err:
            self._matched = []
            self.stat_var.set(err)
            return
        self._matched = self.matched(lo, hi)

    def update_m(self):
        if not self._matched and not self.entries:
            return
        n = len(self._matched)
        d = self.dir_var.get().strip()
        if not d:
            self.stat_var.set(f'{n} matched · no image dir selected')
            return
        if not os.path.isdir(d):
            self.stat_var.set(f'{n} matched · directory not found')
            return
        if self._index_dirty:
            self.stat_var.set(f'{n} matched · indexing directory...')
            if self._idx_after is None:
                self._idx_after = self.app.after(250, self._build_and_count)
            return
        m, multi = self._count_found()
        txt = f'{n} matched · found {m} files'
        if multi:
            txt += f' ({multi} names hit multiple files, first taken)'
        self.stat_var.set(txt)

    def _count_found(self):
        m = multi = 0
        for p, _ in self._matched:
            name = os.path.basename(p).lower()
            hit = self.found_cache.get(name)
            if hit is None:
                hit = len(self.index.get(name, ()))
                self.found_cache[name] = hit
            if hit:
                m += 1
                if hit > 1:
                    multi += 1
        return m, multi

    def resolved(self):
        """[(json path, pred, actual path or None)]; same resolution as the stats line."""
        out = []
        for p, pred in self._matched:
            src, _ = resolve_by_name(p, self.index)
            out.append((p, pred, src))
        return out

    def set_locked(self, locked):
        for w in self.widgets:
            w.config(state='disabled' if locked else 'normal')


# ---------- main window ----------

class App:
    def __init__(self, root):
        self.root = root
        root.title('Collect images by pred range - pred_filter')
        root.geometry('900x660')
        self.cfg = self._load_config()
        self._tabs = []
        self._loaded = set()       # loaded JSONs (normcase), guards against duplicate imports
        self.q = queue.Queue()
        self.running = False
        self.stop_flag = False
        self._m_after = None
        self._worker = None
        self.t0 = time.time()
        self._build()
        self._restore_config()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.root.after(120, self._poll)

    # ---- config ----
    @staticmethod
    def _load_config():
        cfg = {'min': '0', 'max': '1', 'out_dir': '', 'last_dir': ''}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, encoding='utf-8') as f:
                    cfg.update(json.load(f))
            except Exception:
                pass
        return cfg

    def _save_config(self):
        try:
            self.cfg.update({'min': self.gmin_var.get(),
                             'max': self.gmax_var.get(),
                             'out_dir': self.out_var.get()})
        except Exception:
            pass
        tmp = CONFIG_PATH + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.cfg, f, ensure_ascii=False, indent=1)
            if os.path.exists(CONFIG_PATH):
                os.replace(CONFIG_PATH, CONFIG_PATH + '.bak')
            os.replace(tmp, CONFIG_PATH)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _restore_config(self):
        self.gmin_var.set(str(self.cfg.get('min', '0')))
        self.gmax_var.set(str(self.cfg.get('max', '1')))
        self.out_var.set(str(self.cfg.get('out_dir', '')))

    # ---- UI ----
    def _build(self):
        pad = dict(padx=6, pady=4)
        top = ttk.Frame(self.root)
        top.pack(fill='x', **pad)
        self.btn_select = ttk.Button(top, text='Select JSON... (multi-select)',
                                     command=self.pick_jsons)
        self.btn_select.pack(side='left', padx=4)
        self.btn_close_tab = ttk.Button(top, text='x Close current tab', command=self.close_tab)
        self.btn_close_tab.pack(side='left', padx=4)

        tf = ttk.LabelFrame(self.root, text='One tab per JSON (pick the image dir, optional range override)')
        tf.pack(fill='x', **pad)
        self.notebook = ttk.Notebook(tf)
        self.notebook.pack(fill='x', **pad)

        gf = ttk.LabelFrame(self.root, text='Global settings (used when a tab leaves a field empty)')
        gf.pack(fill='x', **pad)
        row = ttk.Frame(gf)
        row.pack(fill='x', **pad)
        ttk.Label(row, text='min').pack(side='left')
        self.gmin_var = tk.StringVar(value='0')
        ge1 = ttk.Entry(row, textvariable=self.gmin_var, width=9)
        ge1.pack(side='left', padx=(2, 12))
        ttk.Label(row, text='max').pack(side='left')
        self.gmax_var = tk.StringVar(value='1')
        ge2 = ttk.Entry(row, textvariable=self.gmax_var, width=9)
        ge2.pack(side='left', padx=(2, 12))
        ttk.Label(row, text='Output dir').pack(side='left')
        self.out_var = tk.StringVar()
        oe = ttk.Entry(row, textvariable=self.out_var)
        oe.pack(side='left', fill='x', expand=True, padx=2)
        ob = ttk.Button(row, text='Browse...', command=self._pick_out)
        ob.pack(side='left', padx=4)
        self.lockable = [ge1, ge2, oe, ob]
        self.gmin_var.trace_add('write', lambda *a: self.refresh_all())
        self.gmax_var.trace_add('write', lambda *a: self.refresh_all())

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill='x', **pad)
        self.btn_start = ttk.Button(ctrl, text='Start copy', command=self.start_run)
        self.btn_start.pack(side='left', padx=6)
        self.btn_stop = ttk.Button(ctrl, text='Stop', command=self.stop_run,
                                   state='disabled')
        self.btn_stop.pack(side='left', padx=6)
        self.progress = ttk.Progressbar(ctrl, length=260, maximum=1, value=0)
        self.progress.pack(side='left', padx=10)
        self.btn_open = ttk.Button(ctrl, text='Open output dir', command=self.open_out,
                                   state='disabled')
        self.btn_open.pack(side='right', padx=6)

        self.status_var = tk.StringVar(value='Ready: select JSON files first')
        ttk.Label(self.root, textvariable=self.status_var, foreground='#036').pack(
            anchor='w', padx=10)

        lf = ttk.LabelFrame(self.root, text='Log')
        lf.pack(fill='both', expand=True, **pad)
        self.logbox = tk.Text(lf, height=9, font=('Consolas', 9), state='disabled',
                              wrap='none')
        self.logbox.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(lf, command=self.logbox.yview)
        sb.pack(side='right', fill='y')
        self.logbox.config(yscrollcommand=sb.set)

    def _pick_out(self):
        d = filedialog.askdirectory(initialdir=self.out_var.get() or
                                    self.cfg.get('last_dir') or BASE)
        if d:
            self.out_var.set(os.path.normpath(d))

    def open_out(self):
        d = self.out_var.get().strip()
        if d and os.path.isdir(d):
            os.startfile(d)

    # ---- tab management ----
    def pick_jsons(self):
        ps = filedialog.askopenfilenames(
            title='Select predict output JSONs (multi-select)',
            initialdir=self.cfg.get('last_dir') or BASE,
            filetypes=[('JSON', '*.json'), ('All files', '*.*')])
        if not ps:
            return
        self.cfg['last_dir'] = os.path.dirname(ps[0])
        self.add_json_tabs(ps)

    def add_json_tabs(self, paths):
        added = skipped = 0
        for p in paths:
            key = os.path.normcase(os.path.normpath(p))
            if key in self._loaded:
                skipped += 1
                continue
            try:
                entries, fmt, bad = load_entries(p)
            except Exception as e:
                messagebox.showerror('Read failed', f'{p}\n{e}')
                continue
            if not entries:
                messagebox.showwarning('Empty JSON', f'{p}\nNo valid entries, skipped.')
                continue
            self._loaded.add(key)
            tab = FilterTab(self, self.notebook, p, entries, fmt, bad)
            self._tabs.append(tab)
            self.notebook.add(tab.frame, text=self._title_for(tab))
            self.notebook.select(tab.frame)
            added += 1
        self.status_var.set(f'{added} JSONs loaded'
                            + (f' ({skipped} duplicates skipped)' if skipped else ''))
        self.refresh_all()

    def _title_for(self, tab):
        stem = tab.title
        titles = {self.notebook.tab(t.frame, 'text')
                  for t in self._tabs if t is not tab}
        if stem not in titles:
            return stem
        i = 2
        while f'{stem} ({i})' in titles:
            i += 1
        return f'{stem} ({i})'

    def close_tab(self):
        if self.running or not self._tabs:
            return
        i = max(0, self.notebook.index(self.notebook.select()))
        tab = self._tabs[i]
        if (tab.dir_var.get().strip() or tab.min_var.get().strip()
                or tab.max_var.get().strip()):
            if not messagebox.askyesno(
                    'Close tab', f'Close "{tab.title}"? Its settings will be discarded.'):
                return
        self.notebook.forget(tab.frame)
        self._tabs.remove(tab)
        self._loaded.discard(os.path.normcase(os.path.normpath(tab.json_path)))

    # ---- live stats ----
    def glob_range(self):
        try:
            s = self.gmin_var.get().strip()
            lo = float(s) if s else None
        except ValueError:
            return None, None, 'global min is not a number'
        try:
            s = self.gmax_var.get().strip()
            hi = float(s) if s else None
        except ValueError:
            return None, None, 'global max is not a number'
        return lo, hi, None

    def refresh_all(self, *_):
        for t in self._tabs:
            t.update_n()
        if self._m_after is not None:
            try:
                self.root.after_cancel(self._m_after)
            except Exception:
                pass
            self._m_after = None
        if self._tabs:
            self._m_after = self.root.after(250, self._update_all_m)

    def _update_all_m(self):
        self._m_after = None
        for t in self._tabs:
            t.update_m()

    def after(self, ms, fn):
        return self.root.after(ms, fn)

    def after_cancel(self, aid):
        try:
            self.root.after_cancel(aid)
        except Exception:
            pass

    # ---- preflight and run ----
    def start_run(self):
        if self.running:
            return
        out_dir = self.out_var.get().strip()
        if not out_dir:
            messagebox.showerror('Cannot start', 'Set the output directory first.')
            return
        if os.path.isfile(out_dir):
            messagebox.showerror('Cannot start', f'output dir points to a file: {out_dir}')
            return
        plans, probs, zeros = [], [], []
        for t in self._tabs:
            lo, hi, err = t.effective_range()
            if err:
                probs.append(f'[{t.title}] {err}')
                continue
            d = t.dir_var.get().strip()
            if not d or not os.path.isdir(d):
                probs.append(f'[{t.title}] image dir missing or not found')
                continue
            if not t.matched(lo, hi):
                zeros.append(t.title)
                continue
            if t._index_dirty:
                t.build_index_now()
            t._matched = t.matched(lo, hi)   # run snapshot: later form edits do not affect this run
            plans.append((t, t.resolved()))
        if probs:
            messagebox.showerror('Preflight failed', '\n'.join(probs))
        if not plans:
            return
        if zeros:
            self._log('[GUI] 0 matched, skipped: ' + ', '.join(zeros) + '\n')
        names = dedup_folder_names([t.json_path for t, _ in plans])
        renamed = [(t.title, names[t.json_path]) for t, _ in plans
                   if names[t.json_path] != t.title]
        if renamed:
            lines = '\n'.join(f'· {a} → {b}' for a, b in renamed)
            if not messagebox.askyesno(
                    'Duplicate output folder names',
                    'The following JSONs share a name; output folders get parent-dir prefixes:\n\n' + lines
                    + '\n\nContinue?'):
                return
        total = sum(len(rows) for _, rows in plans)
        if total == 0:
            messagebox.showerror('Cannot start', 'No locatable images within range in any tab.')
            return
        self._run(plans, names, out_dir, total)

    def _run(self, plans, names, out_dir, total):
        self.running = True
        self.stop_flag = False
        self.t0 = time.time()
        self._lock(True)
        self.progress.config(maximum=total, value=0)
        self.status_var.set('Copying...')
        self._log(f'[GUI] copying {len(plans)} tabs, {total} entries -> {out_dir}\n')
        self._worker = threading.Thread(
            target=self._copy_worker, args=(plans, names, out_dir, total),
            daemon=True)
        self._worker.start()

    def _lock(self, locked):
        st = 'disabled' if locked else 'normal'
        self.btn_select.config(state=st)
        self.btn_close_tab.config(state=st)
        self.btn_start.config(state=st)
        self.btn_stop.config(state='normal' if locked else 'disabled')
        for w in self.lockable:
            w.config(state=st)
        for t in self._tabs:
            t.set_locked(locked)

    def _copy_worker(self, plans, names, out_dir, total):
        done = 0
        t0 = time.time()
        for k, (tab, rows) in enumerate(plans, 1):
            if self.stop_flag:
                break
            folder = os.path.join(out_dir, names[tab.json_path])
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                self.q.put(('log', f'[fail] mkdir {folder}: {e}\n'))
                done += len(rows)
                continue
            self.q.put(('log', f'--- [{k}/{len(plans)}] {tab.title} → {folder}'
                        f' ({len(rows)} matched)\n'))
            copied = missed = renamed_n = failed = 0
            taken = set()
            try:
                mf = open(os.path.join(folder, 'manifest.csv'), 'w',
                          newline='', encoding='utf-8-sig')
            except Exception as e:
                self.q.put(('log', f'[fail] writing manifest.csv: {e}\n'))
                done += len(rows)
                continue
            with mf:
                w = csv.writer(mf)
                w.writerow(['new_name', 'pred', 'json_path', 'copied_path'])
                n_rows = len(rows)
                for i, (rec_path, pred, src) in enumerate(rows, 1):
                    if self.stop_flag:
                        break
                    done += 1
                    if src is None:
                        missed += 1
                    else:
                        ext = os.path.splitext(src)[1]
                        stem = pred_label(pred)
                        name = unique_name(stem, ext, taken)
                        if name.lower() != (stem + ext).lower():
                            renamed_n += 1
                        try:
                            shutil.copy2(src, os.path.join(folder, name))
                            w.writerow([name, pred, rec_path, src])
                            copied += 1
                        except Exception as e:
                            failed += 1
                            self.q.put(('log', f'[fail] {rec_path}: {e}\n'))
                    if i % 50 == 0 or i == n_rows:
                        self.q.put(('prog', (done, total,
                                             f'{k}/{len(plans)} {tab.title}: {i}/{n_rows}')))
            msg = f'copied {copied} · not found {missed} · renamed {renamed_n}'
            if failed:
                msg += f' · failed {failed}'
            if self.stop_flag:
                msg += ' (stopped)'
            self.q.put(('tab_done', f'[{tab.title}] {msg}'))
        self.q.put(('done', round(time.time() - t0)))

    def stop_run(self):
        self.stop_flag = True
        self._log('[GUI] stop requested: finishes the current file...\n')

    # ---- runtime polling (worker threads only enqueue) ----
    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == 'log':
                    self._log(payload)
                elif kind == 'prog':
                    done, total, text = payload
                    self.progress.config(maximum=total, value=done)
                    self.status_var.set(
                        f'{text} · elapsed {int(time.time() - self.t0)} s')
                elif kind == 'tab_done':
                    self._log('[GUI] ' + payload + '\n')
                elif kind == 'done':
                    self._finish_run(payload)
        except queue.Empty:
            pass
        try:
            self.root.after(120, self._poll)
        except Exception:
            pass  # window already destroyed (test teardown)

    def _finish_run(self, sec):
        self.running = False
        self._lock(False)
        tail = ' (stopped, partial)' if self.stop_flag else ''
        self._log(f'[GUI] ===== copy finished, {sec} s{tail} =====\n')
        self.status_var.set(f'Copy finished in {sec} s{tail} (details in log)')
        if os.path.isdir(self.out_var.get().strip()):
            self.btn_open.config(state='normal')
        self._save_config()

    def _log(self, text):
        self.logbox.config(state='normal')
        self.logbox.insert('end', text)
        self.logbox.see('end')
        self.logbox.config(state='disabled')

    # ---- quit ----
    def on_close(self):
        if self.running:
            if not messagebox.askyesno(
                    'Quit', 'Copying still in progress. Stop and quit?\n(Copied files are kept)'):
                return
            self.stop_flag = True
            self._close_t0 = time.time()
            self._wait_close()
            return
        self._save_config()
        self.root.destroy()

    def _wait_close(self):
        alive = (self._worker is not None and self._worker.is_alive()
                 and time.time() - self._close_t0 < 5)
        if alive:
            self.root.after(150, self._wait_close)
            return
        self._save_config()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
