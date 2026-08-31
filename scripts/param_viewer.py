"""param_viewer.py — Model Parameter Viewer (demo tool)

Pick a checkpoint file (.pth/.pt/.bin/.ckpt/.safetensors) or a weight
directory (HF format) and the tool shows: total parameter count, the <2B
compliance verdict, a composition bar chart grouped by top-level name
segment, and the largest tensors. Pure file-based counting — no model is
instantiated (state_dict basis, includes persistent buffers).

Checkpoint structures (probed 2026-08-30):
- checkpoint-h-cur.pth → state dict under the 'model' key ('optimizer' holds
  no tensors; 'args' is a Namespace)
- mirror_phase1.pth    → under 'model_state_dict' ('optimizer_state_dict'
  contains nested Adam moment tensors — failing to recognize the key would
  double-count them)
- dinov3-huge/         → safetensors; parsed with a pure-stdlib header read
  (8-byte little-endian length + JSON), zero tensor loading
.pth load chain: torch.load(mmap=True, weights_only=True) →
weights_only=False → mmap=False. The mmap path only reads tensor metadata
(numel/shape/dtype), so a 3.45 GB file counts in ~0.2 s.

Counting runs in a worker thread; queue events are always 2-tuples
(kind, payload). Config only persists last_dir (atomic write, .bak kept).
Closing the window while a load thread runs is fine (daemon thread).
"""
import json
import math
import os
import queue
import struct
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, 'param_viewer_config.json')

LIMIT = 2_000_000_000          # challenge hard limit: <2B parameters
PRIORITY_KEYS = ('model_state_dict', 'model', 'state_dict', 'module', 'net')
EXCLUDE_KEYS = ('optimizer', 'optimizer_state_dict', 'param_groups', 'scaler',
                'amp', 'scheduler', 'args', 'opt')
WEIGHT_EXTS = ('.safetensors', '.pth', '.pt', '.bin', '.ckpt')
ST_DTYPE = {'F64': 'float64', 'F32': 'float32', 'F16': 'float16',
            'BF16': 'bfloat16', 'I64': 'int64', 'I32': 'int32', 'I16': 'int16',
            'I8': 'int8', 'U8': 'uint8', 'BOOL': 'bool'}

BG = '#f4f6f8'
CARD = '#ffffff'
BORDER = '#e2e6ea'
FG = '#1f2937'
MUT = '#6b7280'
ACC = '#2563eb'
OK = '#0a7d43'
BAD = '#c0392b'
CHIP_BG = '#e8eefc'
BAR_PALETTE = ('#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe')
F_UI = ('Microsoft YaHei UI', 10)
F_UI_S = ('Microsoft YaHei UI', 9)
F_UI_B = ('Microsoft YaHei UI', 10, 'bold')
F_TITLE = ('Microsoft YaHei UI', 15, 'bold')
F_NUM = ('Consolas', 26, 'bold')
F_NUM_S = ('Consolas', 10)


# ---------- Pure functions (unit-testable without GUI) ----------

def fmt_int(n):
    return f'{n:,}'


def fmt_units(n):
    if n >= 1e8:   # use B above 0.1B (855,964,616 → 0.856B, matches docs)
        return f'{n / 1e9:.3f}B'.replace('.000B', 'B')
    if n >= 1e6:
        return f'{n / 1e6:.2f}M'
    if n >= 1e3:
        return f'{n / 1e3:.1f}K'
    return str(n)


def fmt_size(b):
    if b >= 2**30:
        return f'{b / 2**30:.2f} GB'
    if b >= 2**20:
        return f'{b / 2**20:.1f} MB'
    return f'{b / 2**10:.1f} KB'


def _is_tensor_dict(d):
    return (isinstance(d, dict) and bool(d)
            and all(isinstance(v, torch.Tensor) for v in d.values()))


def pick_state_dict(obj):
    """Locate the model weight tensor dict inside an arbitrary checkpoint
    object → (dict, note). Three-tier strategy: bare tensor dict →
    recognized key → whole-dict exclusion of non-model entries."""
    if _is_tensor_dict(obj):
        return dict(obj), 'file is a bare state dict (plain tensor dict)'
    if not isinstance(obj, dict):
        raise ValueError(f'top level is {type(obj).__name__}, not a weight dict')
    for k in PRIORITY_KEYS:
        v = obj.get(k)
        if _is_tensor_dict(v):
            skipped = [str(x) for x in obj if x != k]
            note = f'state dict under "{k}" key'
            if skipped:
                note += ' (skipped: ' + ', '.join(skipped) + ')'
            return dict(v), note
    sd, skipped = {}, []
    for k, v in obj.items():
        ks = str(k).lower()
        if isinstance(v, torch.Tensor):
            sd[k] = v
        elif any(ks == e or ks.startswith(e) for e in EXCLUDE_KEYS):
            skipped.append(str(k))
    if sd:
        note = 'no recognized key; counted the whole dict'
        if skipped:
            note += ' (skipped: ' + ', '.join(skipped) + ')'
        return sd, note
    for k, v in obj.items():      # last resort: any all-tensor sub-dict
        if _is_tensor_dict(v):
            return dict(v), f'using sub-dict "{k}" (remaining entries are not model weights)'
    raise ValueError('no tensor weights found — this may not be a model checkpoint')


def _aggregate(items):
    """items=[(name, shape tuple, dtype str, numel)] → stats dict."""
    total = n = 0
    dtypes, groups, top = {}, {}, []
    for name, shape, dt, num in items:
        total += num
        n += 1
        d = dtypes.setdefault(dt, [0, 0])
        d[0] += 1
        d[1] += num
        seg = name.split('.')[0]
        groups[seg] = groups.get(seg, 0) + num
        top.append((num, name, shape, dt))
    top.sort(key=lambda x: -x[0])
    return {'total': total, 'n_tensors': n, 'dtypes': dtypes, 'groups': groups,
            'top': [(name, '×'.join(map(str, sh)) or 'scalar', dt, num)
                    for num, name, sh, dt in top[:8]]}


def count_state(sd):
    items = [(str(k), tuple(t.shape), str(t.dtype).replace('torch.', ''), t.numel())
             for k, t in sd.items()]
    return _aggregate(items)


def read_safetensors_header(path):
    """Parse a safetensors header with the stdlib only (no tensor loading)
    → {name: (shape, dtype str)}."""
    with open(path, 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(n).decode('utf-8'))
    header.pop('__metadata__', None)
    return {name: (tuple(info['shape']), info['dtype'])
            for name, info in header.items()}


def count_safetensors(path):
    header = read_safetensors_header(path)
    items = [(name, shape, ST_DTYPE.get(dt, dt.lower()), math.prod(shape))
             for name, (shape, dt) in header.items()]
    return _aggregate(items)


def load_torch_obj(path, log):
    """Degradation chain over mmap/weights_only; returns the checkpoint object."""
    attempts = ((True, True), (True, False), (False, False))
    last = None
    for mmap, wo in attempts:
        try:
            obj = torch.load(path, map_location='cpu', mmap=mmap, weights_only=wo)
            log(f'loaded with mmap={mmap}, weights_only={wo}')
            return obj
        except Exception as e:
            last = e
            log(f'load attempt mmap={mmap}, weights_only={wo} failed: {type(e).__name__}')
    raise last


def _merge(dst, src):
    dst['total'] += src['total']
    dst['n_tensors'] += src['n_tensors']
    for dt, (n, num) in src['dtypes'].items():
        d = dst['dtypes'].setdefault(dt, [0, 0])
        d[0] += n
        d[1] += num
    for seg, num in src['groups'].items():
        dst['groups'][seg] = dst['groups'].get(seg, 0) + num
    dst['top'] = sorted(dst['top'] + src['top'], key=lambda x: -x[3])[:8]


def count_path(path, log=lambda s: None):
    """Count parameters of a weight file or directory → dict for the GUI."""
    t0 = time.time()
    if os.path.isdir(path):
        files = []
        for rd, _, fs in os.walk(path):
            files += [os.path.join(rd, f) for f in fs
                      if f.lower().endswith(WEIGHT_EXTS)]
        if not files:
            raise ValueError('directory contains no weight files '
                             '(.safetensors / .pth / .pt / .bin / .ckpt)')
        res = {'total': 0, 'n_tensors': 0, 'dtypes': {}, 'groups': {},
               'top': []}
        for p in sorted(files):
            log(f'counting {os.path.basename(p)}…')
            _merge(res, count_safetensors(p) if p.lower().endswith('.safetensors')
                   else count_state(pick_state_dict(load_torch_obj(p, log))[0]))
        size = sum(os.path.getsize(p) for p in files)
        res['note'] = f'directory: aggregated {len(files)} weight files'
    else:
        if not os.path.isfile(path):
            raise ValueError(f'path does not exist: {path}')
        if not path.lower().endswith(WEIGHT_EXTS):
            raise ValueError('not a weight file (supported: '
                             '.safetensors / .pth / .pt / .bin / .ckpt)')
        size = os.path.getsize(path)
        if path.lower().endswith('.safetensors'):
            res = count_safetensors(path)
            log('safetensors header parsed (no tensor data loaded)')
        else:
            obj = load_torch_obj(path, log)
            sd, note = pick_state_dict(obj)
            res = count_state(sd)
            res['note'] = note
    res.update({'path': os.path.normpath(path),
                'kind': 'dir' if os.path.isdir(path) else 'file',
                'size': size, 'sec': round(time.time() - t0, 2)})
    return res


# ---------- GUI ----------

def card(parent, **pack):
    f = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
    f.pack(**pack)
    return f


class App:
    def __init__(self, root):
        self.root = root
        root.title('Model Parameter Viewer · MIRROR')
        root.geometry('940x700')
        root.minsize(880, 620)
        root.configure(bg=BG)
        self.cfg = self._load_config()
        self.result = None
        self.loading = False
        self._worker_thread = None
        self.q = queue.Queue()
        self._style()
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self._poll()

    # ---- config ----
    @staticmethod
    def _load_config():
        cfg = {'last_dir': ''}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, encoding='utf-8') as f:
                    cfg.update(json.load(f))
            except Exception:
                pass
        return cfg

    def _save_config(self):
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

    # ---- style ----
    def _style(self):
        st = ttk.Style(self.root)
        st.theme_use('clam')
        st.configure('.', background=BG, foreground=FG, font=F_UI)
        st.configure('TButton', padding=(12, 5))
        st.map('TButton', background=[('active', '#dbeafe')])
        st.configure('Treeview', background=CARD, fieldbackground=CARD,
                     rowheight=24, font=F_UI_S, borderwidth=0)
        st.configure('Treeview.Heading', background='#eef1f4', font=F_UI_S,
                     relief='flat', padding=(4, 3))
        st.configure('TProgressbar', troughcolor=BORDER, background=ACC,
                     borderwidth=0)

    # ---- layout ----
    def _build(self):
        # header bar
        head = tk.Frame(self.root, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        head.pack(fill='x')
        tk.Label(head, text='Model Parameter Viewer', bg=CARD, fg=FG,
                 font=F_TITLE).pack(side='left', padx=(16, 6), pady=12)
        tk.Label(head, text='pick a checkpoint → parameter count · <2B compliance',
                 bg=CARD, fg=MUT, font=F_UI_S).pack(side='left', pady=(6, 0))
        btns = tk.Frame(head, bg=CARD)
        btns.pack(side='right', padx=14)
        self.btn_file = ttk.Button(btns, text='+ Load Weight File…',
                                   command=self.pick_file)
        self.btn_file.pack(side='left', padx=4)
        self.btn_dir = ttk.Button(btns, text='+ Load Weight Directory…',
                                  command=self.pick_dir)
        self.btn_dir.pack(side='left', padx=4)

        # content area
        self.body = tk.Frame(self.root, bg=BG)
        self.body.pack(fill='both', expand=True)
        self._render_placeholder()

        # status bar
        sb = tk.Frame(self.root, bg=CARD, highlightbackground=BORDER,
                      highlightthickness=1)
        sb.pack(fill='x', side='bottom')
        self.status_var = tk.StringVar(value='Ready: pick a weight file or directory')
        tk.Label(sb, textvariable=self.status_var, bg=CARD, fg=MUT,
                 font=F_UI_S, anchor='w').pack(side='left', padx=12, pady=4,
                                               fill='x', expand=True)
        self.progress = ttk.Progressbar(sb, length=150, mode='indeterminate')

    def _render_placeholder(self):
        for w in self.body.winfo_children():
            w.destroy()
        box = tk.Frame(self.body, bg=BG)
        box.place(relx=0.5, rely=0.45, anchor='center')
        tk.Label(box, text='No checkpoint loaded yet', bg=BG, fg=MUT,
                 font=('Microsoft YaHei UI', 14)).pack()
        tk.Label(box, text='Supported: .pth / .pt / .bin / .ckpt / .safetensors, '
                           'or a weight directory',
                 bg=BG, fg=MUT, font=F_UI_S).pack(pady=(4, 0))

    def _render(self):
        for w in self.body.winfo_children():
            w.destroy()
        r = self.result
        pad = dict(fill='x', padx=14, pady=(12, 0))

        # info card
        c1 = card(self.body, **pad)
        name = os.path.basename(r['path']) + ('/' if r['kind'] == 'dir' else '')
        tk.Label(c1, text=name, bg=CARD, fg=FG,
                 font=('Microsoft YaHei UI', 12, 'bold')).pack(
            anchor='w', padx=14, pady=(10, 0))
        tk.Label(c1, text=r['path'], bg=CARD, fg=MUT, font=F_NUM_S,
                 wraplength=860, justify='left', anchor='w').pack(
            fill='x', padx=14)
        chips = tk.Frame(c1, bg=CARD)
        chips.pack(anchor='w', padx=14, pady=(6, 10))
        dts = sorted(r['dtypes'].items(), key=lambda kv: -kv[1][1])
        chip_texts = [f"Size {fmt_size(r['size'])}",
                      f"{r['n_tensors']:,} tensors", f"counted in {r['sec']} s"]
        chip_texts += [f'{dt} {fmt_units(num)} · {num / max(r["total"], 1):.0%}'
                       for dt, (_, num) in dts]
        for t in chip_texts:
            tk.Label(chips, text=t, bg=CHIP_BG, fg=ACC, font=F_UI_S,
                     padx=8, pady=2).pack(side='left', padx=(0, 6))
        if r.get('note'):
            tk.Label(c1, text='Structure: ' + r['note'], bg=CARD, fg=MUT,
                     font=F_UI_S, wraplength=860, justify='left',
                     anchor='w').pack(fill='x', padx=14, pady=(0, 4))

        # big number card + compliance card
        row = tk.Frame(self.body, bg=BG)
        row.pack(fill='x', padx=14, pady=(12, 0))
        c2 = tk.Frame(row, bg=CARD, highlightbackground=BORDER,
                      highlightthickness=1)
        c2.pack(side='left', fill='both', expand=True, padx=(0, 8))
        tk.Label(c2, text='Total parameters', bg=CARD, fg=MUT, font=F_UI_S).pack(
            anchor='w', padx=16, pady=(10, 0))
        tk.Label(c2, text=fmt_int(r['total']), bg=CARD, fg=FG,
                 font=F_NUM).pack(anchor='w', padx=16)
        tk.Label(c2, text=f'≈ {fmt_units(r["total"])} params', bg=CARD, fg=ACC,
                 font=('Microsoft YaHei UI', 12, 'bold')).pack(anchor='w', padx=16)
        tk.Label(c2, text='state_dict basis (includes persistent buffers, '
                          'e.g. normalization stats)',
                 bg=CARD, fg=MUT, font=F_UI_S).pack(anchor='w', padx=16,
                                                    pady=(2, 10))
        c3 = tk.Frame(row, bg=CARD, highlightbackground=BORDER,
                      highlightthickness=1)
        c3.pack(side='left', fill='both', expand=True)
        ok = r['total'] <= LIMIT
        color = OK if ok else BAD
        tk.Label(c3, text='<2B Compliance', bg=CARD, fg=MUT, font=F_UI_S).pack(
            anchor='w', padx=16, pady=(10, 0))
        tk.Label(c3, text='✔ Under the 2B limit' if ok else '✘ Over the 2B limit',
                 bg=CARD, fg=color, font=('Microsoft YaHei UI', 15, 'bold')).pack(
            anchor='w', padx=16)
        margin = (LIMIT - r['total']) / LIMIT * 100
        tk.Label(c3, text=(f'Headroom {margin:.1f}% (limit {fmt_int(LIMIT)} params)'
                           if ok else
                           f'Over by {-margin:.1f}% (limit {fmt_int(LIMIT)} params)'),
                 bg=CARD, fg=MUT, font=F_UI_S).pack(anchor='w', padx=16)
        cv = tk.Canvas(c3, bg=CARD, width=300, height=30, highlightthickness=0)
        cv.pack(anchor='w', padx=16, pady=(2, 10))
        cv.create_rectangle(0, 6, 300, 18, fill='#e5e7eb', width=0)
        frac = min(r['total'] / LIMIT, 1.0)
        cv.create_rectangle(0, 6, max(2, round(300 * frac)), 18, fill=color, width=0)
        cv.create_text(2, 24, anchor='w', text=f'{fmt_units(r["total"])} / 2B',
                       font=F_UI_S, fill=MUT)

        # composition card
        c4 = card(self.body, fill='x', padx=14, pady=(12, 0))
        tk.Label(c4, text='Composition (by top-level name segment)', bg=CARD,
                 fg=FG, font=F_UI_B).pack(anchor='w', padx=14, pady=(10, 2))
        groups = sorted(r['groups'].items(), key=lambda kv: -kv[1])
        shown = groups[:6]
        rest = groups[6:]
        if rest:
            shown = shown + [('Other', sum(v for _, v in rest))]
        gmax = max(v for _, v in shown)
        for i, (seg, num) in enumerate(shown):
            grow = tk.Frame(c4, bg=CARD)
            grow.pack(fill='x', padx=14, pady=2)
            tk.Label(grow, text=seg, bg=CARD, fg=FG, font=F_UI_S, width=16,
                     anchor='w').pack(side='left')
            cv = tk.Canvas(grow, bg=CARD, width=340, height=14,
                           highlightthickness=0)
            cv.pack(side='left', padx=(0, 10))
            w = max(3, round(340 * num / gmax))
            cv.create_rectangle(0, 1, 340, 13, fill='#eef1f4', width=0)
            cv.create_rectangle(0, 1, w, 13,
                                fill='#9ca3af' if seg == 'Other' else
                                BAR_PALETTE[min(i, len(BAR_PALETTE) - 1)], width=0)
            tk.Label(grow, text=f'{fmt_units(num)} · {num / r["total"]:.1%}',
                     bg=CARD, fg=MUT, font=F_NUM_S).pack(side='left')

        # top tensors card
        c5 = card(self.body, fill='both', expand=True, padx=14, pady=(12, 14))
        tk.Label(c5, text='Largest tensors (Top 8)', bg=CARD, fg=FG,
                 font=F_UI_B).pack(anchor='w', padx=14, pady=(10, 4))
        tvf = tk.Frame(c5, bg=CARD)
        tvf.pack(fill='both', expand=True, padx=14, pady=(0, 12))
        tv = ttk.Treeview(tvf, columns=('numel', 'name', 'shape', 'dtype'),
                          show='headings', height=8)
        for col, text, w in (('numel', 'Params', 110), ('name', 'Tensor', 430),
                             ('shape', 'Shape', 150), ('dtype', 'dtype', 80)):
            tv.heading(col, text=text)
            tv.column(col, width=w, anchor='w', stretch=(col == 'name'))
        sb = ttk.Scrollbar(tvf, command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self._name_by_iid = {}
        for name, shape, dt, num in r['top']:
            iid = tv.insert('', 'end', values=(fmt_int(num), name, shape, dt))
            self._name_by_iid[iid] = name
        tv.bind('<<TreeviewSelect>>', self._on_tree_select)
        self._tree = tv

    def _on_tree_select(self, _e=None):
        sel = self._tree.selection() if getattr(self, '_tree', None) else ()
        if sel:
            self.status_var.set(self._name_by_iid.get(sel[0], ''))

    # ---- pick & load ----
    def pick_file(self):
        p = filedialog.askopenfilename(
            title='Select weight file',
            initialdir=self.cfg.get('last_dir') or BASE,
            filetypes=[('Weight files', '*.pth *.pt *.bin *.ckpt *.safetensors'),
                       ('All files', '*.*')])
        if p:
            self.start_load(p)

    def pick_dir(self):
        d = filedialog.askdirectory(title='Select weight directory (HF format)',
                                    initialdir=self.cfg.get('last_dir') or BASE)
        if d:
            self.start_load(os.path.normpath(d))

    def start_load(self, path):
        if self.loading:
            return
        self.loading = True
        self.cfg['last_dir'] = path if os.path.isdir(path) else os.path.dirname(path)
        for b in (self.btn_file, self.btn_dir):
            b.config(state='disabled')
        self.progress.pack(side='left', padx=(0, 12))
        self.progress.start(12)
        self.status_var.set(f'counting {os.path.basename(path)}…')
        self._worker_thread = threading.Thread(
            target=self._worker, args=(path,), daemon=True)
        self._worker_thread.start()

    def _worker(self, path):
        try:
            res = count_path(path, log=lambda s: self.q.put(('log', s)))
            self.q.put(('done', res))
        except Exception as e:
            self.q.put(('error', f'{path}\n{e}'))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == 'log':
                    self.status_var.set(payload)
                elif kind == 'done':
                    self.result = payload
                    self._render()
                    self.status_var.set(
                        f'ready · {os.path.basename(payload["path"])} · '
                        f'{fmt_units(payload["total"])} params')
                elif kind == 'error':
                    self._render_placeholder()
                    self.status_var.set('count failed, see dialog')
                    messagebox.showerror('Count failed', payload)
        except queue.Empty:
            pass
        if self.loading and self._worker_idle():
            self.loading = False
            self.progress.stop()
            self.progress.pack_forget()
            for b in (self.btn_file, self.btn_dir):
                b.config(state='normal')
            self._save_config()
        try:
            self.root.after(120, self._poll)
        except Exception:
            pass  # window already destroyed (test teardown)

    def _worker_idle(self):
        return self._worker_thread is None or not self._worker_thread.is_alive()

    def on_close(self):
        self._save_config()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
