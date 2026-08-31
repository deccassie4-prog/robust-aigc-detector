"""pred_filter_gui.py — 按 pred 区间筛选图片并归档（误差分析取图工具）

用途：选多个 predict.py 产出的 JSON（competition / detailed 双格式自动识别），
每个 JSON 一个选项卡：指定对应图片目录、可覆盖全局的分数区间；实时统计
"匹配 n 条 · 找到 m 张"。点"开始复制"后，在输出目录下按 JSON 文件名建文件夹，
把区间内的图片复制进去并重命名为各自的 pred 值（撞名自动加 _2/_3 序号），
每个文件夹附 manifest.csv（新文件名, pred, JSON记录路径, 实际复制路径）。

查找语义：只按文件名匹配，不含任何路径——JSON 里的
image_path 仅取 basename，实际定位全靠所选图片目录的递归索引；数据集搬过家
（换盘、换机器）也能找到。多处同名命中取排序第一个并提示。

零第三方依赖（tkinter/json/threading/queue/csv 均为标准库）。
复制在子线程执行，界面不卡；停止=当前文件完成后停下，已复制的保留。
全局设置持久化在 pred_filter_config.json（原子写，留 .bak）。
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


# ---------- 纯函数（可脱离 GUI 单测） ----------

def load_entries(path):
    """读 JSON → (entries, fmt, n_bad)。entries=[(image_path, pred), ...]。
    competition = 顶层列表 [{"image_path","pred"}]；detailed = {"images":[...]}。
    缺字段 / 非数值 pred 的条目跳过并计数。"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get('images'), list):
        raw, fmt = data['images'], 'detailed'
    elif isinstance(data, list):
        raw, fmt = data, 'competition'
    else:
        raise ValueError('不是可识别的格式（应为列表或含 images 的字典）')
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
    """闭区间 [lo, hi]；任一侧 None 表示不设限。失败项 pred=-1 在 min>=0 时自然排除。"""
    if lo is not None and pred < lo:
        return False
    if hi is not None and pred > hi:
        return False
    return True


def pred_label(pred):
    """pred → 文件名主干：固定 6 位小数去尾零（0.951053→'0.951053'、0.5→'0.5'、
    1.0→'1'），避免 str() 对极小值输出科学计数法（1e-05 不能做文件名主干）。"""
    s = f'{pred:.6f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-') else '0'


def unique_name(stem, ext, taken):
    """返回不冲突的 文件名。taken=已占用文件名集合（小写），撞名追加 _2/_3…"""
    cand = stem + ext
    i = 2
    while cand.lower() in taken:
        cand = f'{stem}_{i}{ext}'
        i += 1
    taken.add(cand.lower())
    return cand


def build_index(root):
    """递归索引图片目录：{文件名小写: [完整路径...]}。"只按文件名匹配"的依据。"""
    idx = {}
    for rd, _, fs in os.walk(root):
        for f in fs:
            idx.setdefault(f.lower(), []).append(os.path.join(rd, f))
    return idx


def resolve_by_name(img_path, index):
    """按文件名解析 → (实际路径 or None, 命中数)。多处命中取排序第一个。"""
    hits = index.get(os.path.basename(img_path).lower())
    if not hits:
        return None, 0
    if len(hits) == 1:
        return hits[0], 1
    return sorted(hits)[0], len(hits)


def dedup_folder_names(paths):
    """JSON 路径 → 输出文件夹名。同名自动加父目录前缀（父目录也同名则继续上溯）。"""
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


# ---------- 选项卡 ----------

class FilterTab:
    """一个 JSON 选项卡：图片目录 + 可覆盖的分数区间（每侧留空=用全局）+ 实时统计。"""

    def __init__(self, app, notebook, json_path, entries, fmt, n_bad):
        self.app = app
        self.json_path = json_path
        self.entries = entries
        self.fmt = fmt
        self.n_bad = n_bad
        self.index = None          # {文件名小写: [完整路径...]}
        self._index_dirty = True
        self._idx_after = None
        self.found_cache = {}      # 文件名 → 命中数（0=未找到），跨区间调整复用
        self._matched = []
        self.dir_var = tk.StringVar()
        self.min_var = tk.StringVar()
        self.max_var = tk.StringVar()
        self.stat_var = tk.StringVar(value='')
        self.frame = ttk.Frame(notebook)
        self.widgets = []          # 运行期间统一禁用
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
        ttk.Label(f, text='图片目录').grid(row=0, column=0, sticky='e', **pad)
        fr = ttk.Frame(f)
        fr.grid(row=0, column=1, sticky='we', **pad)
        de = ttk.Entry(fr, textvariable=self.dir_var)
        de.pack(side='left', fill='x', expand=True)
        db = ttk.Button(fr, text='浏览…', command=self.pick_dir)
        db.pack(side='left', padx=4)
        self.widgets += [de, db]
        info = f'{os.path.basename(self.json_path)} · {len(self.entries):,} 条（{self.fmt}）'
        if self.n_bad:
            info += f' · 无效 {self.n_bad} 条已跳过'
        ttk.Label(f, text=info, foreground='#666').grid(row=0, column=2, sticky='w', **pad)

        ttk.Label(f, text='区间覆盖').grid(row=1, column=0, sticky='e', **pad)
        rf = ttk.Frame(f)
        rf.grid(row=1, column=1, sticky='w', **pad)
        ttk.Label(rf, text='最小').pack(side='left')
        me = ttk.Entry(rf, textvariable=self.min_var, width=9)
        me.pack(side='left', padx=(2, 10))
        ttk.Label(rf, text='最大').pack(side='left')
        xe = ttk.Entry(rf, textvariable=self.max_var, width=9)
        xe.pack(side='left', padx=2)
        self.widgets += [me, xe]
        ttk.Label(f, text='留空 = 用全局值', foreground='#999').grid(
            row=1, column=2, sticky='w', **pad)

        st = tk.Label(f, textvariable=self.stat_var, fg='#0a6', wraplength=560,
                      anchor='w', justify='left')
        st.grid(row=2, column=0, columnspan=3, sticky='we', padx=10, pady=(0, 6))
        f.columnconfigure(1, weight=1)

    # -- 目录 --
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

    # -- 区间与统计 --
    def effective_range(self):
        """(lo, hi, err)：每侧留空回退全局；文本非法或 min>max 返回 err。"""
        try:
            s = self.min_var.get().strip()
            lo = float(s) if s else None
        except ValueError:
            return None, None, '覆盖最小值不是数字'
        try:
            s = self.max_var.get().strip()
            hi = float(s) if s else None
        except ValueError:
            return None, None, '覆盖最大值不是数字'
        gmin, gmax, gerr = self.app.glob_range()
        if gerr:
            return None, None, gerr
        lo = gmin if lo is None else lo
        hi = gmax if hi is None else hi
        if lo is not None and hi is not None and lo > hi:
            return None, None, '最小值大于最大值'
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
            self.stat_var.set(f'匹配 {n} 条 · 未选图片目录')
            return
        if not os.path.isdir(d):
            self.stat_var.set(f'匹配 {n} 条 · 目录不存在')
            return
        if self._index_dirty:
            self.stat_var.set(f'匹配 {n} 条 · 正在索引目录…')
            if self._idx_after is None:
                self._idx_after = self.app.after(250, self._build_and_count)
            return
        m, multi = self._count_found()
        txt = f'匹配 {n} 条 · 找到 {m} 张'
        if multi:
            txt += f'（{multi} 个文件名多处命中，取第一个）'
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
        """[(JSON记录路径, pred, 实际路径 or None)]，与统计同一套解析。"""
        out = []
        for p, pred in self._matched:
            src, _ = resolve_by_name(p, self.index)
            out.append((p, pred, src))
        return out

    def set_locked(self, locked):
        for w in self.widgets:
            w.config(state='disabled' if locked else 'normal')


# ---------- 主窗口 ----------

class App:
    def __init__(self, root):
        self.root = root
        root.title('按 pred 区间收集图片 · pred_filter')
        root.geometry('900x660')
        self.cfg = self._load_config()
        self._tabs = []
        self._loaded = set()       # 已加载 JSON（normcase），防重复导入
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

    # ---- 配置 ----
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

    # ---- 界面 ----
    def _build(self):
        pad = dict(padx=6, pady=4)
        top = ttk.Frame(self.root)
        top.pack(fill='x', **pad)
        self.btn_select = ttk.Button(top, text='选择 JSON…（可多选）',
                                     command=self.pick_jsons)
        self.btn_select.pack(side='left', padx=4)
        self.btn_close_tab = ttk.Button(top, text='✕ 关闭当前', command=self.close_tab)
        self.btn_close_tab.pack(side='left', padx=4)

        tf = ttk.LabelFrame(self.root, text='每个 JSON 一个选项卡（选图片目录、可覆盖分数区间）')
        tf.pack(fill='x', **pad)
        self.notebook = ttk.Notebook(tf)
        self.notebook.pack(fill='x', **pad)

        gf = ttk.LabelFrame(self.root, text='全局设置（选项卡留空的项使用这里的值）')
        gf.pack(fill='x', **pad)
        row = ttk.Frame(gf)
        row.pack(fill='x', **pad)
        ttk.Label(row, text='最小值').pack(side='left')
        self.gmin_var = tk.StringVar(value='0')
        ge1 = ttk.Entry(row, textvariable=self.gmin_var, width=9)
        ge1.pack(side='left', padx=(2, 12))
        ttk.Label(row, text='最大值').pack(side='left')
        self.gmax_var = tk.StringVar(value='1')
        ge2 = ttk.Entry(row, textvariable=self.gmax_var, width=9)
        ge2.pack(side='left', padx=(2, 12))
        ttk.Label(row, text='输出目录').pack(side='left')
        self.out_var = tk.StringVar()
        oe = ttk.Entry(row, textvariable=self.out_var)
        oe.pack(side='left', fill='x', expand=True, padx=2)
        ob = ttk.Button(row, text='浏览…', command=self._pick_out)
        ob.pack(side='left', padx=4)
        self.lockable = [ge1, ge2, oe, ob]
        self.gmin_var.trace_add('write', lambda *a: self.refresh_all())
        self.gmax_var.trace_add('write', lambda *a: self.refresh_all())

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill='x', **pad)
        self.btn_start = ttk.Button(ctrl, text='开始复制', command=self.start_run)
        self.btn_start.pack(side='left', padx=6)
        self.btn_stop = ttk.Button(ctrl, text='停止', command=self.stop_run,
                                   state='disabled')
        self.btn_stop.pack(side='left', padx=6)
        self.progress = ttk.Progressbar(ctrl, length=260, maximum=1, value=0)
        self.progress.pack(side='left', padx=10)
        self.btn_open = ttk.Button(ctrl, text='打开输出目录', command=self.open_out,
                                   state='disabled')
        self.btn_open.pack(side='right', padx=6)

        self.status_var = tk.StringVar(value='就绪：先选择 JSON 文件')
        ttk.Label(self.root, textvariable=self.status_var, foreground='#036').pack(
            anchor='w', padx=10)

        lf = ttk.LabelFrame(self.root, text='日志')
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

    # ---- 选项卡管理 ----
    def pick_jsons(self):
        ps = filedialog.askopenfilenames(
            title='选择 predict 输出的 JSON（可多选）',
            initialdir=self.cfg.get('last_dir') or BASE,
            filetypes=[('JSON', '*.json'), ('所有文件', '*.*')])
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
                messagebox.showerror('读取失败', f'{p}\n{e}')
                continue
            if not entries:
                messagebox.showwarning('空 JSON', f'{p}\n没有有效条目，跳过。')
                continue
            self._loaded.add(key)
            tab = FilterTab(self, self.notebook, p, entries, fmt, bad)
            self._tabs.append(tab)
            self.notebook.add(tab.frame, text=self._title_for(tab))
            self.notebook.select(tab.frame)
            added += 1
        self.status_var.set(f'已加载 {added} 个 JSON'
                            + (f'（跳过重复 {skipped}）' if skipped else ''))
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
                    '关闭选项卡', f'关闭"{tab.title}"？其设置将丢弃。'):
                return
        self.notebook.forget(tab.frame)
        self._tabs.remove(tab)
        self._loaded.discard(os.path.normcase(os.path.normpath(tab.json_path)))

    # ---- 实时统计 ----
    def glob_range(self):
        try:
            s = self.gmin_var.get().strip()
            lo = float(s) if s else None
        except ValueError:
            return None, None, '全局最小值不是数字'
        try:
            s = self.gmax_var.get().strip()
            hi = float(s) if s else None
        except ValueError:
            return None, None, '全局最大值不是数字'
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

    # ---- 预检与运行 ----
    def start_run(self):
        if self.running:
            return
        out_dir = self.out_var.get().strip()
        if not out_dir:
            messagebox.showerror('无法开始', '请先设置输出目录。')
            return
        if os.path.isfile(out_dir):
            messagebox.showerror('无法开始', f'输出目录指向的是一个文件：{out_dir}')
            return
        plans, probs, zeros = [], [], []
        for t in self._tabs:
            lo, hi, err = t.effective_range()
            if err:
                probs.append(f'[{t.title}] {err}')
                continue
            d = t.dir_var.get().strip()
            if not d or not os.path.isdir(d):
                probs.append(f'[{t.title}] 未选图片目录或目录不存在')
                continue
            if not t.matched(lo, hi):
                zeros.append(t.title)
                continue
            if t._index_dirty:
                t.build_index_now()
            t._matched = t.matched(lo, hi)   # 运行快照：之后改表单不影响本次
            plans.append((t, t.resolved()))
        if probs:
            messagebox.showerror('预检未通过', '\n'.join(probs))
        if not plans:
            return
        if zeros:
            self._log('[GUI] 匹配 0 条、已跳过：' + '、'.join(zeros) + '\n')
        names = dedup_folder_names([t.json_path for t, _ in plans])
        renamed = [(t.title, names[t.json_path]) for t, _ in plans
                   if names[t.json_path] != t.title]
        if renamed:
            lines = '\n'.join(f'· {a} → {b}' for a, b in renamed)
            if not messagebox.askyesno(
                    '输出文件夹重名',
                    '以下 JSON 同名，输出文件夹自动加父目录前缀：\n\n' + lines
                    + '\n\n是否继续？'):
                return
        total = sum(len(rows) for _, rows in plans)
        if total == 0:
            messagebox.showerror('无法开始', '所有选项卡在区间内都没有可定位的图片。')
            return
        self._run(plans, names, out_dir, total)

    def _run(self, plans, names, out_dir, total):
        self.running = True
        self.stop_flag = False
        self.t0 = time.time()
        self._lock(True)
        self.progress.config(maximum=total, value=0)
        self.status_var.set('复制中…')
        self._log(f'[GUI] 开始复制 {len(plans)} 个选项卡，共 {total} 条 → {out_dir}\n')
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
                self.q.put(('log', f'[失败] 建目录 {folder}: {e}\n'))
                done += len(rows)
                continue
            self.q.put(('log', f'--- [{k}/{len(plans)}] {tab.title} → {folder}'
                        f'（匹配 {len(rows)} 条）\n'))
            copied = missed = renamed_n = failed = 0
            taken = set()
            try:
                mf = open(os.path.join(folder, 'manifest.csv'), 'w',
                          newline='', encoding='utf-8-sig')
            except Exception as e:
                self.q.put(('log', f'[失败] 写 manifest.csv: {e}\n'))
                done += len(rows)
                continue
            with mf:
                w = csv.writer(mf)
                w.writerow(['新文件名', 'pred', 'JSON记录路径', '实际复制路径'])
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
                            self.q.put(('log', f'[失败] {rec_path}: {e}\n'))
                    if i % 50 == 0 or i == n_rows:
                        self.q.put(('prog', (done, total,
                                             f'{k}/{len(plans)} {tab.title}: {i}/{n_rows}')))
            msg = f'复制 {copied} · 未找到 {missed} · 撞名加序号 {renamed_n}'
            if failed:
                msg += f' · 失败 {failed}'
            if self.stop_flag:
                msg += '（已停止）'
            self.q.put(('tab_done', f'[{tab.title}] {msg}'))
        self.q.put(('done', round(time.time() - t0)))

    def stop_run(self):
        self.stop_flag = True
        self._log('[GUI] 已请求停止：当前文件完成后停下…\n')

    # ---- 运行时轮询（子线程只入队） ----
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
                        f'{text} · 已用 {int(time.time() - self.t0)} 秒')
                elif kind == 'tab_done':
                    self._log('[GUI] ' + payload + '\n')
                elif kind == 'done':
                    self._finish_run(payload)
        except queue.Empty:
            pass
        try:
            self.root.after(120, self._poll)
        except Exception:
            pass  # 窗口已销毁（测试收尾）

    def _finish_run(self, sec):
        self.running = False
        self._lock(False)
        tail = '（已停止，部分完成）' if self.stop_flag else ''
        self._log(f'[GUI] ===== 复制结束，用时 {sec} 秒{tail} =====\n')
        self.status_var.set(f'复制结束，用时 {sec} 秒{tail}（明细见日志）')
        if os.path.isdir(self.out_var.get().strip()):
            self.btn_open.config(state='normal')
        self._save_config()

    def _log(self, text):
        self.logbox.config(state='normal')
        self.logbox.insert('end', text)
        self.logbox.see('end')
        self.logbox.config(state='disabled')

    # ---- 退出 ----
    def on_close(self):
        if self.running:
            if not messagebox.askyesno(
                    '退出', '复制仍在进行，停止并退出？\n（已复制的文件保留）'):
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
