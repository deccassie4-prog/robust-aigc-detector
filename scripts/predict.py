"""
predict.py — 比赛交付物 + 多裁剪(multi-crop)消融实验入口

赛题要求：输入一个图片目录，为每张图输出 AIGC 置信分，写成 JSON（含 image_path 和 pred 字段）。
本脚本在 MIRROR 零样本推理之上支持多种裁剪策略与分数聚合：
  - 文献结论：mean / top-k mean 是主推聚合；max 对局部伪影敏感但推高真图 FP，仅作对照；不用硬投票。
  - center 模式与评测脚本 evaluation/evaluate.py 的 TestDataset 预处理逐像素一致（等价性门禁）。

显存约定（重要）：--batch_size 计的是"每批前向的裁剪块数"，与 evaluation/evaluate.py 同语义。
多裁剪模式下自动换算成 图片数/批 = batch_size // K，保证任何 crop_mode 的前向规模一致。
两套显存口径（2026-08-28 实测，RTX 3080 20GB，fp32）：PyTorch 已分配 ≈130MB/块 + 权重 3.4GB
（64 块 = 11.9 GiB）；任务管理器/nvidia-smi 口径另加 ~3GB（CUDA 上下文 + cuBLAS/cuDNN
workspace + 分配器保留池），64 块外部实测 ~15GB。默认 32 块/批 → 外部 ~10GB，64 为已验证上限。

权重默认从 <仓库根>/data/weights/ 读取（checkpoint-h-cur.pth / mirror_phase1.pth / dinov3-huge/，
获取方式见 README）；传相对路径时按仓库根解析，与启动时的 CWD 无关。

用法示例：
  python scripts/predict.py --data_dir 图片目录 --output_json preds.json
  python scripts/predict.py --data_dir 图片目录 --crop_mode five --aggregate topk
  python scripts/predict.py --data_dir 图片目录 --limit 8 --dump_scores per_crop.csv
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# 仓库根锚定：保证本脚本从任意 CWD 运行时都能导入 src/ 与 evaluation/ 包
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.mirror import build_mirror
from evaluation.evaluate import compress_image  # 复用 q96 格式对齐，保证与评测管线一致

ImageFile.LOAD_TRUNCATED_IMAGES = True

CROP = 224
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
# 每个 crop_mode 固定输出的裁剪块数：批内张量形状必须一致，否则 default_collate 直接报错
N_CROPS = {'center': 1, 'five': 5, 'grid3x3': 9, 'grid4x4': 16, 'grid5x5': 25,
           'multiscale': 3, 'single336': 1}


# ============ 裁剪策略：返回 K 个 (PIL crop, 描述) 列表 ============

def _resize_short(img, target_short=512):
    """与 evaluation/evaluate.py 的 RandomScaleCropOrDirect224 相同的缩放规则：短边>512 才缩到 512。"""
    w, h = img.size
    short = min(w, h)
    if short <= target_short:
        return img
    scale = target_short / short
    return TF.resize(img, [int(round(h * scale)), int(round(w * scale))],
                     interpolation=InterpolationMode.BICUBIC, antialias=True)


def _crop_padded(img, x0, y0, size=CROP):
    """按左上角 (x0, y0) 裁 size×size；越界部分补 0（与 TF.center_crop 的小图行为一致）。"""
    w, h = img.size
    pad_r, pad_b = max(0, x0 + size - w), max(0, y0 + size - h)
    if pad_r or pad_b or x0 < 0 or y0 < 0:
        canvas = Image.new('RGB', (max(w, x0 + size), max(h, y0 + size)), (0, 0, 0))
        canvas.paste(img, (0, 0))
        img = canvas
    return img.crop((x0, y0, x0 + size, y0 + size))


def make_crops(img, mode):
    """输入已缩放的 PIL 图，输出恰好 N_CROPS[mode] 个 crop_PIL。

    小图导致坐标重合时不去重（重复块分数相同，聚合结果不变）——
    固定块数优先于去重，否则批内形状不一致会让 collate 崩掉整批。
    """
    w, h = img.size
    # 中点取整与 torchvision.transforms.center_crop 完全一致（round 而非 floor），
    # 保证 center 模式与原生 inference.py 逐像素可复现
    cx, cy = int(round((w - CROP) / 2.0)), int(round((h - CROP) / 2.0))
    if mode == 'center':
        boxes = [(cx, cy)]
    elif mode == 'five':  # 四角 + 中心
        xs = [0, cx, w - CROP]
        ys = [0, cy, h - CROP]
        boxes = [(xs[0], ys[0]), (xs[2], ys[0]), (xs[0], ys[2]), (xs[2], ys[2]), (xs[1], ys[1])]
    elif mode.startswith('grid'):
        # grid3x3/grid4x4/grid5x5：n×n 均匀网格含边缘，全覆盖带重叠，恒 n² 块。
        # 端点式取起点与旧版 grid3x3 的 [0, cx, w-CROP] 逐值一致（i=1 即 round((w-CROP)/2)=cx），
        # 小图起点 clamp 到 0、越界由 _crop_padded 补 0
        n = int(mode[4])
        xs = [min(max(int(round(i * (w - CROP) / (n - 1))), 0), max(w - CROP, 0)) for i in range(n)]
        ys = [min(max(int(round(i * (h - CROP) / (n - 1))), 0), max(h - CROP, 0)) for i in range(n)]
        boxes = [(x, y) for y in ys for x in xs]
    elif mode == 'multiscale':  # 0.5×/1×/2× 各取中心裁剪
        crops = []
        for s in (0.5, 1.0, 2.0):
            if s == 1.0:
                scaled = img
            else:
                scaled = TF.resize(img, [int(round(h * s)), int(round(w * s))],
                                   interpolation=InterpolationMode.BICUBIC, antialias=True)
            sw, sh = scaled.size
            crops.append(_crop_padded(scaled, int(round((sw - CROP) / 2.0)),
                                      int(round((sh - CROP) / 2.0))))
        return crops
    elif mode == 'single336':  # 对照组：RoPE 自适应的 336 单裁剪，分数分布可能漂移
        return [_crop_padded(img, (w - 336) // 2, (h - 336) // 2, size=336)]
    else:
        raise ValueError(f'unknown crop_mode: {mode}')
    return [_crop_padded(img, x, y) for x, y in boxes]


# ============ 数据集：每张图返回 (idx, [K,3,H,W], 有效标志) ============

class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, root, crop_mode, no_resize=False):
        self.paths = []
        for root_dir, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    self.paths.append(os.path.join(root_dir, f))
        self.paths.sort()
        self.crop_mode = crop_mode
        self.n_crops = N_CROPS[crop_mode]
        self.no_resize = no_resize
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            if not self.no_resize:
                img = _resize_short(img)  # 与原生管线相同的缩放
            crops = make_crops(img, self.crop_mode)
            # 与 TestDataset 一致：先裁剪、后 PNG→JPEG q96 对齐
            tensors = []
            for c in crops:
                if path.lower().endswith(('.png', '.bmp', '.tiff')):
                    c = compress_image(c, quality=96)
                tensors.append(self.to_tensor(c))
            return idx, torch.stack(tensors), torch.tensor(1.0)  # 第三项: 1=有效, 0=读取失败
        except Exception as e:
            # 失败也要返回 n_crops 块，保持批内形状一致（否则一张坏图炸掉整批）
            print(f"[警告] 读取失败 {path}: {e}")
            size = 336 if self.crop_mode == 'single336' else CROP
            return idx, torch.zeros(self.n_crops, 3, size, size), torch.tensor(0.0)


# ============ 聚合策略 ============

def aggregate(scores, method, topk_k):
    """scores: [B, K] 的每裁剪块 P(fake)。返回 [B]。"""
    if method == 'mean':
        return scores.mean(dim=1)
    if method == 'median':
        return scores.median(dim=1).values
    if method == 'max':
        return scores.max(dim=1).values
    if method == 'topk':
        k = min(max(topk_k, 1), scores.shape[1])
        return scores.topk(k, dim=1).values.mean(dim=1)
    raise ValueError(f'unknown aggregate: {method}')


# ============ 主流程 ============

def get_args():
    p = argparse.ArgumentParser('MIRROR multi-crop predict')
    weight_default = os.path.join(REPO_ROOT, 'data', 'weights')
    p.add_argument('--data_dir', required=True, nargs='+',
                   help='一个或多个图片目录；多目录时逐个推理（模型只加载一次），'
                        '每个目录输出一个 JSON 到 --output_dir，文件名为文件夹名')
    p.add_argument('--output_json', default='predictions.json',
                   help='单目录模式的输出 JSON 路径')
    p.add_argument('--output_dir', default='',
                   help='多目录批处理模式的统一输出目录（每文件夹一个 <文件夹名>.json）')
    p.add_argument('--model_path', default=os.path.join(weight_default, 'checkpoint-h-cur.pth'))
    p.add_argument('--memory_path', default=os.path.join(weight_default, 'mirror_phase1.pth'))
    p.add_argument('--backbone_path', default=os.path.join(weight_default, 'dinov3-huge'))
    p.add_argument('--crop_mode', default='center',
                   choices=['center', 'five', 'grid3x3', 'grid4x4', 'grid5x5',
                            'multiscale', 'single336'])
    p.add_argument('--aggregate', default='mean', choices=['mean', 'median', 'topk', 'max'])
    p.add_argument('--topk_k', type=int, default=0, help='0 = ceil(K/3)')
    p.add_argument('--batch_size', type=int, default=32,
                   help='每批前向的裁剪块数（与 evaluation/evaluate.py 同语义）；'
                        '多裁剪模式自动换算为 图片数/批 = batch_size // K。'
                        '默认 32 → 任务管理器口径 ~10GB；64 为已验证上限（~15GB）')
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader 进程数；预处理仅 ~3ms/张、瓶颈在 GPU 前向，'
                        'Windows 下 spawn worker 反而多占内存，默认 0')
    p.add_argument('--device', default='cuda')
    p.add_argument('--use_amp', action='store_true',
                   help='fp16 混合精度，提速明显（溢出问题已修复），推荐开启')
    p.add_argument('--no_resize', action='store_true',
                   help='跳过短边>512→512 的缩放，直接在原图上裁剪（保留高频伪影；'
                        '即 DDA 的推理协议）。默认关闭，与原生管线/等价门禁一致')
    p.add_argument('--dump_scores', default='', help='可选：输出每裁剪块分数的 CSV 路径')
    p.add_argument('--limit', type=int, default=0, help='只处理前 N 张（调试用，0=全部）')
    p.add_argument('--json_format', default='competition', choices=['competition', 'detailed'],
                   help='competition=每图一条 {"image_path","pred"}（比赛交付格式，默认）；'
                        'detailed=参数表头 + 每图 is_fake 布尔（GUI/实验用）')
    args = p.parse_args()
    # 权重相对路径按仓库根解析（GUI 子进程的 CWD 是 scripts/，相对权重路径仍应指向 <仓库>/data/weights）
    for k in ('model_path', 'memory_path', 'backbone_path'):
        if not os.path.isabs(getattr(args, k)):
            setattr(args, k, os.path.join(REPO_ROOT, getattr(args, k)))
    return args


def _run_dir(model, device, args, data_dir):
    """对单个目录推理。返回 (results, crop_rows)。"""
    dataset = PredictDataset(data_dir, args.crop_mode, args.no_resize)
    if args.limit:
        dataset.paths = dataset.paths[:args.limit]
    if not dataset.paths:
        raise SystemExit(f'目录里没有图片: {data_dir}')
    n_crops = N_CROPS[args.crop_mode]
    # 根因修复：batch_size 语义 = 裁剪块/批。前向规模只由块数决定，
    # 多裁剪若按"图片数"解释会把前向放大 K 倍（five=320 块/grid3x3=576 块 → 必然 OOM）
    img_per_batch = max(1, args.batch_size // n_crops)
    n_batches = math.ceil(len(dataset) / img_per_batch)
    est_gb = args.batch_size * 0.13 + 3.4  # 实测标定：~130MB/块（已分配口径）+ 权重
    print(f"    {len(dataset.paths)} 张图 × {n_crops} 裁剪块；"
          f"batch_size={args.batch_size} 块/批 → {img_per_batch} 图/批，共 {n_batches} 批；"
          f"预计已分配显存 ≈{est_gb:.1f} GiB（任务管理器口径另加 ~3GB）", flush=True)

    loader = DataLoader(dataset, batch_size=img_per_batch,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'))

    results, crop_rows = [], []
    t0 = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()  # 只统计推理循环的峰值，排除加载阶段
    with torch.inference_mode():
        for bi, (idx, tensors, ok) in enumerate(loader):
            b, k = tensors.shape[0], tensors.shape[1]
            x = tensors.view(b * k, *tensors.shape[2:]).to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                logits, _, _ = model(x)
            scores = torch.nn.functional.softmax(logits.float(), dim=1)[:, 1].view(b, k).cpu()
            combined = aggregate(scores, args.aggregate, args.topk_k or math.ceil(k / 3))
            ok = ok.bool()
            for i, j in enumerate(idx.tolist()):
                p = dataset.paths[j]
                pred = combined[i].item() if ok[i] else -1.0
                results.append({'image_path': os.path.abspath(p), 'pred': round(pred, 6)})
                if args.dump_scores and ok[i]:
                    for c in range(k):
                        crop_rows.append([os.path.abspath(p), c, round(scores[i, c].item(), 6)])
            if device.type == 'cuda' and bi == 0:
                print(f"    显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB（首批）",
                      flush=True)
            if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
                done = min((bi + 1) * img_per_batch, len(dataset))
                print(f"    [{bi + 1}/{n_batches} 批] {done}/{len(dataset)} 张，"
                      f"用时 {time.time() - t0:.0f} 秒", flush=True)
    return results, crop_rows


def write_results(path, results, meta, fmt):
    """按 json_format 写出。competition=顶层列表（比赛格式，逐字节兼容旧版）；
    detailed=meta 表头 + 每图 is_fake 布尔（失败图 pred=-1 → is_fake=null）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt == 'competition':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        return
    images = [{'image_path': r['image_path'],
               'pred': r['pred'],
               'is_fake': (r['pred'] > 0.5) if r['pred'] >= 0 else None}
              for r in results]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'images': images}, f, ensure_ascii=False, indent=1)


def out_json_name(d, dup_bases):
    """批处理输出文件名 = 文件夹名；同批 basename 冲突时加父目录前缀区分。"""
    name = os.path.basename(os.path.normpath(d))
    if name not in dup_bases:
        return name
    parent = os.path.basename(os.path.normpath(os.path.dirname(os.path.normpath(d))))
    return f'{parent}__{name}'


def main(args):
    device = torch.device(args.device)
    t0 = time.time()

    # 输出参数校验（fail-fast：在加载模型前给出可读报错）
    dirs = args.data_dir
    batch_mode = len(dirs) > 1
    if args.output_dir:
        if os.path.isfile(args.output_dir):
            raise SystemExit(f'--output_dir 指向的是一个文件，请给目录路径：{args.output_dir}')
        os.makedirs(args.output_dir, exist_ok=True)
    elif not batch_mode and os.path.isdir(args.output_json):
        # 目录当文件写会报 PermissionError
        raise SystemExit(f'--output_json 指向的是目录，请给完整文件路径'
                         f'（如 {os.path.join(args.output_json, "predictions.json")}）')
    if batch_mode and not args.output_dir:
        raise SystemExit('多目录批处理需要 --output_dir')

    print(f">>> 加载模型（crop_mode={args.crop_mode}, aggregate={args.aggregate}"
          f"{', no_resize' if args.no_resize else ''}）")
    model = build_mirror(memory_path=args.memory_path, backbone_path=args.backbone_path)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    # transformers 5.x 布局差异修复，同 smoke_test.py / evaluation/evaluate.py
    state = {k.replace("backbone.dino.layer.", "backbone.dino.model.layer."): v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, \
        f"权重加载不完整: missing={len(missing)}, unexpected={len(unexpected)}"
    model.to(device).eval()

    # 重名兜底（GUI 已提前警告，这里是第二道防线）：输出名 = 文件夹名，
    # 同批 basename 冲突时加父目录前缀区分
    by_base = {}
    for d in dirs:
        by_base.setdefault(os.path.basename(os.path.normpath(d)), []).append(d)
    dup_bases = {b for b, ds in by_base.items() if len(ds) > 1}

    param_dict = {k: v for k, v in vars(args).items()
                  if k not in ('data_dir', 'output_json', 'output_dir', 'json_format')}
    ok_dirs, fail_dirs = 0, []
    for k, d in enumerate(dirs):
        name = os.path.basename(os.path.normpath(d))
        print(f"=== [{k + 1}/{len(dirs)}] {name} ===", flush=True)
        try:
            results, crop_rows = _run_dir(model, device, args, d)
        except SystemExit as e:
            print(f"[跳过] {e}", flush=True)
            fail_dirs.append((name, str(e)))
            continue
        except Exception as e:
            print(f"[失败] 文件夹 {name} 推理出错：{e}（继续下一个）", flush=True)
            fail_dirs.append((name, str(e)))
            continue

        out_name = out_json_name(d, dup_bases)
        # 输出目标：--output_dir 优先（每文件夹一个 JSON），其次 --output_json（单文件）
        # 两种模式都支持任意目录数（GUI 子批次 = 单目录 + output_dir 也走这里）
        if args.output_dir:
            out_path = os.path.join(args.output_dir, out_name + '.json')
        else:
            out_path = args.output_json
        meta = {'format_version': 2,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'source_dir': os.path.abspath(d),
                'n_images': len(results),
                'n_failed': sum(1 for r in results if r['pred'] < 0),
                'threshold': 0.5,
                'duration_sec': round(time.time() - t0, 1),
                'params': param_dict}
        write_results(out_path, results, meta, args.json_format)
        if args.dump_scores:
            csv_path = (os.path.join(args.output_dir, out_name + '__per_crop.csv')
                        if args.output_dir else args.dump_scores)
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows([['image_path', 'crop_idx', 'p_fake']] + crop_rows)

        n_fail = meta['n_failed']
        print(f"[完成] {len(results)} 张 -> {out_path}"
              f"（失败 {n_fail} 张记为 pred=-1），用时 {time.time() - t0:.0f} 秒", flush=True)
        if device.type == 'cuda':
            print(f"    显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if results:
            preds = [r['pred'] for r in results if r['pred'] >= 0]
            if preds:
                print(f"    P(fake) 分布: min={min(preds):.4f} mean={np.mean(preds):.4f} "
                      f"max={max(preds):.4f}, >0.5 判 fake: {sum(p > 0.5 for p in preds)} 张",
                      flush=True)
        ok_dirs += 1

    if batch_mode:
        print(f"[批处理完成] 成功 {ok_dirs}/{len(dirs)} 个文件夹"
              + (f"，失败: {fail_dirs}" if fail_dirs else "")
              + f"，输出目录 {args.output_dir}", flush=True)


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    main(get_args())
