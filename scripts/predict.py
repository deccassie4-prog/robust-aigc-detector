"""
predict.py — competition deliverable + multi-crop ablation entry point

Competition requirement: take an image directory, output an AIGC confidence score per
image into a JSON (each item has image_path and pred fields).
Built on MIRROR zero-shot inference, with multiple crop strategies and score aggregations:
  - Literature consensus: mean / top-k mean are the recommended aggregations; max is
    sensitive to local artifacts but raises false positives on real images (baseline only).
  - The center mode is pixel-identical to the TestDataset preprocessing of
    evaluation/evaluate.py (equivalence gate).

VRAM contract (important): --batch_size counts CROPS per forward batch (same semantics as
evaluation/evaluate.py). Multi-crop modes convert it to images/batch = batch_size // K so
every crop_mode gets the same forward scale.
Two VRAM accounting schemes (measured 2026-08-28, RTX 3080 20GB, fp32): PyTorch allocated
~130MB/crop + 3.4GB weights (64 crops = 11.9 GiB); Task-Manager/nvidia-smi adds ~3GB more
(CUDA context + cuBLAS/cuDNN workspace + allocator reserve), i.e. ~15GB external for 64.
Default 32 crops/batch -> ~10GB external; 64 is the verified ceiling.

Weights are read from <repo root>/data/weights/ by default (checkpoint-h-cur.pth /
mirror_phase1.pth / dinov3-huge/, see README for download); relative paths are resolved
against the repo root, independent of the launching CWD.

Examples:
  python scripts/predict.py --data_dir IMAGE_DIR --output_json preds.json
  python scripts/predict.py --data_dir IMAGE_DIR --crop_mode five --aggregate topk
  python scripts/predict.py --data_dir IMAGE_DIR --limit 8 --dump_scores per_crop.csv
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

# repo-root anchor: importable from any CWD (src/ and evaluation/ packages)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.mirror import build_mirror
from evaluation.evaluate import compress_image  # reuse q96 JPEG alignment, identical to the eval pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

CROP = 224
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
# fixed number of crops per crop_mode: batch tensors must keep the same shape or default_collate fails
N_CROPS = {'center': 1, 'five': 5, 'grid3x3': 9, 'grid4x4': 16, 'grid5x5': 25,
           'multiscale': 3, 'single336': 1}


# ============ Crop strategies: return a list of K PIL crops ============

def _resize_short(img, target_short=512):
    """Same resize rule as RandomScaleCropOrDirect224 in evaluation/evaluate.py: shrink to 512 only when short side > 512."""
    w, h = img.size
    short = min(w, h)
    if short <= target_short:
        return img
    scale = target_short / short
    return TF.resize(img, [int(round(h * scale)), int(round(w * scale))],
                     interpolation=InterpolationMode.BICUBIC, antialias=True)


def _crop_padded(img, x0, y0, size=CROP):
    """Crop size x size at top-left (x0, y0); out-of-bounds areas are zero-padded (same as TF.center_crop on small images)."""
    w, h = img.size
    pad_r, pad_b = max(0, x0 + size - w), max(0, y0 + size - h)
    if pad_r or pad_b or x0 < 0 or y0 < 0:
        canvas = Image.new('RGB', (max(w, x0 + size), max(h, y0 + size)), (0, 0, 0))
        canvas.paste(img, (0, 0))
        img = canvas
    return img.crop((x0, y0, x0 + size, y0 + size))


def make_crops(img, mode):
    """Take a resized PIL image, return exactly N_CROPS[mode] crops.

    Overlapping coordinates on small images are NOT deduplicated (duplicate crops score
    the same, aggregation is unchanged) - a fixed crop count takes priority over
    deduplication, otherwise inconsistent batch shapes would crash collate for the batch.
    """
    w, h = img.size
    # midpoint rounding is identical to torchvision.transforms.center_crop (round, not floor),
    # keeping the center mode pixel-reproducible against the original pipeline
    cx, cy = int(round((w - CROP) / 2.0)), int(round((h - CROP) / 2.0))
    if mode == 'center':
        boxes = [(cx, cy)]
    elif mode == 'five':  # four corners + center
        xs = [0, cx, w - CROP]
        ys = [0, cy, h - CROP]
        boxes = [(xs[0], ys[0]), (xs[2], ys[0]), (xs[0], ys[2]), (xs[2], ys[2]), (xs[1], ys[1])]
    elif mode.startswith('grid'):
        # grid3x3/grid4x4/grid5x5: n x n uniform grid including edges, full coverage with
        # overlap, always n^2 crops. Endpoint-style starts match the legacy grid3x3 starts
        # [0, cx, w-CROP] value by value (i=1 gives round((w-CROP)/2)=cx); small-image starts
        # clamp to 0 and out-of-bounds areas are zero-padded by _crop_padded
        n = int(mode[4])
        xs = [min(max(int(round(i * (w - CROP) / (n - 1))), 0), max(w - CROP, 0)) for i in range(n)]
        ys = [min(max(int(round(i * (h - CROP) / (n - 1))), 0), max(h - CROP, 0)) for i in range(n)]
        boxes = [(x, y) for y in ys for x in xs]
    elif mode == 'multiscale':  # center crop at 0.5x / 1x / 2x scales
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
    elif mode == 'single336':  # control group: single 336 crop (RoPE-adaptive), score distribution may shift
        return [_crop_padded(img, (w - 336) // 2, (h - 336) // 2, size=336)]
    else:
        raise ValueError(f'unknown crop_mode: {mode}')
    return [_crop_padded(img, x, y) for x, y in boxes]


# ============ Dataset: each item -> (idx, [K,3,H,W], valid flag) ============

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
                img = _resize_short(img)  # same resize rule as the eval pipeline
            crops = make_crops(img, self.crop_mode)
            # same order as TestDataset: crop first, then PNG -> JPEG q96 alignment
            tensors = []
            for c in crops:
                if path.lower().endswith(('.png', '.bmp', '.tiff')):
                    c = compress_image(c, quality=96)
                tensors.append(self.to_tensor(c))
            return idx, torch.stack(tensors), torch.tensor(1.0)  # third element: 1=valid, 0=read failed
        except Exception as e:
            # even on failure return n_crops zero blocks to keep batch shapes consistent (a single bad image must not kill the batch)
            print(f"[warn] failed to read {path}: {e}")
            size = 336 if self.crop_mode == 'single336' else CROP
            return idx, torch.zeros(self.n_crops, 3, size, size), torch.tensor(0.0)


# ============ Aggregation strategies ============

def aggregate(scores, method, topk_k):
    """scores: [B, K] per-crop P(fake). Returns [B]."""
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


# ============ Main flow ============

def get_args():
    p = argparse.ArgumentParser('MIRROR multi-crop predict')
    weight_default = os.path.join(REPO_ROOT, 'data', 'weights')
    p.add_argument('--data_dir', required=True, nargs='+',
                   help='one or more image directories; processed one by one (model loaded once),'
                        'each directory outputs a JSON into --output_dir named after the folder')
    p.add_argument('--output_json', default='predictions.json',
                   help='output JSON path for single-directory mode')
    p.add_argument('--output_dir', default='',
                   help='unified output directory for multi-directory batch mode (one <folder>.json per folder)')
    p.add_argument('--model_path', default=os.path.join(weight_default, 'checkpoint-h-cur.pth'))
    p.add_argument('--memory_path', default=os.path.join(weight_default, 'mirror_phase1.pth'))
    p.add_argument('--backbone_path', default=os.path.join(weight_default, 'dinov3-huge'))
    p.add_argument('--crop_mode', default='center',
                   choices=['center', 'five', 'grid3x3', 'grid4x4', 'grid5x5',
                            'multiscale', 'single336'])
    p.add_argument('--aggregate', default='mean', choices=['mean', 'median', 'topk', 'max'])
    p.add_argument('--topk_k', type=int, default=0, help='0 = ceil(K/3)')
    p.add_argument('--batch_size', type=int, default=32,
                   help='number of crops per forward batch (same semantics as evaluation/evaluate.py);'
                        'multi-crop modes convert it to images/batch = batch_size // K.'
                        'Default 32 -> ~10GB external accounting; 64 is the verified ceiling (~15GB)')
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader worker processes; preprocessing is ~3ms/image and the GPU forward is the bottleneck,'
                        'spawn workers only add memory on Windows, default 0')
    p.add_argument('--device', default='cuda')
    p.add_argument('--use_amp', action='store_true',
                   help='fp16 mixed precision, noticeably faster (overflow bug fixed), recommended')
    p.add_argument('--no_resize', action='store_true',
                   help='skip the short-side>512->512 resize and crop the original image directly (keeps'
                        'high-frequency artifacts; the DDA inference protocol). Off by default, matching'
                        'the original pipeline / equivalence gate')
    p.add_argument('--dump_scores', default='', help='optional: CSV path for per-crop scores')
    p.add_argument('--limit', type=int, default=0, help='process only the first N images (debugging, 0=all)')
    p.add_argument('--json_format', default='competition', choices=['competition', 'detailed'],
                   help='competition=one {"image_path","pred"} item per image (competition deliverable format, default);'
                        'detailed=meta header + is_fake boolean per image (GUI/experiments)')
    args = p.parse_args()
    # resolve relative weight paths against repo root (the GUI subprocess CWD is scripts/, yet weights should point to <repo>/data/weights)
    for k in ('model_path', 'memory_path', 'backbone_path'):
        if not os.path.isabs(getattr(args, k)):
            setattr(args, k, os.path.join(REPO_ROOT, getattr(args, k)))
    return args


def _run_dir(model, device, args, data_dir):
    """Run inference on one directory. Returns (results, crop_rows)."""
    dataset = PredictDataset(data_dir, args.crop_mode, args.no_resize)
    if args.limit:
        dataset.paths = dataset.paths[:args.limit]
    if not dataset.paths:
        raise SystemExit(f'no images found in: {data_dir}')
    n_crops = N_CROPS[args.crop_mode]
    # root-cause fix: batch_size semantics = crops per batch. Forward scale depends only
    # on the crop count; interpreting it as images/batch would multiply the forward by K
    # (five = 320 crops, grid3x3 = 576 crops -> guaranteed OOM)
    img_per_batch = max(1, args.batch_size // n_crops)
    n_batches = math.ceil(len(dataset) / img_per_batch)
    est_gb = args.batch_size * 0.13 + 3.4  # measured calibration: ~130MB/crop (allocated accounting) + weights
    print(f"    {len(dataset.paths)} images x {n_crops} crops; "
          f"batch_size={args.batch_size} blocks/batch -> {img_per_batch} imgs/batch, {n_batches} batches; "
          f"est. allocated VRAM ~{est_gb:.1f} GiB (+~3GB CUDA context/workspace)", flush=True)

    loader = DataLoader(dataset, batch_size=img_per_batch,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'))

    results, crop_rows = [], []
    t0 = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()  # measure the inference-loop peak only, excluding loading
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
                print(f"    VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB (first batch)",
                      flush=True)
            if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
                done = min((bi + 1) * img_per_batch, len(dataset))
                print(f"    [{bi + 1}/{n_batches} batches] {done}/{len(dataset)} images, "
                      f"{time.time() - t0:.0f} s", flush=True)
    return results, crop_rows


def write_results(path, results, meta, fmt):
    """Write results by json_format. competition=top-level list (deliverable format, byte-compatible);
    detailed=meta header + is_fake boolean per image (failed images pred=-1 -> is_fake=null)."""
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
    """Batch output file name = folder name; duplicate basenames get a parent-dir prefix."""
    name = os.path.basename(os.path.normpath(d))
    if name not in dup_bases:
        return name
    parent = os.path.basename(os.path.normpath(os.path.dirname(os.path.normpath(d))))
    return f'{parent}__{name}'


def main(args):
    device = torch.device(args.device)
    t0 = time.time()

    # output argument validation (fail-fast: readable errors before loading the model)
    dirs = args.data_dir
    batch_mode = len(dirs) > 1
    if args.output_dir:
        if os.path.isfile(args.output_dir):
            raise SystemExit(f'--output_dir points to a file, please provide a directory: {args.output_dir}')
        os.makedirs(args.output_dir, exist_ok=True)
    elif not batch_mode and os.path.isdir(args.output_json):
        # writing to a directory path raises PermissionError
        raise SystemExit(f'--output_json points to a directory, provide the full file path'
                         f' (e.g. {os.path.join(args.output_json, "predictions.json")})')
    if batch_mode and not args.output_dir:
        raise SystemExit('multi-directory batch mode requires --output_dir')

    print(f">>> loading model (crop_mode={args.crop_mode}, aggregate={args.aggregate}"
          f"{', no_resize' if args.no_resize else ''})")
    model = build_mirror(memory_path=args.memory_path, backbone_path=args.backbone_path)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    # transformers 5.x layout fix, same as smoke_test.py / evaluation/evaluate.py
    state = {k.replace("backbone.dino.layer.", "backbone.dino.model.layer."): v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, \
        f"incomplete weight load: missing={len(missing)}, unexpected={len(unexpected)}"
    model.to(device).eval()

    # duplicate-name fallback (the GUI warns earlier; this is the second line of defense):
    # output name = folder name, duplicate basenames in one batch get a parent-dir prefix
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
            print(f"[skip] {e}", flush=True)
            fail_dirs.append((name, str(e)))
            continue
        except Exception as e:
            print(f"[fail] folder {name}: {e} (continuing with the next)", flush=True)
            fail_dirs.append((name, str(e)))
            continue

        out_name = out_json_name(d, dup_bases)
        # output target: --output_dir wins (one JSON per folder), else --output_json (single file)
        # both modes accept any number of directories (GUI sub-batch = single dir + output_dir goes here too)
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
        print(f"[done] {len(results)} images -> {out_path}"
              f" ({n_fail} failed as pred=-1), {time.time() - t0:.0f} s", flush=True)
        if device.type == 'cuda':
            print(f"    VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if results:
            preds = [r['pred'] for r in results if r['pred'] >= 0]
            if preds:
                print(f"    P(fake) stats: min={min(preds):.4f} mean={np.mean(preds):.4f} "
                      f"max={max(preds):.4f}, >0.5 -> fake: {sum(p > 0.5 for p in preds)}",
                      flush=True)
        ok_dirs += 1

    if batch_mode:
        print(f"[batch done] {ok_dirs}/{len(dirs)} folders ok"
              + (f", failed: {fail_dirs}" if fail_dirs else "")
              + f", output dir {args.output_dir}", flush=True)


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    main(get_args())
