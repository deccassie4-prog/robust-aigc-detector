"""
MIRROR environment & weights smoke test
Before full runs, cheaply verify four things:
  1. the three weights load together correctly (every key matched, nothing silently dropped)
  2. the model completes one forward pass on GPU with the right output shape
  3. the actual parameter count (evidence for the competition <2B constraint)
  4. load time and VRAM usage, as a basis for estimating GPU run time
Usage (weights live in <repo root>/data/weights/, see README for download):
  python scripts/smoke_test.py                  # mechanism check with two synthetic images
  python scripts/smoke_test.py IMAGE_DIR [N]    # predict the first N images (default 8) of a dataset
                                               # preprocessing is identical to evaluation/evaluate.py
"""
import os
import sys
import time

import torch
from PIL import Image
from torchvision import transforms

# repo-root anchor: importable from any CWD (src/ and evaluation/ packages)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.mirror import build_mirror
# reuse the eval preprocessing of evaluation/evaluate.py for a pixel-identical path
from evaluation.evaluate import RandomScaleCropOrDirect224, compress_image

WEIGHT_DIR = os.path.join(REPO_ROOT, "data", "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.time()
print(f"[1/4] Building model (backbone: {os.path.join(WEIGHT_DIR, 'dinov3-huge')})...")
model = build_mirror(
    memory_path=os.path.join(WEIGHT_DIR, "mirror_phase1.pth"),
    backbone_path=os.path.join(WEIGHT_DIR, "dinov3-huge"),
)

print("[2/4] Loading Phase 2 detector checkpoint ...")
ckpt = torch.load(os.path.join(WEIGHT_DIR, "checkpoint-h-cur.pth"),
                  map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt.get("state_dict", ckpt))
# transformers 5.x moved the DINOv3 layer list from dino.layer to dino.model.layer,
# while the checkpoint was saved with the old layout: keys must be remapped or the
# 192 LoRA weights get silently dropped by strict=False
state = {k.replace("backbone.dino.layer.", "backbone.dino.model.layer."): v
         for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"    missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
if missing:
    print("    missing examples:", missing[:5])
if unexpected:
    print("    unexpected examples:", unexpected[:5])

model.to(DEVICE).eval()

print("[3/4] Counting parameters ...")
total = sum(p.numel() for p in model.parameters())
by_prefix = {}
for n, p in model.named_parameters():
    top = ".".join(n.split(".")[:2])
    by_prefix[top] = by_prefix.get(top, 0) + p.numel()
print(f"    total params: {total / 1e9:.3f} B  ({total:,})")
for k, v in sorted(by_prefix.items(), key=lambda kv: -kv[1]):
    print(f"    {k:<40} {v / 1e6:10.1f} M")
print(f"    <2B compliant: {'yes' if total < 2e9 else 'NO !!!'}")

# ---------- [4/4] Forward tests: dataset mode or synthetic mode ----------
img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
to_tensor = transforms.ToTensor()
dataset_dir = sys.argv[1] if len(sys.argv) > 1 else None

print("[4/4] Forward test ...")
if dataset_dir:
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    # collect images recursively, sort paths, take first N -> reproducible
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(img_exts):
                paths.append(os.path.join(root, f))
    paths.sort()
    paths = paths[:n_max]
    if not paths:
        sys.exit(f"[error] no images found in: {dataset_dir}")

    pre = RandomScaleCropOrDirect224(infer_policy="short256_center", eval_resize_short=512)
    print(f"    mode: dataset sampling ({len(paths)} images from {dataset_dir})")
    n_fake = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = pre(img)
        if p.lower().endswith(('.png', '.bmp', '.tiff')):
            x = compress_image(x, quality=96)  # same PNG alignment as TestDataset
        x = to_tensor(x).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _, _ = model(x)
        p_fake = torch.softmax(logits, dim=1)[0, 1].item()
        verdict = "fake" if p_fake > 0.5 else "real"
        n_fake += verdict == "fake"
        print(f"    P(fake)={p_fake:.4f} -> {verdict:<5} | {os.path.relpath(p, dataset_dir)}")
    print(f"    judged fake: {n_fake}/{len(paths)}")
else:
    # one noise image + one smooth gradient: mechanism check only, not accuracy
    Image.effect_noise((224, 224), 30).convert("RGB").save("_smoke_noise.png")
    grad = Image.new("L", (224, 224))
    grad.putdata([int(255 * (x + y) / 446) for y in range(224) for x in range(224)])
    grad.convert("RGB").save("_smoke_grad.png")
    print("    mode: synthetic images (no image dir given)")
    for name in ["_smoke_noise.png", "_smoke_grad.png"]:
        x = to_tensor(Image.open(name).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _, _ = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        print(f"    {name}: logits shape {tuple(logits.shape)}, "
              f"P(fake)={prob[1].item():.4f}, P(real)={prob[0].item():.4f}")

if DEVICE == "cuda":
    print(f"    peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"done in {time.time() - t0:.0f} s")
