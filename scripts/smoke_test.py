"""
MIRROR 环境与权重 smoke test
在跑全量推理前，用最小代价确认四件事：
  1. 三个权重能正确组合加载（key 全部对上，没有静默丢失）
  2. 模型能在 GPU 上完成一次前向，输出形状正确
  3. 实测总参数量（比赛 <2B 合规证据）
  4. 顺带记录加载耗时和显存占用，为后面估 GPU 时间提供依据
用法（权重位于 <仓库根>/data/weights/，获取方式见 README）：
  python scripts/smoke_test.py                  # 用两张程序生成的图验证机制
  python scripts/smoke_test.py 图片目录 [N]     # 用现成数据集的前 N 张图（默认 8）验证预测
                                               # 预处理与 evaluation/evaluate.py 完全一致
"""
import os
import sys
import time

import torch
from PIL import Image
from torchvision import transforms

# 仓库根锚定：保证本脚本从任意 CWD 运行时都能导入 src/ 与 evaluation/ 包
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.mirror import build_mirror
# 直接复用 evaluation/evaluate.py 的评测预处理，保证逐像素同一条路径
from evaluation.evaluate import RandomScaleCropOrDirect224, compress_image

WEIGHT_DIR = os.path.join(REPO_ROOT, "data", "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.time()
print(f"[1/4] 构建模型（骨干: {os.path.join(WEIGHT_DIR, 'dinov3-huge')}）...")
model = build_mirror(
    memory_path=os.path.join(WEIGHT_DIR, "mirror_phase1.pth"),
    backbone_path=os.path.join(WEIGHT_DIR, "dinov3-huge"),
)

print("[2/4] 加载 Phase 2 检测器 checkpoint ...")
ckpt = torch.load(os.path.join(WEIGHT_DIR, "checkpoint-h-cur.pth"),
                  map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt.get("state_dict", ckpt))
# transformers 5.x 把 DINOv3 层列表从 dino.layer 挪到了 dino.model.layer，
# 而 checkpoint 按旧布局保存，必须重映射 key，否则 192 条 LoRA 权重会被
# strict=False 静默丢弃
state = {k.replace("backbone.dino.layer.", "backbone.dino.model.layer."): v
         for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"    missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
if missing:
    print("    missing 示例:", missing[:5])
if unexpected:
    print("    unexpected 示例:", unexpected[:5])

model.to(DEVICE).eval()

print("[3/4] 统计参数量 ...")
total = sum(p.numel() for p in model.parameters())
by_prefix = {}
for n, p in model.named_parameters():
    top = ".".join(n.split(".")[:2])
    by_prefix[top] = by_prefix.get(top, 0) + p.numel()
print(f"    总参数量: {total / 1e9:.3f} B  ({total:,})")
for k, v in sorted(by_prefix.items(), key=lambda kv: -kv[1]):
    print(f"    {k:<40} {v / 1e6:10.1f} M")
print(f"    <2B 合规: {'是' if total < 2e9 else '否！！！'}")

# ---------- [4/4] 前向测试：数据集模式 或 合成图模式 ----------
img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
to_tensor = transforms.ToTensor()
dataset_dir = sys.argv[1] if len(sys.argv) > 1 else None

print("[4/4] 前向测试 ...")
if dataset_dir:
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    # 递归收集图片，按路径排序取前 N 张，保证结果可复现
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(img_exts):
                paths.append(os.path.join(root, f))
    paths.sort()
    paths = paths[:n_max]
    if not paths:
        sys.exit(f"[错误] 目录里没找到图片: {dataset_dir}")

    pre = RandomScaleCropOrDirect224(infer_policy="short256_center", eval_resize_short=512)
    print(f"    模式: 数据集抽样（{len(paths)} 张，来自 {dataset_dir}）")
    n_fake = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = pre(img)
        if p.lower().endswith(('.png', '.bmp', '.tiff')):
            x = compress_image(x, quality=96)  # 与 TestDataset 的 PNG 对齐逻辑一致
        x = to_tensor(x).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _, _ = model(x)
        p_fake = torch.softmax(logits, dim=1)[0, 1].item()
        verdict = "fake" if p_fake > 0.5 else "real"
        n_fake += verdict == "fake"
        print(f"    P(fake)={p_fake:.4f} -> {verdict:<5} | {os.path.relpath(p, dataset_dir)}")
    print(f"    判为 fake: {n_fake}/{len(paths)}")
else:
    # 一张噪声图、一张平滑渐变图：只验证机制能跑通，不验证检测准确率
    Image.effect_noise((224, 224), 30).convert("RGB").save("_smoke_noise.png")
    grad = Image.new("L", (224, 224))
    grad.putdata([int(255 * (x + y) / 446) for y in range(224) for x in range(224)])
    grad.convert("RGB").save("_smoke_grad.png")
    print("    模式: 合成图（未指定图片目录）")
    for name in ["_smoke_noise.png", "_smoke_grad.png"]:
        x = to_tensor(Image.open(name).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _, _ = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        print(f"    {name}: logits shape {tuple(logits.shape)}, "
              f"P(fake)={prob[1].item():.4f}, P(real)={prob[0].item():.4f}")

if DEVICE == "cuda":
    print(f"    峰值显存: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"完成，用时 {time.time() - t0:.0f} 秒")
