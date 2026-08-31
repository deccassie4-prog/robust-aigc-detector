"""
organize_dataset.py — 把验证集整理成 MIRROR 要求的 0_real/1_fake 结构

目标结构（evaluation/evaluate.py 的 TestDataset 按叶子文件夹名打标签）：
  <out_dir>/0_real/   COCO val2017 真图
  <out_dir>/1_fake/   WildFake 的 DALL·E Advanced 假图

用法示例：
  python scripts/organize_dataset.py --real_dir <COCO目录> --fake_dir <WildFake解压根目录> --out_dir <输出目录>
默认用硬链接（不占额外空间，要求同一 NTFS 卷），失败自动回退为复制。
"""
import argparse
import os

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
EXPECTED = {'0_real': 4998, '1_fake': 8843}  # 赛题文档给的张数


def collect(root, advanced_only=False):
    paths = []
    for dirpath, _, files in os.walk(root):
        if advanced_only:
            # 检查相对路径的任意一层是否带 Advanced（图片可能在 Advanced/DALLE3/dalle3 深处）
            rel = os.path.relpath(dirpath, root).lower()
            if 'advanced' not in rel.split(os.sep):
                continue
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def place(src, dst_dir, mode, counters, src_root):
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst):
        # 跨会话子目录可能同名（本数据集实测 1808 例）→ 用相对路径展平保证唯一且可溯源
        rel = os.path.relpath(src, src_root)
        dst = os.path.join(dst_dir, rel.replace(os.sep, '__'))
    if os.path.exists(dst):
        counters['skip'] += 1
        return
    if mode == 'move':
        os.replace(src, dst)
    elif mode == 'hardlink':
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
            counters['copied_fallback'] += 1
    else:
        import shutil
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real_dir', required=True, help='COCO val2017 图片目录')
    p.add_argument('--fake_dir', required=True,
                   help='DALL·E 解压后的目录；若传的是含 Typical/Advanced 的上层目录，会自动只取 Advanced')
    p.add_argument('--out_dir', required=True, help='输出目录（0_real/1_fake 的父目录）')
    p.add_argument('--mode', default='hardlink', choices=['hardlink', 'copy', 'move'])
    args = p.parse_args()

    # fake_dir 里如果同时有 Advanced 和 Typical，自动下钻到 Advanced
    sub = [d for d in os.listdir(args.fake_dir)
           if os.path.isdir(os.path.join(args.fake_dir, d))]
    advanced_only = False
    if any('advanced' in d.lower() for d in sub):
        advanced_only = True
        print(f"[提示] fake_dir 含子目录，只收集路径中带 Advanced 的图片")

    os.makedirs(os.path.join(args.out_dir, '0_real'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, '1_fake'), exist_ok=True)

    counters = {'skip': 0, 'copied_fallback': 0}
    for label, src_root in [('0_real', args.real_dir), ('1_fake', args.fake_dir)]:
        paths = collect(src_root, advanced_only=(label == '1_fake' and advanced_only))
        dst_dir = os.path.join(args.out_dir, label)
        for src in paths:
            place(src, dst_dir, args.mode, counters, src_root)
        n = len([f for f in os.listdir(dst_dir) if f.lower().endswith(IMG_EXTS)])
        status = 'OK' if n == EXPECTED[label] else f'注意：预期 {EXPECTED[label]}'
        print(f"[{label}] 收集 {len(paths)} 张，目录现有 {n} 张（重名跳过 {counters['skip']}）— {status}")
        counters['skip'] = 0

    print(f"\n完成。评测用法：python evaluation/evaluate.py --base_data_path {os.path.dirname(args.out_dir)} --benchmarks TechjamVal")


if __name__ == '__main__':
    main()
