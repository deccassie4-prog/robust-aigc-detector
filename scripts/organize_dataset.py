"""
organize_dataset.py — assemble the validation set into the 0_real/1_fake layout

Target layout (TestDataset in evaluation/evaluate.py labels by leaf folder name):
  <out_dir>/0_real/   COCO val2017 real images
  <out_dir>/1_fake/   DALL-E Advanced fake images from WildFake

Example:
  python scripts/organize_dataset.py --real_dir <COCO dir> --fake_dir <WildFake root> --out_dir <output dir>
Hardlinks by default (no extra disk space, same NTFS volume required); falls back to copying.
"""
import argparse
import os

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
EXPECTED = {'0_real': 4998, '1_fake': 8843}  # counts given in the competition brief


def collect(root, advanced_only=False):
    paths = []
    for dirpath, _, files in os.walk(root):
        if advanced_only:
            # require 'advanced' in some level of the relative path (images may sit deep)
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
        # subfolders can share file names across sessions (1808 cases measured) -> flatten rel path
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
    p.add_argument('--real_dir', required=True, help='COCO val2017 real-image directory')
    p.add_argument('--fake_dir', required=True,
                   help='extracted WildFake directory; if given a parent with Typical/Advanced, only Advanced is used')
    p.add_argument('--out_dir', required=True, help='output directory (parent of 0_real/1_fake)')
    p.add_argument('--mode', default='hardlink', choices=['hardlink', 'copy', 'move'])
    args = p.parse_args()

    # if fake_dir contains both Advanced and Typical, drill down to Advanced
    sub = [d for d in os.listdir(args.fake_dir)
           if os.path.isdir(os.path.join(args.fake_dir, d))]
    advanced_only = False
    if any('advanced' in d.lower() for d in sub):
        advanced_only = True
        print(f"[hint] fake_dir has subfolders: only collecting images under Advanced paths")

    os.makedirs(os.path.join(args.out_dir, '0_real'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, '1_fake'), exist_ok=True)

    counters = {'skip': 0, 'copied_fallback': 0}
    for label, src_root in [('0_real', args.real_dir), ('1_fake', args.fake_dir)]:
        paths = collect(src_root, advanced_only=(label == '1_fake' and advanced_only))
        dst_dir = os.path.join(args.out_dir, label)
        for src in paths:
            place(src, dst_dir, args.mode, counters, src_root)
        n = len([f for f in os.listdir(dst_dir) if f.lower().endswith(IMG_EXTS)])
        status = 'OK' if n == EXPECTED[label] else f'warning: expected {EXPECTED[label]}'
        print(f"[{label}] collected {len(paths)}, dir now has {n} (skipped {counters['skip']} duplicates) - {status}")
        counters['skip'] = 0

    print(f"\nDone. Evaluate with: python evaluation/evaluate.py --base_data_path {os.path.dirname(args.out_dir)} --benchmarks TechjamVal")


if __name__ == '__main__':
    main()
