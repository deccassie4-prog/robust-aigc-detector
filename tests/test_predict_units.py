# -*- coding: utf-8 -*-
"""predict.py / predict_gui.py 纯函数单元测试：命令构造（含多目录）、进度解析（含分节）、重名检测。

headless：不开 Tk、不开 GPU、无子进程。所有临时文件与样例图片均用 tempfile 现造。
可直接 `python tests/test_predict_units.py` 运行。
"""
import json
import os
import sys
import tempfile

try:
    import pytest
except ImportError:  # 允许无 pytest 直接运行（走文件尾的 __main__ 入口）
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.predict import out_json_name, write_results
from scripts.predict_gui import (build_command, count_images, find_name_conflicts,
                                 find_out_conflicts, parse_line)


def test_parse_line_single_dir():
    """解析器：真实运行输出逐行喂（单目录）。"""
    state = {}
    lines = [
        "    60 张图 × 5 裁剪块；batch_size=32 块/批 → 6 图/批，共 10 批",
        "    显存峰值 7.6 GiB（首批）",
        "    [3/10 批] 18/60 张，用时 12 秒",
        "[完成] 60 张 -> out.json（失败 1 张记为 pred=-1），用时 18 秒",
        "    P(fake) 分布: min=0.0681 mean=0.6452 max=0.9385, >0.5 判 fake: 44 张",
    ]
    for l in lines:
        parse_line(l, state)
    assert state['total'] == 10 and state['batch'] == 3 and state['vram'] == '7.6'
    assert state['done_n'] == '60' and state['fake_n'] == '44'
    assert state['folders'] == [{'name': '', 'n': '60', 'fake': '44'}], state['folders']


def test_parse_line_batch_sections():
    """批处理分节解析：分节重置进度、累积 folders、统计失败文件夹。"""
    state = {}
    lines = [
        "=== [1/3] real ===",
        "    66 张图 × 1 裁剪块；共 3 批",
        "    [3/3 批] 66/66 张，用时 30 秒",
        "[完成] 66 张 -> real.json（失败 0 张记为 pred=-1），用时 30 秒",
        "    P(fake) 分布: ..., >0.5 判 fake: 2 张",
        "=== [2/3] diffusion ===",
        "    [1/2 批] 40/46 张，用时 10 秒",
        "[跳过] 目录里没有图片: bad",
        "=== [3/3] real ===",
        "[完成] 46 张 -> parent__real.json（失败 0 张记为 pred=-1），用时 5 秒",
        "    P(fake) 分布: ..., >0.5 判 fake: 40 张",
        "[批处理完成] 成功 2/3 个文件夹，失败: [('bad', '...')]",
    ]
    for l in lines:
        parse_line(l, state)
    assert state['sec_n'] == 3 and state['sec_k'] == 3 and state['sec_name'] == 'real'
    assert state['folder_fails'] == 1
    assert [f['name'] for f in state['folders']] == ['real', 'real']
    assert state['folders'][0]['fake'] == '2' and state['folders'][1]['fake'] == '40'
    assert state['total'] == 0 and state['batch'] == 0  # 分节头重置了批进度


def test_build_command_multi_dir():
    """build_command：多目录展开 + EXTRA 标志；注册表参数 --json_format 只出现一次。"""
    params = [
        {"flag": "--data_dir", "widget": "dir", "default": ""},
        {"flag": "--output_json", "widget": "savefile", "default": ""},
        {"flag": "--json_format", "widget": "choice", "default": "detailed",
         "choices": ["detailed", "competition"]},
        {"flag": "--batch_size", "widget": "int", "default": 32},
        {"flag": "--use_amp", "widget": "bool", "default": False},
    ]
    v = {"--data_dir": ["X:\\a\\real", "X:\\b\\diffusion"], "--output_json": "",
         "--json_format": "detailed", "--batch_size": "32", "--use_amp": True,
         "--output_dir": "X:\\out"}
    cmd = build_command(v, params)
    assert cmd.count("--data_dir") == 1, "argparse nargs='+' 重复标志会覆盖！"
    assert cmd.count("--json_format") == 1, "--json_format 重复传参！"
    i = cmd.index("--data_dir")
    assert cmd[i + 1:i + 3] == ["X:\\a\\real", "X:\\b\\diffusion"]
    assert cmd[cmd.index("--output_dir") + 1] == "X:\\out"
    assert cmd[cmd.index("--json_format") + 1] == "detailed"
    assert "--output_json" not in cmd  # 批处理时置空被跳过
    v1 = {"--data_dir": "X:\\one", "--output_json": "x.json", "--batch_size": "32",
          "--use_amp": False}
    cmd1 = build_command(v1, params)
    assert cmd1.count("--data_dir") == 1 and cmd1[cmd1.index("--data_dir") + 1] == "X:\\one"


def test_name_conflicts_and_out_json_name():
    """重名检测与改名（路径断言依赖 Windows 分隔符语义）。"""
    conf = find_name_conflicts(["X:\\a\\real", "X:\\b\\real", "X:\\c\\fake"])
    assert set(conf) == {'real'} and len(conf['real']) == 2
    assert out_json_name("X:\\a\\real", {'real'}) == 'a__real'
    assert out_json_name("X:\\a\\real", set()) == 'real'
    assert out_json_name("X:\\a\\real\\", {'x'}) == 'real'  # 尾部斜杠归一化


def test_out_conflicts():
    """跨子批次输出文件夹冲突检测：大小写不敏感，返回每组首个路径。"""
    if os.name != 'nt':
        # normcase 大小写归一化是 Windows 特有语义
        if pytest is not None:
            pytest.skip('non-Windows')
        return
    conf = find_out_conflicts(["X:\\out\\res", "X:\\OUT\\RES", "X:\\other"])
    assert conf == ["X:\\out\\res"], conf
    assert find_out_conflicts(["X:\\a", "X:\\b"]) == []


def test_write_results_schemas():
    """write_results：detailed schema + competition 兼容（等价门禁同构）。"""
    tmp = tempfile.mkdtemp(prefix='_units_')
    try:
        results = [{"image_path": "a.png", "pred": 0.934},
                   {"image_path": "b.png", "pred": 0.021},
                   {"image_path": "c.png", "pred": -1.0}]
        meta = {"format_version": 2, "source_dir": "X", "threshold": 0.5, "n_images": 3}
        p_det = os.path.join(tmp, 'detailed.json')
        p_comp = os.path.join(tmp, 'comp.json')
        write_results(p_det, results, meta, 'detailed')
        write_results(p_comp, results, meta, 'competition')
        d = json.load(open(p_det, encoding='utf-8'))
        assert set(d) == {'meta', 'images'} and d['meta']['threshold'] == 0.5
        assert d['images'][0]['is_fake'] is True and d['images'][1]['is_fake'] is False
        assert d['images'][2]['is_fake'] is None and d['images'][2]['pred'] == -1.0
        c = json.load(open(p_comp, encoding='utf-8'))
        assert isinstance(c, list) and c[0] == {"image_path": "a.png", "pred": 0.934}
        # competition 输出与等价门禁文件同构（顶层列表、字段 image_path/pred）
        assert set(c[0]) == {'image_path', 'pred'}
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_count_images():
    """count_images 回归：递归计数只看扩展名。原回归依赖本机 test_images 的 7 张图，这里现造等价目录。"""
    tmp = tempfile.mkdtemp(prefix='_units_img_')
    try:
        sub = os.path.join(tmp, 'sub')
        os.makedirs(sub)
        names = ['a.png', 'b.jpg', 'c.jpeg', 'd.bmp', 'e.webp',
                 os.path.join('sub', 'f.png'), os.path.join('sub', 'g.JPG')]
        for n in names:
            open(os.path.join(tmp, n), 'wb').close()
        open(os.path.join(tmp, 'ignored.txt'), 'wb').close()
        assert count_images(tmp) == 7
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    fns = [test_parse_line_single_dir, test_parse_line_batch_sections,
           test_build_command_multi_dir, test_name_conflicts_and_out_json_name,
           test_out_conflicts, test_write_results_schemas, test_count_images]
    for i, fn in enumerate(fns, 1):
        fn()
        print(f'{i} {fn.__name__} OK')
    print('\n全部单元测试通过')
