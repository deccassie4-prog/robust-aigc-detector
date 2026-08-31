"""MIRROR 检测器模型包（基于 handsome-rich/MIRROR，含 transformers 5.x 兼容修复）。"""

from .mirror import MIRROR_Detector, build_mirror

__all__ = ['MIRROR_Detector', 'build_mirror']
