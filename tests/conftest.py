"""pytest root config: inject the repo root into sys.path so tests/ can import
scripts/, src/ and evaluation/ packages.

Also takes effect when running `python tests/test_xxx.py` directly (each test file
keeps a dual-mode __main__ entry)."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
