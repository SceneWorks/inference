"""Refuse task-cache symlinks outside the approved internal directory, including XET."""

import os
from pathlib import Path
import sys


def verify_root(root):
    if root.is_symlink():
        raise ValueError(f"weights root is a symlink: {root}")
    allowed = root.resolve(strict=True)
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders + files:
            path = Path(directory) / name
            if path.is_symlink() and not path.resolve(strict=True).is_relative_to(allowed):
                raise ValueError(f"cache symlink escapes task-owned root: {path}")
    return allowed


if __name__ == "__main__":
    print(verify_root(Path(sys.argv[1])))
