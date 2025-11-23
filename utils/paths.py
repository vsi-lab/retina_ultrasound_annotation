# utils/paths.py
from pathlib import Path
from typing import Union

def resolve_under_root(p: Union[str, Path], root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)

def resolve_under_root_cfg(cfg, p: Union[str, Path]) -> Path:
    """
    Resolve path under cfg["work_root"].
    Works whether p is absolute or relative.
    """
    work_root = Path(cfg["work_root"])
    return resolve_under_root(p.format(**cfg), work_root)