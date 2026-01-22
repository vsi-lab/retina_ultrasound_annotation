# tests/test_usg_train_eval.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.train_seg as train_seg
import training.eval_seg as eval_seg



def _find_ckpt(out_dir: Path):
    hits = list(out_dir.glob("*.ckpt"))
    return hits[0] if hits else None


def _find_metrics_csv(out_dir: Path):
    hits = list(out_dir.glob("metrics.csv"))
    return hits[0] if hits else None


def test_usg_train_smoke_and_ckpt(tmp_usg_repo, monkeypatch):
    """
    Smoke test:
      - run a tiny USG training (epochs controlled by the test config)
      - verify that a checkpoint is written.
    """
    cfg_path, work_root = tmp_usg_repo

    out_dir = work_root / "runs" / "seg_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CLI: train_seg --config CONFIG --out OUT
    argv = [
        "train_seg",
        "--config", str(cfg_path),
        "--out", str(out_dir),
    ]
    monkeypatch.setattr("sys.argv", argv)
    train_seg.main()

    ckpt = _find_ckpt(out_dir)
    assert ckpt is not None, f"No checkpoint found under {out_dir}"
    assert ckpt.stat().st_size > 0


def test_usg_eval_smoke_and_metrics(tmp_usg_repo, monkeypatch):
    """
    Train briefly (as above), then run eval_seg and check that
    a metrics.csv file is written and is non-empty.
    """
    cfg_path, work_root = tmp_usg_repo

    out_dir = work_root / "runs" / "seg_smoke"
    eval_dir = work_root / "runs" / "seg_smoke_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) train once ----
    argv_train = [
        "train_seg",
        "--config", str(cfg_path),
        "--out", str(out_dir),
    ]
    monkeypatch.setattr("sys.argv", argv_train)
    train_seg.main()

    ckpt = _find_ckpt(out_dir)
    assert ckpt is not None, f"No checkpoint found under {out_dir}"

    # ---- 2) eval ----
    argv_eval = [
        "eval_seg",
        "--config", str(cfg_path),
        "--ckpt", str(ckpt),
        "--out", str(eval_dir),
    ]
    monkeypatch.setattr("sys.argv", argv_eval)
    eval_seg.main()

    metrics_csv = _find_metrics_csv(eval_dir)
    assert metrics_csv is not None, f"No metrics.csv written under {eval_dir}"

    import pandas as pd
    df = pd.read_csv(metrics_csv)
    assert not df.empty, "metrics.csv is empty"