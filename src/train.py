# src/train.py
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.models.lstm import LSTMAutoEncoder


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(device_cfg: str) -> torch.device:
    device_cfg = (device_cfg or "auto").lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dir(runs_dir: str, exp_name: str) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(runs_dir) / f"{ts}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@torch.no_grad()
def eval_recon_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch  # (B,T,F)
        x = x.to(device)
        x_hat = model(x)
        loss = criterion(x_hat, x)
        losses.append(loss.detach().float().item())
    return float(np.mean(losses)) if losses else float("nan")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    use_amp: bool = True,
) -> float:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    losses = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch  # (B,T,F)
        x = x.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            x_hat = model(x)
            loss = criterion(x_hat, x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach().float().item())

    return float(np.mean(losses)) if losses else float("nan")


def train_from_arrays(
    cfg,
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
) -> Dict:
    """
    노트북에서 X_train (N,F)만 넘기면
    windowing은 노트북에서 이미 하고 DataLoader를 넘기는 방식이 아니라,
    여기서는 '윈도우까지 이미 만들어진 loader'가 아니라 '배열'을 받는 버전이야.

    지금 단계에선 구조만 확정하려고, loader 생성은 노트북에서 하고
    이 함수 대신 train_from_loaders를 쓰는 걸 추천함.
    """
    raise NotImplementedError("이번 단계에서는 loader 기반 함수(train_from_loaders)를 사용해줘.")


def train_from_loaders(
    cfg,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_features: Optional[int] = None,
) -> Dict:
    # cfg는 dict/OmegaConf 둘 다 허용
    cfg = OmegaConf.create(cfg)

    seed = int(cfg.train.seed)
    set_seed(seed)

    device = pick_device(str(cfg.train.device))
    use_amp = bool(cfg.train.use_amp)

    # n_features 자동 추론 (loader 첫 배치로)
    if n_features is None:
        first = next(iter(train_loader))
        x0 = first[0] if isinstance(first, (list, tuple)) else first
        n_features = int(x0.shape[-1])

    # 모델 생성
    model = LSTMAutoEncoder(
        n_features=n_features,
        hidden_size=int(cfg.model.hidden_size),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    # runs 디렉토리 생성 + 설정 스냅샷 저장
    run_dir = make_run_dir(str(cfg.logging.runs_dir), str(cfg.experiment.name))
    (run_dir / "artifacts").mkdir(exist_ok=True)

    # cfg에 n_features 채워서 저장
    cfg.model.n_features = n_features
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    history = {
        "train_loss": [],
        "val_loss": [],
        "device": str(device),
    }

    best_val = float("inf")
    best_path = run_dir / "model_best.pt"
    last_path = run_dir / "model_last.pt"

    for epoch in range(1, int(cfg.train.epochs) + 1):
        tr = train_one_epoch(model, train_loader, device, optimizer, criterion, use_amp=use_amp)
        history["train_loss"].append(tr)

        if val_loader is not None:
            vl = eval_recon_loss(model, val_loader, device, criterion)
            history["val_loss"].append(vl)

            if vl < best_val:
                best_val = vl
                if bool(cfg.logging.save_model):
                    torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg)}, best_path)
        else:
            history["val_loss"].append(None)

        # 매 epoch 간단 로그 저장
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Epoch {epoch:03d} | train={tr:.6f}" + (f" | val={vl:.6f}" if val_loader is not None else ""))

    if bool(cfg.logging.save_model):
        torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg)}, last_path)

    return {
        "run_dir": str(run_dir),
        "history": history,
        "best_model_path": str(best_path) if best_path.exists() else None,
        "last_model_path": str(last_path) if last_path.exists() else None,
        "n_features": n_features,
    }
