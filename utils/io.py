from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, best_score: float, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'best_score': best_score,
            'config': config,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location=None) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
