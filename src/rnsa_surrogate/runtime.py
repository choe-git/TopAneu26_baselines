"""Reproducibility, EMA, atomic checkpoints, and run metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import atomic_json_dump


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def environment_payload() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def config_digest(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_log(path: Path, message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{timestamp} {message.rstrip()}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach().float(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any], device: torch.device) -> None:
        self.decay = float(state["decay"])
        self.shadow = {
            name: value.detach().to(device=device, dtype=torch.float32).clone()
            for name, value in state["shadow"].items()
        }

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        state.update({name: value.to(state[name].dtype) for name, value in self.shadow.items()})
        model.load_state_dict(state)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        current = model.state_dict()
        backup = {name: current[name].detach().clone() for name in self.shadow}
        self.copy_to(model)
        try:
            yield
        finally:
            restored = model.state_dict()
            restored.update(backup)
            model.load_state_dict(restored)


def write_status(path: Path, status: str, **details: Any) -> None:
    atomic_json_dump({"status": status, "updated_at_utc": utc_now(), **details}, path)
