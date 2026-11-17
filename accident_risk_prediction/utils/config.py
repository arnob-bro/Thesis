"""YAML configuration loader with attribute-style access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """Recursive dict that exposes keys as attributes."""

    def __init__(self, mapping: Dict[str, Any] | None = None) -> None:
        super().__init__()
        if mapping:
            for k, v in mapping.items():
                self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)


def to_plain_dict(value: Any) -> Any:
    """Recursively convert Config objects into YAML-serializable containers."""
    if isinstance(value, Config):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    return Config(raw)


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply ``section.key=value`` overrides (values parsed as YAML)."""
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = Config._wrap(yaml.safe_load(value))
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(to_plain_dict(cfg), fp, sort_keys=False)
