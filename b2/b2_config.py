# b2_config.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class B2GridStyle:
    background: tuple[int, int, int] = (245, 245, 245)
    grid_color: tuple[int, int, int] = (210, 210, 210)
    grid_thickness: int = 1
    major_color: tuple[int, int, int] = (180, 180, 180)
    major_thickness: int = 1
    centerline_color: tuple[int, int, int] = (0, 0, 255)
    centerline_thickness: int = 2
    left_color: tuple[int, int, int] = (0, 160, 255)
    right_color: tuple[int, int, int] = (0, 200, 0)
    column_thickness: int = 2
    pole_color: tuple[int, int, int] = (0, 0, 0)
    pole_radius: int = 6
    pole_label_color: tuple[int, int, int] = (30, 30, 30)
    pole_label_scale: float = 0.65
    pole_label_thickness: int = 2


@dataclass(frozen=True)
class B2GridConfig:
    ppm: float = 100.0
    grid_step_m: float = 1.0
    x_margin_m: float = 1.0
    y_margin_m: float = 1.0
    style: B2GridStyle = field(default_factory=B2GridStyle)


@dataclass(frozen=True)
class VisConfig:
    enable: bool = True
    frame_mode: str = "middle"
    draw_all_poles: bool = True
    draw_observed: bool = True
    emit_bev_observed: bool = False


@dataclass(frozen=True)
class VideoConfig:
    cam1_path: Optional[str] = None
    cam2_path: Optional[str] = None


@dataclass(frozen=True)
class B2Config:
    run_id: str
    b1_result_path: str
    b2_grid: B2GridConfig
    videos: Optional[VideoConfig] = None
    vis: VisConfig = field(default_factory=VisConfig)


def _style_from_dict(d: Dict[str, Any]) -> B2GridStyle:
    base = B2GridStyle()
    merged = _deep_update(base.__dict__, d)
    return B2GridStyle(**merged)


def _grid_from_dict(d: Dict[str, Any]) -> B2GridConfig:
    base = {
        "ppm": 100.0,
        "grid_step_m": 1.0,
        "x_margin_m": 1.0,
        "y_margin_m": 1.0,
        "style": {},
    }
    merged = _deep_update(base, d)
    style = _style_from_dict(merged.get("style", {}))
    return B2GridConfig(
        ppm=float(merged["ppm"]),
        grid_step_m=float(merged["grid_step_m"]),
        x_margin_m=float(merged["x_margin_m"]),
        y_margin_m=float(merged["y_margin_m"]),
        style=style,
    )


def _vis_from_dict(d: Dict[str, Any]) -> VisConfig:
    base = VisConfig()
    merged = _deep_update(base.__dict__, d)
    return VisConfig(**merged)


def _videos_from_dict(d: Dict[str, Any]) -> VideoConfig:
    base = VideoConfig()
    merged = _deep_update(base.__dict__, d)
    return VideoConfig(**merged)


def load_b2_config(path: str) -> B2Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "run_id" not in raw:
        raise KeyError("b2_config: missing required field 'run_id'")
    if "b1_result_path" not in raw:
        raise KeyError("b2_config: missing required field 'b1_result_path'")
    if "b2_grid" not in raw:
        raise KeyError("b2_config: missing required field 'b2_grid'")

    grid_cfg = _grid_from_dict(raw.get("b2_grid", {}))
    vis_cfg = _vis_from_dict(raw.get("vis", {}))

    videos_cfg = None
    if "videos" in raw:
        videos_cfg = _videos_from_dict(raw.get("videos", {}))

    return B2Config(
        run_id=str(raw["run_id"]),
        b1_result_path=str(raw["b1_result_path"]),
        b2_grid=grid_cfg,
        videos=videos_cfg,
        vis=vis_cfg,
    )
