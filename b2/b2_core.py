from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Any

from b2_config import B2GridConfig


def world_points_from_layout(layout: Dict[str, Any]) -> np.ndarray:
    n_poles = int(layout["n_poles"])
    stagger = float(layout["stagger_m"])
    lat_gap = float(layout["lat_gap_m"])
    pts = np.zeros((n_poles, 2), np.float32)
    yL, yR = -0.5 * lat_gap, 0.5 * lat_gap
    for pid in range(1, n_poles + 1):
        x = (pid - 1) * stagger
        y = yL if pid % 2 == 1 else yR
        pts[pid - 1] = (x, y)
    return pts


def compute_world_bbox(layout: Dict[str, Any], margins: Dict[str, float]) -> Dict[str, float]:
    n_poles = int(layout["n_poles"])
    stagger = float(layout["stagger_m"])
    lat_gap = float(layout["lat_gap_m"])
    x_min = -float(margins["x_margin_m"])
    x_max = (n_poles - 1) * stagger + float(margins["x_margin_m"])
    y_min = -0.5 * lat_gap - float(margins["y_margin_m"])
    y_max = 0.5 * lat_gap + float(margins["y_margin_m"])
    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }


def compute_bev_grid(layout: Dict[str, Any], margins: Dict[str, float], grid_cfg: B2GridConfig) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    bbox = compute_world_bbox(layout, margins)
    ppm = float(grid_cfg.ppm)
    x_min, x_max = bbox["x_min"], bbox["x_max"]
    y_min, y_max = bbox["y_min"], bbox["y_max"]

    W2G = np.array([
        [ppm, 0.0, -x_min * ppm],
        [0.0, -ppm, y_max * ppm],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    G2W = np.linalg.inv(W2G).astype(np.float32)

    width = int(round((x_max - x_min) * ppm))
    height = int(round((y_max - y_min) * ppm))

    bev_grid = {
        "ppm": ppm,
        "grid_step_m": float(grid_cfg.grid_step_m),
        "canvas_size_px": {"width": width, "height": height},
        "W2G": W2G.tolist(),
        "G2W": G2W.tolist(),
    }
    return bev_grid, W2G, G2W


def assemble_bundle(
    run_id: str,
    b1_result: Dict[str, Any],
    cfg_path: str,
    grid_cfg: B2GridConfig,
    bev_grid: Dict[str, Any],
    world_bbox: Dict[str, float],
    margins: Dict[str, float],
) -> Dict[str, Any]:
    layout = b1_result["layout"]
    bundle = {
        "schema_version": "b2_calib_bundle_v1",
        "run_id": str(run_id),
        "from_b1_result_path": cfg_path,
        "world_def": b1_result.get("world_def", {}),
        "layout": layout,
        "margins": {
            "x_margin_m": float(margins["x_margin_m"]),
            "y_margin_m": float(margins["y_margin_m"]),
        },
        "world_bbox": world_bbox,
        "b2_grid_resolved": {
            "ppm": float(grid_cfg.ppm),
            "grid_step_m": float(grid_cfg.grid_step_m),
            "x_margin_m": float(grid_cfg.x_margin_m),
            "y_margin_m": float(grid_cfg.y_margin_m),
            "style": grid_cfg.style.__dict__,
        },
        "bev_grid": bev_grid,
        "cameras": b1_result["cameras"],
    }
    if "metrics" in b1_result:
        bundle["metrics"] = b1_result["metrics"]
    return bundle


def world_to_grid(W2G: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), np.float32)
    ph = np.concatenate([pts, ones], axis=1)
    qh = (W2G @ ph.T).T
    return qh[:, :2] / (qh[:, 2:3] + 1e-12)
