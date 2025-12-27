# b2_vis.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any

import cv2
import numpy as np

from b2_core import world_points_from_layout, world_to_grid
from b2_config import B2GridConfig, VisConfig


def _color_tuple(c):
    if isinstance(c, (list, tuple)):
        return tuple(int(x) for x in c)
    return (0, 0, 0)


def _render_grid_image(layout: Dict[str, Any], grid_cfg: B2GridConfig, bev_grid: Dict[str, Any]) -> np.ndarray:
    size = bev_grid["canvas_size_px"]
    W2G = np.asarray(bev_grid["W2G"], np.float32)
    img = np.full((int(size["height"]), int(size["width"]), 3), grid_cfg.style.background, np.uint8)

    ppm = float(bev_grid["ppm"])
    step_m = float(bev_grid["grid_step_m"])
    step_px = step_m * ppm

    h, w = img.shape[:2]

    # grid lines
    if step_px > 0:
        major_every = 5
        for u in np.arange(0, w + 1, step_px):
            color = grid_cfg.style.grid_color
            thick = grid_cfg.style.grid_thickness
            idx = int(round(u / step_px))
            if idx % major_every == 0:
                color = grid_cfg.style.major_color
                thick = grid_cfg.style.major_thickness
            cv2.line(img, (int(round(u)), 0), (int(round(u)), h - 1), color, thick, cv2.LINE_AA)
        for v in np.arange(0, h + 1, step_px):
            color = grid_cfg.style.grid_color
            thick = grid_cfg.style.grid_thickness
            idx = int(round(v / step_px))
            if idx % major_every == 0:
                color = grid_cfg.style.major_color
                thick = grid_cfg.style.major_thickness
            cv2.line(img, (0, int(round(v))), (w - 1, int(round(v))), color, thick, cv2.LINE_AA)

    # centerline and columns (horizontal lines at constant y)
    y_center = 0.0
    lat_gap = float(layout["lat_gap_m"])
    v_center = world_to_grid(W2G, np.array([[0.0, y_center]], np.float32))[0, 1]
    cv2.line(
        img,
        (0, int(round(v_center))),
        (w - 1, int(round(v_center))),
        grid_cfg.style.centerline_color,
        grid_cfg.style.centerline_thickness,
        cv2.LINE_AA,
    )

    y_left = -0.5 * lat_gap
    y_right = 0.5 * lat_gap
    for y_val, color in [(y_left, grid_cfg.style.left_color), (y_right, grid_cfg.style.right_color)]:
        v_line = world_to_grid(W2G, np.array([[0.0, y_val]], np.float32))[0, 1]
        cv2.line(
            img,
            (0, int(round(v_line))),
            (w - 1, int(round(v_line))),
            color,
            grid_cfg.style.column_thickness,
            cv2.LINE_AA,
        )

    # poles
    world_pts = world_points_from_layout(layout)
    grid_pts = world_to_grid(W2G, world_pts)
    for i, (u, v) in enumerate(grid_pts):
        cv2.circle(img, (int(round(u)), int(round(v))), grid_cfg.style.pole_radius, grid_cfg.style.pole_color, -1)
        cv2.putText(
            img,
            f"P{i + 1}",
            (int(round(u + 8)), int(round(v - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            grid_cfg.style.pole_label_scale,
            grid_cfg.style.pole_label_color,
            grid_cfg.style.pole_label_thickness,
            cv2.LINE_AA,
        )

    return img


def draw_grid_overlay(out_path: str, layout: Dict[str, Any], grid_cfg: B2GridConfig, bev_grid: Dict[str, Any]) -> None:
    img = _render_grid_image(layout, grid_cfg, bev_grid)
    cv2.imwrite(out_path, img)


def _middle_frame(cap: cv2.VideoCapture):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = int(np.floor(total / 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to grab middle frame from video")
    return frame


def _project_world_points(H_w2p: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    ones = np.ones((pts_world.shape[0], 1), np.float32)
    wh = np.concatenate([pts_world.astype(np.float32), ones], axis=1)
    ph = (H_w2p @ wh.T).T
    return ph[:, :2] / (ph[:, 2:3] + 1e-12)


def _draw_world_frame(img: np.ndarray, H_w2p: np.ndarray, world_bbox: Dict[str, float]) -> None:
    """Draws the world bounding box and XY axes in pixel space."""
    corners_w = np.array(
        [
            [world_bbox["x_min"], world_bbox["y_min"]],
            [world_bbox["x_max"], world_bbox["y_min"]],
            [world_bbox["x_max"], world_bbox["y_max"]],
            [world_bbox["x_min"], world_bbox["y_max"]],
        ],
        np.float32,
    )
    corners_p = _project_world_points(H_w2p, corners_w)
    pts = corners_p.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [pts], True, (80, 180, 80), 2, cv2.LINE_AA)

    origin_p = _project_world_points(H_w2p, np.array([[0.0, 0.0]], np.float32))[0]
    x_axis = _project_world_points(H_w2p, np.array([[max(1.0, world_bbox["x_max"] * 0.2), 0.0]], np.float32))[0]
    y_axis = _project_world_points(H_w2p, np.array([[0.0, max(1.0, abs(world_bbox["y_max"]) * 0.6)]], np.float32))[0]

    origin_i = tuple(np.round(origin_p).astype(int))
    cv2.drawMarker(img, origin_i, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, origin_i, tuple(np.round(x_axis).astype(int)), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.08)
    cv2.putText(img, "X+", tuple(np.round(x_axis + [6, -6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, origin_i, tuple(np.round(y_axis).astype(int)), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.08)
    cv2.putText(img, "Y+", tuple(np.round(y_axis + [6, -6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


def draw_cam_overlay(cam_result: Dict[str, Any], video_path: str, out_path: str, vis_cfg: VisConfig, world_bbox: Dict[str, float]) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if vis_cfg.frame_mode != "middle":
        raise ValueError("Only frame_mode='middle' is supported in B2")

    frame = _middle_frame(cap)
    cap.release()

    img = frame.copy()
    h, w = img.shape[:2]

    H_w2p = np.asarray(cam_result.get("H_w2p"), np.float32)
    _draw_world_frame(img, H_w2p, world_bbox)

    # all poles
    if vis_cfg.draw_all_poles:
        for pid_str, pt in cam_result.get("poles_px", {}).items():
            x, y = float(pt[0]), float(pt[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (int(round(x)), int(round(y))), 6, (0, 180, 255), -1)
                cv2.putText(img, f"P{pid_str}", (int(round(x + 8)), int(round(y - 8))), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 160, 255), 2, cv2.LINE_AA)

    # observed
    if vis_cfg.draw_observed:
        for pid_str, pt in cam_result.get("observed_px", {}).items():
            x, y = float(pt[0]), float(pt[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (int(round(x)), int(round(y))), 8, (0, 0, 255), -1)
                cv2.putText(img, f"P{pid_str}", (int(round(x + 8)), int(round(y - 8))), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)


def draw_cam1_bev(out_path: str, cam_result: Dict[str, Any], video_path: str, bev_grid: Dict[str, Any], grid_cfg: B2GridConfig, layout: Dict[str, Any]) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame = _middle_frame(cap)
    cap.release()

    H_p2w = np.asarray(cam_result.get("H_p2w"), np.float32)
    W2G = np.asarray(bev_grid["W2G"], np.float32)
    canvas = bev_grid["canvas_size_px"]

    M = W2G @ H_p2w
    bev_img = cv2.warpPerspective(frame, M, (int(canvas["width"]), int(canvas["height"])), flags=cv2.INTER_LINEAR)

    grid_img = _render_grid_image(layout, grid_cfg, bev_grid) if isinstance(grid_cfg, B2GridConfig) else None
    if grid_img is not None:
        bev_img = cv2.addWeighted(bev_img, 0.7, grid_img, 0.3, 0)

    cv2.imwrite(out_path, bev_img)


def draw_bev_observed(out_path: str, b1_result: Dict[str, Any], bev_grid: Dict[str, Any]) -> None:
    W2G = np.asarray(bev_grid["W2G"], np.float32)
    size = bev_grid["canvas_size_px"]
    img = np.full((int(size["height"]), int(size["width"]), 3), 250, np.uint8)

    colors = {"cam1": (0, 160, 255), "cam2": (255, 0, 0)}
    for cam_id, cam in b1_result.get("cameras", {}).items():
        if cam_id not in colors:
            continue
        H = np.asarray(cam["H_p2w"], np.float32)
        obs = cam.get("observed_px", {})
        if not obs:
            continue
        pts_px = np.array(list(obs.values()), np.float32)
        ones = np.ones((pts_px.shape[0], 1), np.float32)
        ph = np.concatenate([pts_px, ones], axis=1)
        qh = (H @ ph.T).T
        world = qh[:, :2] / (qh[:, 2:3] + 1e-12)
        grid_pts = world_to_grid(W2G, world)
        for (u, v), pid_str in zip(grid_pts, obs.keys()):
            cv2.circle(img, (int(round(u)), int(round(v))), 6, colors[cam_id], -1)
            cv2.putText(img, f"{cam_id}-{pid_str}", (int(round(u + 6)), int(round(v - 6))), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colors[cam_id], 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
