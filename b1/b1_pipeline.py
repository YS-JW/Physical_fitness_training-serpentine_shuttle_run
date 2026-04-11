# Usage:
#   python b1_pipeline.py --cfg b1_config.json --cam1 path/to/cam1.mp4 --cam2 path/to/cam2.mp4 --show
#   python b1_pipeline.py --cfg b1_config.json --cam1 ... --cam2 ... --out-json out_b1.json
#   python b1_pipeline.py --cfg b1_config.json --cam1 ... --cam2 ... --save-vis ./vis

from __future__ import annotations

import os
import json
import argparse
from dataclasses import asdict
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

from b1_config import load_b1_config
from pole_detector import YoloV5, PoleDetector

from obs_normalizer import pole_setpx_to_detections
from candidate_generator import CamObs, LayoutSpec
from multicam_resolver import solve_two_cam


# -----------------------------
# strict helpers
# -----------------------------
def _must_exist(path: str, what: str) -> None:
    if not path or not os.path.exists(path):
        raise RuntimeError(f"{what} not found: {path}")


def _read_frame_at_sec_strict(video_path: str, sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame at {sec:.2f}s from {video_path}")
    return frame


def _apply_H(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), np.float32)
    ph = np.concatenate([pts, ones], axis=1)
    qh = (H @ ph.T).T
    return qh[:, :2] / (qh[:, 2:3] + 1e-12)


def _get_H_w2p_from_joint(joint, which: str) -> np.ndarray:
    candidates = []
    if which == "cam1":
        candidates = [
            "H_w2p_cam1",
            "cam1_H_w2p",
            "H_w2p1",
        ]
    elif which == "cam2":
        candidates = [
            "H_w2p_cam2",
            "cam2_H_w2p",
            "H_w2p2",
        ]
    else:
        raise ValueError("which must be cam1/cam2")

    for name in candidates:
        if hasattr(joint, name):
            H = getattr(joint, name)
            H = np.asarray(H, np.float32)
            if H.shape != (3, 3):
                raise RuntimeError(f"{which}: {name} shape invalid: {H.shape}, expect (3,3)")
            return H


    if hasattr(joint, which):
        obj = getattr(joint, which)
        # cam?.fit.H_w2p
        if hasattr(obj, "fit") and hasattr(obj.fit, "H_w2p"):
            H = np.asarray(obj.fit.H_w2p, np.float32)
            if H.shape != (3, 3):
                raise RuntimeError(f"{which}: {which}.fit.H_w2p invalid shape {H.shape}")
            return H
        # cam?.H_w2p
        if hasattr(obj, "H_w2p"):
            H = np.asarray(obj.H_w2p, np.float32)
            if H.shape != (3, 3):
                raise RuntimeError(f"{which}: {which}.H_w2p invalid shape {H.shape}")
            return H

    raise RuntimeError(
        f"Cannot find H_w2p for {which} in joint result. "
        f"Expected one of {candidates} or {which}.fit.H_w2p / {which}.H_w2p"
    )


def _pick_vis_frame_sec_from_cfg(cfg) -> float:
    window_mode = str(getattr(cfg.pole_detector, "window_mode", "start")).lower()
    sample_secs = float(getattr(cfg.pole_detector, "sample_secs", 0.0))
    if sample_secs <= 0:
        sample_secs = float(getattr(cfg.pole_detector, "init_secs"))
    if window_mode == "custom":
        start_sec = float(getattr(cfg.pole_detector, "start_sec"))
    elif window_mode == "center":
        # center 模式下 pole_detector 内部会估计起点；这里无法复现它的估计就别假装正确
        raise RuntimeError("window_mode='center' is not supported for visualization frame alignment (strict mode).")
    else:
        start_sec = 0.0
    return float(start_sec + 0.5 * sample_secs)


# -----------------------------
# visualization
# -----------------------------
def _draw_projection(frame_bgr: np.ndarray, proj: Dict[int, Tuple[float, float]], title: str) -> np.ndarray:
    img = frame_bgr.copy()
    cv2.putText(img, title, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    for pid in range(1, 8):
        x, y = proj[pid]
        xi, yi = int(round(x)), int(round(y))
        cv2.drawMarker(img, (xi, yi), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=18, thickness=2, line_type=cv2.LINE_AA)
        cv2.putText(img, f"P{pid}", (xi + 10, yi - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def _render_bev(layout: LayoutSpec, H_p2w_cam1: np.ndarray, obs1_pts: List[Tuple[float, float]],
                H_p2w_cam2: np.ndarray, obs2_pts: List[Tuple[float, float]],
                width: int = 700, height: int = 540, ppm: float = 18.0) -> np.ndarray:
    img = np.full((height, width, 3), 245, np.uint8)

    W = layout.world_points()  # (7,2)
    x_min, x_max = float(W[:, 0].min()), float(W[:, 0].max())
    y_min, y_max = float(W[:, 1].min()), float(W[:, 1].max())
    mx, my = 2.0, 1.5
    x0, x1 = x_min - mx, x_max + mx
    y0, y1 = y_min - my, y_max + my

    def w2i(x: float, y: float) -> Tuple[int, int]:
        u = int(round((x - x0) * ppm))
        v = int(round((y1 - y) * ppm))
        return u, v

    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (200, 200, 200), 1)

    # template
    for pid in range(1, 8):
        x, y = float(W[pid - 1, 0]), float(W[pid - 1, 1])
        u, v = w2i(x, y)
        cv2.circle(img, (u, v), 7, (0, 0, 0), -1)
        cv2.putText(img, f"P{pid}", (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)

    # obs points -> world (cam1 green, cam2 blue)
    w1 = _apply_H(H_p2w_cam1, np.asarray(obs1_pts, np.float32))
    w2 = _apply_H(H_p2w_cam2, np.asarray(obs2_pts, np.float32))

    for xy in w1:
        u, v = w2i(float(xy[0]), float(xy[1]))
        cv2.circle(img, (u, v), 6, (0, 180, 0), -1)

    for xy in w2:
        u, v = w2i(float(xy[0]), float(xy[1]))
        cv2.circle(img, (u, v), 6, (180, 0, 0), -1)

    cv2.putText(img, "BEV (canonical world)", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    return img


# -----------------------------
# pipeline
# -----------------------------
def run_b1(cfg_path: str, cam1_video: str, cam2_video: str):
    _must_exist(cfg_path, "cfg")
    _must_exist(cam1_video, "cam1 video")
    _must_exist(cam2_video, "cam2 video")

    cfg = load_b1_config(cfg_path)

    # detector
    yolo = YoloV5.from_cfg(cfg.yolo)
    det = PoleDetector(yolo, cfg.pole_detector)

    poles1 = det.run(cam1_video)
    poles2 = det.run(cam2_video)

    pd1 = pole_setpx_to_detections(poles1)
    pd2 = pole_setpx_to_detections(poles2)

    obs1 = CamObs(
        cam_id="cam1",
        role="start",
        poles_px=list(pd1.points_px),
        area=list(pd1.area),
        count=list(pd1.count),
        mad_px=list(pd1.mad_px),
        meta=dict(pd1.meta),
    )
    obs2 = CamObs(
        cam_id="cam2",
        role="end",
        poles_px=list(pd2.points_px),
        area=list(pd2.area),
        count=list(pd2.count),
        mad_px=list(pd2.mad_px),
        meta=dict(pd2.meta),
    )

    layout = LayoutSpec(
        n_poles=int(cfg.layout.n_poles),
        stagger_m=float(cfg.layout.stagger_m),
        lat_gap_m=float(cfg.layout.lat_gap_m),
    )

    topk = int(getattr(cfg.multicam_resolver, "topk_single"))
    thpx = float(getattr(cfg.layout_fitter, "ransac_th_px"))

    joint = solve_two_cam(obs1, obs2, layout, topk_candidates=topk, ransac_th_px=thpx)
    return cfg, layout, obs1, obs2, joint


def main():
    ap = argparse.ArgumentParser("b1_pipeline.py (one-click B1 chain)")
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--cam1", type=str, required=True)
    ap.add_argument("--cam2", type=str, required=True)
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-vis", type=str, default="", help="directory to save visualization images")
    args = ap.parse_args()

    cfg, layout, obs1, obs2, joint = run_b1(args.cfg, args.cam1, args.cam2)

    # fetch H_w2p for both cams
    H_w2p_1 = _get_H_w2p_from_joint(joint, "cam1")
    H_w2p_2 = _get_H_w2p_from_joint(joint, "cam2")
    H_p2w_1 = np.linalg.inv(H_w2p_1).astype(np.float32)
    H_p2w_2 = np.linalg.inv(H_w2p_2).astype(np.float32)

    # project template poles to each cam
    W = layout.world_points()  # (7,2)
    P1 = _apply_H(H_w2p_1, W)
    P2 = _apply_H(H_w2p_2, W)
    proj_cam1 = {pid: (float(P1[pid - 1, 0]), float(P1[pid - 1, 1])) for pid in range(1, 8)}
    proj_cam2 = {pid: (float(P2[pid - 1, 0]), float(P2[pid - 1, 1])) for pid in range(1, 8)}

    print("\n========== B1 FINAL ==========")
    for pid in range(1, 8):
        xw, yw = float(W[pid - 1, 0]), float(W[pid - 1, 1])
        x1, y1 = proj_cam1[pid]
        x2, y2 = proj_cam2[pid]
        print(f"P{pid}: world=({xw:.3f},{yw:.3f})  cam1_px=({x1:.1f},{y1:.1f})  cam2_px=({x2:.1f},{y2:.1f})")

    if args.out_json:
        out = {
            "layout": {
                "n_poles": int(layout.n_poles),
                "stagger_m": float(layout.stagger_m),
                "lat_gap_m": float(layout.lat_gap_m),
            },
            "poles": {
                str(pid): {
                    "world_xy": [float(W[pid - 1, 0]), float(W[pid - 1, 1])],
                    "cam1_px": [float(proj_cam1[pid][0]), float(proj_cam1[pid][1])],
                    "cam2_px": [float(proj_cam2[pid][0]), float(proj_cam2[pid][1])],
                }
                for pid in range(1, 8)
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("\nWrote:", args.out_json)

    # visualization
    if args.show or args.save_vis:
        frame_sec = _pick_vis_frame_sec_from_cfg(cfg)

        f1 = _read_frame_at_sec_strict(args.cam1, frame_sec)
        f2 = _read_frame_at_sec_strict(args.cam2, frame_sec)

        vis1 = _draw_projection(f1, proj_cam1, "cam1 projection of P1..P7 (from joint)")
        vis2 = _draw_projection(f2, proj_cam2, "cam2 projection of P1..P7 (from joint)")
        bev = _render_bev(layout, H_p2w_1, obs1.poles_px, H_p2w_2, obs2.poles_px)

        target_h = 540
        panels = [vis1, vis2, bev]
        resized = []
        for im in panels:
            h, w = im.shape[:2]
            scale = target_h / float(h)
            resized.append(cv2.resize(im, (int(round(w * scale)), target_h)))
        canvas = cv2.hconcat(resized)

        if args.save_vis:
            os.makedirs(args.save_vis, exist_ok=True)
            p_canvas = os.path.join(args.save_vis, "b1_joint_canvas.png")
            p_cam1 = os.path.join(args.save_vis, "b1_cam1.png")
            p_cam2 = os.path.join(args.save_vis, "b1_cam2.png")
            p_bev = os.path.join(args.save_vis, "b1_bev.png")
            cv2.imwrite(p_canvas, canvas)
            cv2.imwrite(p_cam1, vis1)
            cv2.imwrite(p_cam2, vis2)
            cv2.imwrite(p_bev, bev)
            print("Saved vis to:", args.save_vis)

        if args.show:
            cv2.imshow("B1 (cam1 | cam2 | BEV)", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
