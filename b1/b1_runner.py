# b1_runner.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

from b1_config import load_b1_config
from pole_detector import YoloV5, PoleDetector, PoleSetPx

from candidate_generator import CamObs, LayoutSpec
from multicam_resolver import solve_two_cam, JointSolution


# -----------------------------
# Result bundle
# -----------------------------
@dataclass(frozen=True)
class B1RunResult:
    poles_cam1: PoleSetPx
    poles_cam2: PoleSetPx
    joint: JointSolution
    layout: LayoutSpec


# -----------------------------
# Small helpers
# -----------------------------
def _apply_H(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, np.float32)
    ones = np.ones((pts.shape[0], 1), np.float32)
    ph = np.concatenate([pts, ones], axis=1)
    qh = (H @ ph.T).T
    return qh[:, :2] / (qh[:, 2:3] + 1e-12)


def _in_bounds(x: float, y: float, w: int, h: int, margin: int = 2) -> bool:
    return (margin <= x < (w - margin)) and (margin <= y < (h - margin))


def _read_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    return ok, frame


# -----------------------------
# Drawing: overlay joint result on one camera
# -----------------------------
def _draw_joint_on_cam(
    frame_bgr: np.ndarray,
    obs: CamObs,
    joint_cam_sol,   # JointSolution.cam1 or cam2 (CamSolution)
    color_obs=(0, 255, 0),
    color_miss=(0, 0, 255),
) -> None:
    """
    Draw final JOINT result on this camera frame:

    - Observed + inlier: draw at observed pixel (green circle + Pid)
    - Missing: draw predicted pixel from homography projection (red cross + Pid)
    - Outlier obs: skip (avoid "floating points")
    """
    h, w = frame_bgr.shape[:2]

    cand = joint_cam_sol.candidate
    fit = joint_cam_sol.fit

    # 1) draw observed inliers at obs pixels
    for i, (px, pid) in enumerate(zip(obs.poles_px, cand.pole_ids_in_obs_order)):
        pid = int(pid)
        if pid <= 0:
            continue
        if i >= len(fit.inlier_mask_obs) or (not fit.inlier_mask_obs[i]):
            continue  # skip outliers

        x, y = float(px[0]), float(px[1])
        if not _in_bounds(x, y, w, h):
            continue

        xi, yi = int(round(x)), int(round(y))
        cv2.circle(frame_bgr, (xi, yi), 7, color_obs, -1)
        cv2.putText(frame_bgr, f"P{pid}", (xi + 10, yi - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_obs, 2, cv2.LINE_AA)

    # 2) draw missing as predicted pixels (template -> pixel via H_w2p)
    for pid, (x, y) in joint_cam_sol.pred_px_missing.items():
        x, y = float(x), float(y)
        if not _in_bounds(x, y, w, h):
            continue
        xi, yi = int(round(x)), int(round(y))
        cv2.drawMarker(frame_bgr, (xi, yi), color_miss, markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=18, thickness=2, line_type=cv2.LINE_AA)
        cv2.putText(frame_bgr, f"P{int(pid)}", (xi + 10, yi - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_miss, 2, cv2.LINE_AA)

    # header line
    cv2.putText(frame_bgr,
                f"{obs.cam_id} (green=inlier obs, red=missing)  inliers={joint_cam_sol.fit.inliers}/{joint_cam_sol.fit.total}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)


# -----------------------------
# Drawing: BEV (canonical world)
# -----------------------------
def _render_joint_bev(
    layout: LayoutSpec,
    joint: JointSolution,
    cam1_obs: CamObs,
    cam2_obs: CamObs,
    width: int = 700,
    height: int = 540,
    ppm: float = 18.0,
) -> np.ndarray:
    img = np.full((height, width, 3), 245, np.uint8)

    world_p = layout.world_points()  # (7,2)
    x_min, x_max = float(world_p[:, 0].min()), float(world_p[:, 0].max())
    y_min, y_max = float(world_p[:, 1].min()), float(world_p[:, 1].max())

    mx, my = 2.0, 1.5
    x0, x1 = x_min - mx, x_max + mx
    y0, y1 = y_min - my, y_max + my

    def w2i(x: float, y: float) -> Tuple[int, int]:
        u = int(round((x - x0) * ppm))
        v = int(round((y1 - y) * ppm))
        return u, v

    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (200, 200, 200), 1)

    # canonical poles
    for pid in range(1, layout.n_poles + 1):
        x, y = float(world_p[pid - 1, 0]), float(world_p[pid - 1, 1])
        u, v = w2i(x, y)
        cv2.circle(img, (u, v), 7, (0, 0, 0), -1)
        cv2.putText(img, f"P{pid}", (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)

    # cam points in world (use inlier obs only)
    def pid_world_map(obs: CamObs, sol) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        cand = sol.candidate
        fit = sol.fit
        for i, pid in enumerate(cand.pole_ids_in_obs_order):
            pid = int(pid)
            if pid <= 0:
                continue
            if i >= len(fit.inlier_mask_obs) or (not fit.inlier_mask_obs[i]):
                continue
            px = np.asarray([obs.poles_px[i]], np.float32)
            wxy = _apply_H(fit.H_p2w, px)[0]
            out[pid] = wxy
        return out

    d1 = pid_world_map(cam1_obs, joint.cam1)
    d2 = pid_world_map(cam2_obs, joint.cam2)

    # draw cam1 points (green-ish)
    for pid, wxy in d1.items():
        u, v = w2i(float(wxy[0]), float(wxy[1]))
        cv2.circle(img, (u, v), 6, (0, 180, 0), -1)

    # draw cam2 points (blue-ish in BGR)
    for pid, wxy in d2.items():
        u, v = w2i(float(wxy[0]), float(wxy[1]))
        cv2.circle(img, (u, v), 6, (180, 0, 0), -1)

    # shared lines
    shared = sorted(list(set(d1.keys()) & set(d2.keys())))
    for pid in shared:
        u1, v1 = w2i(float(d1[pid][0]), float(d1[pid][1]))
        u2, v2 = w2i(float(d2[pid][0]), float(d2[pid][1]))
        cv2.line(img, (u1, v1), (u2, v2), (0, 160, 200), 2, cv2.LINE_AA)

    # title
    cv2.putText(img, "Joint BEV (canonical world)", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, f"coverage={joint.coverage}/7 union={joint.union_ids}", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(img,
                f"shared={joint.shared_count} mean={joint.shared_world_err_mean_m:.3f}m max={joint.shared_world_err_max_m:.3f}m",
                (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (40, 40, 40), 2, cv2.LINE_AA)

    # legend
    cv2.circle(img, (20, height - 70), 6, (0, 180, 0), -1)
    cv2.putText(img, "cam1->world (inliers)", (34, height - 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.circle(img, (20, height - 40), 6, (180, 0, 0), -1)
    cv2.putText(img, "cam2->world (inliers)", (34, height - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)

    return img


# -----------------------------
# Pipeline
# -----------------------------
def _require(obj, field: str, where: str):
    if obj is None:
        raise KeyError(f"Missing config section: {where}")
    if not hasattr(obj, field):
        raise KeyError(f"Missing config field: {where}.{field}")
    return getattr(obj, field)


def _layout_from_cfg(cfg) -> LayoutSpec:
    lay = _require(cfg, "layout", "cfg")

    n_poles = int(_require(lay, "n_poles", "cfg.layout"))
    same_col = float(_require(lay, "same_col_step_m", "cfg.layout"))
    stagger = float(_require(lay, "stagger_m", "cfg.layout"))
    lat_gap = float(_require(lay, "lat_gap_m", "cfg.layout"))

    # 严格一致性：交错布局下应满足同列间距 = 2 * stagger
    if abs(same_col - 2.0 * stagger) > 1e-6:
        raise ValueError(
            f"layout inconsistent: expected same_col_step_m == 2*stagger_m, "
            f"got same_col_step_m={same_col}, stagger_m={stagger}"
        )

    # LayoutSpec.step_x 就是 P1->P2 的步长（等于 stagger_m）
    return LayoutSpec(n_poles=n_poles, step_x=stagger, lat_gap=lat_gap)


def _solver_params_from_cfg(cfg) -> tuple[int, float]:
    mc = _require(cfg, "multicam_resolver", "cfg")

    topk = int(_require(mc, "topk_single", "cfg.multicam_resolver"))
    thpx = float(_require(mc, "ransac_th_px", "cfg.multicam_resolver"))

    # 角色也严格校验（你已在 _validate 做过，这里再兜一层）
    cam1_role = str(_require(mc, "cam1_role", "cfg.multicam_resolver"))
    cam2_role = str(_require(mc, "cam2_role", "cfg.multicam_resolver"))
    if cam1_role != "start" or cam2_role != "end":
        raise ValueError(f"multicam_resolver roles must be start/end, got {cam1_role}/{cam2_role}")

    return topk, thpx




def run_b1(cfg_path: str, cam1_video: str, cam2_video: str) -> B1RunResult:
    cfg = load_b1_config(cfg_path)

    # 1) detect poles
    yolo = YoloV5.from_cfg(cfg.yolo)
    det = PoleDetector(yolo, cfg.pole_detector)

    poles1 = det.run(cam1_video)
    poles2 = det.run(cam2_video)

    # 2) build obs for new pipeline
    obs1 = CamObs(cam_id="cam1", role="start",
                  poles_px=poles1.poles_px_ordered,
                  pole_area=poles1.pole_area_ordered)
    obs2 = CamObs(cam_id="cam2", role="end",
                  poles_px=poles2.poles_px_ordered,
                  pole_area=poles2.pole_area_ordered)

    layout = _layout_from_cfg(cfg)
    topk, thpx = _solver_params_from_cfg(cfg)

    # 3) joint solve
    joint = solve_two_cam(obs1, obs2, layout, topk_candidates=topk, ransac_th_px=thpx)

    return B1RunResult(poles_cam1=poles1, poles_cam2=poles2, joint=joint, layout=layout)


def _print_summary(res: B1RunResult) -> None:
    j = res.joint
    print("\n========== B1 Summary (NEW) ==========")
    print("coverage:", f"{j.coverage}/7", "union:", j.union_ids)
    print("shared  :", j.shared_count, "ids:", j.shared_ids,
          f"mean={j.shared_world_err_mean_m:.4f}m", f"max={j.shared_world_err_max_m:.4f}m")
    print("joint_score:", f"{j.joint_score:.4f}")

    print("\n--- cam1 ---")
    print("ids_in_obs_order:", j.cam1.candidate.pole_ids_in_obs_order)
    print("present:", j.cam1.present_ids, "missing:", j.cam1.missing_ids)
    print("reproj :", f"mean={j.cam1.fit.reproj_mean_m:.4f}m max={j.cam1.fit.reproj_max_m:.4f}m",
          f"(px mean={j.cam1.fit.reproj_mean_px:.2f} max={j.cam1.fit.reproj_max_px:.2f})",
          f"inliers={j.cam1.fit.inliers}/{j.cam1.fit.total}")

    print("\n--- cam2 ---")
    print("ids_in_obs_order:", j.cam2.candidate.pole_ids_in_obs_order)
    print("present:", j.cam2.present_ids, "missing:", j.cam2.missing_ids)
    print("reproj :", f"mean={j.cam2.fit.reproj_mean_m:.4f}m max={j.cam2.fit.reproj_max_m:.4f}m",
          f"(px mean={j.cam2.fit.reproj_mean_px:.2f} max={j.cam2.fit.reproj_max_px:.2f})",
          f"inliers={j.cam2.fit.inliers}/{j.cam2.fit.total}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("b1_runner.py (one-click B1, NEW joint pipeline)")
    ap.add_argument("--cfg", type=str, default="b1_config.json")
    ap.add_argument("--cam1", type=str, default=r"..\正常跑1 20.6s\正常跑前视角-1.mp4")
    ap.add_argument("--cam2", type=str, default=r"..\正常跑1 20.6s\正常跑后视角.mp4")
    ap.add_argument("--show", action="store_true", help="show a 3-panel joint visualization (no saving)")
    args = ap.parse_args()

    res = run_b1(args.cfg, args.cam1, args.cam2)
    _print_summary(res)

    if args.show:
        # Rebuild obs for drawing (same as run_b1)
        cam1_obs = CamObs("cam1", "start", res.poles_cam1.poles_px_ordered, res.poles_cam1.pole_area_ordered)
        cam2_obs = CamObs("cam2", "end", res.poles_cam2.poles_px_ordered, res.poles_cam2.pole_area_ordered)

        ok1, f1 = _read_first_frame(args.cam1)
        ok2, f2 = _read_first_frame(args.cam2)

        panels = []
        target_h = 540

        if ok1:
            vis1 = f1.copy()
            _draw_joint_on_cam(vis1, cam1_obs, res.joint.cam1)
            panels.append(vis1)

        if ok2:
            vis2 = f2.copy()
            _draw_joint_on_cam(vis2, cam2_obs, res.joint.cam2)
            panels.append(vis2)

        bev = _render_joint_bev(res.layout, res.joint, cam1_obs, cam2_obs, width=700, height=540, ppm=18.0)
        panels.append(bev)

        resized = []
        for img in panels:
            h, w = img.shape[:2]
            scale = target_h / float(h)
            resized.append(cv2.resize(img, (int(round(w * scale)), target_h)))

        canvas = cv2.hconcat(resized)
        cv2.imshow("B1 joint result (cam1 | cam2 | BEV)", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
