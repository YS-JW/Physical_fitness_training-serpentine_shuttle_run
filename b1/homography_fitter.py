# homography_fitter.py
# python .\homography_fitter.py --cfg .\b1_config.json --video '..\正常跑1 20.6s\正常跑后视角.mp4' --role end --show
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Homography fitter (single camera).

World template (canonical):
  - P1..P7
  - x increases along the track, step = stagger_m (default 5m)
  - y = -lat_gap/2 for odd, +lat_gap/2 for even  (y=0 is centerline)

This module fits homography using RANSAC:
  - Fit H_w2p (world -> pixel) via cv2.findHomography(..., RANSAC, threshold in pixels)
  - Invert to H_p2w (pixel -> world)

It does NOT decide IDs. IDs come from candidate_generator.IdCandidate.
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

from candidate_generator import CamObs, LayoutSpec, IdCandidate, generate_id_candidates
from obs_normalizer import pole_setpx_to_detections


# -----------------------------
# Output types
# -----------------------------
@dataclass(frozen=True)
class HomographyFit:
    """
    H_w2p: world -> pixel
    H_p2w: pixel -> world (inverse of H_w2p)

    inlier_mask_obs:
      length M (len(obs.poles_px)), True means this obs point is an inlier (if assigned a pid).
    """
    H_w2p: np.ndarray
    H_p2w: np.ndarray

    inlier_mask_obs: List[bool]

    inliers: int
    total_corr: int  # number of correspondences used (pid>0)

    reproj_mean_px: float
    reproj_max_px: float

    reproj_mean_m: float
    reproj_max_m: float


@dataclass(frozen=True)
class SingleCamSolution:
    cand: IdCandidate
    fit: HomographyFit
    score: float   # lower is better


# -----------------------------
# Small helpers
# -----------------------------
def _apply_H(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Apply homography H to Nx2 points."""
    pts = np.asarray(pts_xy, np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), np.float32)
    ph = np.concatenate([pts, ones], axis=1)  # (N,3)
    qh = (H @ ph.T).T
    return qh[:, :2] / (qh[:, 2:3] + 1e-12)


def _corr_from_obs_and_candidate(
    obs: CamObs,
    cand: IdCandidate,
    layout: LayoutSpec,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Build correspondences for homography.

    Returns:
      world_xy (N,2), pixel_xy (N,2), obs_indices (N,), pid_list (N,)
    """
    M = len(obs.poles_px)
    if len(cand.pole_ids_in_obs_order) != M:
        raise ValueError("candidate length mismatch with obs length")

    W_all = layout.world_points()  # (7,2)

    world_xy: List[np.ndarray] = []
    pixel_xy: List[Tuple[float, float]] = []
    obs_indices: List[int] = []
    pid_list: List[int] = []

    for i in range(M):
        pid = int(cand.pole_ids_in_obs_order[i])
        if pid <= 0:
            continue
        if not (1 <= pid <= layout.n_poles):
            continue
        world_xy.append(W_all[pid - 1])
        pixel_xy.append(obs.poles_px[i])
        obs_indices.append(i)
        pid_list.append(pid)

    if len(world_xy) < 4:
        raise RuntimeError(f"Too few correspondences for homography: N={len(world_xy)} (<4)")

    return (
        np.asarray(world_xy, np.float32),
        np.asarray(pixel_xy, np.float32),
        obs_indices,
        pid_list,
    )


def _poly_area(pts: np.ndarray) -> float:
    """Area of convex hull (for degeneracy check)."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    hull = cv2.convexHull(pts)
    return float(cv2.contourArea(hull))


def _degeneracy_check(world_xy: np.ndarray, pixel_xy: np.ndarray) -> None:
    """
    RANSAC may fail if points are near-collinear.
    This check avoids passing clearly degenerate sets to findHomography.
    """
    a_w = _poly_area(world_xy)
    a_p = _poly_area(pixel_xy)
    if a_w < 1e-4:
        raise RuntimeError(f"Degenerate world correspondences (convex hull area too small): {a_w:.3e}")
    if a_p < 5.0:  # pixels^2, very small hull -> almost collinear in image
        raise RuntimeError(f"Degenerate pixel correspondences (convex hull area too small): {a_p:.3f}")


# -----------------------------
# Core API
# -----------------------------
def fit_homography_ransac(
    obs: CamObs,
    cand: IdCandidate,
    layout: LayoutSpec,
    ransac_th_px: float,
    max_iters: int = 4000,
    confidence: float = 0.995,
) -> HomographyFit:
    """
    Fit homography with RANSAC.
    """
    world_xy, pixel_xy, obs_indices, _ = _corr_from_obs_and_candidate(obs, cand, layout)
    _degeneracy_check(world_xy, pixel_xy)

    H_w2p, mask = cv2.findHomography(
        world_xy,
        pixel_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_th_px),
        maxIters=int(max_iters),
        confidence=float(confidence),
    )
    if H_w2p is None or mask is None:
        raise RuntimeError("cv2.findHomography failed: H is None.")

    H_w2p = H_w2p.astype(np.float32)

    try:
        H_p2w = np.linalg.inv(H_w2p).astype(np.float32)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Homography inversion failed: {e}")

    mask = mask.reshape(-1).astype(np.uint8)  # length N
    N = int(mask.shape[0])
    inliers = int(mask.sum())

    # per-obs inlier mask
    M = len(obs.poles_px)
    inlier_mask_obs = [False] * M
    for k in range(N):
        if mask[k]:
            inlier_mask_obs[int(obs_indices[k])] = True

    # pixel reprojection error: world -> pixel
    pred_px = _apply_H(H_w2p, world_xy)
    err_px = np.linalg.norm(pred_px - pixel_xy, axis=1)

    # world reprojection error: pixel -> world
    pred_w = _apply_H(H_p2w, pixel_xy)
    err_m = np.linalg.norm(pred_w - world_xy, axis=1)

    if inliers > 0:
        sel = mask.astype(bool)
        reproj_mean_px = float(err_px[sel].mean())
        reproj_max_px = float(err_px[sel].max())
        reproj_mean_m = float(err_m[sel].mean())
        reproj_max_m = float(err_m[sel].max())
    else:
        reproj_mean_px = float(err_px.mean())
        reproj_max_px = float(err_px.max())
        reproj_mean_m = float(err_m.mean())
        reproj_max_m = float(err_m.max())

    return HomographyFit(
        H_w2p=H_w2p,
        H_p2w=H_p2w,
        inlier_mask_obs=inlier_mask_obs,
        inliers=inliers,
        total_corr=N,
        reproj_mean_px=reproj_mean_px,
        reproj_max_px=reproj_max_px,
        reproj_mean_m=reproj_mean_m,
        reproj_max_m=reproj_max_m,
    )

def _missing_role_cost(missing_ids: List[int], role: str, n_poles: int) -> float:
    """
    越小越“合理”。

    一个可用的默认：缺远端更合理（cost小）
      - start 相机：远端是大 pid（P7），所以 missing pid 越大 cost 越小
      - end   相机：远端是小 pid（P1），所以 missing pid 越小 cost 越小
    """
    if not missing_ids:
        return 0.0
    N = float(n_poles)

    if role == "start":
        # pid 大 -> 更远 -> 更合理 -> cost 小
        return float(np.mean([(N + 1.0 - float(pid)) / N for pid in missing_ids]))
    else:  # "end"
        # pid 小 -> 更远 -> 更合理 -> cost 小
        return float(np.mean([float(pid) / N for pid in missing_ids]))



def _solution_score(cand: IdCandidate, fit: HomographyFit, role: str, n_poles: int) -> float:
    """
    score 越小越好
    - 几何项：world reprojection（米）
    - 惩罚 outlier 数量（RANSAC 踢掉的点越多越可疑）
    - 加入 missing_ids 的 role 先验（仅在几何接近时拉开差距）
    """
    geom = float(fit.reproj_mean_m + 0.5 * fit.reproj_max_m)

    outliers = int(fit.total_corr - fit.inliers)
    outlier_pen_m = 0.05 * float(outliers)  # 0.05m/个，可调

    prior = _missing_role_cost(cand.missing_ids, role=role, n_poles=n_poles)
    prior_w_m = 0.25  # 0.25m * prior，可调（建议 0.1~0.5 之间试）

    # 轻微奖励更多 inliers（保留你原来的思路）
    reward = 0.01 * float(max(0, min(7, fit.inliers) - 4))

    return geom + outlier_pen_m + prior_w_m * float(prior) - reward



def fit_topk_solutions(
    obs: CamObs,
    layout: LayoutSpec,
    topk_candidates: int,
    ransac_th_px: float,
    max_return: int = 20,
) -> List[SingleCamSolution]:
    """
    Convenience wrapper for multicam_resolver:
      obs -> candidates -> (try fit) -> sort by solution score
    """
    cands = generate_id_candidates(obs, layout, max_return=int(topk_candidates))
    sols: List[SingleCamSolution] = []

    for cand in cands:
        try:
            fit = fit_homography_ransac(obs, cand, layout, ransac_th_px=ransac_th_px)
        except Exception:
            continue
        sols.append(SingleCamSolution(
            cand=cand,
            fit=fit,
            score=_solution_score(cand, fit, role=obs.role, n_poles=layout.n_poles)
        ))

    sols.sort(key=lambda s: s.score)
    return sols[: max(1, int(max_return))]


# -----------------------------
# Visualization (debug)
# -----------------------------
def project_all_poles_px(layout: LayoutSpec, H_w2p: np.ndarray) -> Dict[int, Tuple[float, float]]:
    W = layout.world_points()  # (7,2)
    P = _apply_H(H_w2p, W)     # (7,2)
    return {pid: (float(P[pid - 1, 0]), float(P[pid - 1, 1])) for pid in range(1, layout.n_poles + 1)}


def draw_solution_overlay(
    frame_bgr: np.ndarray,
    obs: CamObs,
    sol: SingleCamSolution,
    layout: LayoutSpec,
) -> np.ndarray:
    """
    Overlay:
      - observed points: green if inlier, orange if outlier
      - label each observed point with assigned P#
      - draw projected template P1..P7 as red crosses (so you can see the model)
    """
    img = frame_bgr.copy()
    cand = sol.cand
    fit = sol.fit

    cv2.putText(img, f"score={sol.score:.4f}  inliers={fit.inliers}/{fit.total_corr}  "
                     f"mean_m={fit.reproj_mean_m:.4f} max_m={fit.reproj_max_m:.4f}",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)

    # projected template points
    proj = project_all_poles_px(layout, fit.H_w2p)
    for pid, (x, y) in proj.items():
        xi, yi = int(round(x)), int(round(y))
        cv2.drawMarker(img, (xi, yi), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=18, thickness=2, line_type=cv2.LINE_AA)
        cv2.putText(img, f"P{pid}", (xi + 10, yi - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # observed points + assigned pid
    for i, (px, pid) in enumerate(zip(obs.poles_px, cand.pole_ids_in_obs_order)):
        pid = int(pid)
        if pid <= 0:
            continue
        x, y = float(px[0]), float(px[1])
        xi, yi = int(round(x)), int(round(y))
        inl = bool(fit.inlier_mask_obs[i])
        color = (0, 220, 0) if inl else (0, 170, 255)  # green / orange
        cv2.circle(img, (xi, yi), 7, color, -1)
        cv2.putText(img, f"det{i}:P{pid}", (xi + 10, yi - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    return img


def _read_frame_at_sec(video_path: str, sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame at {sec:.2f}s: {video_path}")
    return frame


# -----------------------------
# Real-video single-cam demo
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("homography_fitter.py (real-video single-cam demo)")
    ap.add_argument("--cfg", type=str, required=True, help="Path to b1_config.json")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--role", type=str, choices=["start", "end"], required=True)
    ap.add_argument("--cam-id", type=str, default="cam1")
    ap.add_argument("--topk-cand", type=int, default=30)
    ap.add_argument("--topk-fit", type=int, default=8)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # cfg must exist; load fails -> raise
    from b1_config import load_b1_config
    from pole_detector import YoloV5, PoleDetector

    cfg = load_b1_config(args.cfg)

    # detect poles from video
    yolo = YoloV5.from_cfg(cfg.yolo)
    det = PoleDetector(yolo, cfg.pole_detector)
    poleset = det.run(args.video)

    pd = pole_setpx_to_detections(poleset)
    obs = CamObs(
        cam_id=str(args.cam_id),
        role=str(args.role),
        poles_px=list(pd.points_px),
        area=list(pd.area),
        count=list(pd.count),
        mad_px=list(pd.mad_px),
        meta=dict(pd.meta),
    )

    layout = LayoutSpec(
        n_poles=int(cfg.layout.n_poles),
        stagger_m=float(cfg.layout.stagger_m),
        lat_gap_m=float(cfg.layout.lat_gap_m),
    )

    # match a representative frame to the detector sampling window (for visualization)
    window_mode = str(getattr(cfg.pole_detector, "window_mode", "start")).lower()
    sample_secs = float(getattr(cfg.pole_detector, "sample_secs", 0.0))
    if sample_secs <= 0:
        sample_secs = float(getattr(cfg.pole_detector, "init_secs"))
    if window_mode == "custom":
        start_sec = float(getattr(cfg.pole_detector, "start_sec"))
    else:
        start_sec = 0.0
    frame_sec = float(start_sec + 0.5 * sample_secs)

    # thresholds from cfg
    th_px = float(getattr(cfg.layout_fitter, "ransac_th_px"))
    sols = fit_topk_solutions(
        obs=obs,
        layout=layout,
        topk_candidates=int(args.topk_cand),
        ransac_th_px=th_px,
        max_return=int(args.topk_fit),
    )
    if not sols:
        raise RuntimeError("All candidate homography fits failed (need >=4 non-degenerate correspondences).")

    print("\n=== Single-cam solutions (top) ===")
    for i, s in enumerate(sols, 1):
        f = s.fit

        out = []
        for det_i, (pid, inl) in enumerate(zip(s.cand.pole_ids_in_obs_order, f.inlier_mask_obs)):
            pid = int(pid)
            if pid > 0 and (not inl):
                out.append(f"det{det_i}:P{pid}")

        out_str = "[" + ",".join(out) + "]" if out else "[]"

        print(f"[{i:02d}] score={s.score:.4f}  inliers={f.inliers}/{f.total_corr}  "
              f"mean_m={f.reproj_mean_m:.4f} max_m={f.reproj_max_m:.4f}  "
              f"mean_px={f.reproj_mean_px:.2f} max_px={f.reproj_max_px:.2f}  "
              f"missing={s.cand.missing_ids}  outliers={out_str}  ids={s.cand.pole_ids_in_obs_order}")

    best = sols[0]
    print("\n=== Best ===")
    print("ids:", best.cand.pole_ids_in_obs_order)
    print("missing:", best.cand.missing_ids)
    print("H_w2p:")
    for r in range(3):
        print("  ", ["{:+0.6f}".format(float(v)) for v in best.fit.H_w2p[r]])
    print("H_p2w:")
    for r in range(3):
        print("  ", ["{:+0.6f}".format(float(v)) for v in best.fit.H_p2w[r]])

    if args.show:
        frame0 = _read_frame_at_sec(args.video, frame_sec)
        vis = draw_solution_overlay(frame0, obs, best, layout)
        cv2.imshow("homography_fitter: best solution overlay", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
