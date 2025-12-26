# multicam_resolver.py
# python .\multicam_resolver.py --cfg .\b1_config.json --cam1 '..\正常跑1 20.6s\正常跑前视角-1.mp4' --cam2 '..\正常跑1 20.6s\正常跑后视角.mp4' --show
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Multi-camera resolver (B1 final module).

Goal:
  Use two cameras' observations to produce a unified P1..P7 localization result.

Default output is "result-only" (no debug fields).
Debug info is only printed/used in __main__ demo.

Assumptions (hard priors) are enforced upstream in candidate_generator:
  - cam1 role == "start", cam2 role == "end"
  - cam1 left column = odd, cam2 left column = even, etc.
So this module focuses on:
  - fitting homography for top candidates in each camera
  - selecting the best pair by joint consistency

Output:
  MultiCamResult { cam1: CameraCalibResult, cam2: CameraCalibResult }
Where each CameraCalibResult provides:
  - H_p2w / H_w2p (canonical world <-> pixel)
  - poles_px: projection of ALL P1..P7 onto this camera (always 7 points)
  - observed_px: only those PIDs that are actually observed+assigned in this camera
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set, Iterable

import numpy as np
import cv2

from candidate_generator import CamObs, LayoutSpec
from homography_fitter import (
    SingleCamSolution,
    fit_topk_solutions,
    project_all_poles_px,
)


# -----------------------------
# Public output (result-only)
# -----------------------------
@dataclass(frozen=True)
class CameraCalibResult:
    cam_id: str
    role: str

    H_p2w: np.ndarray  # (3,3) pixel -> canonical world
    H_w2p: np.ndarray  # (3,3) canonical world -> pixel

    poles_px: Dict[int, Tuple[float, float]]      # pid=1..7 (always full)
    observed_px: Dict[int, Tuple[float, float]]   # pid subset (actual detected points)


@dataclass(frozen=True)
class MultiCamResult:
    cam1: CameraCalibResult
    cam2: CameraCalibResult


# -----------------------------
# Internal helpers
# -----------------------------
def _apply_H(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), np.float32)
    ph = np.concatenate([pts, ones], axis=1)
    qh = (H @ ph.T).T
    return qh[:, :2] / (qh[:, 2:3] + 1e-12)


def _assigned_ids(cand_ids_in_obs_order: Iterable[int]) -> List[int]:
    ids = [int(pid) for pid in cand_ids_in_obs_order if int(pid) > 0]
    # keep unique, stable order
    seen = set()
    out = []
    for pid in ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _pid_to_obs_px(obs: CamObs, cand_ids_in_obs_order: List[int]) -> Dict[int, Tuple[float, float]]:
    """
    Return {pid: observed_px} for pids assigned by candidate.
    If duplicates exist (shouldn't), keep the first occurrence.
    """
    if len(obs.poles_px) != len(cand_ids_in_obs_order):
        raise ValueError("obs/candidate length mismatch")

    out: Dict[int, Tuple[float, float]] = {}
    for px, pid in zip(obs.poles_px, cand_ids_in_obs_order):
        pid = int(pid)
        if pid <= 0:
            continue
        if pid not in out:
            out[pid] = (float(px[0]), float(px[1]))
    return out


def _shared_world_mean_err(
    obs1: CamObs,
    sol1: SingleCamSolution,
    obs2: CamObs,
    sol2: SingleCamSolution,
) -> Tuple[float, float, int]:
    """
    Compare world coords (via each camera's H_p2w) for shared PIDs.

    Returns (mean_err_m, max_err_m, shared_count).
    If no shared, returns (0,0,0).
    """
    ids1 = set(_assigned_ids(sol1.cand.pole_ids_in_obs_order))
    ids2 = set(_assigned_ids(sol2.cand.pole_ids_in_obs_order))
    shared = sorted(list(ids1 & ids2))
    if not shared:
        return 0.0, 0.0, 0

    d1 = _pid_to_obs_px(obs1, sol1.cand.pole_ids_in_obs_order)
    d2 = _pid_to_obs_px(obs2, sol2.cand.pole_ids_in_obs_order)

    errs: List[float] = []
    for pid in shared:
        if pid not in d1 or pid not in d2:
            continue
        px1 = np.asarray([d1[pid]], np.float32)
        px2 = np.asarray([d2[pid]], np.float32)
        w1 = _apply_H(sol1.fit.H_p2w, px1)[0]
        w2 = _apply_H(sol2.fit.H_p2w, px2)[0]
        errs.append(float(np.linalg.norm(w1 - w2)))

    if not errs:
        return 0.0, 0.0, 0

    return float(np.mean(errs)), float(np.max(errs)), int(len(errs))


def _build_cam_result(obs: CamObs, sol: SingleCamSolution, layout: LayoutSpec) -> CameraCalibResult:
    poles_px = project_all_poles_px(layout, sol.fit.H_w2p)  # full P1..P7
    observed_px = _pid_to_obs_px(obs, sol.cand.pole_ids_in_obs_order)
    return CameraCalibResult(
        cam_id=str(obs.cam_id),
        role=str(obs.role),
        H_p2w=sol.fit.H_p2w,
        H_w2p=sol.fit.H_w2p,
        poles_px=poles_px,
        observed_px=observed_px,
    )


# -----------------------------
# Public API
# -----------------------------
def solve_two_cam(
    obs1: CamObs,
    obs2: CamObs,
    layout: LayoutSpec,
    topk_candidates: int,
    ransac_th_px: float,
    *,
    coverage_penalty_m: float = 0.15,
    shared_consistency_w: float = 0.30,
    min_coverage: int = 4,
    max_return_per_cam: int = 20,
    return_debug: bool = False,
) -> MultiCamResult | Tuple[MultiCamResult, dict]:
    """
    Two-camera joint selection.

    - Generates top candidates per camera (via candidate_generator)
    - Fits homography for each candidate (via homography_fitter)
    - Picks best pair by joint score

    Joint score (lower is better):
      score = sol1.score + sol2.score
            + coverage_penalty_m * (N - coverage)
            + shared_consistency_w * shared_mean_err

    return_debug=True -> returns (result, debug_dict)
    """
    N = int(layout.n_poles)

    sols1 = fit_topk_solutions(
        obs=obs1,
        layout=layout,
        topk_candidates=int(topk_candidates),
        ransac_th_px=float(ransac_th_px),
        max_return=int(max_return_per_cam),
    )
    if not sols1:
        raise RuntimeError("cam1: all candidate homography fits failed (need >=4 non-degenerate correspondences).")

    sols2 = fit_topk_solutions(
        obs=obs2,
        layout=layout,
        topk_candidates=int(topk_candidates),
        ransac_th_px=float(ransac_th_px),
        max_return=int(max_return_per_cam),
    )
    if not sols2:
        raise RuntimeError("cam2: all candidate homography fits failed (need >=4 non-degenerate correspondences).")

    best_pair: Optional[Tuple[SingleCamSolution, SingleCamSolution]] = None
    best_score = float("inf")
    best_cov = -1
    best_shared = (0.0, 0.0, 0)
    best_union_ids: List[int] = []
    best_shared_ids: List[int] = []

    # brute force over top-k solutions (small K)
    for s1 in sols1:
        ids1 = set(_assigned_ids(s1.cand.pole_ids_in_obs_order))
        for s2 in sols2:
            ids2 = set(_assigned_ids(s2.cand.pole_ids_in_obs_order))

            union = sorted(list(ids1 | ids2))
            coverage = len(union)
            if coverage < int(min_coverage):
                continue

            shared = sorted(list(ids1 & ids2))
            shared_mean, shared_max, shared_cnt = _shared_world_mean_err(obs1, s1, obs2, s2)

            score = float(s1.score + s2.score)
            score += float(coverage_penalty_m * (N - coverage))
            score += float(shared_consistency_w * shared_mean)

            # tie-break: prefer larger coverage, then smaller shared_mean
            if (score < best_score) or (abs(score - best_score) < 1e-9 and coverage > best_cov) or (
                abs(score - best_score) < 1e-9 and coverage == best_cov and shared_mean < best_shared[0]
            ):
                best_score = score
                best_pair = (s1, s2)
                best_cov = coverage
                best_shared = (shared_mean, shared_max, shared_cnt)
                best_union_ids = union
                best_shared_ids = shared

    if best_pair is None:
        raise RuntimeError("No feasible joint solution under current constraints (min_coverage too strict?).")

    s1_best, s2_best = best_pair
    res = MultiCamResult(
        cam1=_build_cam_result(obs1, s1_best, layout),
        cam2=_build_cam_result(obs2, s2_best, layout),
    )

    if not return_debug:
        return res

    debug = {
        "best_joint_score": float(best_score),
        "coverage": int(best_cov),
        "union_ids": best_union_ids,
        "shared_ids": best_shared_ids,
        "shared_mean_err_m": float(best_shared[0]),
        "shared_max_err_m": float(best_shared[1]),
        "shared_count": int(best_shared[2]),
        "cam1": {
            "cand_ids_in_obs_order": list(map(int, s1_best.cand.pole_ids_in_obs_order)),
            "missing_ids": list(map(int, getattr(s1_best.cand, "missing_ids", []))),
            "single_score": float(s1_best.score),
            "inliers": int(s1_best.fit.inliers),
            "total_corr": int(s1_best.fit.total_corr),
            "reproj_mean_m": float(s1_best.fit.reproj_mean_m),
            "reproj_max_m": float(s1_best.fit.reproj_max_m),
            "reproj_mean_px": float(s1_best.fit.reproj_mean_px),
            "reproj_max_px": float(s1_best.fit.reproj_max_px),
        },
        "cam2": {
            "cand_ids_in_obs_order": list(map(int, s2_best.cand.pole_ids_in_obs_order)),
            "missing_ids": list(map(int, getattr(s2_best.cand, "missing_ids", []))),
            "single_score": float(s2_best.score),
            "inliers": int(s2_best.fit.inliers),
            "total_corr": int(s2_best.fit.total_corr),
            "reproj_mean_m": float(s2_best.fit.reproj_mean_m),
            "reproj_max_m": float(s2_best.fit.reproj_max_m),
            "reproj_mean_px": float(s2_best.fit.reproj_mean_px),
            "reproj_max_px": float(s2_best.fit.reproj_max_px),
        },
    }
    return res, debug


def solve_two_cam_cfg(obs1: CamObs, obs2: CamObs, cfg, *, return_debug: bool = False):
    """
    cfg-driven wrapper (B1Config).

    Uses:
      - cfg.layout -> LayoutSpec
      - cfg.layout_fitter.ransac_th_px
      - cfg.multicam_resolver.topk_single / coverage_penalty_m / shared_consistency_w / min_coverage
    """
    layout = LayoutSpec(
        n_poles=int(cfg.layout.n_poles),
        stagger_m=float(cfg.layout.stagger_m),
        lat_gap_m=float(cfg.layout.lat_gap_m),
    )

    th_px = float(cfg.layout_fitter.ransac_th_px)
    topk = int(cfg.multicam_resolver.topk_single)

    return solve_two_cam(
        obs1=obs1,
        obs2=obs2,
        layout=layout,
        topk_candidates=topk,
        ransac_th_px=th_px,
        coverage_penalty_m=float(cfg.multicam_resolver.coverage_penalty_m),
        shared_consistency_w=float(cfg.multicam_resolver.shared_consistency_w),
        min_coverage=int(cfg.multicam_resolver.min_coverage),
        max_return_per_cam=min(40, topk),
        return_debug=return_debug,
    )


# -----------------------------
# Demo / self-test (real video)
# -----------------------------
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


def _draw_joint_on_frame(frame_bgr: np.ndarray, cam_res: CameraCalibResult) -> np.ndarray:
    """
    Draw final joint result projection on a camera frame:
      - observed pid: green circle
      - missing pid: red cross
    """
    img = frame_bgr.copy()

    # full P1..P7
    for pid in range(1, 8):
        x, y = cam_res.poles_px[pid]
        xi, yi = int(round(x)), int(round(y))
        if pid in cam_res.observed_px:
            cv2.circle(img, (xi, yi), 7, (0, 220, 0), -1)
            cv2.putText(img, f"P{pid}", (xi + 10, yi - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, (xi, yi), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=18, thickness=2, line_type=cv2.LINE_AA)
            cv2.putText(img, f"P{pid}", (xi + 10, yi - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser("multicam_resolver.py (B1 final, real-video demo)")
    ap.add_argument("--cfg", type=str, required=True, help="Path to b1_config.json")
    ap.add_argument("--cam1", type=str, required=True)
    ap.add_argument("--cam2", type=str, required=True)
    ap.add_argument("--show", action="store_true", help="show 2-panel overlay (no saving)")
    args = ap.parse_args()

    from b1_config import load_b1_config
    from pole_detector import YoloV5, PoleDetector
    from obs_normalizer import pole_setpx_to_detections

    cfg = load_b1_config(args.cfg)

    # --- detect poles for both cams ---
    yolo = YoloV5.from_cfg(cfg.yolo)
    det = PoleDetector(yolo, cfg.pole_detector)

    poles1 = det.run(args.cam1)
    poles2 = det.run(args.cam2)

    d1 = pole_setpx_to_detections(poles1)
    d2 = pole_setpx_to_detections(poles2)

    # build CamObs (roles are HARD)
    obs1 = CamObs(
        cam_id="cam1",
        role="start",
        poles_px=list(d1.points_px),
        area=list(d1.area),
        count=list(d1.count),
        mad_px=list(d1.mad_px),
        meta=dict(d1.meta),
    )
    obs2 = CamObs(
        cam_id="cam2",
        role="end",
        poles_px=list(d2.points_px),
        area=list(d2.area),
        count=list(d2.count),
        mad_px=list(d2.mad_px),
        meta=dict(d2.meta),
    )

    # --- solve ---
    res, dbg = solve_two_cam_cfg(obs1, obs2, cfg, return_debug=True)

    print("\n=== MultiCam Debug (demo only) ===")
    print("joint_score:", f"{dbg['best_joint_score']:.4f}")
    print("coverage   :", dbg["coverage"], "/7  union:", dbg["union_ids"])
    print("shared     :", dbg["shared_count"],
          f"mean={dbg['shared_mean_err_m']:.4f}m",
          f"max={dbg['shared_max_err_m']:.4f}m  ids={dbg['shared_ids']}")
    print("\ncam1:")
    print("  ids_in_obs_order:", dbg["cam1"]["cand_ids_in_obs_order"])
    print("  inliers:", f"{dbg['cam1']['inliers']}/{dbg['cam1']['total_corr']}",
          f"mean_m={dbg['cam1']['reproj_mean_m']:.4f} max_m={dbg['cam1']['reproj_max_m']:.4f}")
    print("cam2:")
    print("  ids_in_obs_order:", dbg["cam2"]["cand_ids_in_obs_order"])
    print("  inliers:", f"{dbg['cam2']['inliers']}/{dbg['cam2']['total_corr']}",
          f"mean_m={dbg['cam2']['reproj_mean_m']:.4f} max_m={dbg['cam2']['reproj_max_m']:.4f}")

    if args.show:
        # Try to show a frame around the detector sampling window center (for visual consistency)
        window_mode = str(getattr(cfg.pole_detector, "window_mode", "start")).lower()
        sample_secs = float(getattr(cfg.pole_detector, "sample_secs", 0.0))
        if sample_secs <= 0:
            sample_secs = float(getattr(cfg.pole_detector, "init_secs"))
        if window_mode == "custom":
            start_sec = float(getattr(cfg.pole_detector, "start_sec"))
        else:
            start_sec = 0.0
        frame_sec = float(start_sec + 0.5 * sample_secs)

        f1 = _read_frame_at_sec(args.cam1, frame_sec)
        f2 = _read_frame_at_sec(args.cam2, frame_sec)

        vis1 = _draw_joint_on_frame(f1, res.cam1)
        vis2 = _draw_joint_on_frame(f2, res.cam2)

        cv2.putText(vis1, "cam1 JOINT (green=observed, red=missing)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(vis2, "cam2 JOINT (green=observed, red=missing)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)

        # unify height then concat
        target_h = 720
        def _resize_to_h(img):
            h, w = img.shape[:2]
            s = target_h / float(h)
            return cv2.resize(img, (int(round(w * s)), target_h))

        canvas = cv2.hconcat([_resize_to_h(vis1), _resize_to_h(vis2)])
        cv2.imshow("multicam_resolver (joint P1..P7 projection)", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
