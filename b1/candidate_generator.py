# python .\candidate_generator.py --cfg .\b1_config.json --video '..\正常跑1 20.6s\正常跑前视角-1.mp4' --role start --cam-id cam1 --show --show-norm
from __future__ import annotations

"""
Candidate generator (single camera): assign pole IDs to detections under hard priors.

Hard priors (as requested):
  role="start": left column -> odd (1,3,5,7), right column -> even (2,4,6)
  role="end"  : left column -> even (2,4,6), right column -> odd (1,3,5,7)

Key constraint to prevent "P1 behind P7":
- Within EACH column, we define "near -> far" by pixel y DESC (larger y is nearer in typical camera).
- The assigned IDs must follow the correct near->far order:
    start: near->far IDs increase  (odd: 1<3<5<7, even: 2<4<6)
    end  : near->far IDs decrease  (odd: 7>5>3>1, even: 6>4>2)
This still allows missing P1/P2 etc. It only forbids reversed ordering.

This module DOES NOT fit homography. Homography fitting is in homography_fitter.py
"""

import argparse
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from obs_normalizer import (
    PoleDetections,
    pole_setpx_to_detections,
    normalize_obs_xsweep,
    visualize_normalized_on_frame,
)


# -----------------------------
# Public types
# -----------------------------
@dataclass(frozen=True)
class CamObs:
    cam_id: str
    role: str  # "start" | "end"
    poles_px: List[Tuple[float, float]]
    area: Optional[List[float]] = None
    count: Optional[List[int]] = None
    mad_px: Optional[List[float]] = None
    meta: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class LayoutSpec:
    n_poles: int = 7
    stagger_m: float = 5.0
    lat_gap_m: float = 2.5

    def world_points(self) -> np.ndarray:
        """
        Canonical world template P1..P7.
        x = (pid-1) * stagger_m
        y = -lat_gap/2 for odd, +lat_gap/2 for even
        """
        W = np.zeros((self.n_poles, 2), np.float32)
        yL = -0.5 * float(self.lat_gap_m)
        yR = +0.5 * float(self.lat_gap_m)
        for pid in range(1, self.n_poles + 1):
            x = float((pid - 1) * self.stagger_m)
            y = yL if (pid % 2 == 1) else yR
            W[pid - 1] = (x, y)
        return W


@dataclass(frozen=True)
class IdCandidate:
    pole_ids_in_obs_order: List[int]  # length M aligned with obs.poles_px; 0 means unassigned
    missing_ids: List[int]  # ids in 1..7 not used
    score: float
    debug: Dict[str, float]


# -----------------------------
# Helpers
# -----------------------------
def _obs_to_detections(obs: CamObs) -> PoleDetections:
    pts = list(obs.poles_px)
    M = len(pts)
    if M == 0:
        raise RuntimeError("CamObs has no points.")

    area = list(obs.area) if obs.area is not None else [1.0] * M
    cnt = list(obs.count) if obs.count is not None else [1] * M
    mad = list(obs.mad_px) if obs.mad_px is not None else [1.0] * M
    meta = dict(obs.meta) if obs.meta is not None else {}

    if not (len(area) == len(cnt) == len(mad) == M):
        raise RuntimeError("CamObs: area/count/mad length mismatch.")

    return PoleDetections(points_px=pts, area=area, count=cnt, mad_px=mad, meta=meta)


def _near_to_far_sort_by_y_desc(col_points) -> list:
    # col_points: list of NormalizedPoint (from obs_normalizer), each has px=(x,y), det_idx, t, ...
    return sorted(col_points, key=lambda p: float(p.px[1]), reverse=True)


def _gap_cost_abs(t_vals: List[float], pid_vals: List[int]) -> float:
    """
    Compare abs gaps in t (pixel main-axis coordinate) vs gaps in pid sequence.
    Sign-free (we use abs), because we don't want the "t axis flip" to create mirrored candidates.
    """
    if len(t_vals) < 2:
        return 0.0
    t = np.asarray(t_vals, np.float32)
    p = np.asarray(pid_vals, np.float32)

    dt = np.abs(np.diff(t))
    dp = np.abs(np.diff(p))

    med_dt = float(np.median(dt)) + 1e-6
    med_dp = float(np.median(dp)) + 1e-6

    dt_n = dt / med_dt
    dp_n = dp / med_dp
    return float(np.mean(np.abs(dt_n - dp_n)))


def _linear_fit_cost(t_all: np.ndarray, x_all: np.ndarray) -> float:
    """
    Fit t ≈ a*x + b and return normalized RMSE.
    This is a soft scoring term, not a hard constraint.
    """
    if t_all.size < 2:
        return 0.0
    x = x_all.astype(np.float32)
    t = t_all.astype(np.float32)
    A = np.stack([x, np.ones_like(x)], axis=1)  # (N,2)
    # least squares
    sol, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    pred = a * x + b
    rmse = float(np.sqrt(np.mean((pred - t) ** 2)))
    # normalize by typical step in t (use median abs diff)
    if t.size >= 2:
        step = float(np.median(np.abs(np.diff(np.sort(t))))) + 1e-6
    else:
        step = 1.0
    return float(rmse / step)

def _missing_role_cost(missing_ids: List[int], role: str, N: int) -> float:
    if not missing_ids:
        return 0.0
    if role == "start":
        return float(np.mean([(N + 1 - pid) / N for pid in missing_ids]))
    else:  # "end"
        return float(np.mean([pid / N for pid in missing_ids]))



# -----------------------------
# Public API
# -----------------------------
def generate_id_candidates(
        obs: CamObs,
        layout: LayoutSpec,
        max_return: int = 50,
        min_per_col: int = 2,
        imbalance_penalty_px: float = 2.0,
) -> List[IdCandidate]:
    """
    Input:
      obs: one camera detections
      layout: canonical layout spec
    Output:
      sorted list of IdCandidate (ascending score)
    """
    if obs.role not in ("start", "end"):
        raise ValueError("obs.role must be 'start' or 'end'.")

    det = _obs_to_detections(obs)
    norm = normalize_obs_xsweep(det, min_per_col=int(min_per_col),
                                imbalance_penalty_px=float(imbalance_penalty_px),
                                robust="median")

    # 1) column split (from obs_normalizer)
    left_raw = list(norm.left_col)
    right_raw = list(norm.right_col)

    # 2) FIX near->far ordering by pixel y (DESC)
    left = _near_to_far_sort_by_y_desc(left_raw)
    right = _near_to_far_sort_by_y_desc(right_raw)

    # 3) hard id pools in near->far order (ONLY order that prevents reversal)
    if obs.role == "start":
        left_pool_order = [1, 3, 5, 7]  # odd near->far
        right_pool_order = [2, 4, 6]  # even near->far
    else:  # end
        left_pool_order = [6, 4, 2]  # even near->far (near is larger pid)
        right_pool_order = [7, 5, 3, 1]  # odd near->far (near is larger pid)

    if len(left) > len(left_pool_order) or len(right) > len(right_pool_order):
        raise RuntimeError(
            f"split sizes incompatible with hard priors: "
            f"left={len(left)} (pool {len(left_pool_order)}), "
            f"right={len(right)} (pool {len(right_pool_order)})"
        )

    M = len(det.points_px)
    N = int(layout.n_poles)

    # scoring weights (simple + stable, does NOT force P1/P2 presence)
    w_missing_cnt = 0.2
    w_missing_role = 1.2
    w_gap = 0.8
    w_lin = 0.6


    cands: List[IdCandidate] = []

    L_det_idx = [int(p.det_idx) for p in left]
    R_det_idx = [int(p.det_idx) for p in right]
    L_t = [float(p.t) for p in left]
    R_t = [float(p.t) for p in right]

    # enumerate order-preserving subsequences from the ordered pools
    for L_ids in combinations(left_pool_order, len(L_det_idx)):
        for R_ids in combinations(right_pool_order, len(R_det_idx)):
            pid_in_obs = [0] * M
            used = set()

            for di, pid in zip(L_det_idx, L_ids):
                pid_in_obs[di] = int(pid)
                used.add(int(pid))
            for di, pid in zip(R_det_idx, R_ids):
                pid_in_obs[di] = int(pid)
                used.add(int(pid))

            missing = [pid for pid in range(1, N + 1) if pid not in used]

            miss_cnt = float(len(missing))
            miss_role = _missing_role_cost(missing, obs.role, N)

            # column gap costs
            gapL = _gap_cost_abs(L_t, list(map(int, L_ids)))
            gapR = _gap_cost_abs(R_t, list(map(int, R_ids)))

            # global linear fit cost: use all assigned points' (t, x_world)
            t_all = []
            x_all = []
            for p in left:
                di = int(p.det_idx)
                pid = int(pid_in_obs[di])
                if pid > 0:
                    t_all.append(float(p.t))
                    x_all.append(float((pid - 1) * layout.stagger_m))
            for p in right:
                di = int(p.det_idx)
                pid = int(pid_in_obs[di])
                if pid > 0:
                    t_all.append(float(p.t))
                    x_all.append(float((pid - 1) * layout.stagger_m))
            lin_cost = _linear_fit_cost(np.asarray(t_all, np.float32), np.asarray(x_all, np.float32))

            score = (
                    w_missing_cnt * miss_cnt +
                    w_missing_role * miss_role +
                    w_gap * float(gapL + gapR) +
                    w_lin * float(lin_cost)
            )

            cands.append(
                IdCandidate(
                    pole_ids_in_obs_order=pid_in_obs,
                    missing_ids=missing,
                    score=float(score),
                    debug={
                        "missing_cnt": miss_cnt,
                        "missing_role": float(miss_role),
                        "gapL": float(gapL),
                        "gapR": float(gapR),
                        "lin": float(lin_cost),
                        "y_order": 1.0,
                    },
                )
            )

    cands.sort(key=lambda c: c.score)
    return cands[: max(1, int(max_return))]


# -----------------------------
# Visualization: one candidate per image (n/p)
# -----------------------------
def _draw_candidate_on_frame(frame_bgr: np.ndarray, obs: CamObs, cand: IdCandidate, header: str) -> np.ndarray:
    img = frame_bgr.copy()
    cv2.putText(img, header, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, f"score={cand.score:.4f} missing={cand.missing_ids} dbg={cand.debug}",
                (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)

    for det_idx, ((x, y), pid) in enumerate(zip(obs.poles_px, cand.pole_ids_in_obs_order)):
        x_i, y_i = int(round(float(x))), int(round(float(y)))
        pid = int(pid)
        if pid > 0:
            color = (0, 255, 0)
            label = f"P{pid}"
        else:
            color = (160, 160, 160)
            label = "NA"
        cv2.circle(img, (x_i, y_i), 7, color, -1)
        cv2.putText(img, f"det{det_idx}:{label}", (x_i + 10, y_i - 10),
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


def _browse_candidates(video_path: str, frame_sec: float, obs: CamObs, cands: List[IdCandidate], topk: int) -> None:
    frame0 = _read_frame_at_sec(video_path, frame_sec)
    K = min(int(topk), len(cands))
    if K <= 0:
        raise RuntimeError("No candidates to browse.")
    win = "candidate_generator (n/p/q)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i = 0
    while True:
        cand = cands[i]
        header = f"[{i + 1}/{K}] role={obs.role} cam={obs.cam_id}"
        img = _draw_candidate_on_frame(frame0, obs, cand, header)
        cv2.imshow(win, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("n"), ord("d"), 83):
            i = min(i + 1, K - 1)
        elif key in (ord("p"), ord("a"), 81):
            i = max(i - 1, 0)

    cv2.destroyAllWindows()


# -----------------------------
# Demo: real video -> pole_detector -> candidates
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("candidate_generator.py (real-video demo)")
    ap.add_argument("--cfg", type=str, required=True, help="Path to b1_config.json")
    ap.add_argument("--video", type=str, required=True, help="Single camera video path")
    ap.add_argument("--role", type=str, choices=["start", "end"], required=True)
    ap.add_argument("--cam-id", type=str, default="cam1")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--show-norm", action="store_true", help="show normalization debug overlay")
    ap.add_argument("--show", action="store_true", help="browse candidates (one per image)")
    args = ap.parse_args()

    # load cfg (must exist)
    from b1_config import load_b1_config
    from pole_detector import YoloV5, PoleDetector

    cfg = load_b1_config(args.cfg)

    # detect poles from video
    yolo = YoloV5.from_cfg(cfg.yolo)
    det = PoleDetector(yolo, cfg.pole_detector)
    poleset = det.run(args.video)

    # build detections and CamObs (treat order as arbitrary)
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

    # representative frame time: center of sampling window
    window_mode = str(getattr(cfg.pole_detector, "window_mode", "start")).lower()
    sample_secs = float(getattr(cfg.pole_detector, "sample_secs", 0.0))
    if sample_secs <= 0:
        sample_secs = float(getattr(cfg.pole_detector, "init_secs"))

    if window_mode == "custom":
        start_sec = float(getattr(cfg.pole_detector, "start_sec"))
    else:
        start_sec = 0.0
    frame_sec = float(start_sec + 0.5 * sample_secs)

    if args.show_norm:
        frame0 = _read_frame_at_sec(args.video, frame_sec)
        norm = normalize_obs_xsweep(_obs_to_detections(obs), min_per_col=2, imbalance_penalty_px=2.0, robust="median")
        vis = visualize_normalized_on_frame(frame0, norm)
        cv2.imshow("candidate_generator: normalization", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # generate candidates
    cands = generate_id_candidates(
        obs=obs,
        layout=layout,
        max_return=max(50, int(args.topk)),
        min_per_col=2,
        imbalance_penalty_px=2.0,
    )

    print("\n=== Candidates (topK) ===")
    K = min(int(args.topk), len(cands))
    for i in range(K):
        c = cands[i]
        print(f"[{i + 1:02d}] score={c.score:.4f} missing={c.missing_ids} ids={c.pole_ids_in_obs_order} dbg={c.debug}")

    if args.show:
        _browse_candidates(args.video, frame_sec, obs, cands[:K], topk=K)
