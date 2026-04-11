# python .\obs_normalizer.py --cfg .\b1_config.json --video "..\正常跑1 20.6s\正常跑前视角-1.mp4" --show
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class PoleDetections:
    """
    A single camera's pole detections (clustered points).
    points_px[i] is a representative pixel point of one pole cluster.

    area/count/mad are per-point cluster statistics from pole_detector:
      - area: mean bbox area within this cluster (roughly correlates with "near" but not strictly monotonic)
      - count: number of raw detections supporting this cluster
      - mad_px: median distance to cluster center (cluster spread, large => unreliable/merged/split)
    """
    points_px: List[Tuple[float, float]]
    area: List[float]
    count: List[int]
    mad_px: List[float]
    meta: Dict[str, float]


@dataclass(frozen=True)
class ObsPoint:
    det_idx: int
    px: Tuple[float, float]
    area: float
    count: int
    mad_px: float
    t: float
    s: float


@dataclass(frozen=True)
class NormalizedObs:
    """
    Normalized observation for downstream modules.
    - left_col/right_col are sorted by t (along main_dir)
    - main_dir/lat_dir are unit vectors in pixel space
    - mean_xy is the center between two columns (preferred over global mean)
    - split_debug stores split decision details (k, scores, etc.)
    """
    raw: PoleDetections
    left_col: List[ObsPoint]
    right_col: List[ObsPoint]
    main_dir: Tuple[float, float]
    lat_dir: Tuple[float, float]
    mean_xy: Tuple[float, float]
    split_debug: Dict[str, float]


# -----------------------------
# Geometry helpers
# -----------------------------
def _fit_line_cv(pts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """cv2.fitLine -> (v_unit(2,), p0(2,)) or None if too few points."""
    pts = np.asarray(pts, np.float32)
    if pts.shape[0] < 2:
        return None
    pts2 = pts.reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)
    v = np.array([float(vx), float(vy)], np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    v = v / n
    p0 = np.array([float(x0), float(y0)], np.float32)
    return v, p0


def _point_line_dists(pts: np.ndarray, v: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Perpendicular distances from pts to infinite line (p0 + t*v)."""
    d = pts - p0[None, :]
    # 2D cross magnitude: |dx*vy - dy*vx|
    return np.abs(d[:, 0] * v[1] - d[:, 1] * v[0]).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


# -----------------------------
# Split by x-sort + sweep cut
# -----------------------------
def split_columns_by_xsweep(
    pts: np.ndarray,
    min_per_col: int = 2,
    imbalance_penalty_px: float = 2.0,
    robust: str = "median",  # "median" | "mean"
) -> Tuple[List[int], List[int], Dict[str, float]]:
    """
    Split M points into left/right columns by:
      1) sort by x ascending
      2) sweep cut position k: left = first k, right = rest
      3) fit 2 lines and score by robust point-to-line distances (+ imbalance penalty)

    Returns:
      left_idx, right_idx (indices in original pts order),
      debug dict
    """
    pts = np.asarray(pts, np.float32)
    M = int(pts.shape[0])
    if M < 2 * min_per_col:
        raise RuntimeError(f"split_columns_by_xsweep: need >= {2*min_per_col} points, got {M}")

    xs = pts[:, 0]
    idx_sorted = np.argsort(xs)  # ascending x

    def agg(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 1e9
        if robust == "mean":
            return float(arr.mean())
        return float(np.median(arr))

    best_score = 1e18
    best_left: Optional[List[int]] = None
    best_right: Optional[List[int]] = None
    best_dbg: Dict[str, float] = {}

    # k in [min_per_col, M-min_per_col]
    for k in range(min_per_col, M - min_per_col + 1):
        Ls = idx_sorted[:k]
        Rs = idx_sorted[k:]

        L = pts[Ls]
        R = pts[Rs]

        fitL = _fit_line_cv(L)
        fitR = _fit_line_cv(R)
        if fitL is None or fitR is None:
            continue
        vL, p0L = fitL
        vR, p0R = fitR

        dL = _point_line_dists(L, vL, p0L)
        dR = _point_line_dists(R, vR, p0R)

        errL = agg(dL)
        errR = agg(dR)

        imbalance = abs(int(len(Ls)) - int(len(Rs)))
        score = float(errL + errR + imbalance_penalty_px * imbalance)

        if score < best_score:
            best_score = score
            best_left = Ls.tolist()
            best_right = Rs.tolist()
            best_dbg = {
                "k": float(k),
                "errL_px": float(errL),
                "errR_px": float(errR),
                "imbalance": float(imbalance),
                "score": float(score),
            }

    if best_left is None or best_right is None:
        raise RuntimeError("split_columns_by_xsweep: no feasible split found.")

    # Ensure "left" truly means smaller x on average (defensive)
    mean_xL = float(np.mean(pts[best_left, 0])) if best_left else 1e9
    mean_xR = float(np.mean(pts[best_right, 0])) if best_right else 1e9
    if mean_xL > mean_xR:
        best_left, best_right = best_right, best_left

    return best_left, best_right, best_dbg


# -----------------------------
# Normalization
# -----------------------------
def normalize_obs_xsweep(
    det: PoleDetections,
    min_per_col: int = 2,
    imbalance_penalty_px: float = 2.0,
    robust: str = "median",
) -> NormalizedObs:
    pts = np.asarray(det.points_px, np.float32)
    M = int(pts.shape[0])
    if M == 0:
        raise RuntimeError("normalize_obs_xsweep: empty detections.")

    if M < 4:
        raise RuntimeError(f"normalize_obs_xsweep: need >=4 points, got {M} (too few for 2-column geometry).")

    left_idx, right_idx, dbg = split_columns_by_xsweep(
        pts,
        min_per_col=min_per_col,
        imbalance_penalty_px=imbalance_penalty_px,
        robust=robust,
    )

    left_pts = pts[left_idx]
    right_pts = pts[right_idx]

    # mean_xy should be the center of two columns, not global mean
    mean_xy = 0.5 * (left_pts.mean(axis=0) + right_pts.mean(axis=0))

    fitL = _fit_line_cv(left_pts)
    fitR = _fit_line_cv(right_pts)
    if fitL is None or fitR is None:
        # With >=2 points per col this shouldn't happen
        raise RuntimeError("normalize_obs_xsweep: fitLine failed unexpectedly.")

    vL, p0L = fitL
    vR, p0R = fitR
    # align directions to avoid cancellation
    if float(vL @ vR) < 0:
        vR = -vR

    main_dir = _unit(vL + vR)  # average direction
    lat_dir = _unit(np.array([-main_dir[1], main_dir[0]], np.float32))

    # Make lat_dir sign consistent: right column should have larger s
    Xc = pts - mean_xy[None, :]
    s_all = (Xc @ lat_dir).astype(np.float32)
    sL = float(np.mean(s_all[left_idx]))
    sR = float(np.mean(s_all[right_idx]))
    if sR < sL:
        lat_dir = -lat_dir
        s_all = -s_all

    t_all = (Xc @ main_dir).astype(np.float32)

    def make_point(i: int) -> ObsPoint:
        return ObsPoint(
            det_idx=int(i),
            px=(float(pts[i, 0]), float(pts[i, 1])),
            area=float(det.area[i]),
            count=int(det.count[i]),
            mad_px=float(det.mad_px[i]),
            t=float(t_all[i]),
            s=float(s_all[i]),
        )

    left_col = [make_point(i) for i in left_idx]
    right_col = [make_point(i) for i in right_idx]
    left_col.sort(key=lambda p: p.t)
    right_col.sort(key=lambda p: p.t)

    # enrich debug (optional but useful)
    dbg2 = dict(dbg)
    dbg2.update({
        "mean_x_left": float(np.mean(left_pts[:, 0])),
        "mean_x_right": float(np.mean(right_pts[:, 0])),
        "main_dx": float(main_dir[0]),
        "main_dy": float(main_dir[1]),
        "lat_dx": float(lat_dir[0]),
        "lat_dy": float(lat_dir[1]),
        "mean_x": float(mean_xy[0]),
        "mean_y": float(mean_xy[1]),
    })

    return NormalizedObs(
        raw=det,
        left_col=left_col,
        right_col=right_col,
        main_dir=(float(main_dir[0]), float(main_dir[1])),
        lat_dir=(float(lat_dir[0]), float(lat_dir[1])),
        mean_xy=(float(mean_xy[0]), float(mean_xy[1])),
        split_debug=dbg2,
    )


# -----------------------------
# Adapter from pole_detector.PoleSetPx
# -----------------------------
def pole_setpx_to_detections(poleset) -> PoleDetections:
    """
    Convert from your current pole_detector.PoleSetPx into PoleDetections.

    NOTE:
    - pole_detector currently returns `poles_px_ordered`, but we treat it as an arbitrary order.
      This normalizer does NOT rely on that order.
    """
    # Lazy attribute reads (avoid hard import coupling)
    pts = list(getattr(poleset, "poles_px"))
    area = list(getattr(poleset, "pole_area"))
    cnt = list(getattr(poleset, "pole_count"))
    mad = list(getattr(poleset, "pole_spread_mad_px"))
    meta = dict(getattr(poleset, "meta", {}))
    if not (len(pts) == len(area) == len(cnt) == len(mad)):
        raise RuntimeError("pole_setpx_to_detections: length mismatch in PoleSetPx fields.")
    return PoleDetections(points_px=pts, area=area, count=cnt, mad_px=mad, meta=meta)


# -----------------------------
# Visualization (debug)
# -----------------------------
def _draw_arrow(img, p0: Tuple[float, float], v: Tuple[float, float], length: float, color, label: str):
    x0, y0 = float(p0[0]), float(p0[1])
    vx, vy = float(v[0]), float(v[1])
    p1 = (int(round(x0 + vx * length)), int(round(y0 + vy * length)))
    p0i = (int(round(x0)), int(round(y0)))
    cv2.arrowedLine(img, p0i, p1, color, 2, cv2.LINE_AA, tipLength=0.12)
    cv2.putText(img, label, (p1[0] + 6, p1[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _draw_line(img, v: np.ndarray, p0: np.ndarray, color):
    # draw a long segment through p0 along v
    h, w = img.shape[:2]
    L = float(max(h, w)) * 2.0
    a = (p0 - v * L).astype(np.float32)
    b = (p0 + v * L).astype(np.float32)
    ax, ay = int(round(float(a[0]))), int(round(float(a[1])))
    bx, by = int(round(float(b[0]))), int(round(float(b[1])))
    cv2.line(img, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)


def visualize_normalized_on_frame(frame_bgr: np.ndarray, norm: NormalizedObs) -> np.ndarray:
    """
    Show:
      - left col points (green), right col points (blue)
      - mean_xy (red)
      - main_dir (yellow arrow), lat_dir (magenta arrow)
      - fitted line for each column (same colors)
      - det_idx + (area,count,mad) small text
    """
    img = frame_bgr.copy()
    pts = np.asarray(norm.raw.points_px, np.float32)

    left_idx = [p.det_idx for p in norm.left_col]
    right_idx = [p.det_idx for p in norm.right_col]

    # fit lines for drawing
    fitL = _fit_line_cv(pts[left_idx])
    fitR = _fit_line_cv(pts[right_idx])
    if fitL is not None:
        vL, p0L = fitL
        _draw_line(img, vL, p0L, (0, 200, 0))
    if fitR is not None:
        vR, p0R = fitR
        _draw_line(img, vR, p0R, (200, 0, 0))

    # points
    for p in norm.left_col:
        x, y = p.px
        cv2.circle(img, (int(round(x)), int(round(y))), 7, (0, 255, 0), -1)
        cv2.putText(img, f"det{p.det_idx}", (int(round(x)) + 8, int(round(y)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    for p in norm.right_col:
        x, y = p.px
        cv2.circle(img, (int(round(x)), int(round(y))), 7, (255, 0, 0), -1)
        cv2.putText(img, f"det{p.det_idx}", (int(round(x)) + 8, int(round(y)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # mean
    mx, my = norm.mean_xy
    cv2.circle(img, (int(round(mx)), int(round(my))), 8, (0, 0, 255), -1)
    cv2.putText(img, "mean", (int(round(mx)) + 10, int(round(my)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # axes
    _draw_arrow(img, norm.mean_xy, norm.main_dir, 160.0, (0, 255, 255), "main")
    _draw_arrow(img, norm.mean_xy, norm.lat_dir, 120.0, (255, 0, 255), "lat")

    # header
    k = int(round(norm.split_debug.get("k", -1)))
    s = float(norm.split_debug.get("score", -1.0))
    eL = float(norm.split_debug.get("errL_px", -1.0))
    eR = float(norm.split_debug.get("errR_px", -1.0))
    cv2.putText(img, f"x-sweep split: k={k}  score={s:.2f}  errL={eL:.2f}px  errR={eR:.2f}px",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, f"left={left_idx}  right={right_idx}",
                (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    return img


# -----------------------------
# Self-test (real video via pole_detector)
# -----------------------------
def _read_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser("obs_normalizer.py (x-sweep split, no PCA)")
    ap.add_argument("--cfg", type=str, required=True, help="Path to b1_config.json")
    ap.add_argument("--video", type=str, required=True, help="Video path (single cam)")
    ap.add_argument("--show", action="store_true", help="visualize on first frame")
    args = ap.parse_args()

    # Import your existing cfg + detector (fail fast if not found)
    from b1_config import load_b1_config
    from pole_detector import YoloV5, PoleDetector

    cfg = load_b1_config(args.cfg)

    yolo = YoloV5.from_cfg(cfg.yolo)
    det = PoleDetector(yolo, cfg.pole_detector)

    poleset = det.run(args.video)
    dets = pole_setpx_to_detections(poleset)

    # Print raw detections (in their current order, no assumptions)
    print("\n=== Raw PoleDetections ===")
    for i, (p, a, c, m) in enumerate(zip(dets.points_px, dets.area, dets.count, dets.mad_px)):
        print(f"det{i}: px=({p[0]:.2f},{p[1]:.2f})  area={a:.1f}  count={c}  mad={m:.2f}px")
    print("meta:", dets.meta)

    # Normalize (x-sweep split)
    norm = normalize_obs_xsweep(
        dets,
        min_per_col=2,
        imbalance_penalty_px=2.0,
        robust="median",
    )

    print("\n=== NormalizedObs ===")
    print("split_debug:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in norm.split_debug.items()})
    print("mean_xy:", tuple(round(x, 3) for x in norm.mean_xy))
    print("main_dir:", tuple(round(x, 6) for x in norm.main_dir))
    print("lat_dir :", tuple(round(x, 6) for x in norm.lat_dir))

    print("\nleft_col (sorted by t):")
    for p in norm.left_col:
        print(f"  det{p.det_idx}: t={p.t:8.2f} s={p.s:8.2f}  px=({p.px[0]:.1f},{p.px[1]:.1f})"
              f"  area={p.area:.0f} cnt={p.count} mad={p.mad_px:.2f}")

    print("\nright_col (sorted by t):")
    for p in norm.right_col:
        print(f"  det{p.det_idx}: t={p.t:8.2f} s={p.s:8.2f}  px=({p.px[0]:.1f},{p.px[1]:.1f})"
              f"  area={p.area:.0f} cnt={p.count} mad={p.mad_px:.2f}")

    if args.show:
        frame0 = _read_first_frame(args.video)
        vis = visualize_normalized_on_frame(frame0, norm)
        cv2.imshow("obs_normalizer (x-sweep split)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
