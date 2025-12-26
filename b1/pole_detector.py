# pole_detector.py
# python .\pole_detector.py --cfg .\b1_config.json --video '..\正常跑1 20.6s\正常跑前视角-1.mp4' --show
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2


# -----------------------------
# Output data (UNORDERED)
# -----------------------------
@dataclass
class PoleDetections:
    """
    Unordered pole detections aggregated from a video clip.

    poles_px:
      M points (M<=n_poles), each is the estimated pole "bottom" point in pixel coords.

    NOTE:
      - This list is NOT ordered along track direction.
      - Downstream modules must do: column split / along-track ordering / id assignment.
      - We only apply a deterministic "stability sort" so repeated runs are reproducible.
        This sort has NO geometric meaning. Do not rely on index semantics.
    """
    poles_px: List[Tuple[float, float]]                 # length M
    pole_area: List[float]                              # length M, mean bbox area per cluster (px^2)
    pole_count: List[int]                               # length M, number of raw detections supporting the cluster
    pole_spread_mad_px: List[float]                     # length M, MAD of points in the cluster (px)
    meta: Dict[str, float]


# -----------------------------
# Minimal k-means (2D) with kmeans++ init
# -----------------------------
def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    N = X.shape[0]
    C = np.empty((K, X.shape[1]), dtype=np.float32)
    idx = rng.integers(0, N)
    C[0] = X[idx]
    d2 = np.sum((X - C[0]) ** 2, axis=1)
    for k in range(1, K):
        p = d2 / (d2.sum() + 1e-12)
        idx = rng.choice(N, p=p)
        C[k] = X[idx]
        d2 = np.minimum(d2, np.sum((X - C[k]) ** 2, axis=1))
    return C


def kmeans_2d(X: np.ndarray, K: int, iters: int = 60, seed: int = 2025) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, np.float32)
    rng = np.random.default_rng(seed)
    C = _kmeans_pp_init(X, K, rng)
    labels = np.zeros((X.shape[0],), np.int32)

    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(K):
            m = (labels == k)
            if m.any():
                C[k] = X[m].mean(axis=0)
            else:
                C[k] = X[rng.integers(0, X.shape[0])]
    return C.astype(np.float32), labels.astype(np.int32)


# -----------------------------
# YOLOv5 wrapper (torch hub)
# -----------------------------
class YoloV5:
    """
    Minimal YOLOv5 wrapper via torch.hub.

    - source="local": use local yolov5 repo (recommended)
    - source="remote": downloads from ultralytics/yolov5 (may require internet)
    """
    def __init__(
        self,
        weights: str,
        yolov5_repo: Optional[str] = None,
        source: str = "local",            # "local" | "remote"
        class_name: Optional[str] = "pole",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 960,
    ):
        import torch

        self.class_name = class_name
        self.imgsz = int(imgsz)

        if source == "local":
            if not yolov5_repo:
                raise ValueError("source=local 需要提供 yolov5_repo 路径")
            self.model = torch.hub.load(yolov5_repo, "custom", path=weights, source="local")
        elif source == "remote":
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)
        else:
            raise ValueError("source must be 'local' or 'remote'")

        self.model.conf = float(conf)
        self.model.iou = float(iou)
        self.model.max_det = 200
        self.names = getattr(self.model, "names", None)

    @classmethod
    def from_cfg(cls, yolo_cfg) -> "YoloV5":
        repo = getattr(yolo_cfg, "yolov5_repo", "")
        return cls(
            weights=getattr(yolo_cfg, "weights"),
            yolov5_repo=repo if repo else None,
            source=getattr(yolo_cfg, "source", "local"),
            class_name=getattr(yolo_cfg, "class_name", "pole"),
            conf=float(getattr(yolo_cfg, "conf", 0.25)),
            iou=float(getattr(yolo_cfg, "iou", 0.45)),
            imgsz=int(getattr(yolo_cfg, "imgsz", 960)),
        )

    def detect_xyxy(self, frame_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float, str]]:
        """Return list of (x1,y1,x2,y2,conf,name)."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=self.imgsz)

        dets = []
        try:
            df = results.pandas().xyxy[0]
            for _, r in df.iterrows():
                name = str(r.get("name", ""))
                if self.class_name and self.names and name != self.class_name:
                    continue
                x1, y1, x2, y2 = float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])
                conf = float(r["confidence"])
                dets.append((x1, y1, x2, y2, conf, name))
        except Exception:
            for *xyxy, conf, cls in results.xyxy[0].tolist():
                cls = int(cls)
                name = self.names[cls] if self.names else str(cls)
                if self.class_name and self.names and name != self.class_name:
                    continue
                x1, y1, x2, y2 = map(float, xyxy)
                dets.append((x1, y1, x2, y2, float(conf), name))
        return dets


# -----------------------------
# Pole detector (video -> unordered pole clusters)
# -----------------------------
class PoleDetector:
    """
    cfg 期望字段（直接对应 cfg.pole_detector）：
      - n_poles, frame_stride, seed, kmeans_iters
      - min_points_per_pole, max_mad_px
      - window_mode ("start"|"center"|"custom"), start_sec, sample_secs (<=0 means use init_secs), init_secs
    """
    def __init__(self, yolo: YoloV5, cfg):
        self.yolo = yolo
        self.cfg = cfg

    @staticmethod
    def _bottom_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        return ((x1 + x2) * 0.5, y2)

    @staticmethod
    def _area(x1: float, y1: float, x2: float, y2: float) -> float:
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    @staticmethod
    def _robust_center(pts: np.ndarray) -> np.ndarray:
        return np.median(pts, axis=0).astype(np.float32)

    @staticmethod
    def _stable_sort_indices(centers: np.ndarray, area_k: np.ndarray, count_k: np.ndarray) -> List[int]:
        """
        Deterministic "stability sort" only (NO geometric meaning):
          primary: higher count first (more supported)
          secondary: larger area first
          tertiary: x ascending
          quaternary: y ascending
        """
        items = []
        for i in range(centers.shape[0]):
            x, y = float(centers[i, 0]), float(centers[i, 1])
            items.append((i, -int(count_k[i]), -float(area_k[i]), x, y))
        items.sort(key=lambda t: (t[1], t[2], t[3], t[4]))
        return [t[0] for t in items]

    def run(self, video_path: str) -> PoleDetections:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1e-3 else 30.0

        window_mode = str(getattr(self.cfg, "window_mode", "start")).lower()
        sample_secs = float(getattr(self.cfg, "sample_secs", 0.0))
        if sample_secs <= 0:
            sample_secs = float(getattr(self.cfg, "init_secs", 5.0))

        start_sec = 0.0
        if window_mode == "start":
            start_sec = 0.0
        elif window_mode == "custom":
            start_sec = float(getattr(self.cfg, "start_sec", 0.0))
            start_sec = max(0.0, start_sec)
        elif window_mode == "center":
            frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_cnt and frame_cnt > 0:
                duration = float(frame_cnt) / float(fps)
                start_sec = max(0.0, 0.5 * (duration - sample_secs))
            else:
                start_sec = 0.0
        else:
            start_sec = 0.0

        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        max_frames = int(np.ceil(sample_secs * fps))

        pts_all: List[Tuple[float, float]] = []
        area_all: List[float] = []
        used_frames = 0
        total_frames_read = 0

        fi = 0
        stride = int(getattr(self.cfg, "frame_stride"))
        while fi < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames_read += 1

            if fi % stride != 0:
                fi += 1
                continue

            dets = self.yolo.detect_xyxy(frame)
            for x1, y1, x2, y2, conf, name in dets:
                pts_all.append(self._bottom_center(x1, y1, x2, y2))
                area_all.append(self._area(x1, y1, x2, y2))

            used_frames += 1
            fi += 1

        cap.release()

        if len(pts_all) < 4:
            raise RuntimeError(f"Too few detections: total_points={len(pts_all)} (<4)")

        pts = np.asarray(pts_all, np.float32)
        areas = np.asarray(area_all, np.float32)

        K = int(getattr(self.cfg, "n_poles"))
        iters = int(getattr(self.cfg, "kmeans_iters"))
        seed = int(getattr(self.cfg, "seed"))

        centers_init, labels = kmeans_2d(pts, K, iters=iters, seed=seed)

        centers = np.zeros_like(centers_init)
        area_k = np.zeros((K,), np.float32)
        count_k = np.zeros((K,), np.int32)
        mad_k = np.zeros((K,), np.float32)

        for k in range(K):
            m = (labels == k)
            count_k[k] = int(m.sum())
            if m.any():
                c = self._robust_center(pts[m])
                centers[k] = c
                area_k[k] = float(np.mean(areas[m]))
                d = np.linalg.norm(pts[m] - c[None, :], axis=1)
                mad_k[k] = float(np.median(d))
            else:
                centers[k] = centers_init[k]
                area_k[k] = 0.0
                mad_k[k] = 1e9

        # keep only reliable clusters -> output M<=K
        min_pts = int(getattr(self.cfg, "min_points_per_pole"))
        max_mad = float(getattr(self.cfg, "max_mad_px"))
        keep = [k for k in range(K) if (count_k[k] >= min_pts) and (mad_k[k] <= max_mad)]

        if len(keep) == 0:
            raise RuntimeError(
                "No reliable pole clusters found. "
                "Try larger sample_secs / lower yolo.conf / higher yolo.imgsz / relax min_points_per_pole,max_mad_px."
            )

        centers = centers[keep]
        area_k = area_k[keep]
        count_k = count_k[keep]
        mad_k = mad_k[keep]

        # stability sort only (no PCA, no geometry semantics)
        order = self._stable_sort_indices(centers, area_k, count_k)
        centers_s = centers[order]
        area_s = area_k[order]
        count_s = count_k[order]
        mad_s = mad_k[order]

        return PoleDetections(
            poles_px=[(float(x), float(y)) for x, y in centers_s],
            pole_area=[float(a) for a in area_s],
            pole_count=[int(c) for c in count_s],
            pole_spread_mad_px=[float(s) for s in mad_s],
            meta={
                "fps": float(fps),
                "window_start_sec": float(start_sec),
                "sample_secs": float(sample_secs),
                "frame_stride": float(stride),
                "used_frames": float(used_frames),
                "total_frames_read": float(total_frames_read),
                "total_points": float(len(pts_all)),
                "returned_poles": float(len(centers_s)),
                "min_points_per_pole": float(min_pts),
                "max_mad_px": float(max_mad),
                "imgsz": float(getattr(self.yolo, "imgsz", -1)),
            },
        )


# -----------------------------
# Visualization (optional, no file output)
# -----------------------------
def draw_poles(frame_bgr: np.ndarray, poles: PoleDetections) -> np.ndarray:
    vis = frame_bgr.copy()
    for i, (x, y) in enumerate(poles.poles_px):
        cv2.circle(vis, (int(round(x)), int(round(y))), 6, (0, 255, 0), -1)
        cv2.putText(vis, f"det{i}", (int(round(x)) + 8, int(round(y)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


# -----------------------------
# Self-test (cfg-driven, no file output)
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("pole_detector.py")
    ap.add_argument("--cfg", type=str, required=True, help="Path to b1_config.json")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    from b1_config import load_b1_config

    cfg = load_b1_config(args.cfg)

    yolo = YoloV5.from_cfg(cfg.yolo)
    det = PoleDetector(yolo, cfg.pole_detector)

    poles = det.run(args.video)

    print("PoleDetections:")
    print("  meta:", poles.meta)
    for i, (p, a, c, s) in enumerate(zip(
        poles.poles_px, poles.pole_area, poles.pole_count, poles.pole_spread_mad_px
    )):
        print(f"  det{i}: px=({p[0]:.2f},{p[1]:.2f})  area={a:.1f}  count={c}  mad={s:.2f}px")

    if args.show:
        cap = cv2.VideoCapture(args.video)
        ok, frame0 = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Cannot read first frame for visualization.")
        vis = draw_poles(frame0, poles)
        cv2.imshow("pole_detector", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
